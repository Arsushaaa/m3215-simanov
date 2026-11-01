"""Internal LR3 boundaries, not an A2A implementation or authorization proof.

Only RouteDecision is model-owned. The application constructs AgentRequest,
AgentResult and TraceSpan from trusted identity, validated facts and observed
operations. A valid DTO never establishes ownership or that an operation ran.
"""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from app.mcp.contracts import (
    AnswerText,
    AwareDateTime,
    ErrorCode,
    ScheduleResult,
    ShortText,
    TimezoneName,
    WeatherResult,
    WeekResult,
)

RouteTarget = Literal["schedule", "knowledge", "lifestyle", "legacy_study", "clarify"]
RouteReason = Literal[
    "schedule_request", "materials_request", "weather_request", "general_study", "ambiguous"
]
Component = Literal[
    "supervisor",
    "router",
    "schedule",
    "knowledge",
    "lifestyle",
    "legacy_study",
    "translate",
    "quiz",
    "briefing",
    "reminder_delivery",
    "telegram",
    "lifecycle",
]
MCPTool = Literal["get_schedule", "add_reminder", "get_weather", "search_materials"]
MCPResource = Literal["schedule://current-week", "materials://catalog"]
ResultError = (
    ErrorCode | Literal["INVALID_ROUTE", "POLICY_DENIED", "MODEL_UNAVAILABLE", "CANCELLED"]
)
ActorId = Annotated[int, Field(strict=True, gt=0, le=2**63 - 1)]
Confidence = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]
DocumentId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
ModelName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.:/-]{1,128}$")]
UserText = Annotated[str, StringConstraints(min_length=1, max_length=16000)]

_REASONS = {
    "schedule": "schedule_request",
    "knowledge": "materials_request",
    "lifestyle": "weather_request",
    "legacy_study": "general_study",
    "clarify": "ambiguous",
}


class ClosedModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, revalidate_instances="always"
    )


class RouteDecision(ClosedModel):
    target: RouteTarget
    reason: RouteReason
    confidence: Confidence

    @model_validator(mode="after")
    def reason_matches_target(self) -> "RouteDecision":
        if self.reason != _REASONS[self.target]:
            raise ValueError("Route reason does not match the target.")
        return self


class AgentRequest(ClosedModel):
    request_id: UUID
    trace_id: UUID
    actor_id: ActorId = Field(repr=False)
    text: UserText = Field(repr=False)
    intent: Literal["user_query", "daily_briefing"]
    occurred_at: AwareDateTime
    timezone: TimezoneName | None

    @field_validator("text")
    @classmethod
    def nonblank_original_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Request text must not be blank.")
        return value  # Evidence must refer to the original, untrimmed user text.

    def model_context(self) -> dict:
        """Explicit projection; do not serialize host DTOs wholesale to a model."""
        return {
            "text": self.text,
            "intent": self.intent,
            "occurred_at": self.occurred_at.isoformat(),
            "timezone": self.timezone,
        }


class EmptyFacts(ClosedModel):
    kind: Literal["empty"] = "empty"


class _MCPFacts(ClosedModel):
    @field_validator("data", mode="before", check_fields=False)
    @classmethod
    def reconstruct_existing_models(cls, value):
        # Legacy MCP models do not revalidate instances. Reconstruct even an
        # existing instance so unsafe model_construct objects cannot pass as facts.
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json", warnings=False)
        return value


class ScheduleFacts(_MCPFacts):
    kind: Literal["schedule_day"] = "schedule_day"
    data: ScheduleResult = Field(repr=False)


class WeekFacts(_MCPFacts):
    kind: Literal["schedule_week"] = "schedule_week"
    data: WeekResult = Field(repr=False)


class WeatherFacts(_MCPFacts):
    kind: Literal["weather"] = "weather"
    data: WeatherResult = Field(repr=False)


class MaterialMatch(ClosedModel):
    document_id: DocumentId
    title: ShortText
    version: Digest
    snippet: Annotated[str, StringConstraints(min_length=1, max_length=1200)] = Field(repr=False)
    score: Confidence


class KnowledgeFacts(ClosedModel):
    kind: Literal["knowledge"] = "knowledge"
    catalog_version: Digest
    matches: Annotated[tuple[MaterialMatch, ...], Field(max_length=5)]


AgentFacts = Annotated[
    EmptyFacts | ScheduleFacts | WeekFacts | WeatherFacts | KnowledgeFacts,
    Field(discriminator="kind"),
]


class AgentResult(ClosedModel):
    request_id: UUID
    trace_id: UUID
    status: Literal[
        "success", "needs_clarification", "needs_confirmation", "not_found", "failed", "cancelled"
    ]
    message: AnswerText = Field(repr=False)
    confidence: Confidence | None = None
    facts: AgentFacts = Field(repr=False)
    used_tools: Annotated[tuple[MCPTool, ...], Field(max_length=1)] = ()
    used_resources: Annotated[tuple[MCPResource, ...], Field(max_length=1)] = ()
    error_code: ResultError | None = None
    pending_action_id: UUID | None = Field(default=None, repr=False)

    @model_validator(mode="after")
    def consistent_result(self) -> "AgentResult":
        if len(self.used_tools) + len(self.used_resources) > 1:
            raise ValueError("One specialist result cannot contain multiple MCP calls.")
        if self.status == "failed" and self.error_code is None:
            raise ValueError("Failed results need a fixed error code.")
        if self.status in {"success", "needs_confirmation"} and self.error_code is not None:
            raise ValueError("A successful result cannot carry an error.")
        if (self.status == "needs_confirmation") != (self.pending_action_id is not None):
            raise ValueError("Only a confirmation result references a pending action.")
        if isinstance(self.facts, KnowledgeFacts):
            expected = "success" if self.facts.matches else "not_found"
            if self.status != expected:
                raise ValueError("Knowledge status must match search results.")
        elif isinstance(self.facts, _MCPFacts):
            raw = self.facts.data.root
            expected = {
                "ok": "success",
                "ambiguous": "needs_clarification",
                "not_found": "not_found",
                "outside_coverage": "not_found",
                "error": "failed",
            }[getattr(raw, "status", "ok")]
            if self.status != expected:
                raise ValueError("Status must match validated MCP facts.")
            if expected == "failed" and self.error_code != raw.error.code:
                raise ValueError("Error code must match validated MCP facts.")
        return self


class TraceSpan(ClosedModel):
    trace_id: UUID
    span_id: UUID
    parent_span_id: UUID | None
    request_id: UUID
    component: Component
    operation: Literal[
        "receive",
        "route",
        "agent",
        "model_decide",
        "model_final",
        "tool_call",
        "resource_read",
        "prepare_notification",
        "claim_notification",
        "send_message",
        "command",
    ]
    status: Literal["running", "ok", "failed", "cancelled", "interrupted"]
    started_at: AwareDateTime
    finished_at: AwareDateTime | None
    duration_ms: NonnegativeInt | None
    target: RouteTarget | None = None
    confidence: Confidence | None = None
    tool_name: MCPTool | None = None
    resource_uri: MCPResource | None = None
    model_name: ModelName | None = None
    input_tokens: NonnegativeInt | None = None
    output_tokens: NonnegativeInt | None = None
    error_code: ResultError | None = None
    document_ids: Annotated[tuple[DocumentId, ...], Field(max_length=5)] = ()
    cache_hit: bool | None = None

    @model_validator(mode="after")
    def lifecycle(self) -> "TraceSpan":
        if self.parent_span_id == self.span_id:
            raise ValueError("A span cannot be its own parent.")
        running = self.status == "running"
        if running != (self.finished_at is None and self.duration_ms is None):
            raise ValueError("Only a running span has no finish/duration.")
        if not running and (self.finished_at is None or self.duration_ms is None):
            raise ValueError("Finished spans require finish and duration.")
        if self.finished_at is not None and self.finished_at < self.started_at:
            raise ValueError("A span cannot finish before it starts.")
        if self.status == "ok" and self.error_code is not None:
            raise ValueError("Successful spans cannot carry an error.")
        return self
