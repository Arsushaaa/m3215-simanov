"""Stage1: LR3 boundary contracts only; no Router execution or external APIs."""

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError
from test_mcp_contracts import SCHEDULE, WEATHER

from app.mcp.contracts import WeatherOk, WeatherResult
from app.routing.contracts import (
    AgentRequest,
    AgentResult,
    EmptyFacts,
    KnowledgeFacts,
    MaterialMatch,
    RouteDecision,
    ScheduleFacts,
    TraceSpan,
    WeatherFacts,
)

NOW = datetime(2026, 11, 1, 9, tzinfo=UTC)


def request(**changes):
    return {
        "request_id": uuid4(),
        "trace_id": uuid4(),
        "actor_id": 7,
        "text": "  Не забудь напомнить про отчёт  ",
        "intent": "user_query",
        "occurred_at": NOW,
        "timezone": "Europe/Moscow",
        **changes,
    }


def result(**changes):
    return {
        "request_id": uuid4(),
        "trace_id": uuid4(),
        "status": "success",
        "message": "Проверенный ответ",
        "facts": EmptyFacts(),
        **changes,
    }


def span(**changes):
    return {
        "trace_id": uuid4(),
        "span_id": uuid4(),
        "parent_span_id": None,
        "request_id": uuid4(),
        "component": "router",
        "operation": "route",
        "status": "running",
        "started_at": NOW,
        "finished_at": None,
        "duration_ms": None,
        **changes,
    }


@pytest.mark.parametrize(
    "target,reason",
    [
        ("schedule", "schedule_request"),
        ("knowledge", "materials_request"),
        ("lifestyle", "weather_request"),
        ("legacy_study", "general_study"),
        ("clarify", "ambiguous"),
    ],
)
def test_router_has_closed_targets_and_fixed_nonprivate_reasons(target, reason):
    dto = RouteDecision(target=target, reason=reason, confidence=0.8)
    assert RouteDecision.model_validate_json(dto.model_dump_json()) == dto
    Draft202012Validator(dto.model_json_schema()).validate(dto.model_dump(mode="json"))


@pytest.mark.parametrize(
    "extra",
    [
        {"actor_id": 99},
        {"trace_id": str(uuid4())},
        {"destination_chat_id": 99},
        {"tool": "add_reminder"},
        {"intent": "daily_briefing"},
        {"capability": "TEST-SECRET"},
    ],
)
def test_router_cannot_return_host_identity_delivery_or_authority_fields(extra):
    with pytest.raises(ValidationError):
        RouteDecision.model_validate(
            {
                "target": "schedule",
                "reason": "schedule_request",
                "confidence": 1.0,
                **extra,
            }
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"target": "administrator"},
        {"reason": "TEST-PRIVATE-MESSAGE"},
        {"target": "schedule", "reason": "weather_request"},
        {"confidence": True},
        {"confidence": "0.9"},
        {"confidence": -0.1},
        {"confidence": 1.1},
        {"confidence": float("inf")},
        {"confidence": float("nan")},
    ],
)
def test_router_rejects_unknown_or_inconsistent_decisions(changes):
    with pytest.raises(ValidationError):
        RouteDecision.model_validate(
            {
                "target": "knowledge",
                "reason": "materials_request",
                "confidence": 0.9,
                **changes,
            }
        )


def test_request_keeps_original_evidence_and_has_explicit_model_projection():
    source = request()
    dto = AgentRequest.model_validate(source)
    assert dto.text == source["text"]
    assert dto.model_context() == {
        "text": source["text"],
        "intent": "user_query",
        "occurred_at": NOW.isoformat(),
        "timezone": "Europe/Moscow",
    }
    assert (
        not {"actor_id", "request_id", "trace_id", "destination_chat_id"}
        & dto.model_context().keys()
    )
    assert source["text"] not in repr(dto) and "actor_id" not in repr(dto)
    assert AgentRequest.model_validate_json(dto.model_dump_json()) == dto
    assert AgentRequest.model_validate(request(timezone=None)).timezone is None


@pytest.mark.parametrize(
    "changes",
    [
        {"actor_id": True},
        {"actor_id": "7"},
        {"actor_id": 0},
        {"actor_id": 2**63},
        {"text": "   "},
        {"text": "x" * 16001},
        {"intent": "new_system_job"},
        {"occurred_at": NOW.replace(tzinfo=None)},
        {"timezone": "localtime"},
        {"timezone": "Not/AZone"},
        {"destination_chat_id": 999},
    ],
)
def test_request_rejects_bad_identity_time_and_hidden_delivery_fields(changes):
    with pytest.raises(ValidationError):
        AgentRequest.model_validate(request(**changes))


def test_discriminated_result_reuses_checked_mcp_facts():
    dto = AgentResult.model_validate(
        result(
            facts=WeatherFacts(data=WEATHER),
            used_tools=("get_weather",),
            confidence=None,
        )
    )
    assert dto.facts.kind == "weather" and dto.facts.data.root.temperature_c == 8.5
    assert "Проверенный ответ" not in repr(dto)
    assert AgentResult.model_validate_json(dto.model_dump_json()) == dto
    Draft202012Validator(dto.model_json_schema()).validate(dto.model_dump(mode="json"))
    empty_day = AgentResult.model_validate(result(facts=ScheduleFacts(data=SCHEDULE)))
    assert empty_day.status == "success" and not empty_day.facts.data.root.lessons


@pytest.mark.parametrize(
    "change",
    [
        {"status": "failed", "error_code": None},
        {"error_code": "UPSTREAM_ERROR"},
        {"facts": {"kind": "unknown"}},
        {"used_tools": ("arbitrary_python",)},
        {"used_tools": ("get_weather", "add_reminder")},
        {"used_resources": ("file:///etc/passwd",)},
        {"used_tools": ("search_materials",), "used_resources": ("materials://catalog",)},
        {"status": "needs_confirmation"},
        {"pending_action_id": uuid4()},
        {"confidence": True},
        {"destination_chat_id": 99},
    ],
)
def test_result_rejects_untyped_facts_false_success_and_unbounded_operations(change):
    with pytest.raises(ValidationError):
        AgentResult.model_validate(result(**change))


def test_result_status_is_derived_from_weather_and_schedule_outcome():
    ambiguous = {
        "status": "ambiguous",
        "query": "Ростов",
        "source": "Open-Meteo",
        "candidates": [
            {"name": "Ростов-на-Дону", "country": "Россия", "region": "Ростовская область"},
            {"name": "Ростов Великий", "country": "Россия", "region": "Ярославская область"},
        ],
    }
    with pytest.raises(ValidationError):
        AgentResult.model_validate(result(facts=WeatherFacts(data=ambiguous)))
    dto = AgentResult.model_validate(
        result(
            status="needs_clarification",
            facts=WeatherFacts(data=ambiguous),
        )
    )
    assert dto.status == "needs_clarification"
    error = {
        "status": "error",
        "error": {"code": "UPSTREAM_ERROR", "message": "Недоступно", "retryable": True},
    }
    with pytest.raises(ValidationError):
        AgentResult.model_validate(
            result(
                status="failed",
                facts=WeatherFacts(data=error),
                error_code="TIMEOUT",
            )
        )
    assert (
        AgentResult.model_validate(
            result(
                status="failed",
                facts=WeatherFacts(data=error),
                error_code="UPSTREAM_ERROR",
            )
        ).status
        == "failed"
    )


def test_pending_proposal_is_not_reported_as_an_already_saved_reminder():
    dto = AgentResult.model_validate(result(status="needs_confirmation", pending_action_id=uuid4()))
    assert dto.pending_action_id is not None and not dto.used_tools


def test_existing_unsafe_mcp_model_instances_are_reconstructed_and_validated():
    unsafe = WeatherResult.model_construct(
        root=WeatherOk.model_construct(
            **{**deepcopy(WEATHER), "temperature_c": "NOT-A-NUMBER"},
        )
    )
    with pytest.raises(ValidationError):
        WeatherFacts(data=unsafe)
    unsafe_route = RouteDecision.model_construct(
        target="administrator",
        reason="schedule_request",
        confidence=0.9,
    )
    with pytest.raises(ValidationError):
        RouteDecision.model_validate(unsafe_route)


def test_knowledge_facts_require_bounded_named_sources_and_do_not_invent_a_match():
    facts = KnowledgeFacts(catalog_version="a" * 64, matches=())
    with pytest.raises(ValidationError):
        AgentResult.model_validate(result(facts=facts))
    assert AgentResult.model_validate(result(status="not_found", facts=facts)).status == "not_found"
    match = MaterialMatch(
        document_id="mcp_basics",
        title="Основы MCP",
        version="b" * 64,
        snippet="TEST-PRIVATE-DOCUMENT-FRAGMENT",
        score=0.7,
    )
    assert "TEST-PRIVATE" not in repr(match)
    assert KnowledgeFacts(catalog_version="a" * 64, matches=(match,)).matches[0] == match
    with pytest.raises(ValidationError):
        KnowledgeFacts(catalog_version="a" * 64, matches=(match,) * 6)
    with pytest.raises(ValidationError):
        MaterialMatch(
            document_id="../../etc/passwd",
            title="X",
            version="b" * 64,
            snippet="X",
            score=0.7,
        )


@pytest.mark.parametrize(
    "change",
    [
        {"prompt": "TEST-PRIVATE"},
        {"text": "TEST-PRIVATE"},
        {"actor_id": 7},
        {"attributes": {"content": "TEST-PRIVATE"}},
        {"tool_name": "delete_everything"},
        {"resource_uri": "file:///etc/passwd"},
        {"input_tokens": True},
        {"input_tokens": -1},
        {"duration_ms": 0},
        {"finished_at": NOW},
        {"status": "ok"},
        {"component": "__import__"},
    ],
)
def test_trace_has_closed_safe_fields_and_checked_running_state(change):
    with pytest.raises(ValidationError):
        TraceSpan.model_validate(span(**change))


def test_trace_finished_lifecycle_tokens_and_parent_are_checked():
    dto = TraceSpan.model_validate(
        span(
            status="ok",
            finished_at=NOW + timedelta(seconds=1),
            duration_ms=1000,
            target="lifestyle",
            input_tokens=None,
            output_tokens=None,
        )
    )
    assert dto.input_tokens is None  # Missing provider usage is not fabricated as zero.
    assert TraceSpan.model_validate_json(dto.model_dump_json()) == dto
    for change in (
        {"finished_at": NOW - timedelta(seconds=1)},
        {"error_code": "TIMEOUT"},
        {"parent_span_id": dto.span_id},
        {"output_tokens": float("inf")},
    ):
        with pytest.raises(ValidationError):
            TraceSpan.model_validate({**dto.model_dump(), **change})
