"""Host-selected MCP scopes; discovery cannot grant new authority."""

from copy import deepcopy

from app.agent_repository import valid_capability_context
from app.llm import ModelError
from app.mcp.client import MCPClientError
from app.tracing.runtime import span

TOOLS = {
    "schedule": frozenset({"get_schedule", "add_reminder"}),
    "lifestyle": frozenset({"get_weather"}),
    "knowledge": frozenset({"search_materials"}),
}
RESOURCES = {
    "schedule": frozenset({"schedule://current-week"}),
    "knowledge": frozenset({"materials://catalog"}),
}
DENIED = "Выбранное действие недоступно этому специалисту. Уточни запрос."


class PolicyDenied(ModelError):
    trace_code = "POLICY_DENIED"

    def __init__(self):
        super().__init__(DENIED)


class Budget:
    def __init__(self):
        self.llm_calls = 0
        self.mcp_calls = 0

    def claim_llm(self):
        if self.llm_calls >= 3:
            raise ModelError("Достигнут предел обращений к модели.")
        self.llm_calls += 1

    def claim_mcp(self):
        if self.mcp_calls >= 1:
            raise MCPClientError("UNAUTHORIZED", "Достигнут предел обращений к инструментам.")
        self.mcp_calls += 1


class ScopedMCP:
    def __init__(self, mcp, component, *, budget=None, confirmation=False):
        if confirmation and component != "schedule":
            raise ValueError("Only a Schedule callback may write")
        self._mcp, self.component = mcp, component
        self.budget, self.confirmation = budget, confirmation
        self.observation = None

    @property
    def available(self):
        return self._mcp.available

    @property
    def tools(self):
        return deepcopy(
            {
                k: v
                for k, v in self._mcp.tools.items()
                if k in TOOLS.get(self.component, ()) and v.get("name") == k
            }
        )

    @property
    def resources(self):
        return deepcopy(
            {k: v for k, v in self._mcp.resources.items() if k in RESOURCES.get(self.component, ())}
        )

    def allows_decision(self, decision):
        # Preparation/control adapters also need authority, even without an MCP call.
        action = decision.action
        if action in {"clarify_reminder", "use_confirmation", "cancel_reminder"}:
            return self.component == "schedule"
        if action == "call_tool":
            return decision.name in self.tools
        if action == "read_resource":
            return decision.uri in self.resources
        return action in {"respond", "clarify"}

    def decision_schema(self, schema):
        allowed = {"RespondDecision", "ClarifyDecision"}
        if self.component == "schedule":
            allowed |= {
                "ScheduleDecision",
                "ResourceDecision",
                "ReminderDecision",
                "ClarifyReminderDecision",
                "ConfirmReminderDecision",
                "CancelReminderDecision",
            }
        elif self.component == "lifestyle":
            allowed.add("WeatherDecision")
        result = deepcopy(schema)
        result.pop("discriminator", None)
        result["oneOf"] = [{"$ref": f"#/$defs/{name}"} for name in sorted(allowed)]
        definitions = result["$defs"]
        reachable = set()

        def visit(value):
            if isinstance(value, list):
                for item in value:
                    visit(item)
            elif isinstance(value, dict):
                reference = value.get("$ref", "")
                if reference.startswith("#/$defs/"):
                    name = reference.removeprefix("#/$defs/")
                    if name not in reachable:
                        reachable.add(name)
                        visit(definitions[name])
                for item in value.values():
                    visit(item)

        visit(result["oneOf"])
        result["$defs"] = {name: definitions[name] for name in sorted(reachable)}
        return result

    async def call_tool(self, name, arguments, **kwargs):
        allowed = name in self.tools and (name != "add_reminder" or self.confirmation)
        if name == "add_reminder":
            allowed = allowed and valid_capability_context(kwargs.get("action_context"))
        if not allowed:
            await self._deny()
        if self.budget is not None:
            self.budget.claim_mcp()
        async with span(self.component, "tool_call", tool_name=name):
            result = await self._mcp.call_tool(name, arguments, **kwargs)
            self.observation = ("tool", name, deepcopy(result))
            return result

    async def read_resource(self, uri):
        if uri not in self.resources or self.confirmation:
            await self._deny()
        if self.budget is not None:
            self.budget.claim_mcp()
        async with span(self.component, "resource_read", resource_uri=uri):
            result = await self._mcp.read_resource(uri)
            self.observation = ("resource", uri, deepcopy(result))
            return result

    async def _deny(self):
        async with span(
            self.component if self.component in TOOLS else "supervisor", "agent"
        ) as metadata:
            metadata.update(status="failed", error_code="POLICY_DENIED")
            raise MCPClientError("UNAUTHORIZED", DENIED)
