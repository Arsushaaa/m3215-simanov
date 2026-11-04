"""Adversarial execution tests: model replies never establish tool authority."""

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.agent import RoutingContext
from app.llm import Completion, ModelError, RoutingCompletion
from app.mcp.client import MCPClientError
from app.mcp.contracts import tool_schemas
from app.repository import UserSettings
from app.routing.contracts import AgentRequest
from app.routing.policy import Budget, ScopedMCP
from app.routing.router import ROUTING_ERROR, Router
from app.routing.supervisor import Supervisor
from app.tracing.runtime import TraceManager, current_trace, span

NOW = datetime(2026, 11, 1, 10, tzinfo=UTC)
DAY = {
    "status": "ok",
    "date": "2026-11-01",
    "timezone": "Europe/Moscow",
    "lessons": [],
    "source": "local_schedule",
}
REASONS = {
    "schedule": "schedule_request",
    "lifestyle": "weather_request",
    "knowledge": "materials_request",
    "legacy_study": "general_study",
    "clarify": "ambiguous",
}


def choice(target, native=False):
    value = {"target": target, "reason": REASONS[target], "confidence": 0.9}
    return tool("select_route", value) if native else RoutingCompletion(json.dumps(value), [])


def tool(name, arguments):
    return RoutingCompletion(
        None,
        [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments),
                },
            }
        ],
    )


@pytest.fixture
def setup_supervisor(bot_settings):
    llm = SimpleNamespace(decide=AsyncMock(), complete=AsyncMock(return_value=Completion("Итог")))
    mcp = SimpleNamespace(
        available=True,
        tools={
            name: {"name": name, "description": name, **tool_schemas(name)}
            for name in ("get_weather", "get_schedule", "add_reminder")
        },
        resources={"schedule://current-week": {}},
        call_tool=AsyncMock(return_value=DAY),
        read_resource=AsyncMock(),
    )
    repository = SimpleNamespace(
        timezone=AsyncMock(return_value="Europe/Moscow"),
        start_request=AsyncMock(),
        has_processed=AsyncMock(return_value=False),
        record=AsyncMock(),
    )
    supervisor = Supervisor(llm, mcp, repository, bot_settings, now=lambda: NOW)
    return supervisor, llm, mcp, repository


async def ask(supervisor, text="Занятия на сегодня", **kwargs):
    return await supervisor.respond(
        UserSettings(9191919191, "study", 0.3, 0),
        1,
        text,
        prompt="You are Labbi.",
        history=[],
        **kwargs,
    )


class MemoryTraces:
    def __init__(self):
        self.headers, self.spans = {}, {}

    async def start(self, owner, trace_id, request_id, started):
        self.headers[trace_id] = {"owner": owner, "request_id": request_id, "status": "running"}

    async def span(self, value):
        self.spans[value.span_id] = value

    async def finish(self, identifier, status, finished):
        self.headers[identifier]["status"] = status


@pytest.mark.parametrize("native", [False, True])
async def test_schedule_routes_once_and_restricts_schema_and_model_context(
    setup_supervisor, native
):
    supervisor, llm, mcp, repo = setup_supervisor
    supervisor.settings = replace(
        supervisor.settings, llm_tool_mode="native" if native else "structured"
    )
    specialist = (
        tool("get_schedule", {"date": "2026-11-01"})
        if native
        else RoutingCompletion(
            '{"action":"call_tool","name":"get_schedule","arguments":{"date":"2026-11-01"},"reason":"Classes"}',
            [],
        )
    )
    llm.decide.side_effect = [choice("schedule", native), specialist]
    memory = MemoryTraces()
    async with TraceManager(memory).request(9191919191) as trace:
        reply = await ask(supervisor)
        assert reply.agent_result.request_id == trace.request_id
        assert reply.agent_result.trace_id == trace.trace_id
    assert reply.agent_result.status == "success"
    assert reply.agent_result.used_tools == ("get_schedule",)
    assert llm.decide.await_count == 2 and llm.complete.await_count == 1
    mcp.call_tool.assert_awaited_once_with("get_schedule", {"date": "2026-11-01"})
    payloads = json.dumps([c.args for c in llm.decide.call_args_list])
    assert "9191919191" not in payloads
    assert "get_weather" not in json.dumps(llm.decide.call_args_list[1].kwargs)
    assert "ScheduleAgent" in llm.decide.call_args_list[1].args[0][0]["content"]
    assert repo.record.call_args.kwargs["llm_calls"] == 2  # Specialist budget, no Router.
    spans = list(memory.spans.values())
    assert sum(s.operation.startswith("model_") for s in spans) == 3
    assert sum(s.operation == "tool_call" for s in spans) == 1
    assert all(s.status == "ok" for s in spans)
    assert all("Занятия" not in s.model_dump_json() for s in spans)
    final = llm.complete.call_args
    assert final.kwargs["disable_tools"] is True
    if native:
        assert any(m.get("tool_call_id") == "call_1" for m in final.args[0])


@pytest.mark.parametrize("target,calls", [("legacy_study", 2), ("knowledge", 1), ("clarify", 1)])
async def test_non_tool_routes_do_not_fall_back_to_combined_agent(setup_supervisor, target, calls):
    supervisor, llm, mcp, _ = setup_supervisor
    llm.decide.return_value = choice(target)
    reply = await ask(supervisor)
    assert llm.decide.await_count + llm.complete.await_count == calls
    mcp.call_tool.assert_not_awaited()
    mcp.read_resource.assert_not_awaited()
    if target == "knowledge":
        assert reply.agent_result.status == "failed"
        assert reply.agent_result.error_code == "SOURCE_UNAVAILABLE"
    if target == "legacy_study":
        assert llm.complete.call_args.kwargs["disable_tools"]


@pytest.mark.parametrize(
    "bad",
    [
        '{"target":"admin","reason":"ambiguous","confidence":1}',
        '{"target":"schedule","reason":"weather_request","confidence":1}',
        '{"target":"schedule","reason":"schedule_request","confidence":true}',
        '{"target":"schedule","reason":"schedule_request","confidence":NaN}',
        '{"target":"schedule","reason":"schedule_request","confidence":1,"actor_id":7}',
        '{"target":"schedule","target":"lifestyle","reason":"weather_request","confidence":1}',
        "[]",
        "null",
        "not JSON",
        "x" * 4097,
    ],
)
async def test_invalid_route_safe_clarification_no_retry_no_tool(setup_supervisor, bad):
    supervisor, llm, mcp, _ = setup_supervisor
    llm.decide.return_value = RoutingCompletion(bad, [])
    memory = MemoryTraces()
    async with TraceManager(memory).request(1):
        reply = await ask(supervisor)
        assert reply.text == ROUTING_ERROR
        assert reply.agent_result.status == "failed"
        assert reply.agent_result.error_code == "INVALID_ROUTE"
    llm.decide.assert_awaited_once()
    llm.complete.assert_not_awaited()
    mcp.call_tool.assert_not_awaited()
    assert any(s.error_code == "INVALID_ROUTE" for s in memory.spans.values())
    assert next(iter(memory.headers.values()))["status"] == "failed"


@pytest.mark.parametrize("native", [False, True])
async def test_lifestyle_cannot_prepare_or_revoke_card_using_model_output(setup_supervisor, native):
    supervisor, llm, mcp, _ = setup_supervisor
    supervisor.settings = replace(
        supervisor.settings, llm_tool_mode="native" if native else "structured"
    )
    llm.decide.side_effect = [
        choice("lifestyle", native),
        tool("cancel_reminder", {"reason": "Override"})
        if native
        else RoutingCompletion('{"action":"cancel_reminder","reason":"Override"}', []),
    ]
    with pytest.raises(ModelError):
        await ask(supervisor)
    mcp.call_tool.assert_not_awaited()
    llm.complete.assert_not_awaited()
    serialized = json.dumps(llm.decide.call_args_list[1].kwargs)
    assert "cancel_reminder" not in serialized and "get_schedule" not in serialized


@pytest.mark.parametrize(
    "role,name",
    [
        ("router", "get_weather"),
        ("knowledge", "get_weather"),
        ("lifestyle", "add_reminder"),
        ("lifestyle", "get_schedule"),
        ("schedule", "get_weather"),
        ("legacy_study", "get_schedule"),
        ("schedule", "run_shell"),
    ],
)
async def test_scoped_mcp_denies_actual_calls_even_if_catalog_advertises_them(
    setup_supervisor, role, name
):
    _, _, mcp, _ = setup_supervisor
    mcp.tools["run_shell"] = {"name": "run_shell", "description": "All rights"}
    proxy = ScopedMCP(mcp, role, budget=Budget())
    with pytest.raises(MCPClientError):
        await proxy.call_tool(name, {})
    mcp.call_tool.assert_not_awaited()
    assert "run_shell" not in proxy.tools


async def test_scope_rechecks_changed_discovery_at_execution_and_copies_catalog(setup_supervisor):
    _, _, mcp, _ = setup_supervisor
    proxy = ScopedMCP(mcp, "lifestyle")
    view = proxy.tools
    view["get_weather"]["name"] = "run_shell"
    assert proxy.tools["get_weather"]["name"] == "get_weather"
    del mcp.tools["get_weather"]
    with pytest.raises(MCPClientError):
        await proxy.call_tool("get_weather", {})
    mcp.call_tool.assert_not_awaited()


@pytest.mark.parametrize("role", ["schedule", "lifestyle", "knowledge", "router"])
async def test_query_scope_cannot_write_even_with_capability(setup_supervisor, role):
    _, _, mcp, _ = setup_supervisor
    with pytest.raises(MCPClientError):
        await ScopedMCP(mcp, role).call_tool(
            "add_reminder", {}, action_context={"action_id": str(uuid4()), "capability": "a" * 43}
        )
    mcp.call_tool.assert_not_awaited()


async def test_budget_is_reserved_before_fourth_llm_and_second_mcp(setup_supervisor):
    _, _, mcp, _ = setup_supervisor
    budget = Budget()
    for _ in range(3):
        budget.claim_llm()
    with pytest.raises(ModelError):
        budget.claim_llm()
    proxy = ScopedMCP(mcp, "schedule", budget=budget)
    await proxy.call_tool("get_schedule", {"date": "2026-11-01"})
    with pytest.raises(MCPClientError):
        await proxy.read_resource("schedule://current-week")
    mcp.call_tool.assert_awaited_once()
    mcp.read_resource.assert_not_awaited()


async def test_router_failure_revokes_pending_before_failed_audit_without_specialist(
    setup_supervisor,
):
    supervisor, llm, mcp, repo = setup_supervisor
    revised = UserSettings(9191919191, "study", 0.3, 2)
    service = SimpleNamespace(begin_edit=AsyncMock(return_value=(None, revised, 3)))
    context = RoutingContext(service, SimpleNamespace(), 2)
    llm.decide.side_effect = ModelError("Provider failed")
    with pytest.raises(ModelError):
        await ask(supervisor, confirmation_context=context)
    assert context.revoked and context.epoch == 3 and context.revision == 2
    service.begin_edit.assert_awaited_once()
    assert repo.record.call_args.kwargs["status"] == "FAILED"
    mcp.call_tool.assert_not_awaited()


async def test_cancellation_is_traced_and_context_is_restored():
    memory = MemoryTraces()
    with pytest.raises(asyncio.CancelledError):
        async with TraceManager(memory).request(1):
            async with span("router", "model_decide"):
                raise asyncio.CancelledError
    assert current_trace.get() is None
    assert all(s.status == "cancelled" for s in memory.spans.values())
    assert next(iter(memory.headers.values()))["status"] == "cancelled"


async def test_parallel_traces_have_separate_parents_and_owners():
    memory = MemoryTraces()

    async def run(owner):
        async with TraceManager(memory).request(owner):
            async with span("lifestyle", "agent"):
                await asyncio.sleep(0)
                async with span("lifestyle", "tool_call", tool_name="get_weather"):
                    await asyncio.sleep(0)

    await asyncio.gather(run(1), run(2))
    assert len(memory.headers) == 2
    for item in memory.spans.values():
        if item.parent_span_id:
            parent = memory.spans[item.parent_span_id]
            assert (parent.trace_id, parent.request_id) == (item.trace_id, item.request_id)


async def test_router_native_rejects_free_text_and_multiple_or_unknown_calls(bot_settings):
    request = AgentRequest(
        request_id=uuid4(),
        trace_id=uuid4(),
        actor_id=1,
        text="Привет",
        intent="user_query",
        occurred_at=NOW,
        timezone=None,
    )
    valid = choice("schedule", native=True)
    llm = SimpleNamespace(decide=AsyncMock())
    router = Router(llm, replace(bot_settings, llm_tool_mode="native"))
    for invalid in (
        RoutingCompletion("Ответ", []),
        replace(valid, truncated=True),
        replace(valid, tool_calls=valid.tool_calls * 2),
        replace(valid, text="Override"),
        tool("run_shell", {}),
    ):
        llm.decide.return_value = invalid
        assert await router.route(request, [], 0.3) is None


async def test_native_router_requires_function_and_has_no_execution_tools(bot_settings):
    from app.llm import OpenRouterClient

    llm = OpenRouterClient(bot_settings)
    llm._request = AsyncMock(return_value=choice("legacy_study", native=True))
    router = Router(llm, replace(bot_settings, llm_tool_mode="native"))
    request = AgentRequest(
        request_id=uuid4(),
        trace_id=uuid4(),
        actor_id=123,
        text="Чем список отличается от кортежа в Python?",
        intent="user_query",
        occurred_at=NOW,
        timezone="Europe/Moscow",
    )
    result = await router.route(request, [], 0.2)
    assert result.target == "legacy_study"
    payload = llm._request.call_args.args[0]
    assert payload["tool_choice"] == "required"
    assert [t["function"]["name"] for t in payload["tools"]] == ["select_route"]
    assert "actor_id" not in json.dumps(payload)
    llm._request.assert_awaited_once()
