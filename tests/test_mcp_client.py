"""Граница клиента: строгий discovery, один вызов и владение stdio-контекстом."""

import asyncio
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

import pytest
from mcp import types
from mcp.shared.exceptions import MCPError

from app.mcp import client as module
from app.mcp.client import MCPClient, MCPClientError
from app.mcp.contracts import tool_schemas
from app.mcp.schedule import ScheduleService

DAY = {
    "status": "ok",
    "date": "2026-10-18",
    "timezone": "Europe/Moscow",
    "lessons": [],
    "source": "local_schedule",
}
REMINDER_ARGUMENTS = {"text": "Отправить отчёт", "remind_at": "2030-01-01T10:00:00+03:00"}
ACTION_CONTEXT = {
    "action_id": "a1cf8b47-b6fb-4e95-9931-b21dd239b782",
    "capability": "a" * 43,
}
REMINDER_RECEIPT = {
    "status": "ok",
    "reminder_id": "f4b780fd-4c99-42be-8cab-30ffb8502f5e",
    "remind_at": REMINDER_ARGUMENTS["remind_at"],
    "timezone": "Europe/Moscow",
    "created": True,
}


def tool_result(value, *, is_error=False):
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(value))],
        structuredContent=value,
        isError=is_error,
    )


def catalog():
    return types.ListToolsResult(
        tools=[
            types.Tool(
                name=name,
                description="Проверенный учебный инструмент.",
                **tool_schemas(name),
                annotations=types.ToolAnnotations(readOnlyHint=name != "add_reminder"),
            )
            for name in ("get_weather", "get_schedule", "add_reminder")
        ]
    )


class FakeSDK:
    def __init__(self):
        self.session = self
        self.owner = None
        self.exited = False
        self.list_tools = AsyncMock(return_value=catalog())
        self.list_resources = AsyncMock(
            return_value=types.ListResourcesResult(
                resources=[
                    types.Resource(uri=module.WEEK_URI, name="week", mimeType="application/json")
                ]
            )
        )
        self.call_tool = AsyncMock(return_value=tool_result(DAY))
        service = ScheduleService.from_file(module.ROOT / "data/schedule.json", "Europe/Moscow")
        week = service.current_week(now=datetime(2026, 10, 18, tzinfo=UTC)).model_dump_json()
        self.read_resource = AsyncMock(
            return_value=types.ReadResourceResult(
                contents=[
                    types.TextResourceContents(
                        uri=module.WEEK_URI, mimeType="application/json", text=week
                    )
                ]
            )
        )

    async def __aenter__(self):
        self.owner = asyncio.current_task()
        return self

    async def __aexit__(self, *_args):
        assert asyncio.current_task() is self.owner
        self.exited = True


@pytest.fixture
def setup_client(bot_settings, monkeypatch):
    sdk = FakeSDK()
    factory = Mock(return_value=sdk)
    monkeypatch.setattr(module, "Client", factory)
    return MCPClient(bot_settings), sdk, factory


async def test_discovery_is_canonical_and_child_environment_excludes_bot_secrets(setup_client):
    client, sdk, factory = setup_client
    await asyncio.gather(client.start(), client.start())
    try:
        assert client.available
        assert set(client.tools) == {"get_weather", "get_schedule", "add_reminder"}
        assert set(client.resources) == {module.WEEK_URI}
        assert (
            client.tools["get_weather"]["inputSchema"] == tool_schemas("get_weather")["inputSchema"]
        )
        changed = client.tools
        changed["get_weather"]["inputSchema"].clear()
        assert client.tools["get_weather"]["inputSchema"]
        parameters = factory.call_args.args[0]
        assert parameters.command == sys.executable
        assert parameters.args == ["-m", "app.mcp.server"]
        assert parameters.cwd == module.ROOT
        assert "BOT_TOKEN" not in parameters.env
        assert "OPENROUTER_API_KEY" not in parameters.env
        assert "TELEGRAM_PROXY_URL" not in parameters.env
        assert parameters.env["SCHEDULE_FILE"] == "data/schedule.json"
        assert factory.call_args.kwargs["cache"] is None
        factory.assert_called_once()
    finally:
        await asyncio.create_task(client.close())
    assert sdk.exited
    assert not client.available


async def test_start_failure_is_nonfatal_and_never_retried(bot_settings, monkeypatch, caplog):
    factory = Mock(side_effect=OSError("private-key connection failed"))
    monkeypatch.setattr(module, "Client", factory)
    client = MCPClient(bot_settings)
    await client.start()
    await client.start()
    await client.close()
    assert not client.available
    assert client.tools == client.resources == {}
    factory.assert_called_once()
    assert "private-key" not in caplog.text
    with pytest.raises(MCPClientError, match="недоступны"):
        await client.read_resource(module.WEEK_URI)


@pytest.mark.parametrize(
    "defect",
    ["missing-tool", "extra-tool", "input-schema", "remote-ref", "read-only", "mime", "pagination"],
)
async def test_unexpected_catalog_disables_tools(setup_client, defect):
    client, sdk, _factory = setup_client
    listing = sdk.list_tools.return_value
    if defect == "missing-tool":
        listing.tools.pop()
    elif defect == "extra-tool":
        listing.tools.append(listing.tools[0].model_copy(update={"name": "run_shell"}))
    elif defect == "input-schema":
        listing.tools[0].input_schema["additionalProperties"] = True
    elif defect == "remote-ref":
        listing.tools[0].output_schema = {"$ref": "https://private-key.example/schema"}
    elif defect == "read-only":
        listing.tools[0].annotations.read_only_hint = False
    elif defect == "mime":
        sdk.list_resources.return_value.resources[0].mime_type = "text/plain"
    else:
        listing.next_cursor = "more"
    await client.start()
    await client.close()
    assert not client.available
    assert client.tools == client.resources == {}
    sdk.call_tool.assert_not_awaited()


@pytest.mark.parametrize(
    "name,args,code",
    [
        ("run_shell", {}, "UNKNOWN_TOOL"),
        ("get_weather", {"city": "   "}, "INVALID_ARGUMENTS"),
        (
            "get_weather",
            {"city": "Казань", "url": "https://private-key.example"},
            "INVALID_ARGUMENTS",
        ),
        ("get_schedule", {"date": "2026-02-30"}, "INVALID_ARGUMENTS"),
    ],
)
async def test_invalid_calls_never_reach_sdk(setup_client, name, args, code):
    client, sdk, _factory = setup_client
    await client.start()
    try:
        with pytest.raises(MCPClientError) as exc:
            await client.call_tool(name, args)
        assert exc.value.code == code
        assert "private-key" not in str(exc.value)
        sdk.call_tool.assert_not_awaited()
        with pytest.raises(MCPClientError):
            await client.read_resource("file:///private-key/data.json")
        sdk.read_resource.assert_not_awaited()
    finally:
        await client.close()


async def test_one_call_and_one_read_return_validated_json(setup_client):
    client, sdk, _factory = setup_client
    await client.start()
    try:
        assert await client.call_tool("get_schedule", {"date": "2026-10-18"}) == DAY
        week = await client.read_resource(module.WEEK_URI)
        assert week["week_start"] == "2026-10-12"
        assert week["week_end"] == "2026-10-18"
        assert len(week["days"]) == 7
        sdk.call_tool.assert_awaited_once_with(
            "get_schedule", {"date": "2026-10-18"}, allow_input_required=True
        )
        sdk.read_resource.assert_awaited_once_with(module.WEEK_URI, allow_input_required=True)
    finally:
        await client.close()


@pytest.mark.parametrize(
    "defect", ["missing-data", "extra-data", "date", "flag", "size", "input-required"]
)
async def test_invalid_tool_results_are_not_returned_as_facts(setup_client, defect):
    client, sdk, _factory = setup_client
    if defect == "missing-data":
        sdk.call_tool.return_value = types.CallToolResult(content=[])
    elif defect == "extra-data":
        sdk.call_tool.return_value = tool_result({**DAY, "private-key": "hidden"})
    elif defect == "date":
        sdk.call_tool.return_value = tool_result({**DAY, "date": "2026-02-30"})
    elif defect == "flag":
        sdk.call_tool.return_value = tool_result(DAY, is_error=True)
    elif defect == "size":
        sdk.call_tool.return_value.content[0].text = "x" * 32768
    else:
        sdk.call_tool.return_value = types.InputRequiredResult(requestState="opaque-state")
    await client.start()
    try:
        with pytest.raises(MCPClientError) as exc:
            await client.call_tool("get_schedule", {"date": "2026-10-18"})
        assert exc.value.code == "INVALID_RESULT"
        assert "private-key" not in str(exc.value)
        sdk.call_tool.assert_awaited_once()
    finally:
        await client.close()


@pytest.mark.parametrize(
    "defect", ["uri", "mime", "json", "duplicate-key", "size", "input-required"]
)
async def test_invalid_resource_results_are_rejected_without_retry(setup_client, defect):
    client, sdk, _factory = setup_client
    result = sdk.read_resource.return_value
    content = result.contents[0]
    if defect == "uri":
        content.uri = "file:///private-key"
    elif defect == "mime":
        content.mime_type = "text/plain"
    elif defect == "json":
        content.text = "private-key invalid JSON"
    elif defect == "duplicate-key":
        content.text = content.text.replace("{", '{"week_start":"private-key",', 1)
    elif defect == "size":
        content.text = "x" * 32768
    else:
        sdk.read_resource.return_value = types.InputRequiredResult(requestState="opaque-state")
    await client.start()
    try:
        with pytest.raises(MCPClientError) as exc:
            await client.read_resource(module.WEEK_URI)
        assert exc.value.code == "INVALID_RESULT"
        assert "private-key" not in str(exc.value)
        sdk.read_resource.assert_awaited_once()
    finally:
        await client.close()


async def test_domain_error_keeps_code_but_does_not_echo_server_message(setup_client):
    client, sdk, _factory = setup_client
    sdk.call_tool.return_value = tool_result(
        {
            "status": "error",
            "error": {
                "code": "SOURCE_UNAVAILABLE",
                "message": "private-key upstream details",
                "retryable": True,
            },
        },
        is_error=True,
    )
    await client.start()
    try:
        with pytest.raises(MCPClientError) as exc:
            await client.call_tool("get_schedule", {"date": "2026-10-18"})
        assert exc.value.code == "SOURCE_UNAVAILABLE"
        assert "private-key" not in exc.value.user_message
        assert client.available
    finally:
        await client.close()


async def test_header_mismatch_does_not_trigger_hidden_discovery_or_retry(setup_client):
    client, sdk, _factory = setup_client
    sdk.call_tool.side_effect = MCPError(types.HEADER_MISMATCH, "private-key")
    await client.start()
    with pytest.raises(MCPClientError) as exc:
        await client.call_tool("get_schedule", {"date": "2026-10-18"})
    await client.close()
    assert exc.value.code == "MCP_UNAVAILABLE"
    assert "private-key" not in str(exc.value)
    sdk.list_tools.assert_awaited_once()
    sdk.call_tool.assert_awaited_once()
    assert not client.available


async def test_cancelling_one_request_preserves_shared_channel_and_other_requests(setup_client):
    client, sdk, factory = setup_client
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def execute(_name, args, **_kwargs):
        if args["date"] == "2026-10-12":
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        return tool_result(DAY)

    sdk.call_tool.side_effect = execute
    await client.start()
    first = asyncio.create_task(client.call_tool("get_schedule", {"date": "2026-10-12"}))
    try:
        await asyncio.wait_for(started.wait(), 2)
        second = asyncio.create_task(client.call_tool("get_schedule", {"date": "2026-10-18"}))
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        await asyncio.wait_for(cancelled.wait(), 2)
        assert await second == DAY
        assert client.available
        factory.assert_called_once()
    finally:
        await client.close()


async def test_request_deadline_does_not_close_or_retry_channel(setup_client):
    client, sdk, _factory = setup_client
    await client.start()
    client.settings = replace(client.settings, mcp_timeout_seconds=0.05)

    async def blocked(*_args, **_kwargs):
        await asyncio.Event().wait()

    sdk.call_tool.side_effect = blocked
    try:
        with pytest.raises(MCPClientError) as exc:
            await client.call_tool("get_schedule", {"date": "2026-10-18"})
        assert exc.value.code == "TIMEOUT"
        assert client.available
        sdk.call_tool.assert_awaited_once()
    finally:
        await client.close()


async def test_close_settles_active_request_and_owns_sdk_exit(setup_client):
    client, sdk, _factory = setup_client
    started = asyncio.Event()

    async def blocked(*_args, **_kwargs):
        started.set()
        await asyncio.Event().wait()

    sdk.call_tool.side_effect = blocked
    await client.start()
    request = asyncio.create_task(client.call_tool("get_schedule", {"date": "2026-10-18"}))
    await asyncio.wait_for(started.wait(), 2)
    await asyncio.create_task(client.close())
    with pytest.raises(MCPClientError) as exc:
        await request
    assert exc.value.code == "MCP_UNAVAILABLE"
    assert sdk.exited


async def test_close_during_start_and_discovery_timeout_do_not_leak_context(setup_client):
    client, sdk, _factory = setup_client
    client.settings = replace(client.settings, mcp_timeout_seconds=0.05)
    started = asyncio.Event()

    async def blocked():
        started.set()
        await asyncio.Event().wait()

    sdk.list_tools.side_effect = blocked
    await client.start()
    await client.close()
    assert started.is_set()
    assert not client.available
    assert sdk.exited


async def test_close_before_worker_first_tick_is_safe(setup_client):
    client, _sdk, _factory = setup_client
    start = asyncio.create_task(client.start())
    await asyncio.sleep(0)
    await client.close()
    await asyncio.wait_for(start, 2)
    assert not client.available


async def test_real_stdio_start_and_close_from_different_tasks(bot_settings):
    settings = replace(
        bot_settings, postgres_port=1, schedule_file=str(module.ROOT / "data/schedule.json")
    )
    client = MCPClient(settings)
    async with asyncio.timeout(25):
        await asyncio.create_task(client.start())
        try:
            assert client.available
            cancelled = asyncio.create_task(
                client.call_tool("get_schedule", {"date": "2026-10-12"})
            )
            # Дать worker начать запрос; отдельный subprocess обслуживает настоящий MCP.
            for _ in range(3):
                await asyncio.sleep(0)
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled
            assert client.available
            assert await client.call_tool("get_schedule", {"date": "2026-10-18"}) == DAY
            week = await client.read_resource(module.WEEK_URI)
            assert len(week["days"]) == 7
            with pytest.raises(MCPClientError) as exc:
                await client.call_tool(
                    "add_reminder", {"text": "test", "remind_at": "2030-01-01T10:00:00+03:00"}
                )
            assert exc.value.code == "UNAUTHORIZED"
            assert client.available
        finally:
            await asyncio.create_task(client.close())
    assert not client.available


async def test_confirmed_write_sends_capability_only_in_trusted_metadata(setup_client):
    client, sdk, _ = setup_client
    sdk.call_tool.return_value = tool_result(REMINDER_RECEIPT)
    await client.start()
    try:
        assert (
            await client.call_tool(
                "add_reminder", REMINDER_ARGUMENTS, action_context=ACTION_CONTEXT
            )
            == REMINDER_RECEIPT
        )
        sdk.call_tool.assert_awaited_once_with(
            "add_reminder",
            REMINDER_ARGUMENTS,
            meta={"labbi/action-capability": ACTION_CONTEXT},
            allow_input_required=True,
        )
        assert set(sdk.call_tool.call_args.args[1]) == {"text", "remind_at"}
        assert "capability" not in json.dumps(REMINDER_RECEIPT)
    finally:
        await client.close()


@pytest.mark.parametrize(
    "context",
    [
        None,
        {},
        {**ACTION_CONTEXT, "user_id": 42},
        {**ACTION_CONTEXT, "action_id": "invalid"},
        {**ACTION_CONTEXT, "capability": "a" * 42},
        {**ACTION_CONTEXT, "capability": "я" * 43},
    ],
)
async def test_invalid_write_context_is_refused_before_sdk(setup_client, context):
    client, sdk, _ = setup_client
    await client.start()
    try:
        with pytest.raises(MCPClientError) as error:
            await client.call_tool("add_reminder", REMINDER_ARGUMENTS, action_context=context)
        assert error.value.code == "UNAUTHORIZED"
        sdk.call_tool.assert_not_awaited()
    finally:
        await client.close()


async def test_owner_in_model_arguments_or_metadata_on_read_is_not_permitted(setup_client):
    client, sdk, _ = setup_client
    await client.start()
    try:
        with pytest.raises(MCPClientError) as error:
            await client.call_tool(
                "add_reminder", {**REMINDER_ARGUMENTS, "user_id": 42}, action_context=ACTION_CONTEXT
            )
        assert error.value.code == "INVALID_ARGUMENTS"
        with pytest.raises(MCPClientError) as error:
            await client.call_tool(
                "get_schedule", {"date": "2026-10-18"}, action_context=ACTION_CONTEXT
            )
        assert error.value.code == "INVALID_ARGUMENTS"
        sdk.call_tool.assert_not_awaited()
    finally:
        await client.close()


@pytest.mark.parametrize(
    "failure",
    [
        MCPError(types.REQUEST_TIMEOUT, "SECRET"),
        MCPError(types.HEADER_MISMATCH, "SECRET"),
        RuntimeError("SECRET protocol failure"),
    ],
)
async def test_write_transport_failure_reports_unknown_outcome_without_retry(
    setup_client, failure, caplog
):
    client, sdk, _ = setup_client
    sdk.call_tool.side_effect = failure
    await client.start()
    try:
        with pytest.raises(MCPClientError) as error:
            await client.call_tool(
                "add_reminder", REMINDER_ARGUMENTS, action_context=ACTION_CONTEXT
            )
        assert error.value.code == "OUTCOME_UNKNOWN"
        assert "SECRET" not in error.value.user_message
        assert ACTION_CONTEXT["capability"] not in caplog.text
        sdk.call_tool.assert_awaited_once()
        sdk.list_tools.assert_awaited_once()
    finally:
        await client.close()


async def test_write_outer_timeout_is_unknown_and_does_not_retry(setup_client):
    client, sdk, _ = setup_client
    await client.start()
    client.settings = replace(client.settings, mcp_timeout_seconds=0.02)

    async def waiting(*_args, **_kwargs):
        await asyncio.Event().wait()

    sdk.call_tool.side_effect = waiting
    try:
        with pytest.raises(MCPClientError) as error:
            await client.call_tool(
                "add_reminder", REMINDER_ARGUMENTS, action_context=ACTION_CONTEXT
            )
        assert error.value.code == "OUTCOME_UNKNOWN"
        assert client.available
        sdk.call_tool.assert_awaited_once()
    finally:
        await client.close()


async def test_malformed_write_receipt_is_unknown_not_false_rollback(setup_client):
    client, sdk, _ = setup_client
    sdk.call_tool.return_value = tool_result({"status": "ok", "untrusted": "SECRET"})
    await client.start()
    try:
        with pytest.raises(MCPClientError) as error:
            await client.call_tool(
                "add_reminder", REMINDER_ARGUMENTS, action_context=ACTION_CONTEXT
            )
        assert error.value.code == "OUTCOME_UNKNOWN"
        assert "SECRET" not in error.value.user_message
        sdk.call_tool.assert_awaited_once()
    finally:
        await client.close()


async def test_operation_repr_cannot_expose_authorization_or_reminder_text():
    operation = module._Operation(
        "tools/call",
        "add_reminder",
        {"text": "SECRET reminder"},
        asyncio.get_running_loop().create_future(),
        action_context=ACTION_CONTEXT,
    )
    assert "SECRET" not in repr(operation)
    assert ACTION_CONTEXT["capability"] not in repr(operation)
