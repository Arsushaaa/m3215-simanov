"""Preregistered E2E helper uses production routing with fake HTTP/MCP boundaries."""

import asyncio
import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.llm import ModelError, OpenRouterClient, RoutingCompletion
from app.mcp.contracts import tool_schemas
from app.mcp.server import TOOLS
from scripts.lab2_comparison_live import (
    EXPECTED_DATE,
    QUERY,
    LiveLLM,
    LiveMCP,
    run_live,
)
from scripts.lab2_experiment import checkpoint

SCHEDULE = {
    "status": "ok",
    "date": EXPECTED_DATE,
    "timezone": "Europe/Moscow",
    "lessons": [{"start": "10:40", "end": "12:10", "title": "Алгоритмы", "location": "103"}],
    "source": "local_schedule",
}


def source():
    return SimpleNamespace(
        available=True,
        tools={
            name: {"name": name, "description": item[2], **tool_schemas(name)}
            for name, item in TOOLS.items()
        },
        resources={"schedule://current-week": {"uri": "schedule://current-week"}},
        call_tool=AsyncMock(return_value=deepcopy(SCHEDULE)),
        read_resource=AsyncMock(return_value={"source": "local_schedule"}),
    )


@pytest.fixture
def fake_http(monkeypatch):
    payloads, responses = [], []
    started, closed = AsyncMock(), AsyncMock()

    async def request(client, payload):
        payloads.append(deepcopy(payload))
        if responses:
            response = responses.pop(0)
            if isinstance(response, BaseException):
                raise response
        else:
            response = (
                json.dumps(
                    {
                        "action": "call_tool",
                        "name": "get_schedule",
                        "arguments": {"date": EXPECTED_DATE},
                        "reason": "Нужно расписание на указанный день.",
                    }
                )
                if len(payloads) == 1
                else "13 октября: 10:40–12:10 — Алгоритмы. Источник: local_schedule."
            )
        client._observe_response(
            {
                "id": f"generation-{len(payloads)}",
                "model": payload["model"],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0},
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": response,
                            "reasoning": "PRIVATE hidden reasoning must not be persisted",
                        },
                    }
                ],
            }
        )
        return RoutingCompletion(response, [])

    monkeypatch.setattr(OpenRouterClient, "_request", request)
    monkeypatch.setattr(OpenRouterClient, "start", started)
    monkeypatch.setattr(OpenRouterClient, "close", closed)
    return SimpleNamespace(payloads=payloads, responses=responses, started=started, closed=closed)


async def test_live_read_uses_real_production_flow_and_saves_public_evidence(
    bot_settings, tmp_path, fake_http
):
    output, mcp = tmp_path / "live.json", source()
    checkpoints = []

    async def read(name, arguments):
        snapshot = json.loads(output.read_text())
        checkpoints.append(snapshot)
        assert len(snapshot["raw_data"]) == 1
        assert snapshot["raw_data"][0]["status"] == "ok"
        assert snapshot["mcp_calls"][0]["status"] == "in_progress"
        return deepcopy(SCHEDULE)

    mcp.call_tool.side_effect = read
    report = await run_live(bot_settings, mcp, output, temperature=0)
    assert report["metrics"]["acceptance"] == "passed"
    assert report["metrics"]["llm_calls"] == len(fake_http.payloads) == 2
    assert report["metrics"]["mcp_application_calls"] == 1
    assert report["metrics"]["api_cost"] == 0
    assert report["metrics"]["database_writes"] == report["metrics"]["telegram_calls"] == 0
    assert all(report["metrics"]["criteria"].values())
    mcp.call_tool.assert_awaited_once_with("get_schedule", {"date": EXPECTED_DATE})
    mcp.read_resource.assert_not_awaited()
    assert [item["role"] for item in fake_http.payloads[0]["messages"]] == ["system", "user"]
    assert fake_http.payloads[0]["messages"][-1]["content"] == QUERY
    assert all(payload["temperature"] == 0 for payload in fake_http.payloads)
    assert "tools" not in fake_http.payloads[0]
    assert fake_http.payloads[1]["tools"] == []
    assert "untrusted_mcp_result" in fake_http.payloads[1]["messages"][-1]["content"]
    assert all("expected" not in json.dumps(payload) for payload in fake_http.payloads)
    assert len(report["raw_data"]) == 2
    for index, call in enumerate(report["raw_data"], start=1):
        assert call["id"] == f"generation-{index}"
        assert call["model"] == bot_settings.openrouter_model
        assert call["usage"]["cost"] == 0
        assert call["raw_content"]
        assert len(call["input_sha256"]) == len(call["wire"]["request_sha256"]) == 64
        assert "messages" not in call
    serialized = output.read_text()
    assert "PRIVATE" not in serialized
    assert bot_settings.openrouter_api_key not in serialized
    assert json.loads(serialized) == report
    fake_http.started.assert_awaited_once()
    fake_http.closed.assert_awaited_once()
    original = output.read_bytes()
    again = await run_live(bot_settings, mcp, output, temperature=0)
    assert again == report and output.read_bytes() == original
    assert len(fake_http.payloads) == 2
    fake_http.started.assert_awaited_once()


async def test_checkpoint_exists_before_each_http_dispatch(bot_settings, tmp_path, monkeypatch):
    output, observed = tmp_path / "live.json", []

    async def request(client, payload):
        document = json.loads(output.read_text())
        call = document["raw_data"][-1]
        assert call["status"] == "in_progress"
        assert call["wire"]["request_sha256"]
        observed.append(document)
        raise ModelError("Fake transport failure")

    monkeypatch.setattr(OpenRouterClient, "_request", request)
    monkeypatch.setattr(OpenRouterClient, "start", AsyncMock())
    monkeypatch.setattr(OpenRouterClient, "close", AsyncMock())
    report = await run_live(bot_settings, source(), output, temperature=0.3)
    assert len(observed) == report["metrics"]["llm_calls"] == 1
    assert report["status"] == "failed"
    assert report["metrics"]["acceptance"] == "not_passed"
    assert report["raw_data"][0]["error"]["type"] == "ModelError"
    await run_live(bot_settings, source(), output, temperature=0.3)
    assert len(observed) == 1


async def test_interrupted_checkpoint_is_returned_without_dispatch(
    bot_settings, tmp_path, fake_http
):
    output = tmp_path / "live.json"
    report = await run_live(bot_settings, source(), output, temperature=0.3)
    report["status"] = "in_progress"
    report["raw_data"][-1]["status"] = "in_progress"
    checkpoint(output, report)
    before = output.read_bytes()
    assert await run_live(bot_settings, source(), output, temperature=0.3) == report
    assert output.read_bytes() == before
    assert len(fake_http.payloads) == 2


@pytest.mark.parametrize("phase", ["decide", "complete"])
async def test_api_failure_never_retries_or_exceeds_two_calls(
    bot_settings, tmp_path, fake_http, phase
):
    if phase == "complete":
        fake_http.responses.append(
            json.dumps(
                {
                    "action": "call_tool",
                    "name": "get_schedule",
                    "arguments": {"date": EXPECTED_DATE},
                    "reason": "Чтение.",
                }
            )
        )
    fake_http.responses.append(ModelError("Fake rate limit"))
    mcp = source()
    report = await run_live(bot_settings, mcp, tmp_path / "failure.json", temperature=0.3)
    assert len(fake_http.payloads) == (1 if phase == "decide" else 2)
    assert mcp.call_tool.await_count == (0 if phase == "decide" else 1)
    assert report["metrics"]["acceptance"] == "not_passed"
    fake_http.closed.assert_awaited_once()


@pytest.mark.parametrize("field,value", [("date", "2026-10-14"), ("source", "invented")])
async def test_wrong_result_fails_acceptance(bot_settings, tmp_path, fake_http, field, value):
    mcp = source()
    mcp.call_tool.return_value[field] = value
    report = await run_live(bot_settings, mcp, tmp_path / "wrong.json", temperature=0.3)
    assert report["metrics"]["acceptance"] == "not_passed"
    assert not report["metrics"]["criteria"]["result_matches_date_and_source"]


async def test_mcp_failure_stops_before_final_llm_without_repeat(bot_settings, tmp_path, fake_http):
    mcp = source()
    mcp.call_tool.side_effect = TimeoutError("PRIVATE transport diagnostics")
    report = await run_live(bot_settings, mcp, tmp_path / "timeout.json", temperature=0.3)
    assert len(fake_http.payloads) == mcp.call_tool.await_count == 1
    assert report["metrics"]["acceptance"] == "not_passed"
    assert "PRIVATE" not in json.dumps(report)


async def test_wrappers_enforce_read_only_and_hard_budgets_before_dispatch():
    document = {"raw_data": [], "mcp_calls": []}
    mcp_source = source()
    mcp = LiveMCP(mcp_source, document, lambda: None)
    with pytest.raises(ModelError, match="Writes are forbidden"):
        await mcp.call_tool("add_reminder", {"text": "x"})
    assert not document["mcp_calls"]
    await mcp.read_resource("schedule://current-week")
    with pytest.raises(ModelError, match="budget exceeded"):
        await mcp.call_tool("get_schedule", {"date": EXPECTED_DATE})
    mcp_source.call_tool.assert_not_awaited()
    mcp_source.read_resource.assert_awaited_once()
    client = SimpleNamespace(
        observation={},
        decide=AsyncMock(return_value=RoutingCompletion("X", [])),
        complete=AsyncMock(return_value=RoutingCompletion("Y", [])),
    )
    llm = LiveLLM(client, document, lambda: None)
    await llm.decide([], 0.3)
    with pytest.raises(ModelError, match="final tools"):
        await llm.complete([], 0.3)
    await llm.complete([], 0.3, disable_tools=True)
    with pytest.raises(ModelError, match="budget/order"):
        await llm.complete([], 0.3, disable_tools=True)
    client.decide.assert_awaited_once()
    client.complete.assert_awaited_once()


@pytest.mark.parametrize("temperature", [True, None, "0.3", -1, 2, float("nan")])
async def test_invalid_parameters_fail_before_output_or_api(
    bot_settings, tmp_path, fake_http, temperature
):
    output = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        await run_live(bot_settings, source(), output, temperature=temperature)
    assert not output.exists() and not fake_http.payloads


async def test_paid_model_or_missing_mcp_never_consumes_quota(bot_settings, tmp_path, fake_http):
    with pytest.raises(ValueError):
        await run_live(
            replace(bot_settings, openrouter_model="example/paid"),
            source(),
            tmp_path / "paid.json",
            temperature=0.3,
        )
    mcp = source()
    mcp.available = False
    with pytest.raises(ValueError, match="discovery"):
        await run_live(bot_settings, mcp, tmp_path / "no-mcp.json", temperature=0.3)
    fake_http.started.assert_not_awaited()
    assert not fake_http.payloads


async def test_changed_protocol_cannot_overwrite_existing_result(bot_settings, tmp_path, fake_http):
    output = tmp_path / "live.json"
    await run_live(bot_settings, source(), output, temperature=0)
    before = output.read_bytes()
    with pytest.raises(ValueError, match="different protocol"):
        await run_live(bot_settings, source(), output, temperature=0.3)
    assert output.read_bytes() == before and len(fake_http.payloads) == 2


async def test_legacy_live_protocol_cannot_be_replayed(bot_settings, tmp_path, fake_http):
    output = tmp_path / "old-live.json"
    report = await run_live(bot_settings, source(), output, temperature=0)
    assert report["protocol"]["version"] == 2
    assert "app/reminder_intent.py" not in report["protocol"]["source_sha256"]
    report["protocol"]["version"] = 1
    checkpoint(output, report)
    original = output.read_bytes()

    with pytest.raises(ValueError, match="different protocol"):
        await run_live(bot_settings, source(), output, temperature=0)

    assert output.read_bytes() == original
    assert len(fake_http.payloads) == 2


async def test_cancellation_consumes_slot_and_never_replays(bot_settings, tmp_path, monkeypatch):
    output = tmp_path / "interrupted.json"

    async def cancelled(*args, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(OpenRouterClient, "_request", cancelled)
    monkeypatch.setattr(OpenRouterClient, "start", AsyncMock())
    monkeypatch.setattr(OpenRouterClient, "close", AsyncMock())
    with pytest.raises(asyncio.CancelledError):
        await run_live(bot_settings, source(), output, temperature=0.3)
    document = json.loads(output.read_text())
    assert document["status"] == "failed"
    assert len(document["raw_data"]) == 1
    assert document["raw_data"][0]["error"]["type"] == "CancelledError"
    assert await run_live(bot_settings, source(), output, temperature=0.3) == document
