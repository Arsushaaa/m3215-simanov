"""Six synthetic fake-model cases exercise the preregistered smoke boundaries."""

import asyncio
import json
from copy import deepcopy
from dataclasses import replace

import pytest

from app.llm import RoutingCompletion
from app.mcp.contracts import tool_schemas
from app.mcp.server import TOOLS
from scripts import semantic_routing_smoke as smoke
from scripts.lab2_experiment import CatalogOnlyMCP, RecordingClient


def catalog():
    return CatalogOnlyMCP(
        {
            name: {"name": name, "description": entry[2], **tool_schemas(name)}
            for name, entry in TOOLS.items()
        },
        {"schedule://current-week": {"uri": "schedule://current-week"}},
    )


class FakeClient(RecordingClient):
    def __init__(self, settings, *, overrides=None, before_dispatch=None):
        super().__init__(settings)
        self.requests = []
        self.started = self.closed = 0
        self.overrides = overrides or {}
        self.before_dispatch = before_dispatch

    async def start(self):
        self.started += 1

    async def close(self):
        self.closed += 1

    async def _request(self, payload):
        self.requests.append(deepcopy(payload))
        current = json.loads(payload["messages"][-1]["content"])
        assert current["message_index"] == 0
        case = next(c for c in smoke.CASES if c["query"] == current["text"])
        if self.before_dispatch:
            await self.before_dispatch(case)
        decision = {"action": case["expected_action"], "reason": "Synthetic fixture."}
        if decision["action"] in {"add_reminder", "get_weather"}:
            decision.update(
                action="call_tool",
                name=case["expected_action"],
                arguments=case["expected_arguments"],
            )
        if case["expected_action"] == "add_reminder":
            user = {"source": "user", "message_index": 0, "quote": case["query"]}
            decision["evidence"] = {k: dict(user) for k in ("request", "text", "date", "time")}
            if "pending_card" in case:
                decision["evidence"].update(
                    text={"source": "pending_card"}, date={"source": "pending_card"}
                )
        elif decision["action"] == "respond":
            decision["text"] = "Пирог можно приготовить по простому рецепту."
        elif decision["action"] == "clarify_reminder":
            decision["question"] = "Во сколько завтра напомнить?"
        value = self.overrides.get(case["id"], decision)
        if isinstance(value, BaseException):
            raise value
        calls = []
        if "tools" not in payload or isinstance(value, str):
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        elif value["action"] == "respond":
            text = value["text"]
        else:
            text = None
            if value["action"] == "call_tool":
                name = value["name"]
                arguments = value["arguments"]
                if name == "add_reminder":
                    name = "prepare_reminder"
                    arguments = {"arguments": arguments, "evidence": value["evidence"]}
            else:
                name = value["action"]
                arguments = {key: item for key, item in value.items() if key != "action"}
            calls = [
                {
                    "id": "synthetic-tool-call",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            ]
        self.observation = {}
        self._observe_response(
            {
                "model": payload["model"],
                "id": "synthetic-generation",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 30,
                    "cost": 0,
                    "reasoning": "PRIVATE-reasoning-usage",
                },
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": text,
                            "tool_calls": calls,
                            "reasoning": "PRIVATE-hidden-chain",
                        },
                    }
                ],
            }
        )
        return RoutingCompletion(text, calls)


@pytest.fixture
def committed(monkeypatch):
    observed = []
    monkeypatch.setattr(smoke, "assert_committed", lambda path: observed.append(path))
    return observed


async def preregister(settings, folder):
    return await smoke.prepare(
        settings, folder, catalog=catalog(), quota={"limit": 50, "used": 44, "remaining": 6}
    )


@pytest.mark.parametrize("tool_mode", ["structured", "native"])
async def test_all_six_cases_use_production_routing_and_preserve_only_safe_evidence(
    bot_settings, tmp_path, committed, tool_mode
):
    bot_settings = replace(bot_settings, llm_tool_mode=tool_mode)
    manifest = await preregister(bot_settings, tmp_path)
    assert manifest["protocol"]["tool_mode"] == tool_mode
    assert len(manifest["plan"]["cases"]) == 6
    client = FakeClient(bot_settings)
    report = await smoke.run(bot_settings, tmp_path, catalog=catalog(), client=client)
    assert report["metrics"]["acceptance"] == "passed"
    assert report["metrics"]["cases"] == report["metrics"]["actions"] == {"correct": 6, "total": 6}
    assert report["metrics"]["arguments"] == {"correct": 4, "total": 4}
    assert report["metrics"]["llm_calls"] == len(client.requests) == 6
    assert report["metrics"]["api_cost"] == 0
    assert report["metrics"]["database_writes"] == report["metrics"]["telegram_calls"] == 0
    assert report["metrics"]["executed_mcp_calls"] == 0
    assert client.started == client.closed == 1
    assert report["attempts"][2]["status"] == "captured_before_mcp"
    assert report["attempts"][-1]["former_card_revoked"] is True
    assert [json.loads(p["messages"][-1]["content"])["text"] for p in client.requests] == [
        c["query"] for c in smoke.CASES
    ]
    for payload in client.requests:
        assert payload["temperature"] == 0.2
        assert payload["provider"]["max_price"] == {"prompt": 0, "completion": 0}
        if tool_mode == "structured":
            assert "tools" not in payload
            assert payload["response_format"]["json_schema"]["strict"] is True
        else:
            assert "response_format" not in payload
            assert payload["tool_choice"] == "auto"
            assert "parallel_tool_calls" not in payload
            assert "prepare_reminder" in {tool["function"]["name"] for tool in payload["tools"]}
        assert "expected_action" not in json.dumps(payload)
        assert "expected_arguments" not in json.dumps(payload)
    serialized = (tmp_path / "results.json").read_text()
    for secret in (
        "PRIVATE",
        bot_settings.bot_token,
        bot_settings.openrouter_api_key,
        bot_settings.postgres_password,
    ):
        assert secret not in serialized
    unchanged = (tmp_path / "results.json").read_bytes()
    assert await smoke.run(bot_settings, tmp_path, catalog=catalog(), client=client) == report
    assert (tmp_path / "results.json").read_bytes() == unchanged
    assert len(client.requests) == 6 and client.started == 1 and len(committed) == 2
    if tool_mode == "native":
        assert [
            row["calls"][0]["raw_tool_calls"][0]["function"]["name"]
            if row["calls"][0]["raw_tool_calls"]
            else None
            for row in report["attempts"]
        ] == [
            "prepare_reminder",
            None,
            "get_weather",
            "clarify_reminder",
            "prepare_reminder",
            "prepare_reminder",
        ]


@pytest.mark.parametrize("remaining", [5, 0, -1, True, "6", None])
async def test_insufficient_or_invalid_quota_stops_before_discovery(
    bot_settings, tmp_path, monkeypatch, remaining
):
    async def unexpected(*_args):
        pytest.fail("No discovery/API is allowed after an insufficient quota check.")

    monkeypatch.setattr(smoke, "discover", unexpected)
    with pytest.raises(ValueError, match="six remaining"):
        await smoke.prepare(bot_settings, tmp_path, quota={"remaining": remaining})
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize("model", ["example/paid", ":free", "example/model:free "])
async def test_nonfree_model_refuses_before_any_metadata_or_discovery(
    bot_settings, tmp_path, monkeypatch, model
):
    async def unexpected(*_args):
        pytest.fail("No metadata request may use an invalid model configuration.")

    monkeypatch.setattr(smoke, "quota_status", unexpected)
    monkeypatch.setattr(smoke, "discover", unexpected)
    with pytest.raises(ValueError, match="free model"):
        await smoke.prepare(replace(bot_settings, openrouter_model=model), tmp_path)
    assert not (tmp_path / "manifest.json").exists()


async def test_existing_manifest_cannot_be_overwritten(bot_settings, tmp_path, monkeypatch):
    await preregister(bot_settings, tmp_path)
    path = tmp_path / "manifest.json"
    original = path.read_bytes()

    async def unexpected(*_args):
        pytest.fail("Check existing manifest before metadata request.")

    monkeypatch.setattr(smoke, "quota_status", unexpected)
    with pytest.raises(ValueError, match="Never overwrite"):
        await smoke.prepare(bot_settings, tmp_path)
    assert path.read_bytes() == original


async def test_uncommitted_or_changed_source_refuses_before_client_start(
    bot_settings, tmp_path, monkeypatch
):
    await preregister(bot_settings, tmp_path)
    client = FakeClient(bot_settings)

    def uncommitted(_path):
        raise ValueError("Commit the exact preregistration")

    monkeypatch.setattr(smoke, "assert_committed", uncommitted)
    with pytest.raises(ValueError, match="Commit the exact"):
        await smoke.run(bot_settings, tmp_path, catalog=catalog(), client=client)
    monkeypatch.setattr(smoke, "source_snapshot", lambda: {"changed": "hash"})
    with pytest.raises(ValueError, match="sources changed"):
        await smoke.run(bot_settings, tmp_path, catalog=catalog(), client=client)
    assert not client.requests and client.started == 0
    assert not (tmp_path / "results.json").exists()


async def test_wrong_output_remains_in_fixed_denominator_without_retry(
    bot_settings, tmp_path, committed
):
    await preregister(bot_settings, tmp_path)
    client = FakeClient(bot_settings, overrides={"english_weather": "BROKEN DECISION"})
    report = await smoke.run(bot_settings, tmp_path, catalog=catalog(), client=client)
    assert report["metrics"]["cases"] == {"correct": 5, "total": 6}
    assert report["metrics"]["arguments"] == {"correct": 3, "total": 4}
    assert report["metrics"]["acceptance"] == "not_passed"
    assert len(client.requests) == 6
    failed = report["attempts"][2]
    assert failed["calls"][0]["raw_content"] == "BROKEN DECISION"
    assert failed["status"] == "failed" and not failed["score"]["passed"]
    await smoke.run(bot_settings, tmp_path, catalog=catalog(), client=client)
    assert len(client.requests) == 6


async def test_started_slot_is_durable_and_cancelled_case_is_not_replayed(
    bot_settings, tmp_path, committed
):
    await preregister(bot_settings, tmp_path)

    async def cancelled(_case):
        saved = json.loads((tmp_path / "results.json").read_text())
        assert saved["attempts"][-1]["status"] == "in_progress"
        assert len(saved["attempts"][-1]["calls"]) == 1
        raise asyncio.CancelledError

    interrupted = FakeClient(bot_settings, before_dispatch=cancelled)
    with pytest.raises(asyncio.CancelledError):
        await smoke.run(bot_settings, tmp_path, catalog=catalog(), client=interrupted)
    resumed = FakeClient(bot_settings)
    report = await smoke.run(bot_settings, tmp_path, catalog=catalog(), client=resumed)
    assert len(interrupted.requests) == 1 and len(resumed.requests) == 5
    assert report["metrics"]["llm_calls"] == 6
    assert report["metrics"]["cases"] == {"correct": 5, "total": 6}
    assert report["attempts"][0]["error"]["type"] == "CancelledError"
    assert all(
        json.loads(p["messages"][-1]["content"])["text"] != smoke.CASES[0]["query"]
        for p in resumed.requests
    )
    assert interrupted.closed == 1


async def test_concurrent_invocation_cannot_dispatch_the_same_cases(
    bot_settings, tmp_path, committed
):
    await preregister(bot_settings, tmp_path)
    started, resume = asyncio.Event(), asyncio.Event()

    async def pause(_case):
        started.set()
        await resume.wait()

    first, second = FakeClient(bot_settings, before_dispatch=pause), FakeClient(bot_settings)
    task = asyncio.create_task(smoke.run(bot_settings, tmp_path, catalog=catalog(), client=first))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        with pytest.raises(ValueError, match="Another process owns"):
            await smoke.run(bot_settings, tmp_path, catalog=catalog(), client=second)
        assert second.started == 0 and not second.requests
    finally:
        resume.set()
        report = await task
    assert report["metrics"]["acceptance"] == "passed" and len(first.requests) == 6
