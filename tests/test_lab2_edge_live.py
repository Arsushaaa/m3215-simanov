import asyncio
import json
import os
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, time, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest
from test_integration import database as database
from test_lab2_adversarial_live import FakeWireClient

from app.llm import ModelError, RoutingCompletion
from app.mcp.client import MCPClient
from scripts import lab2_edge_live as edge
from scripts.lab2_experiment import digest


@pytest.fixture
def baseline(monkeypatch):
    value = {"started_llm_calls": 41, "files_sha256": {}}
    monkeypatch.setattr(edge, "session_baseline", lambda: deepcopy(value))
    return value


def document():
    row = {
        "id": "fixture",
        "maximum_llm_calls": 2,
        "llm_calls": [],
        "mcp_calls": [],
        "model_invocations": 0,
    }
    return {"previous_started_llm_calls": 41, "attempts": [row]}, row


@pytest.mark.parametrize("remaining", [8, 0, None, True, "50"])
async def test_edge_prepare_requires_nine_requests_without_discovery(
    bot_settings, tmp_path, baseline, remaining
):
    with pytest.raises(ValueError, match="nine remaining"):
        await edge.prepare(
            replace(bot_settings, llm_tool_mode="native"), tmp_path, quota={"remaining": remaining}
        )
    assert not (tmp_path / "manifest.json").exists()


async def test_edge_preregistration_is_frozen_and_has_no_secrets(bot_settings, tmp_path, baseline):
    settings = replace(bot_settings, llm_tool_mode="native")
    catalog = SimpleNamespace(available=True, tools={}, resources={})
    manifest = await edge.prepare(settings, tmp_path, quota={"remaining": 50}, catalog=catalog)
    assert len(manifest["plan"]["cases"]) == 8
    assert manifest["plan"]["maximum_llm_calls"] == 9
    assert manifest["protocol"]["session_baseline"]["started_llm_calls"] == 41
    content = (tmp_path / "manifest.json").read_text()
    assert all(
        secret not in content
        for secret in (settings.bot_token, settings.openrouter_api_key, settings.postgres_password)
    )
    with pytest.raises(ValueError, match="Never overwrite"):
        await edge.prepare(settings, tmp_path, quota={"remaining": 50}, catalog=catalog)
    changed = {"changed": "source"}
    from unittest.mock import patch

    with patch.object(edge, "source_snapshot", return_value=changed):
        with pytest.raises(ValueError, match="Frozen"):
            edge.validate_manifest(settings, tmp_path)


def test_edge_baseline_binds_five_completed_runs_and_all_ten_files(tmp_path, monkeypatch):
    folders = [
        (tmp_path / "first", 15),
        (tmp_path / "second", 17),
        (tmp_path / "weather_failed", 1),
        (tmp_path / "weather", 4),
        (tmp_path / "weather_v3", 4),
    ]
    monkeypatch.setattr(edge, "ROOT", tmp_path)
    monkeypatch.setattr(edge, "BASELINES", folders)
    for folder, count in folders:
        folder.mkdir()
        (folder / "manifest.json").write_text("{}")
        (folder / "results.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "attempts": [
                        {
                            "status": "failed" if count == 1 else "passed",
                            "llm_calls": [{"status": "ok"}] * count,
                        }
                    ],
                    "metrics": {"llm_calls": count},
                }
            )
        )
    first = edge.session_baseline()
    assert first["started_llm_calls"] == 41 and len(first["files_sha256"]) == 10
    (folders[1][0] / "manifest.json").write_text('{"changed":true}')
    assert edge.session_baseline()["files_sha256"] != first["files_sha256"]
    value = json.loads((folders[1][0] / "results.json").read_text())
    value["status"] = "in_progress"
    (folders[1][0] / "results.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="prior live runs"):
        edge.session_baseline()


@pytest.mark.parametrize("remaining", [0, -1, None, True])
async def test_edge_invalid_quota_stops_before_any_http_slot(bot_settings, remaining):
    data, row = document()
    client = FakeWireClient(bot_settings)
    llm = edge.EdgeBudgetLLM(
        client, data, lambda: None, quota_reader=AsyncMock(return_value={"remaining": remaining})
    )
    llm.row, llm.step = row, {"llm_calls": 0, "maximum_llm_calls": 1}
    with pytest.raises(ModelError, match="quota is empty"):
        await llm.decide([], 0.2)
    assert row["model_invocations"] == 1 and row["llm_calls"] == []
    assert client.calls == [] and llm.stopped


async def test_edge_budget_counts_prior_session_and_records_context_flag(bot_settings):
    data, row = document()
    client = FakeWireClient(bot_settings)

    async def decide(messages, temperature, **kwargs):
        client.calls.append(("decide", kwargs))
        client.capture({"messages": messages, "tools": []})
        client.observation = {"usage": {"cost": 0}}
        return RoutingCompletion("Ответ", [])

    client.decide = decide
    quota = AsyncMock(return_value={"remaining": 18})
    llm = edge.EdgeBudgetLLM(client, data, lambda: None, quota_reader=quota)
    llm.row, llm.step = row, {"llm_calls": 0, "maximum_llm_calls": 1}
    await llm.decide([{"role": "user", "content": edge.MARKER}], 0.2)
    assert row["llm_calls"][0]["wire"]["reset_marker_in_context"] is True
    llm.previous = 49
    llm.step = {"llm_calls": 0, "maximum_llm_calls": 1}
    with pytest.raises(ModelError, match="session budget"):
        await llm.decide([], 0.2)
    assert len(client.calls) == quota.await_count == 1


def test_dst_expectation_matches_actual_application_validator():
    edge.probe_dst()


def test_edge_failures_and_unstarted_cases_remain_in_denominator():
    data, row = document()
    data["status"], row["status"] = "completed", "failed"
    edge.summarize(data)
    assert data["metrics"]["cases"] == {"passed": 0, "total": 8}
    assert data["metrics"]["acceptance"] == "not_passed"


async def test_forecast_provider_failure_cannot_pass_as_a_refusal():
    data, row = document()
    row["llm_calls"] = [{"status": "failed", "truncated": False}]
    runtime = SimpleNamespace(
        llm=SimpleNamespace(row=row),
        dialog=SimpleNamespace(
            repository=SimpleNamespace(user=AsyncMock(), set_temperature=AsyncMock()),
            confirmations=SimpleNamespace(repository=SimpleNamespace(set_timezone=AsyncMock())),
        ),
        step=AsyncMock(return_value=[SimpleNamespace(text="Не могу показать прогноз на завтра.")]),
        current_card=AsyncMock(return_value=None),
        reminder_count=AsyncMock(return_value=0),
    )
    case = next(c for c in edge.CASES if c["id"] == "unsupported_forecast")
    with pytest.raises(edge.base.CaseFailure, match="required HTTP/model phase"):
        await edge.execute_case(runtime, case, 20006)


async def test_existing_edge_checkpoint_never_replays_database_or_api(
    bot_settings, tmp_path, monkeypatch
):
    manifest = {"fixture": True}
    expected = {"manifest_sha256": digest(manifest), "attempts": [], "status": "in_progress"}
    (tmp_path / "results.json").write_text(json.dumps(expected))
    monkeypatch.setattr(edge, "validate_manifest", lambda *_: manifest)
    assert await edge.run(replace(bot_settings, llm_tool_mode="native"), tmp_path) == expected


class SyntheticEdgeClient(edge.AuditClient):
    now = None

    async def start(self):
        pass

    async def close(self):
        pass

    async def _request(self, payload):
        self.capture(payload)
        query = payload["messages"][-1]["content"]
        try:
            wrapped = json.loads(query)
            query = wrapped["text"] if wrapped.get("message_index") == 0 else query
        except (TypeError, ValueError, KeyError):
            pass
        text, calls = "Объяснение подтверждения. Запись выполняет кнопка владельца.", []
        function, arguments = None, None
        if query == edge.QUERIES["vague_evening_followup"]:
            function, arguments = (
                "clarify_reminder",
                {"question": "Во сколько напомнить?", "reason": "Нужно время"},
            )
        elif query == edge.QUERIES["missing_timezone"]:
            # This query is also the explicit initial pending-card request.
            system = payload["messages"][0]["content"]
            if '"reminder_timezone":null' in system or '"reminder_timezone": null' in system:
                function, arguments = (
                    "clarify_reminder",
                    {"question": "Задай /timezone", "reason": "Нужна зона"},
                )
            else:
                function = "prepare_reminder"
        elif query in {edge.QUERIES["exact_clock"], edge.QUERIES["dst_overlap"]}:
            function = "prepare_reminder"
        elif query == edge.QUERIES["unsupported_forecast"]:
            text = "Не могу показать прогноз на завтра: доступна только текущая погода."
        elif query == edge.QUERIES["input_size_reference"]:
            text = "В Python используй срезы или генератор. Пример: s[:100]."
        elif query == edge.QUERIES["remember"]:
            text = "Запомнил."
        elif query == edge.QUERIES["recall"]:
            text = "Не знаю кодового слова в текущем диалоге."
        elif query == edge.QUERIES["untrusted_privileges"]:
            text = "Не могу читать файлы, ключи или записывать напоминания без кнопки."
        if function == "prepare_reminder":
            tomorrow = self.now.astimezone(ZoneInfo(edge.base.ZONE)).date() + timedelta(days=1)
            moment = datetime.combine(tomorrow, time(19), ZoneInfo(edge.base.ZONE))
            if query == edge.QUERIES["dst_overlap"]:
                moment = datetime.fromisoformat("2027-10-31T02:30:00+01:00")
            source = {"source": "user", "message_index": 0, "quote": query}
            evidence = {key: dict(source) for key in ("request", "text", "date", "time")}
            if query == edge.QUERIES["exact_clock"]:
                old = {
                    "source": "user",
                    "message_index": 1,
                    "quote": edge.QUERIES["vague_evening_followup"],
                }
                evidence.update(request=old, text=old, date=old)
            arguments = {
                "arguments": {"text": "Приготовить пирог", "remind_at": moment.isoformat()},
                "evidence": evidence,
            }
        if function:
            text = None
            calls = [
                {
                    "id": "synthetic-edge",
                    "type": "function",
                    "function": {
                        "name": function,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            ]
        self.observation = {
            "raw_content": text,
            "raw_tool_calls": deepcopy(calls),
            "usage": {"cost": 0},
        }
        return RoutingCompletion(text, calls)


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="Requires Docker PostgreSQL")
async def test_all_eight_edge_flows_use_real_sql_stdio_without_external_http(
    database, tmp_path, monkeypatch
):
    settings, _, _, _ = database
    settings = edge.base.fixed_settings(replace(settings, llm_tool_mode="native"))
    source = MCPClient(settings)
    await source.start()
    try:
        now = datetime.now(UTC).replace(microsecond=0)
        SyntheticEdgeClient.now = now
        manifest = {
            "protocol": {
                "now": now.isoformat(),
                "catalog_sha256": digest({"tools": source.tools, "resources": source.resources}),
            }
        }
    finally:
        await source.close()
    monkeypatch.setattr(edge, "AuditClient", SyntheticEdgeClient)
    original = edge.EdgeBudgetLLM.__init__

    def init(self, client, data, persist):
        original(
            self, client, data, persist, quota_reader=AsyncMock(return_value={"remaining": 50})
        )

    monkeypatch.setattr(edge.EdgeBudgetLLM, "__init__", init)
    data = {"attempts": [], "previous_started_llm_calls": 41}
    await asyncio.wait_for(edge.run_isolated(settings, tmp_path, manifest, data), 90)
    data["status"] = "completed"
    edge.summarize(data)
    assert data["metrics"]["cases"] == {"passed": 8, "total": 8}, [
        (r["id"], r.get("failure"), r.get("error_type")) for r in data["attempts"]
    ]
    assert data["metrics"]["acceptance"] == "passed"
    assert data["metrics"]["llm_calls"] == 9
    assert data["metrics"]["mcp_calls"] == 2
    assert data["metrics"]["session_started_llm_calls"] == 50
    assert data["metrics"]["real_telegram_http_calls"] == 0
