import asyncio
import json
import os
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_integration import database as database
from test_lab2_adversarial_live import FakeWireClient

from app.llm import ModelError, RoutingCompletion
from app.mcp.client import MCPClient
from scripts import weather_specificity_regression as verify
from scripts.lab2_experiment import digest


@pytest.mark.parametrize("remaining", [15, 0, True, None])
async def test_prepare_checks_joint_sixteen_call_budget_before_discovery(
    bot_settings, tmp_path, remaining
):
    with pytest.raises(ValueError, match="Sixteen"):
        await verify.prepare(
            replace(bot_settings, llm_tool_mode="native"), tmp_path, quota={"remaining": remaining}
        )
    assert not (tmp_path / "manifest.json").exists()


async def test_preregistration_survives_json_roundtrip_and_binds_related_smoke(
    bot_settings, tmp_path, monkeypatch
):
    monkeypatch.setattr(verify, "related_smoke", lambda: {"path": "fixture", "sha256": "hash"})
    settings = replace(bot_settings, llm_tool_mode="native")
    catalog = SimpleNamespace(available=True, tools={}, resources={})
    manifest = await verify.prepare(settings, tmp_path, quota={"remaining": 50}, catalog=catalog)
    restored = json.loads((tmp_path / "manifest.json").read_text())
    assert digest(restored["plan"]) == digest(verify.PLAN)
    assert manifest["plan"]["maximum_llm_calls"] == 10
    assert restored["protocol"]["related_smoke"]["path"] == "fixture"
    with pytest.raises(ValueError, match="Never overwrite"):
        await verify.prepare(settings, tmp_path, quota={"remaining": 50}, catalog=catalog)
    monkeypatch.setattr(verify, "sources", lambda: {"changed": "source"})
    with pytest.raises(ValueError, match="Frozen"):
        verify.validate(settings, tmp_path)


async def test_budget_refuses_eleventh_call_before_http_or_quota(bot_settings, monkeypatch):
    monkeypatch.setattr(verify, "smoke_calls", lambda: 6)
    row = {"maximum_llm_calls": 10, "llm_calls": []}
    data = {"attempts": [row]}
    client = FakeWireClient(bot_settings)
    quota = AsyncMock(return_value={"remaining": 50})
    llm = verify.WeatherBudget(client, data, lambda: None, quota_reader=quota)
    llm.row = row
    for _ in range(5):
        llm.step = {"maximum_llm_calls": 2, "llm_calls": 0}
        await llm.decide([], 0.2, tools=[])
        await llm.complete([], 0.2, disable_tools=True, tools=[])
    with pytest.raises(ModelError, match="session budget"):
        await llm.decide([], 0.2, tools=[])
    assert len(client.calls) == quota.await_count == 10


def test_failures_and_unstarted_flows_stay_in_denominator(monkeypatch):
    monkeypatch.setattr(verify, "smoke_calls", lambda: 6)
    result = {
        "status": "completed",
        "attempts": [
            {"status": "passed", "llm_calls": [], "mcp_calls": []},
        ],
    }
    verify.summarize(result)
    assert result["metrics"]["cases"] == {"passed": 1, "total": 3}
    assert result["metrics"]["acceptance"] == "not_passed"


async def test_checkpoint_never_replays_any_started_flow(bot_settings, tmp_path, monkeypatch):
    manifest = {"test": "manifest"}
    expected = {"manifest_sha256": digest(manifest), "status": "in_progress", "attempts": []}
    (tmp_path / "results.json").write_text(json.dumps(expected))
    monkeypatch.setattr(verify, "validate", lambda *_: manifest)
    monkeypatch.setattr(verify.base, "isolated_database", lambda *_: pytest.fail("Replay"))
    assert await verify.run(replace(bot_settings, llm_tool_mode="native"), tmp_path) == expected


@pytest.mark.parametrize("wrong_city", ["Ростов-на-Дону", "Ростов, Россия"])
async def test_scorer_rejects_expansion_and_added_country_even_with_good_weather(wrong_city):
    row = {
        "llm_calls": [
            {"status": "ok"},
            {"status": "ok", "disable_tools": True, "wire": {"tool_choice": "none"}},
        ],
        "mcp_calls": [{"target": "get_weather", "arguments": {"city": wrong_city}}],
    }
    runtime = SimpleNamespace(
        llm=SimpleNamespace(row=row),
        dialog=SimpleNamespace(
            repository=SimpleNamespace(user=AsyncMock(), set_temperature=AsyncMock()),
            confirmations=SimpleNamespace(repository=SimpleNamespace(set_timezone=AsyncMock())),
        ),
        step=AsyncMock(return_value=[SimpleNamespace(text="Уточни место, Open-Meteo")]),
    )
    with pytest.raises(verify.base.CaseFailure, match="expanded"):
        await verify.execute_case(runtime, verify.CASES[0], 101)


class FixtureMCP(MCPClient):
    async def call_tool(self, name, arguments, **kwargs):
        assert name == "get_weather"
        city = arguments["city"]
        if "Дону" in city:
            return {
                "status": "ok",
                "place": {
                    "name": "Ростов-на-Дону",
                    "country": "Россия",
                    "region": "Ростовская область",
                },
                "temperature_c": 7.4,
                "condition": {"code": 0, "description": "Ясно"},
                "wind": {"speed": 3.6, "unit": "km/h"},
                "observed_at": "2026-11-01T10:00:00+03:00",
                "timezone": "Europe/Moscow",
                "source": "Open-Meteo",
            }
        return {
            "status": "ambiguous",
            "query": city,
            "source": "Open-Meteo",
            "candidates": [
                {"name": "Ростов-на-Дону", "country": "Россия", "region": "Ростовская область"},
                {"name": "Ростов Великий", "country": "Россия", "region": "Ярославская область"},
            ],
        }


class FixtureClient(verify.base.AuditClient):
    async def start(self):
        pass

    async def close(self):
        pass

    async def _request(self, payload):
        self.capture(payload)
        calls, text = [], "Модель попыталась выбрать город сама."
        if payload.get("tool_choice") == "auto":
            query = json.loads(payload["messages"][-1]["content"])["text"]
            step = next(s for c in verify.CASES for s in c["steps"] if s["query"] == query)
            text = None
            calls = [
                {
                    "id": "fixture",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": step["city"]}),
                    },
                }
            ]
        self.observation = {
            "usage": {"cost": 0},
            "raw_content": text,
            "raw_tool_calls": deepcopy(calls),
        }
        return RoutingCompletion(text, calls)


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="Needs isolated Docker")
async def test_all_flows_use_real_sql_discovery_and_substituted_weather_llm(
    database, tmp_path, monkeypatch
):
    settings, _, _, _ = database
    settings = verify.base.fixed_settings(replace(settings, llm_tool_mode="native"))
    source = FixtureMCP(settings)
    await source.start()
    try:
        manifest = {
            "protocol": {
                "now": datetime.now(UTC).isoformat(),
                "catalog_sha256": digest(
                    {
                        "tools": source.tools,
                        "resources": source.resources,
                    }
                ),
            }
        }
    finally:
        await source.close()
    monkeypatch.setattr(verify, "MCPClient", FixtureMCP)
    monkeypatch.setattr(verify.base, "AuditClient", FixtureClient)
    monkeypatch.setattr(verify, "smoke_calls", lambda: 6)
    original = verify.WeatherBudget.__init__

    def init(self, client, data, persist):
        original(
            self, client, data, persist, quota_reader=AsyncMock(return_value={"remaining": 50})
        )

    monkeypatch.setattr(verify.WeatherBudget, "__init__", init)
    data = {"attempts": [], "status": "in_progress"}
    await asyncio.wait_for(verify.run_isolated(settings, tmp_path, manifest, data), timeout=60)
    data["status"] = "completed"
    verify.summarize(data)
    assert data["metrics"]["acceptance"] == "passed", data["attempts"]
    assert data["metrics"]["llm_calls"] == 10
    assert data["metrics"]["session_started_llm_calls"] == 32
    assert data["metrics"]["mcp_calls"] == 5
