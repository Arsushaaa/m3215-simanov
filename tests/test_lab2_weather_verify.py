import asyncio
import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_integration import database as database
from test_lab2_adversarial_live import FakeWireClient, LocalMCP, SyntheticAuditClient

from app.llm import ModelError
from scripts import lab2_weather_verify as verify
from scripts.lab2_experiment import digest


@pytest.mark.parametrize("remaining", [3, 0, None, True])
async def test_weather_verification_refuses_insufficient_quota(bot_settings, tmp_path, remaining):
    with pytest.raises(ValueError, match="Four remaining"):
        await verify.prepare(
            replace(bot_settings, llm_tool_mode="native"),
            tmp_path,
            quota={"remaining": remaining},
        )


async def test_manifest_binds_previous_32_attempts_without_changing_reports(bot_settings, tmp_path):
    settings = replace(bot_settings, llm_tool_mode="native")
    before = verify.baseline()
    catalog = SimpleNamespace(available=True, tools={}, resources={})
    manifest = await verify.prepare(settings, tmp_path, quota={"remaining": 18}, catalog=catalog)
    assert manifest["protocol"]["session_baseline"] == before == verify.baseline()
    assert before["started_llm_calls"] == 32 and len(before["files_sha256"]) == 4
    assert manifest["plan"]["maximum_llm_calls"] == 4
    assert len(manifest["plan"]["queries"]) == 2
    with pytest.raises(ValueError, match="Never overwrite"):
        await verify.prepare(settings, tmp_path, quota={"remaining": 18}, catalog=catalog)


async def test_weather_budget_refuses_fifth_request_before_metadata_or_generation(bot_settings):
    row = {"maximum_llm_calls": 4, "llm_calls": []}
    data = {"attempts": [row]}
    client, quota = FakeWireClient(bot_settings), AsyncMock(return_value={"remaining": 18})
    llm = verify.WeatherBudget(client, data, lambda: None, quota_reader=quota)
    llm.row = row
    for _ in range(2):
        llm.step = {"maximum_llm_calls": 2, "llm_calls": 0}
        await llm.decide([], 0.2, tools=[])
        await llm.complete([], 0.2, tools=[], disable_tools=True)
    with pytest.raises(ModelError, match="session budget"):
        await llm.decide([], 0.2, tools=[])
    assert len(client.calls) == quota.await_count == 4


async def test_weather_verification_never_replays_an_existing_checkpoint(
    bot_settings, tmp_path, monkeypatch
):
    manifest = {"fixed": "source"}
    result = {"manifest_sha256": digest(manifest), "status": "in_progress", "attempts": []}
    (tmp_path / "results.json").write_text(json.dumps(result))
    monkeypatch.setattr(verify, "validate", lambda *_: manifest)

    def forbidden(*_):
        pytest.fail("A checkpoint cannot create another database or repeat model requests.")

    monkeypatch.setattr(verify.base, "isolated_database", forbidden)
    assert await verify.run(replace(bot_settings, llm_tool_mode="native"), tmp_path) == result


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="Requires isolated Docker")
async def test_weather_verification_uses_unchanged_case_real_sql_stdio_and_fake_model(
    database, tmp_path, monkeypatch
):
    settings, _, _, _ = database
    settings = verify.base.fixed_settings(replace(settings, llm_tool_mode="native"))
    source = LocalMCP(settings)
    await source.start()
    try:
        now = datetime.now(UTC).replace(microsecond=0)
        SyntheticAuditClient.now = now
        manifest = {
            "protocol": {
                "now": now.isoformat(),
                "catalog_sha256": digest({"tools": source.tools, "resources": source.resources}),
            }
        }
    finally:
        await source.close()
    monkeypatch.setattr(verify, "MCPClient", LocalMCP)
    monkeypatch.setattr(verify.base, "AuditClient", SyntheticAuditClient)
    original = verify.WeatherBudget.__init__

    def init(self, client, data, persist):
        original(
            self, client, data, persist, quota_reader=AsyncMock(return_value={"remaining": 18})
        )

    monkeypatch.setattr(verify.WeatherBudget, "__init__", init)
    data = {"attempts": [], "status": "in_progress"}
    await asyncio.wait_for(verify.run_isolated(settings, tmp_path, manifest, data), timeout=60)
    data["status"] = "completed"
    verify.summarize(data)
    assert data["metrics"]["acceptance"] == "passed", data["attempts"]
    assert data["metrics"]["llm_calls"] == 4
    assert data["metrics"]["session_started_llm_calls"] == 36
    assert data["metrics"]["mcp_calls"] == 2
