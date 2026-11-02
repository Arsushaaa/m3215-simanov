"""Control-plane limits and actual application plumbing, no real model requests."""

import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from zoneinfo import ZoneInfo

import pytest
from test_integration import database as database
from test_supervisor import tool

from app.config import Settings
from app.llm import ModelError, OpenRouterClient, RoutingCompletion
from scripts import lab3_redteam_console as redteam


async def test_extra_credentials_cannot_exceed_fifty_attempts_even_across_turns(monkeypatch):
    engine = SimpleNamespace(document={"used": {"key2": 49}}, persist=Mock())
    client = redteam.SlotClient(Settings(bot_token="", postgres_password=""), "key2", engine)
    wire = AsyncMock(return_value=RoutingCompletion("ok", []))
    monkeypatch.setattr(OpenRouterClient, "_request", wire)
    token = redteam.ACTIVE.set({"llm_calls": [], "max_llm": 3})
    try:
        await client._request({})
        with pytest.raises(ModelError):
            await client._request({})
    finally:
        redteam.ACTIVE.reset(token)
    assert engine.document["used"]["key2"] == 50
    wire.assert_awaited_once()


async def test_zero_budget_commands_cannot_start_http_or_spend_primary_reserve(monkeypatch):
    engine = SimpleNamespace(document={"used": {"key2": 0}}, persist=Mock())
    client = redteam.SlotClient(Settings(bot_token="", postgres_password=""), "key2", engine)
    wire = AsyncMock()
    monkeypatch.setattr(OpenRouterClient, "_request", wire)
    token = redteam.ACTIVE.set({"llm_calls": [], "max_llm": 0})
    try:
        with pytest.raises(ModelError):
            await client._request({})
    finally:
        redteam.ACTIVE.reset(token)
    wire.assert_not_awaited()
    assert engine.document["used"]["key2"] == 0
    assert "OPENROUTER_API_KEY" not in redteam.SLOTS.values()


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="Isolated Docker required")
async def test_console_real_private_handlers_and_foreign_callback_without_model_http(
    database, monkeypatch, tmp_path
):
    settings, _, _, _ = database
    monkeypatch.setattr(redteam, "FOLDER", tmp_path)
    text = "Напомни завтра в 11:00 проверить отчёт"
    moment = (datetime.now(UTC).astimezone(ZoneInfo("Europe/Moscow")) + timedelta(days=1)).replace(
        hour=11, minute=0, second=0, microsecond=0
    )
    quote = {"source": "user", "message_index": 0, "quote": text}
    wire = AsyncMock(
        side_effect=[
            tool(
                "select_route",
                {"target": "schedule", "reason": "schedule_request", "confidence": 0.9},
            ),
            tool(
                "prepare_reminder",
                {
                    "arguments": {"text": "Проверить отчёт", "remind_at": moment.isoformat()},
                    "evidence": {k: quote for k in ("request", "text", "date", "time")},
                },
            ),
        ]
    )
    monkeypatch.setattr(OpenRouterClient, "_request", wire)
    engine = redteam.Engine(
        replace(settings, llm_tool_mode="native"),
        {"key2": "fake2", "key3": "fake3"},
        {"used": {"key2": 0, "key3": 0}, "turns": []},
    )
    try:
        await engine.initialize()
        result = await engine.command({"op": "say", "actor": "alpha", "text": text})
        assert result["status"] == "completed" and result["state_after"]["pending"]
        action = result["state_after"]["pending"]["id"]
        result = await engine.command(
            {"op": "callback", "actor": "beta", "data": "rem:confirm:" + action, "max_llm": 0}
        )
        assert result["status"] == "completed"
        assert (await engine.state("alpha"))["pending"]["id"] == action
        assert (await engine.state("alpha"))["reminders"] == []
        assert (await engine.state("beta"))["reminders"] == []
        assert engine.document["used"] == {"key2": 2, "key3": 0}
    finally:
        await engine.close()


async def test_existing_session_journal_is_never_replayed(monkeypatch, tmp_path):
    import json

    monkeypatch.setattr(redteam, "FOLDER", tmp_path)
    monkeypatch.setattr(redteam, "snapshot", lambda: {})
    monkeypatch.setattr(redteam, "assert_committed", lambda path: None)
    (tmp_path / "protocol.json").write_text(json.dumps({"source_sha256": {}}))
    original = '{"used":{"key2":4},"turns":[{"status":"completed"}]}'
    (tmp_path / "journal.json").write_text(original)
    with pytest.raises(ValueError, match="never replay"):
        await redteam.run()
    assert (tmp_path / "journal.json").read_text() == original


async def test_session_credentials_must_match_frozen_mapping(monkeypatch, tmp_path):
    import json

    monkeypatch.setattr(redteam, "FOLDER", tmp_path)
    monkeypatch.setattr(redteam, "snapshot", lambda: {})
    monkeypatch.setattr(redteam, "assert_committed", lambda path: None)
    (tmp_path / "protocol.json").write_text(
        json.dumps({"source_sha256": {}, "credential_slots": {"key2": "OTHER"}})
    )
    settings = Mock(side_effect=AssertionError("Must reject before loading credentials"))
    monkeypatch.setattr(redteam, "settings_and_keys", settings)
    with pytest.raises(ValueError, match="Credential slots"):
        await redteam.run()
    settings.assert_not_called()


async def test_session_cannot_exceed_its_registered_remaining_quota(monkeypatch):
    engine = SimpleNamespace(document={"used": {"key2": 2}, "limits": {"key2": 2}}, persist=Mock())
    client = redteam.SlotClient(Settings(bot_token="", postgres_password=""), "key2", engine)
    wire = AsyncMock()
    monkeypatch.setattr(OpenRouterClient, "_request", wire)
    token = redteam.ACTIVE.set({"llm_calls": [], "max_llm": 3})
    try:
        with pytest.raises(ModelError, match="budget exhausted"):
            await client._request({})
    finally:
        redteam.ACTIVE.reset(token)
    wire.assert_not_awaited()
    assert engine.document["used"]["key2"] == 2


def test_preparation_diagnostic_locates_wrong_index_without_persisting_quotes():
    import json

    private = "Synthetic private instruction text must never be persisted " * 4
    refs = {
        key: {"source": "user", "message_index": 0, "quote": private}
        for key in ("request", "text", "date", "time")
    }
    diagnostic = redteam.preparation_diagnostic(
        json.dumps(
            {
                "arguments": {"text": private, "remind_at": "2026-11-03T18:00:00+03:00"},
                "evidence": refs,
            }
        ),
        {0: "В 18:00.", 1: private},
    )
    assert diagnostic["evidence"]["text"] == {
        "source": "user",
        "declared_index": 0,
        "matches_declared": False,
        "found_at": [1],
    }
    assert private not in json.dumps(diagnostic, ensure_ascii=False)
    assert diagnostic["text_sha256"] == redteam.digest(private.encode())


def test_preparation_diagnostic_checks_actual_visible_user_wrappers_only():
    import json

    sources = redteam.visible_sources(
        {
            "messages": [
                {"role": "system", "content": json.dumps({"message_index": 2, "text": "hidden"})},
                {"role": "user", "content": "not a wrapper"},
                {"role": "user", "content": json.dumps({"message_index": 0, "text": "В 18:00."})},
                {
                    "role": "user",
                    "content": json.dumps({"message_index": 1, "text": "Завтра купить хлеб"}),
                },
            ]
        }
    )
    assert sources == {0: "В 18:00.", 1: "Завтра купить хлеб"}
    assert redteam.preparation_diagnostic("{", sources) == {"error": "invalid_preparation_shape"}
