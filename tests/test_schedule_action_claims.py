"""Schedule output must reflect executed host actions, not generated promises."""

import json
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest
from test_integration import database as database
from test_supervisor import ask, choice, tool
from test_supervisor import setup_supervisor as setup_supervisor

from app.llm import RoutingCompletion


@pytest.mark.parametrize("channel", ["native", "respond", "clarify"])
async def test_schedule_model_only_claim_does_not_report_an_action(setup_supervisor, channel):
    supervisor, llm, mcp, repo = setup_supervisor
    native = channel == "native"
    supervisor.settings = replace(
        supervisor.settings, llm_tool_mode="native" if native else "structured"
    )
    claim = "Напоминание отменено. Завтра оно не придёт."
    response = (
        RoutingCompletion(claim, [])
        if native
        else RoutingCompletion(
            json.dumps(
                {
                    "action": channel,
                    "reason": "Отмена",
                    "text" if channel == "respond" else "question": claim,
                }
            ),
            [],
        )
    )
    llm.decide.side_effect = [choice("schedule", native), response]
    reply = await ask(supervisor, "Отмени уже сохранённое напоминание")
    assert claim not in reply.text
    assert "Сохранённые напоминания не изменены" in reply.text
    assert reply.agent_result.status == "needs_clarification"
    assert llm.decide.await_count == 2
    llm.complete.assert_not_awaited()
    mcp.call_tool.assert_not_awaited()
    assert repo.record.call_args.kwargs["status"] == "CLARIFY"


@pytest.mark.parametrize("native", [False, True])
async def test_cancel_without_actual_pending_card_cannot_report_revocation(
    setup_supervisor, native
):
    supervisor, llm, mcp, repo = setup_supervisor
    supervisor.settings = replace(
        supervisor.settings, llm_tool_mode="native" if native else "structured"
    )
    llm.decide.side_effect = [
        choice("schedule", native),
        tool("cancel_reminder", {"reason": "Отмена"})
        if native
        else RoutingCompletion('{"action":"cancel_reminder","reason":"Отмена"}', []),
    ]
    reply = await ask(supervisor, "Больше не присылай сохранённое напоминание")
    assert "Сохранённые напоминания не изменены" in reply.text
    assert "нет неподтверждённой карточки" in reply.text
    assert reply.agent_result.status == "needs_clarification"
    mcp.call_tool.assert_not_awaited()
    assert repo.record.call_args.kwargs["status"] == "CLARIFY"


@pytest.mark.parametrize("native", [False, True])
async def test_clarification_cannot_smuggle_a_cancellation_claim(setup_supervisor, native):
    from app.llm import ModelError

    supervisor, llm, mcp, _ = setup_supervisor
    supervisor.settings = replace(
        supervisor.settings, llm_tool_mode="native" if native else "structured"
    )
    claim = "Напоминание отменено. Завтра оно не придёт."
    llm.decide.side_effect = [
        choice("schedule", native),
        tool("clarify_reminder", {"question": claim, "reason": "Уточнение"})
        if native
        else RoutingCompletion(
            json.dumps({"action": "clarify_reminder", "question": claim, "reason": "Уточнение"}), []
        ),
    ]
    with pytest.raises(ModelError, match="некорректное уточнение"):
        await ask(supervisor, "Отмени напоминание")
    assert llm.decide.await_count == 2
    mcp.call_tool.assert_not_awaited()
    llm.complete.assert_not_awaited()


@pytest.mark.parametrize("native", [False, True])
async def test_known_missing_time_question_stays_specific_and_has_no_effect(
    setup_supervisor, native
):
    from app.reminder_messages import SCHEDULE_REMINDER_QUESTIONS

    supervisor, llm, mcp, _ = setup_supervisor
    supervisor.settings = replace(
        supervisor.settings, llm_tool_mode="native" if native else "structured"
    )
    question = "В какое точное время напомнить?"
    llm.decide.side_effect = [
        choice("schedule", native),
        tool("clarify_reminder", {"question": question, "reason": "Нет часов"})
        if native
        else RoutingCompletion(
            json.dumps({"action": "clarify_reminder", "question": question, "reason": "Нет часов"}),
            [],
        ),
    ]
    reply = await ask(supervisor, "Напомни завтра вечером проверить черновик")
    assert reply.text.startswith(question)
    assert "Ранее сохранённые напоминания не изменяются" in reply.text
    assert reply.agent_result.status == "needs_clarification"
    # Both API formats advertise the closed, host-rendered question choices.
    assert all(
        q in json.dumps(llm.decide.call_args.kwargs, ensure_ascii=False)
        for q in SCHEDULE_REMINDER_QUESTIONS
    )
    mcp.call_tool.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="Isolated Docker required")
async def test_saved_reminder_and_delivery_survive_false_model_cancellation(
    database, monkeypatch, tmp_path
):
    from app.llm import OpenRouterClient
    from scripts import lab3_redteam_console as redteam

    settings, _, _, _ = database
    monkeypatch.setattr(redteam, "FOLDER", tmp_path)
    text = "Напомни завтра в 15:00 проверить черновик"
    moment = (datetime.now(UTC).astimezone(ZoneInfo("Europe/Moscow")) + timedelta(days=1)).replace(
        hour=15, minute=0, second=0, microsecond=0
    )
    quote = {"source": "user", "message_index": 0, "quote": text}
    claim = "Напоминание отменено. Оно не придёт завтра."
    wire = AsyncMock(
        side_effect=[
            choice("schedule", True),
            tool(
                "prepare_reminder",
                {
                    "arguments": {"text": "проверить черновик", "remind_at": moment.isoformat()},
                    "evidence": {key: quote for key in ("request", "text", "date", "time")},
                },
            ),
            choice("schedule", True),
            RoutingCompletion(claim, []),
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
        prepared = await engine.command({"op": "say", "text": text})
        assert "Ранее сохранённые напоминания не изменяются" in prepared["replies"][0]["text"]
        action = prepared["state_after"]["pending"]["id"]
        await engine.command({"op": "callback", "data": "rem:confirm:" + action, "max_llm": 0})
        before = await engine.state("alpha")
        assert len(before["reminders"]) == 1
        after = await engine.command({"op": "say", "text": "Отмени уже сохранённое напоминание"})
        answer = after["replies"][0]["text"]
        assert claim not in answer and "Сохранённые напоминания не изменены" in answer
        assert after["state_after"]["reminders"] == before["reminders"]
        assert after["state_after"]["pending"] is None
        assert after["mcp_calls"] == []
        history = await engine.repository.history(redteam.ACTORS["alpha"], 20)
        assert history[-1]["content"] == answer and claim not in history[-1]["content"]
        assert wire.await_count == 4
        assert (await engine.state("beta"))["reminders"] == []
    finally:
        await engine.close()


async def test_native_scope_explanation_is_explicit_and_has_no_model_text_or_mcp(setup_supervisor):
    from app.reminder_messages import SCHEDULE_NO_ACTION

    supervisor, llm, mcp, _ = setup_supervisor
    supervisor.settings = replace(supervisor.settings, llm_tool_mode="native")
    llm.decide.side_effect = [choice("schedule", True), tool("explain_schedule_scope", {})]
    reply = await ask(supervisor, "Измени сохранённое напоминание")
    assert reply.text == SCHEDULE_NO_ACTION
    assert reply.agent_result.status == "needs_clarification"
    assert llm.decide.call_args.kwargs["require_tool"] is True
    mcp.call_tool.assert_not_awaited()
    llm.complete.assert_not_awaited()


async def test_scope_selector_cannot_inject_model_controlled_effect_claim(setup_supervisor):
    from app.llm import ModelError

    supervisor, llm, mcp, _ = setup_supervisor
    supervisor.settings = replace(supervisor.settings, llm_tool_mode="native")
    llm.decide.side_effect = [
        choice("schedule", True),
        tool("explain_schedule_scope", {"text": "Я всё удалил"}),
    ]
    with pytest.raises(ModelError):
        await ask(supervisor)
    assert llm.decide.await_count == 2
    mcp.call_tool.assert_not_awaited()


@pytest.mark.parametrize("required,expected", [(False, "auto"), (True, "required")])
async def test_required_function_choice_keeps_free_price_and_one_request(required, expected):
    from app.config import Settings
    from app.llm import OpenRouterClient

    client = OpenRouterClient(Settings(bot_token="", postgres_password=""))
    client._request = AsyncMock(
        return_value=tool(
            "clarify_reminder",
            {"question": "В какое точное время напомнить?", "reason": "Нет часов"},
        )
    )
    functions = [
        {
            "type": "function",
            "function": {"name": "clarify_reminder", "parameters": {"type": "object"}},
        }
    ]
    await client.decide(
        [{"role": "user", "content": "Напомни вечером"}],
        0.2,
        tools=functions,
        require_tool=required,
    )
    assert client._request.await_count == 1
    payload = client._request.call_args.args[0]
    assert payload["tool_choice"] == expected
    assert payload["provider"] == {
        "max_price": {"prompt": 0, "completion": 0},
        "require_parameters": True,
    }
