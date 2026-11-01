"""Trace actual Telegram outcomes without using Telegram HTTP or an LLM endpoint."""

import json
from unittest.mock import AsyncMock

import pytest
from aiogram.methods import SendMessage
from test_confirmation_handlers import sent, text_update
from test_confirmation_handlers import setup_ui as setup_ui
from test_supervisor import MemoryTraces

from app.handlers.assistant import DATABASE_ERROR
from app.tracing.runtime import TraceManager


async def test_handler_records_delivered_reply_without_message_contents(setup_ui):
    dispatcher, bot, dialog = setup_ui
    memory = MemoryTraces()
    dialog.traces = TraceManager(memory)
    dialog.respond.return_value = "private answer secret"
    await dispatcher.feed_update(bot, text_update("private user secret"), dialog=dialog)
    assert sent(bot)[0].text == "private answer secret"
    spans = list(memory.spans.values())
    assert any(s.operation == "send_message" and s.status == "ok" for s in spans)
    assert "private" not in json.dumps([s.model_dump(mode="json") for s in spans])


async def test_telegram_delivery_failure_is_not_a_successful_trace(setup_ui):
    dispatcher, bot, dialog = setup_ui
    memory = MemoryTraces()
    dialog.traces = TraceManager(memory)

    async def fail_send(*args, **kwargs):
        if isinstance(args[1], SendMessage):
            raise RuntimeError("transport failed")

    bot.session.side_effect = fail_send
    with pytest.raises(RuntimeError, match="transport failed"):
        await dispatcher.feed_update(bot, text_update("Привет"), dialog=dialog)
    assert next(iter(memory.headers.values()))["status"] == "failed"
    assert any(
        s.operation == "send_message" and s.status == "failed" for s in memory.spans.values()
    )
    dialog.respond.assert_awaited_once()  # No regeneration because Telegram failed.


async def test_trace_view_does_not_replace_last_and_never_calls_model(setup_ui):
    dispatcher, bot, dialog = setup_ui
    memory = MemoryTraces()
    dialog.traces = TraceManager(memory)
    dialog.trace = AsyncMock(return_value="Трасса: 123")
    await dispatcher.feed_update(bot, text_update("/trace last"), dialog=dialog)
    assert memory.headers == {}
    dialog.trace.assert_awaited_once_with(1, "last")
    dialog.respond.assert_not_awaited()
    assert sent(bot)[0].text == "Трасса: 123"


async def test_trace_database_failure_stops_before_llm_and_reports_error(setup_ui):
    dispatcher, bot, dialog = setup_ui
    memory = MemoryTraces()
    memory.start = AsyncMock(side_effect=RuntimeError("database down private details"))
    dialog.traces = TraceManager(memory)
    await dispatcher.feed_update(bot, text_update("Вопрос"), dialog=dialog)
    dialog.respond.assert_not_awaited()
    assert sent(bot)[0].text == DATABASE_ERROR


async def test_whitespace_message_does_not_crash_trace_command_detection(setup_ui):
    dispatcher, bot, dialog = setup_ui
    memory = MemoryTraces()
    dialog.traces = TraceManager(memory)
    await dispatcher.feed_update(bot, text_update("   "), dialog=dialog)
    assert len(memory.headers) == 1
