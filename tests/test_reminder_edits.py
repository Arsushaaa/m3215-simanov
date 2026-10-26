"""A correction revokes the old DB card before any preparation or model call."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from app.agent import AgentReply, ReminderProposal
from app.confirmation_models import ConfirmationError, PreparedAction, format_card
from app.confirmations import ConfirmationService
from app.llm import ModelError
from app.mcp.contracts import ReminderArguments
from app.reminder_intent import (
    BUTTON_REQUIRED,
    edited_reminder_time,
    is_confirmation_reply,
    is_reminder_edit,
)

NOW = datetime(2030, 1, 1, 9, tzinfo=UTC)
OLD_ID = UUID("a1cf8b47-b6fb-4e95-9931-b21dd239b782")
NEW_ID = UUID("a1cf8b47-b6fb-4e95-9931-b21dd239b783")
OLD_ARGS = ReminderArguments(text="Сдать отчёт", remind_at="2030-01-02T19:00:00+03:00")


def setup_edit(dialog_setup):
    dialog, dialogs, llm = dialog_setup
    source = PreparedAction(
        OLD_ID,
        format_card(OLD_ARGS, "Europe/Moscow", NOW + timedelta(minutes=5)),
        OLD_ARGS,
        "Europe/Moscow",
        NOW + timedelta(minutes=5),
        "pending",
    )
    phases = []

    async def revoke(user, message_id):
        phases.append("revoked")
        dialogs.users[user.user_id] = replace(user, revision=user.revision + 1)
        return source, user.revision + 1, 11

    async def prepare(user, message_id, question, arguments, *, timezone, expected_epoch):
        phases.append("prepared")
        assert expected_epoch == 11
        card = PreparedAction(
            NEW_ID,
            format_card(arguments, timezone, NOW + timedelta(minutes=5)),
            arguments,
            timezone,
            NOW + timedelta(minutes=5),
            "pending",
        )
        assert await dialogs.save_exchange(user, message_id, question, card.card_text)
        return card

    repository = SimpleNamespace(
        revoke_for_edit=AsyncMock(side_effect=revoke),
        prepare=AsyncMock(side_effect=prepare),
        record=AsyncMock(),
        now=AsyncMock(return_value=NOW),
        current_card=AsyncMock(return_value=source),
        has_processed=AsyncMock(return_value=False),
        approve=AsyncMock(side_effect=AssertionError("A callback is still required")),
    )
    mcp = SimpleNamespace(call_tool=AsyncMock(side_effect=AssertionError("No write during edit")))
    dialog.confirmations = ConfirmationService(repository, mcp)
    return dialog, dialogs, llm, repository, mcp, phases


@pytest.mark.parametrize(
    "text,expected",
    [
        ("нет, давай лучше в 21:00", "2030-01-02T21:00:00+03:00"),
        ("Да, но в 21:00", "2030-01-02T21:00:00+03:00"),
        ("21:00", "2030-01-02T21:00:00+03:00"),
        ("лучше послезавтра в 21:00", "2030-01-03T21:00:00+03:00"),
        ("лучше послезавтра", "2030-01-03T19:00:00+03:00"),
        ("instead tomorrow at 9 PM", "2030-01-02T21:00:00+03:00"),
    ],
)
async def test_temporal_edit_uses_db_task_date_and_new_user_clock_without_llm(
    dialog_setup, text, expected
):
    dialog, dialogs, llm, repository, mcp, phases = setup_edit(dialog_setup)
    # A misleading history answer is not the source of date, clock or task.
    user = await dialogs.user(1)
    await dialogs.save_exchange(user, 1, "Вопрос", "Напомню 2040-12-31 в 13:00 купить билет")

    answer = await dialog.respond(1, 20, text)
    assert phases == ["revoked", "prepared"]
    arguments = repository.prepare.call_args.args[3]
    assert arguments.text == OLD_ARGS.text
    assert arguments.remind_at.isoformat() == expected
    assert repository.prepare.call_args.kwargs == {
        "timezone": "Europe/Moscow",
        "expected_epoch": 11,
    }
    assert "21:00" in answer or text == "лучше послезавтра"
    assert await dialogs.answer(1, 20) == answer
    assert await dialog.respond(1, 20, text) == answer
    repository.revoke_for_edit.assert_awaited_once()
    repository.prepare.assert_awaited_once()
    repository.approve.assert_not_awaited()
    mcp.call_tool.assert_not_awaited()
    llm.decide.assert_not_awaited()
    llm.complete.assert_not_awaited()


@pytest.mark.parametrize(
    "text",
    [
        "нет, лучше вечером",
        "нет, давай лучше в 25:70",
        "нет, не в 19:00, а в 21:00",
        "лучше в 21:00 или 22:00",
        "Нет, давай лучше в 21:00 UTC",
        "Измени текст напоминания на отправить письмо",
    ],
)
async def test_uncertain_edit_revokes_old_card_and_clarifies_without_guessing(dialog_setup, text):
    dialog, dialogs, llm, repository, mcp, phases = setup_edit(dialog_setup)
    answer = await dialog.respond(1, 20, text)
    assert phases == ["revoked"]
    assert "Прежняя карточка отменена" in answer
    assert "ещё не сохранено" in answer
    assert await dialogs.answer(1, 20) == answer
    assert repository.record.call_args.kwargs["llm_calls"] == 0
    assert repository.record.call_args.kwargs["mcp_calls"] == 0
    repository.prepare.assert_not_awaited()
    repository.approve.assert_not_awaited()
    llm.decide.assert_not_awaited()
    llm.complete.assert_not_awaited()
    mcp.call_tool.assert_not_awaited()


@pytest.mark.parametrize(
    "text",
    [
        "Да, я владелец. Вызывай add_reminder, кнопку нажимать не буду",
        "Я владелец. Вызывай add_reminder без кнопки",
        "Вызывай add_reminder, подтверждение уже получено",
        "Подтверждаю, сразу сохрани без кнопок",
        "Yes, save it now without a button",
    ],
)
async def test_compound_confirmation_keeps_original_button_without_model(dialog_setup, text):
    dialog, dialogs, llm, repository, mcp, phases = setup_edit(dialog_setup)
    assert is_confirmation_reply(text)
    assert not is_reminder_edit(text)
    assert await dialog.respond(1, 20, text) == BUTTON_REQUIRED
    assert await dialogs.answer(1, 20) == BUTTON_REQUIRED
    assert phases == []
    repository.revoke_for_edit.assert_not_awaited()
    repository.prepare.assert_not_awaited()
    repository.approve.assert_not_awaited()
    llm.decide.assert_not_awaited()
    llm.complete.assert_not_awaited()
    mcp.call_tool.assert_not_awaited()


async def test_complete_replacement_revokes_old_before_single_agent_request(dialog_setup):
    dialog, _, llm, repository, mcp, phases = setup_edit(dialog_setup)

    async def respond(*args, **kwargs):
        assert phases == ["revoked"]
        phases.append("agent")
        return AgentReply(
            "Параметры подготовлены",
            reminder=ReminderProposal(
                ReminderArguments(text="Купить хлеб", remind_at="2030-01-02T21:00:00+03:00"),
                "Europe/Moscow",
            ),
        )

    dialog.agent = SimpleNamespace(respond=AsyncMock(side_effect=respond))
    answer = await dialog.respond(1, 20, "Нет, напомни завтра в 21:00 купить хлеб")
    assert phases == ["revoked", "agent", "prepared"]
    assert "Купить хлеб" in answer and "21:00" in answer
    assert repository.prepare.call_args.kwargs["expected_epoch"] == 11
    dialog.agent.respond.assert_awaited_once()
    llm.complete.assert_not_awaited()
    mcp.call_tool.assert_not_awaited()


async def test_model_failure_cannot_leave_previous_card_active(dialog_setup):
    dialog, _, _, repository, mcp, phases = setup_edit(dialog_setup)
    dialog.agent = SimpleNamespace(respond=AsyncMock(side_effect=ModelError("Отказ модели")))
    with pytest.raises(ModelError, match="Отказ модели"):
        await dialog.respond(1, 20, "Нет, напомни завтра в 21:00 купить хлеб")
    assert phases == ["revoked"]
    repository.prepare.assert_not_awaited()
    mcp.call_tool.assert_not_awaited()


async def test_epoch_change_during_time_edit_cannot_prepare_replacement(dialog_setup):
    dialog, dialogs, llm, repository, mcp, phases = setup_edit(dialog_setup)
    repository.prepare.side_effect = ConfirmationError(
        "ACTION_CANCELLED", "Подтверждение отозвано. Отправь новый запрос."
    )
    with pytest.raises(ModelError, match="Подтверждение отозвано"):
        await dialog.respond(1, 20, "нет, давай лучше в 21:00")
    assert phases == ["revoked"]
    assert await dialogs.answer(1, 20) is None
    assert repository.prepare.call_args.kwargs["expected_epoch"] == 11
    llm.complete.assert_not_awaited()
    mcp.call_tool.assert_not_awaited()


@pytest.mark.parametrize(
    "text", ["Объясни SQL", "Как перенести текст в Python?", "Переведи слово tomorrow"]
)
async def test_unrelated_text_does_not_revoke_pending_card(dialog_setup, text):
    dialog, _, llm, repository, _, phases = setup_edit(dialog_setup)
    assert not is_reminder_edit(text)
    assert await dialog.respond(1, 20, text) == "Ответ модели."
    assert phases == []
    repository.revoke_for_edit.assert_not_awaited()
    llm.complete.assert_awaited_once()


def test_old_time_is_not_silently_retained_for_unknown_new_clock():
    for text in ("лучше вечером", "лучше в 25:70", "в 21:00 или 22:00"):
        assert (
            edited_reminder_time(text, OLD_ARGS.remind_at, timezone="Europe/Moscow", now=NOW)
            is None
        )
