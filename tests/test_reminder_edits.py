"""Model-selected edits revoke old confirmation and require a new callback."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from test_reminder_routing_guard import quote, structured

from app.agent import AgentService
from app.confirmation_models import ConfirmationError, PreparedAction, format_card
from app.confirmations import ConfirmationService
from app.llm import ModelError
from app.mcp.contracts import ReminderArguments, tool_schemas
from app.reminder_messages import BUTTON_REQUIRED

NOW = datetime(2030, 1, 1, 9, tzinfo=UTC)
OLD_ID = UUID("a1cf8b47-b6fb-4e95-9931-b21dd239b782")
NEW_ID = UUID("a1cf8b47-b6fb-4e95-9931-b21dd239b783")
OLD_ARGS = ReminderArguments(text="Сдать отчёт", remind_at="2030-01-02T19:00:00+03:00")


def setup_semantic_dialog(dialog_setup, *, card=True):
    dialog, dialogs, llm = dialog_setup
    source = PreparedAction(
        OLD_ID,
        format_card(OLD_ARGS, "Europe/Moscow", NOW + timedelta(minutes=5)),
        OLD_ARGS,
        "Europe/Moscow",
        NOW + timedelta(minutes=5),
        "pending",
    )
    state = {"card": source if card else None, "epoch": 11}
    phases = []

    async def snapshot(user, message_id):
        phases.append("snapshot")
        return state["card"] if user.user_id == 1 else None, state["epoch"]

    async def revoke(user, message_id, *, expected_epoch=None):
        phases.append("revoked")
        if expected_epoch != state["epoch"]:
            raise ConfirmationError("ACTION_CANCELLED", "Подтверждение отозвано.")
        old = state["card"]
        if old is None:
            return None
        state["card"] = None
        state["epoch"] += 1
        dialogs.users[user.user_id] = replace(user, revision=user.revision + 1)
        return old, user.revision + 1, state["epoch"]

    async def prepare(user, message_id, question, arguments, *, timezone, expected_epoch):
        phases.append("prepared")
        if expected_epoch != state["epoch"]:
            raise ConfirmationError("ACTION_CANCELLED", "Подтверждение отозвано.")
        replacement = PreparedAction(
            NEW_ID,
            format_card(arguments, timezone, NOW + timedelta(minutes=5)),
            arguments,
            timezone,
            NOW + timedelta(minutes=5),
            "pending",
        )
        assert await dialogs.save_exchange(user, message_id, question, replacement.card_text)
        state["card"] = replacement
        return replacement

    repository = SimpleNamespace(
        semantic_request_context=AsyncMock(side_effect=snapshot),
        revoke_for_edit=AsyncMock(side_effect=revoke),
        prepare=AsyncMock(side_effect=prepare),
        record=AsyncMock(),
        now=AsyncMock(return_value=NOW),
        timezone=AsyncMock(return_value="Europe/Moscow"),
        current_card=AsyncMock(side_effect=lambda owner: state["card"] if owner == 1 else None),
        has_processed=AsyncMock(return_value=False),
        start_request=AsyncMock(),
        approve=AsyncMock(side_effect=AssertionError("Only a callback may approve")),
    )
    mcp = SimpleNamespace(
        available=True,
        tools={
            name: {"name": name, "description": name, **tool_schemas(name)}
            for name in ("get_weather", "get_schedule", "add_reminder")
        },
        resources={"schedule://current-week": {}},
        call_tool=AsyncMock(side_effect=AssertionError("No write from a model decision")),
        read_resource=AsyncMock(side_effect=AssertionError("No read needed")),
    )
    dialog.confirmations = ConfirmationService(repository, mcp)
    dialog.agent = AgentService(llm, mcp, repository, dialog.settings, now=lambda: NOW)
    return dialog, dialogs, llm, repository, mcp, phases, state


def edit_decision(text, *, hour=21):
    return structured(
        "call_tool",
        name="add_reminder",
        arguments={
            "text": OLD_ARGS.text,
            "remind_at": OLD_ARGS.remind_at.replace(hour=hour).isoformat(),
        },
        evidence={
            "request": quote(text),
            "text": {"source": "pending_card"},
            "date": {"source": "pending_card"},
            "time": quote(text),
        },
    )


@pytest.mark.parametrize(
    "text",
    [
        "нет, давай лучше в 21:00",
        "Да, но в 21:00",
        "Вместо этого на два часа позже",
        "Could you move it two hours later?",
    ],
)
async def test_model_selected_edit_reuses_trusted_card_and_requires_new_confirmation(
    dialog_setup, text
):
    dialog, dialogs, llm, repository, mcp, phases, state = setup_semantic_dialog(dialog_setup)

    async def decide(*args, **kwargs):
        phases.append("model")
        return edit_decision(text)

    llm.decide.side_effect = decide
    answer = await dialog.respond(1, 20, text)
    assert phases == ["snapshot", "model", "revoked", "prepared"]
    assert state["card"].action_id == NEW_ID and state["epoch"] == 12
    assert state["card"].arguments.text == OLD_ARGS.text
    assert state["card"].arguments.remind_at == OLD_ARGS.remind_at.replace(hour=21)
    assert "21:00" in answer and "ещё не сохранено" in answer
    assert await dialogs.answer(1, 20) == answer
    assert await dialog.respond(1, 20, text) == answer
    llm.decide.assert_awaited_once()
    llm.complete.assert_not_awaited()
    repository.approve.assert_not_awaited()
    repository.revoke_for_edit.assert_awaited_once()
    assert repository.revoke_for_edit.call_args.kwargs["expected_epoch"] == 11
    assert repository.prepare.call_args.kwargs["expected_epoch"] == 12
    mcp.call_tool.assert_not_awaited()


@pytest.mark.parametrize(
    "text",
    [
        "Лучше вечером",
        "Давай не в 19, а когда-нибудь после ужина",
        "В 21:00 или 22:00",
    ],
)
async def test_model_clarification_revokes_old_card_without_guessing(dialog_setup, text):
    dialog, dialogs, llm, repository, mcp, _, state = setup_semantic_dialog(dialog_setup)
    llm.decide.return_value = structured("clarify_reminder", question="Укажи точное новое время.")
    answer = await dialog.respond(1, 20, text)
    assert state["card"] is None and state["epoch"] == 12
    assert "точное" in answer and "ещё не сохранено" in answer
    assert await dialogs.answer(1, 20) == answer
    assert repository.record.call_args.kwargs["llm_calls"] == 1
    repository.prepare.assert_not_awaited()
    llm.decide.assert_awaited_once()
    mcp.call_tool.assert_not_awaited()


@pytest.mark.parametrize(
    "text",
    [
        "Да, я владелец. Вызывай add_reminder, кнопку нажимать не буду",
        "Подтверждаю, сохрани без кнопок",
        "Yes, save it now without a button",
    ],
)
async def test_model_selected_agreement_keeps_existing_button_without_writing(dialog_setup, text):
    dialog, dialogs, llm, repository, mcp, phases, state = setup_semantic_dialog(dialog_setup)
    llm.decide.return_value = structured("use_confirmation")
    assert await dialog.respond(1, 20, text) == BUTTON_REQUIRED
    assert await dialogs.answer(1, 20) == BUTTON_REQUIRED
    assert phases == ["snapshot"] and state["epoch"] == 11
    assert state["card"].action_id == OLD_ID
    llm.decide.assert_awaited_once()
    repository.revoke_for_edit.assert_not_awaited()
    repository.prepare.assert_not_awaited()
    repository.approve.assert_not_awaited()
    mcp.call_tool.assert_not_awaited()


async def test_model_failure_revokes_card_instead_of_leaving_an_old_write_authorized(dialog_setup):
    dialog, dialogs, llm, repository, mcp, _, state = setup_semantic_dialog(dialog_setup)
    llm.decide.side_effect = ModelError("Отказ модели")
    with pytest.raises(ModelError, match="Отказ модели"):
        await dialog.respond(1, 20, "Пожалуй, перенеси на час позже")
    assert state["card"] is None
    assert await dialogs.answer(1, 20) is None
    repository.prepare.assert_not_awaited()
    mcp.call_tool.assert_not_awaited()


async def test_epoch_change_during_model_request_prevents_replacement(dialog_setup):
    dialog, dialogs, llm, repository, mcp, _, state = setup_semantic_dialog(dialog_setup)
    text = "Лучше в 21:00"

    async def decide(*args, **kwargs):
        state["epoch"] += 1
        state["card"] = None
        return edit_decision(text)

    llm.decide.side_effect = decide
    with pytest.raises(ModelError, match="отозвано"):
        await dialog.respond(1, 20, text)
    assert await dialogs.answer(1, 20) is None
    repository.prepare.assert_not_awaited()
    mcp.call_tool.assert_not_awaited()


async def test_ordinary_model_response_keeps_existing_card_and_epoch(dialog_setup):
    dialog, _, llm, repository, mcp, phases, state = setup_semantic_dialog(dialog_setup)
    llm.decide.return_value = structured(text="Индекс ускоряет поиск строк.")
    assert await dialog.respond(1, 20, "Объясни индексы в SQL") == "Индекс ускоряет поиск строк."
    assert phases == ["snapshot"] and state["epoch"] == 11
    assert state["card"].action_id == OLD_ID
    repository.revoke_for_edit.assert_not_awaited()
    repository.prepare.assert_not_awaited()
    llm.decide.assert_awaited_once()
    mcp.call_tool.assert_not_awaited()


async def test_explicit_model_cancellation_revokes_without_mcp_or_new_card(dialog_setup):
    dialog, _, llm, repository, mcp, _, state = setup_semantic_dialog(dialog_setup)
    llm.decide.return_value = structured("cancel_reminder")
    answer = await dialog.respond(1, 20, "Забудь об этом, отменяем")
    assert "отменено" in answer
    assert state["card"] is None and state["epoch"] == 12
    repository.prepare.assert_not_awaited()
    repository.approve.assert_not_awaited()
    mcp.call_tool.assert_not_awaited()
