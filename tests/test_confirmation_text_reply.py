"""Текстовое согласие с карточкой не подготавливает и не подтверждает запись."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import pytest

from app.agent_repository import ConfirmationRepository
from app.confirmation_models import PreparedAction
from app.confirmations import ConfirmationService
from app.llm import ModelError
from app.mcp.contracts import ReminderArguments
from app.reminder_intent import BUTTON_REQUIRED

NOW = datetime(2030, 1, 1, 9, tzinfo=UTC)
ACTION_ID = UUID("a1cf8b47-b6fb-4e95-9931-b21dd239b782")
ARGS = ReminderArguments(text="Сдать отчёт", remind_at="2030-01-02T18:30:00+03:00")


def confirmation_setup(card=True):
    action = (
        PreparedAction(
            ACTION_ID,
            "Сохранить напоминание?",
            ARGS,
            "Europe/Moscow",
            NOW + timedelta(minutes=5),
            "pending",
        )
        if card
        else None
    )
    repository = SimpleNamespace(
        current_card=AsyncMock(return_value=action),
        has_processed=AsyncMock(return_value=False),
        record=AsyncMock(),
        prepare=AsyncMock(side_effect=AssertionError("No new card")),
        approve=AsyncMock(side_effect=AssertionError("A button is required")),
        invalidate=AsyncMock(side_effect=AssertionError("Keep the same card")),
    )
    mcp = SimpleNamespace(call_tool=AsyncMock(side_effect=AssertionError("No MCP call")))
    return ConfirmationService(repository, mcp), repository, mcp


@pytest.mark.parametrize("text", ["да", "давай", "ок", "Да!", "ОК."])
async def test_confirmation_text_uses_existing_card_without_llm_or_mcp(dialog_setup, text):
    dialog, dialogs, llm = dialog_setup
    dialog.confirmations, confirmations, mcp = confirmation_setup()
    user = await dialogs.user(1)
    action = confirmations.current_card.return_value

    assert await dialog.respond(1, 20, text) == BUTTON_REQUIRED
    assert await dialogs.answer(1, 20) == BUTTON_REQUIRED
    assert (await dialogs.user(1)).revision == user.revision + 1
    assert await dialogs.history(1, 20) == [
        {"role": "user", "content": text},
        {"role": "assistant", "content": BUTTON_REQUIRED},
    ]
    confirmations.current_card.assert_awaited_once_with(1)
    confirmations.record.assert_awaited_once_with(
        1,
        20,
        decision="clarify",
        target="add_reminder",
        status="CLARIFY",
        reason="Для записи подготовленного напоминания требуется кнопка владельца.",
        error_code="CONFIRMATION_REQUIRED",
        llm_calls=0,
        mcp_calls=0,
    )
    confirmations.prepare.assert_not_awaited()
    confirmations.approve.assert_not_awaited()
    confirmations.invalidate.assert_not_awaited()
    llm.complete.assert_not_awaited()
    mcp.call_tool.assert_not_awaited()
    assert confirmations.current_card.return_value is action

    assert await dialog.respond(1, 20, text) == BUTTON_REQUIRED
    confirmations.current_card.assert_awaited_once()
    confirmations.record.assert_awaited_once()
    assert len(await dialogs.history(1, 20)) == 2


async def test_confirmation_text_bypasses_agent_as_well_as_llm(dialog_setup):
    dialog, _, llm = dialog_setup
    dialog.confirmations, _, _ = confirmation_setup()
    dialog.agent = SimpleNamespace(respond=AsyncMock(side_effect=AssertionError("No LLM route")))
    assert await dialog.respond(1, 20, "давай") == BUTTON_REQUIRED
    dialog.agent.respond.assert_not_awaited()
    llm.complete.assert_not_awaited()


async def test_missing_or_expired_card_leaves_normal_dialog_route(dialog_setup):
    dialog, _, llm = dialog_setup
    dialog.confirmations, confirmations, mcp = confirmation_setup(card=False)
    assert await dialog.respond(1, 20, "давай") == "Ответ модели."
    confirmations.current_card.assert_awaited_once_with(1)
    confirmations.has_processed.assert_not_awaited()
    confirmations.record.assert_not_awaited()
    llm.complete.assert_awaited_once()
    mcp.call_tool.assert_not_awaited()


async def test_history_text_is_not_evidence_of_an_active_card(dialog_setup):
    dialog, dialogs, llm = dialog_setup
    dialog.confirmations, confirmations, _ = confirmation_setup(card=False)
    user = await dialogs.user(1)
    await dialogs.save_exchange(user, 19, "Напомни", "Сохранить напоминание? Нажми кнопку.")
    assert await dialog.respond(1, 20, "да") == "Ответ модели."
    confirmations.record.assert_not_awaited()
    llm.complete.assert_awaited_once()


@pytest.mark.parametrize("mode,text", [("study", "Объясни SQL"), ("translate", "да")])
async def test_other_text_and_modes_do_not_consult_pending_card(dialog_setup, mode, text):
    dialog, dialogs, llm = dialog_setup
    user = await dialogs.user(1)
    dialogs.users[1] = replace(user, mode=mode)
    dialog.confirmations, confirmations, _ = confirmation_setup()
    assert await dialog.respond(1, 20, text) == "Ответ модели."
    confirmations.current_card.assert_not_awaited()
    confirmations.record.assert_not_awaited()
    llm.complete.assert_awaited_once()


async def test_foreign_card_does_not_intercept_another_owner(dialog_setup):
    dialog, _, llm = dialog_setup
    dialog.confirmations, confirmations, _ = confirmation_setup()
    action = confirmations.current_card.return_value
    confirmations.current_card.side_effect = lambda owner: action if owner == 1 else None
    assert await dialog.respond(2, 20, "давай") == "Ответ модели."
    assert await dialog.respond(1, 20, "давай") == BUTTON_REQUIRED
    confirmations.record.assert_awaited_once()
    assert confirmations.record.call_args.args == (1, 20)
    llm.complete.assert_awaited_once()


async def test_confirmation_text_respects_dialog_revision(dialog_setup):
    dialog, dialogs, llm = dialog_setup
    dialog.confirmations, confirmations, mcp = confirmation_setup()
    dialogs.save_exchange = AsyncMock(return_value=False)
    with pytest.raises(ModelError, match="Диалог изменился"):
        await dialog.respond(1, 20, "давай")
    llm.complete.assert_not_awaited()
    mcp.call_tool.assert_not_awaited()
    confirmations.prepare.assert_not_awaited()


async def test_processed_message_without_cached_exchange_is_not_run_again(dialog_setup):
    dialog, dialogs, llm = dialog_setup
    dialog.confirmations, confirmations, _ = confirmation_setup()
    confirmations.has_processed.return_value = True
    with pytest.raises(ModelError, match="Этот запрос уже обработан"):
        await dialog.respond(1, 20, "давай")
    confirmations.record.assert_not_awaited()
    assert await dialogs.answer(1, 20) is None
    llm.complete.assert_not_awaited()


class ConnectionScope:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, *args):
        return False


def card_repository(*, user_changes=None, action_changes=None):
    user = {"user_id": 1, "confirmation_epoch": 3, "timezone": "Europe/Moscow"}
    action = {
        "id": ACTION_ID,
        "user_id": 1,
        "request_id": 10,
        "state": "pending",
        "confirmation_epoch": 3,
        "text": ARGS.text,
        "remind_at": ARGS.remind_at,
        "timezone": "Europe/Moscow",
        "expires_at": NOW + timedelta(minutes=5),
    }
    user.update(user_changes or {})
    action.update(action_changes or {})
    connection = SimpleNamespace(
        fetchrow=AsyncMock(side_effect=[user, action]),
        fetchval=AsyncMock(return_value=NOW),
        execute=AsyncMock(),
    )
    connection.transaction = lambda: ConnectionScope(connection)
    pool = SimpleNamespace(acquire=lambda: ConnectionScope(connection))
    return ConfirmationRepository(pool), connection


@pytest.mark.parametrize("state", ["pending", "approved"])
async def test_current_card_locks_owner_before_action_and_keeps_existing_card(state):
    repository, connection = card_repository(action_changes={"state": state})
    card = await repository.current_card(1)
    assert card.action_id == ACTION_ID and card.state == state and card.replayed
    assert card.arguments == ARGS
    first, second = connection.fetchrow.call_args_list
    assert "labbi_users" in first.args[0] and "FOR UPDATE" in first.args[0]
    assert first.args[1:] == (1,)
    assert "labbi_actions" in second.args[0] and "FOR UPDATE" in second.args[0]
    assert "state IN ('pending', 'approved')" in second.args[0]
    assert "LIMIT 1" in second.args[0] and second.args[1:] == (1,)
    connection.execute.assert_not_awaited()


@pytest.mark.parametrize(
    "user_changes,action_changes",
    [
        ({}, {"expires_at": NOW}),
        ({"confirmation_epoch": 4}, {}),
        ({"timezone": "Europe/Berlin"}, {}),
        ({}, {"remind_at": NOW}),
        ({}, {"state": "committed"}),
        ({}, {"state": "cancelled"}),
        ({"timezone": "Missing/Zone"}, {"timezone": "Missing/Zone"}),
    ],
)
async def test_current_card_checks_ttl_epoch_zone_future_and_state(user_changes, action_changes):
    repository, _ = card_repository(user_changes=user_changes, action_changes=action_changes)
    assert await repository.current_card(1) is None


@pytest.mark.parametrize("owner", [0, True, None, 2**63])
async def test_current_card_invalid_owner_does_not_access_database(owner):
    pool = Mock()
    pool.acquire.side_effect = AssertionError("No database access")
    assert await ConfirmationRepository(pool).current_card(owner) is None
    pool.acquire.assert_not_called()


@pytest.mark.parametrize("rows", [[None], [{"user_id": 1}, None]])
async def test_current_card_absent_owner_or_action_returns_none(rows):
    repository, connection = card_repository()
    connection.fetchrow.side_effect = rows
    assert await repository.current_card(1) is None
    assert connection.fetchrow.await_count == len(rows)
    connection.fetchval.assert_not_awaited()
    connection.execute.assert_not_awaited()


async def test_existing_lookup_card_keeps_source_message_filter():
    repository, connection = card_repository()
    assert (await repository.lookup_card(1, 19)).action_id == ACTION_ID
    query = connection.fetchrow.call_args_list[1]
    assert "telegram_message_id=$2" in query.args[0]
    assert query.args[1:] == (1, 19)
