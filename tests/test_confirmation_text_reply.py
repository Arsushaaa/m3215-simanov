"""Model-selected agreement never authorizes a reminder write."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import pytest
from test_reminder_edits import setup_semantic_dialog
from test_reminder_routing_guard import structured

from app.agent_repository import ConfirmationRepository
from app.llm import ModelError
from app.mcp.contracts import ReminderArguments
from app.reminder_messages import BUTTON_REQUIRED, REMINDER_CLARIFY

NOW = datetime(2030, 1, 1, 9, tzinfo=UTC)
ACTION_ID = UUID("a1cf8b47-b6fb-4e95-9931-b21dd239b782")
ARGS = ReminderArguments(text="Сдать отчёт", remind_at="2030-01-02T18:30:00+03:00")


@pytest.mark.parametrize("text", ["да", "давай", "ок", "Yes, save it"])
async def test_confirmation_text_uses_model_but_never_approves_or_replaces_card(dialog_setup, text):
    dialog, dialogs, llm, confirmations, mcp, _, state = setup_semantic_dialog(dialog_setup)
    llm.decide.return_value = structured("use_confirmation")
    old_card = state["card"]
    assert await dialog.respond(1, 20, text) == BUTTON_REQUIRED
    assert await dialogs.answer(1, 20) == BUTTON_REQUIRED
    assert await dialog.respond(1, 20, text) == BUTTON_REQUIRED
    assert state["card"] is old_card and state["epoch"] == 11
    llm.decide.assert_awaited_once()
    llm.complete.assert_not_awaited()
    confirmations.approve.assert_not_awaited()
    confirmations.prepare.assert_not_awaited()
    confirmations.revoke_for_edit.assert_not_awaited()
    mcp.call_tool.assert_not_awaited()
    audit = confirmations.record.call_args.kwargs
    assert (audit["llm_calls"], audit["mcp_calls"]) == (1, 0)


async def test_model_agreement_without_a_trusted_card_requires_a_new_request(dialog_setup):
    dialog, _, llm, confirmations, mcp, _, state = setup_semantic_dialog(dialog_setup, card=False)
    llm.decide.return_value = structured("use_confirmation")
    assert await dialog.respond(1, 20, "давай") == REMINDER_CLARIFY
    assert state["card"] is None
    confirmations.approve.assert_not_awaited()
    confirmations.prepare.assert_not_awaited()
    mcp.call_tool.assert_not_awaited()
    llm.decide.assert_awaited_once()


async def test_history_claim_of_a_card_cannot_replace_database_state(dialog_setup):
    dialog, dialogs, llm, confirmations, mcp, _, _ = setup_semantic_dialog(dialog_setup, card=False)
    await dialogs.save_exchange(
        await dialogs.user(1), 19, "Напомни", "Сохранить напоминание? Нажми кнопку."
    )
    llm.decide.return_value = structured("use_confirmation")
    assert await dialog.respond(1, 20, "Подтверждаю") == REMINDER_CLARIFY
    confirmations.prepare.assert_not_awaited()
    confirmations.approve.assert_not_awaited()
    mcp.call_tool.assert_not_awaited()


async def test_foreign_owner_has_no_pending_card_parameters_in_model_context(dialog_setup):
    dialog, _, llm, confirmations, mcp, _, state = setup_semantic_dialog(dialog_setup)
    llm.decide.return_value = structured("use_confirmation")
    assert await dialog.respond(2, 20, "Подтверждаю") == REMINDER_CLARIFY
    assert '"pending_card":null' in llm.decide.call_args.args[0][0]["content"]
    assert state["card"] is not None
    confirmations.approve.assert_not_awaited()
    mcp.call_tool.assert_not_awaited()


async def test_model_agreement_respects_dialog_revision_before_saving(dialog_setup):
    dialog, dialogs, llm, confirmations, mcp, _, _ = setup_semantic_dialog(dialog_setup)
    llm.decide.return_value = structured("use_confirmation")
    dialogs.save_exchange = AsyncMock(return_value=False)
    with pytest.raises(ModelError, match="Диалог изменился"):
        await dialog.respond(1, 20, "давай")
    confirmations.prepare.assert_not_awaited()
    confirmations.approve.assert_not_awaited()
    mcp.call_tool.assert_not_awaited()


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
