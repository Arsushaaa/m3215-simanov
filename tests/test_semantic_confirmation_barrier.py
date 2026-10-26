"""Semantic routing may pause a card, but never authorizes it with model text."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from app.agent_repository import SEMANTIC_REQUEST_WAIT, ConfirmationRepository
from app.confirmation_models import Authorization, ConfirmationError, Outcome
from app.confirmations import ConfirmationService
from app.repository import UserSettings

NOW = datetime(2030, 1, 1, 9, tzinfo=UTC)


class Scope:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *args):
        pass


def blocked_repository(*, processing=True, committed=False):
    action = {
        "id": uuid4(),
        "request_id": uuid4(),
        "user_id": 101,
        "state": "committed" if committed else "pending",
        "timezone": "Europe/Moscow",
        "remind_at": NOW + timedelta(days=1),
        "text": "Сдать отчёт",
    }
    user = {"timezone": "Europe/Moscow", "confirmation_epoch": 7}

    async def value(sql, *args):
        if "clock_timestamp" in sql:
            return NOW
        if "incoming.status='processing'" in sql:
            return processing
        raise AssertionError("Unexpected SQL")

    connection = SimpleNamespace(fetchval=AsyncMock(side_effect=value), execute=AsyncMock())
    connection.transaction = lambda: Scope(None)
    pool = SimpleNamespace(acquire=lambda: Scope(connection))
    repository = ConfirmationRepository(pool)
    repository._locked = AsyncMock(return_value=(user, action))
    repository._current = AsyncMock(return_value=None)
    return repository, connection, action


@pytest.mark.parametrize("method", ["approve", "reconcile"])
async def test_processing_reply_keeps_card_recoverable_without_issuing_capability(method):
    repository, connection, action = blocked_repository()

    outcome = await getattr(repository, method)(101, action["id"])

    assert outcome.status == "error" and outcome.code == "ACTION_CONFLICT"
    assert outcome.message == SEMANTIC_REQUEST_WAIT
    assert not isinstance(outcome, Authorization)
    connection.execute.assert_not_awaited()


async def test_confirm_busy_result_does_not_call_mcp_and_keeps_the_button():
    repository = SimpleNamespace(
        approve=AsyncMock(return_value=Outcome("error", "ACTION_CONFLICT", SEMANTIC_REQUEST_WAIT))
    )
    mcp = SimpleNamespace(call_tool=AsyncMock(side_effect=AssertionError("No write while routing")))

    reply = await ConfirmationService(repository, mcp).confirm(101, str(uuid4()))

    assert reply == SEMANTIC_REQUEST_WAIT
    assert reply.dismiss_card is False
    mcp.call_tool.assert_not_awaited()


async def test_committed_receipt_is_available_even_with_a_new_processing_request():
    repository, connection, action = blocked_repository(committed=True)
    previous = Outcome(
        "committed",
        "OK",
        "Напоминание сохранено.",
        reminder_id=uuid4(),
        remind_at=action["remind_at"],
        timezone=action["timezone"],
    )
    repository._receipt = AsyncMock(return_value=previous)

    assert await repository.approve(101, action["id"]) == previous

    connection.fetchval.assert_not_awaited()


@pytest.mark.parametrize("owner,message_id", [(0, 1), (True, 1), (101, 0), (101, True)])
async def test_invalid_snapshot_identity_does_not_touch_database(owner, message_id):
    pool = Mock()
    pool.acquire.side_effect = AssertionError("Invalid identity must not use DB")

    with pytest.raises(ConfirmationError):
        await ConfirmationRepository(pool).semantic_request_context(
            UserSettings(owner, "study", 0.3, 0), message_id
        )

    pool.acquire.assert_not_called()
