"""Подтверждения: безопасные DTO и локальные проверки до PostgreSQL."""

from datetime import UTC, datetime
from unittest.mock import Mock
from uuid import uuid4

import pytest

from app.agent_repository import ConfirmationRepository
from app.confirmation_models import Authorization, ConfirmationError, format_card
from app.mcp.contracts import ReminderArguments
from app.repository import UserSettings


@pytest.mark.parametrize(
    "changes",
    [
        {"user": UserSettings(0, "study", 0.3, 0)},
        {"user": UserSettings(True, "study", 0.3, 0)},
        {"message_id": 0},
        {"message_id": True},
        {"message_id": 2**63},
        {"question": ""},
        {"question": "   "},
        {"question": None},
        {"question": "x" * 16001},
        {"arguments": {"text": "raw"}},
    ],
)
async def test_invalid_preparation_never_touches_database(changes):
    pool = Mock()
    pool.acquire.side_effect = AssertionError("Database must not be used")
    arguments = {
        "user": UserSettings(101, "study", 0.3, 0),
        "message_id": 1,
        "question": "Напомни отправить отчёт",
        "arguments": ReminderArguments(
            text="Отправить отчёт", remind_at="2030-01-01T18:30:45+03:00"
        ),
    }
    arguments.update(changes)
    with pytest.raises(ConfirmationError) as error:
        await ConfirmationRepository(pool).prepare(**arguments)
    assert error.value.code == "INVALID_ARGUMENTS"
    pool.acquire.assert_not_called()


@pytest.mark.parametrize("method", ["approve", "reconcile", "cancel"])
@pytest.mark.parametrize(
    "user_id, action_id", [(0, str(uuid4())), (True, str(uuid4())), (101, "bad")]
)
async def test_invalid_callback_identity_is_denied_without_database(method, user_id, action_id):
    pool = Mock()
    pool.acquire.side_effect = AssertionError("Database must not be used")
    outcome = await getattr(ConfirmationRepository(pool), method)(user_id, action_id)
    assert outcome.status == "error" and outcome.code == "UNAUTHORIZED"
    assert outcome.reminder_id is None and outcome.remind_at is None and outcome.timezone is None
    pool.acquire.assert_not_called()


@pytest.mark.parametrize("iana", [None, 5, "Unknown/Zone", "localtime", "right/Europe/Moscow", ""])
async def test_invalid_timezone_never_changes_database(iana):
    pool = Mock()
    pool.acquire.side_effect = AssertionError("Database must not be used")
    with pytest.raises(ConfirmationError) as error:
        await ConfirmationRepository(pool).set_timezone(101, iana)
    assert error.value.code == "INVALID_ARGUMENTS"
    pool.acquire.assert_not_called()


@pytest.mark.parametrize(
    "moment, expected",
    [
        ("2030-01-01T18:30:00+03:00", "1 января 2030, 18:30 (UTC+03:00)"),
        (
            "2030-01-01T18:30:45.123456+03:00",
            "1 января 2030, 18:30:45.123456 (UTC+03:00)",
        ),
    ],
)
def test_card_preserves_exact_time_offset_and_text(moment, expected):
    arguments = ReminderArguments(text="Отправить отчёт", remind_at=moment)
    card = format_card(arguments, "Europe/Moscow", datetime(2030, 1, 1, tzinfo=UTC))
    assert "Когда: " + expected in card
    assert arguments.text in card and "Europe/Moscow" in card
    assert "5 минут с подготовки" in card
    assert "ещё не сохранено" in card


def test_card_converts_utc_arguments_to_the_selected_zone_before_display():
    arguments = ReminderArguments(text="Отправить отчёт", remind_at="2030-01-01T15:30:00Z")
    card = format_card(arguments, "Europe/Moscow", datetime(2030, 1, 1, tzinfo=UTC))
    assert "Когда: 1 января 2030, 18:30 (UTC+03:00)" in card
    assert arguments.remind_at.isoformat() == "2030-01-01T15:30:00+00:00"


def test_authorization_repr_hides_capability_and_reminder_text():
    secret = "opaque-capability-must-never-be-logged"
    arguments = ReminderArguments(text="Личное напоминание", remind_at="2030-01-01T18:30:45+03:00")
    identifier = uuid4()
    authorization = Authorization(
        identifier, arguments, {"action_id": str(identifier), "capability": secret}
    )
    assert secret not in repr(authorization)
    assert arguments.text not in repr(authorization)
