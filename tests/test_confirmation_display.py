"""Закрывать карточку позволяет только проверенный окончательный результат БД."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.confirmation_models import ConfirmationText, Outcome
from app.confirmations import UNKNOWN_OUTCOME, ConfirmationService


@pytest.mark.parametrize(
    "status,close",
    [("cancelled", True), ("expired", True), ("error", False), ("unknown", False)],
)
def test_only_terminal_known_outcome_dismisses_card(status, close):
    result = ConfirmationService._outcome(Outcome(status, None, "Результат действия"))
    assert isinstance(result, str) and isinstance(result, ConfirmationText)
    assert result.dismiss_card is close
    if status == "unknown":
        assert result == UNKNOWN_OUTCOME


def test_valid_committed_receipt_dismisses_card_but_malformed_receipt_does_not():
    receipt = Outcome(
        "committed",
        None,
        "Записано",
        uuid4(),
        datetime(2030, 1, 1, 15, tzinfo=UTC),
        "Europe/Moscow",
    )
    result = ConfirmationService._outcome(receipt)
    assert result.dismiss_card and "1 января 2030, 18:00 (UTC+03:00)" in result
    assert ConfirmationService._outcome(Outcome("committed", None, "Записано")) == UNKNOWN_OUTCOME
    assert not getattr(
        ConfirmationService._outcome(Outcome("committed", None, "Записано")), "dismiss_card", False
    )


async def test_timezone_current_clock_does_not_show_database_microseconds():
    repository = SimpleNamespace(
        timezone=AsyncMock(return_value="Europe/Moscow"),
        now=AsyncMock(return_value=datetime(2030, 1, 1, 15, 23, 45, 123456, tzinfo=UTC)),
    )
    answer = await ConfirmationService(repository, SimpleNamespace()).timezone(101)
    assert "1 января 2030, 18:23:45 (UTC+03:00)" in answer
    assert "123456" not in answer
