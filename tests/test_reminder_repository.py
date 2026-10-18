"""Авторизация и неопределённый исход без настоящей БД или внешних сервисов."""

from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import asyncpg
import pytest

from app.agent_repository import ReminderRepository
from app.mcp.contracts import ReminderArguments


@pytest.mark.parametrize(
    "context",
    [
        None,
        "not an object",
        {},
        {"action_id": str(uuid4())},
        {"action_id": str(uuid4()), "capability": "a" * 43, "user_id": 101},
        {"action_id": str(uuid4()), "capability": "a" * 42},
        {"action_id": str(uuid4()), "capability": "a" * 129},
        {"action_id": str(uuid4()), "capability": "ы" * 43},
        {"action_id": str(uuid4()), "capability": "/" * 43},
        {"action_id": str(uuid4()), "capability": b"a" * 43},
        {"action_id": "not-a-uuid", "capability": "a" * 43},
        {"action_id": uuid4(), "capability": "a" * 43},
        {"action_id": "{" + str(uuid4()) + "}", "capability": "a" * 43},
    ],
)
async def test_invalid_capability_context_never_touches_database(context):
    pool = Mock()
    pool.acquire.side_effect = AssertionError("Database must not be used")
    result = await ReminderRepository(pool).add_reminder(
        ReminderArguments(text="Отправить отчёт", remind_at="2030-01-01T18:30:00+03:00"),
        context,
    )
    assert result.root.status == "error"
    assert result.root.error.code == "UNAUTHORIZED"
    pool.acquire.assert_not_called()


@pytest.mark.parametrize(
    ("failure", "code"),
    [
        (asyncpg.ConnectionDoesNotExistError("private database diagnostics"), "OUTCOME_UNKNOWN"),
        (asyncpg.InterfaceError("private connection diagnostics"), "OUTCOME_UNKNOWN"),
        (TimeoutError("private timeout diagnostics"), "OUTCOME_UNKNOWN"),
        (OSError("private network diagnostics"), "OUTCOME_UNKNOWN"),
        (asyncpg.CheckViolationError("private SQL diagnostics"), "SOURCE_UNAVAILABLE"),
    ],
)
async def test_database_failures_are_safe_and_never_trigger_a_retry(failure, code):
    connection = Mock()
    connection.transaction.return_value = AsyncMock()
    acquisition = AsyncMock()
    acquisition.__aenter__.return_value = connection
    pool = Mock()
    pool.acquire.return_value = acquisition
    repository = ReminderRepository(pool)
    repository._add = AsyncMock(side_effect=failure)
    context = {"action_id": str(uuid4()), "capability": "a" * 43}
    result = await repository.add_reminder(
        ReminderArguments(text="Отправить отчёт", remind_at="2030-01-01T18:30:00+03:00"),
        context,
    )
    assert result.root.status == "error"
    assert result.root.error.code == code
    assert "private" not in result.model_dump_json()
    repository._add.assert_awaited_once()
