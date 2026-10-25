"""Границы безопасного аудита; транзакции и конкуренция проверяются на PostgreSQL."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest

from app.agent_repository import AgentRepository, ConfirmationRepository
from app.confirmation_models import ConfirmationError


@pytest.mark.parametrize(
    "changes",
    [
        {"user_id": 0},
        {"user_id": -1},
        {"user_id": True},
        {"user_id": 2**63},
        {"message_id": 0},
        {"message_id": "12"},
        {"decision": "confirm"},
        {"decision": {"action": "respond"}},
        {"target": "run_shell"},
        {"target": "https://private.example"},
        {"target": ["get_weather"]},
        {"status": "completed"},
        {"status": None},
        {"reason": ""},
        {"reason": "   "},
        {"reason": "x" * 501},
        {"reason": " " * 500 + "x"},
        {"reason": "Первая строка\nВторая строка"},
        {"reason": "Текст\x00"},
        {"reason": {"raw": "model reason"}},
        {"error_code": ""},
        {"error_code": "raw error diagnostics"},
        {"error_code": "a"},
        {"error_code": "A" * 65},
        {"error_code": 7},
        {"llm_calls": True},
        {"llm_calls": -1},
        {"llm_calls": 3},
        {"mcp_calls": 2},
        {"mcp_calls": 0.0},
        {"duration_ms": -1},
        {"duration_ms": False},
        {"duration_ms": 2**63},
        {"argument_summary": "x" * 501},
        {"argument_summary": "Первая строка\nВторая строка"},
        {"argument_summary": {"raw": "model reason"}},
    ],
)
async def test_invalid_audit_is_rejected_before_database_access(changes):
    pool = Mock()
    pool.acquire.side_effect = AssertionError("Database must not be used")
    arguments = {
        "user_id": 101,
        "message_id": 12,
        "decision": "respond",
        "target": None,
        "status": "OK",
        "reason": "Подготовлен обычный ответ.",
    }
    arguments.update(changes)
    with pytest.raises(ValueError, match="Некорректные параметры аудита"):
        await AgentRepository(pool).record(**arguments)
    pool.acquire.assert_not_called()


@pytest.mark.parametrize(
    "stored, expected",
    [
        (None, None),
        ("Europe/Moscow", "Europe/Moscow"),
        ("Asia/Tokyo", "Asia/Tokyo"),
        ("Unknown/Zone", None),
        ("localtime", None),
        ("right/Europe/Moscow", None),
        (3, None),
    ],
)
async def test_timezone_returns_only_saved_valid_iana_name(stored, expected):
    pool = Mock(fetchval=AsyncMock(return_value=stored))
    assert await AgentRepository(pool).timezone(101) == expected
    pool.fetchval.assert_awaited_once()


@pytest.mark.parametrize("user_id", [0, -1, True, "101", 2**63])
async def test_timezone_rejects_invalid_identity_without_database_access(user_id):
    pool = Mock(fetchval=AsyncMock(side_effect=AssertionError("Database must not be used")))
    with pytest.raises(ValueError, match="Некорректный владелец"):
        await AgentRepository(pool).timezone(user_id)
    pool.fetchval.assert_not_awaited()


@pytest.mark.parametrize("processed", [False, True])
async def test_processed_lookup_returns_terminal_flag(processed):
    pool = Mock(fetchval=AsyncMock(return_value=processed))
    assert await AgentRepository(pool).has_processed(101, 12) is processed
    pool.fetchval.assert_awaited_once()


@pytest.mark.parametrize("user_id, message_id", [(0, 1), (1, 0), (True, 1), (1, "2"), (2**63, 1)])
async def test_processed_lookup_rejects_invalid_identity_without_database(user_id, message_id):
    pool = Mock(fetchval=AsyncMock(side_effect=AssertionError("Database must not be used")))
    with pytest.raises(ValueError, match="Некорректный идентификатор запроса"):
        await AgentRepository(pool).has_processed(user_id, message_id)
    pool.fetchval.assert_not_awaited()


@pytest.mark.parametrize("user_id, message_id", [(0, 1), (1, 0), (True, 1), (1, "2"), (2**63, 1)])
async def test_start_request_rejects_invalid_identity_without_database(user_id, message_id):
    pool = Mock()
    pool.acquire.side_effect = AssertionError("Database must not be used")
    with pytest.raises(ValueError, match="Некорректный идентификатор запроса"):
        await AgentRepository(pool).start_request(user_id, message_id)
    pool.acquire.assert_not_called()


def stored_event(*, pending=None, created_at=datetime(2026, 10, 20, 16, tzinfo=UTC)):
    return {
        "created_at": created_at,
        "decision": "call_tool",
        "target": "get_weather",
        "status": "OK",
        "reason": "Погода получена из источника.",
        "argument_summary": "Город: Москва",
        "error_code": None,
        "llm_calls": 2,
        "mcp_calls": 1,
        "duration_ms": 10,
        "request_created_at": created_at,
        "pending_created_at": pending,
    }


@pytest.mark.parametrize("has_old_event", [False, True])
async def test_new_request_without_audit_never_exposes_older_event_as_latest(has_old_event):
    now = datetime(2026, 10, 20, 16, tzinfo=UTC)
    row = stored_event(pending=now + timedelta(seconds=1))
    if not has_old_event:
        row = dict.fromkeys(row)
        row["pending_created_at"] = now
    pool = Mock(fetchrow=AsyncMock(return_value=row))
    with pytest.raises(ConfirmationError, match="Итог последнего запроса ещё не сохранён") as error:
        await ConfirmationRepository(pool).latest_event(101)
    assert error.value.code == "SOURCE_UNAVAILABLE"
    assert "Москва" not in error.value.user_message
    assert pool.fetchrow.call_args.args[1:] == (101,)


@pytest.mark.parametrize("pending_age", [None, 0, -1])
async def test_latest_saved_event_is_returned_when_no_newer_incomplete_request(pending_age):
    now = datetime(2026, 10, 20, 16, tzinfo=UTC)
    row = stored_event(
        pending=now + timedelta(seconds=pending_age) if pending_age is not None else None
    )
    pool = Mock(fetchrow=AsyncMock(return_value=row))
    event = await ConfirmationRepository(pool).latest_event(101)
    assert event.argument_summary == "Город: Москва"
    assert event.created_at == now


async def test_latest_event_empty_registry_returns_none():
    pool = Mock(fetchrow=AsyncMock(return_value=dict.fromkeys(stored_event())))
    assert await ConfirmationRepository(pool).latest_event(101) is None


async def test_old_request_completed_after_new_start_cannot_mask_missing_new_audit():
    now = datetime(2026, 10, 20, 16, tzinfo=UTC)
    row = stored_event(pending=now + timedelta(seconds=1), created_at=now + timedelta(seconds=2))
    row["request_created_at"] = now
    pool = Mock(fetchrow=AsyncMock(return_value=row))
    with pytest.raises(ConfirmationError, match="Итог последнего запроса ещё не сохранён"):
        await ConfirmationRepository(pool).latest_event(101)
