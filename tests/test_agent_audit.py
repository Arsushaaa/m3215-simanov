"""Границы безопасного аудита; транзакции и конкуренция проверяются на PostgreSQL."""

from unittest.mock import AsyncMock, Mock

import pytest

from app.agent_repository import AgentRepository


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
