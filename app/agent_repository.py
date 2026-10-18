"""Транзакционная запись уже подтверждённого действия MCP.

Подготовка карточки и Telegram callback относятся к отдельному слою. Здесь право —
token_urlsafe(32) (либо более длинный URL-safe токен до 128 символов); в БД хранится
SHA-256 ASCII-токена. Владелец всегда выводится из действия, не из аргументов.
"""

import hashlib
import hmac
import re
import time
from dataclasses import dataclass, field
from datetime import UTC
from uuid import UUID
from zoneinfo import ZoneInfo

import asyncpg

from app.mcp.contracts import (
    ErrorCode,
    ErrorDetails,
    ErrorResult,
    ReminderArguments,
    ReminderOk,
    ReminderResult,
    ReminderTimeError,
    validate_local_datetime,
    validate_reminder_time,
)


@dataclass(frozen=True)
class _Capability:
    action_id: UUID
    digest: bytes = field(repr=False)


def _context(value: object) -> _Capability | None:
    if type(value) is not dict or set(value) != {"action_id", "capability"}:
        return None
    action_id, token = value["action_id"], value["capability"]
    if not isinstance(action_id, str) or len(action_id) != 36 or not isinstance(token, str):
        return None
    if not re.fullmatch(r"[A-Za-z0-9_-]{43,128}", token):
        return None
    try:
        identifier = UUID(action_id)
    except ValueError:
        return None
    if str(identifier) != action_id:
        return None
    return _Capability(identifier, hashlib.sha256(token.encode("ascii")).digest())


def valid_capability_context(value: object) -> bool:
    """Allow lazy DB connection after shape checks; this does not authenticate."""
    return _context(value) is not None


def _error(code: ErrorCode, message: str, *, retryable: bool = False) -> ReminderResult:
    return ReminderResult(
        ErrorResult(
            status="error", error=ErrorDetails(code=code, message=message, retryable=retryable)
        )
    )


class ReminderRepository:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def add_reminder(self, arguments: ReminderArguments, context: object) -> ReminderResult:
        capability = _context(context)
        if capability is None:
            return _error("UNAUTHORIZED", "Нет действующего права на запись напоминания.")
        started = time.monotonic()
        try:
            async with self.pool.acquire() as connection, connection.transaction():
                return await self._add(connection, arguments, capability, started)
        except (asyncpg.PostgresConnectionError, asyncpg.InterfaceError, OSError, TimeoutError):
            # A lost connection/commit acknowledgement is not evidence of rollback.
            # The caller must recover this same action, never issue a new write.
            return _error(
                "OUTCOME_UNKNOWN",
                "Результат записи пока неизвестен. Проверь состояние того же подтверждения.",
                retryable=True,
            )
        except asyncpg.PostgresError:
            # PostgreSQL rejected the transaction; its writes have been rolled back.
            # Do not expose SQL, arguments, connection details or database diagnostics.
            return _error("SOURCE_UNAVAILABLE", "Хранилище не смогло завершить операцию.")

    async def _add(self, connection, arguments, capability, started) -> ReminderResult:
        owner = await connection.fetchval(
            "SELECT user_id FROM labbi_actions WHERE id=$1", capability.action_id
        )
        if owner is None:
            return _error("UNAUTHORIZED", "Нет действующего права на запись напоминания.")
        # All writers/revocations use this lock order: user, then action.
        user = await connection.fetchrow(
            "SELECT timezone, confirmation_epoch FROM labbi_users WHERE user_id=$1 FOR UPDATE",
            owner,
        )
        action = await connection.fetchrow(
            "SELECT * FROM labbi_actions WHERE id=$1 AND user_id=$2 FOR UPDATE",
            capability.action_id,
            owner,
        )
        if (
            user is None
            or action is None
            or action["capability_hash"] is None
            or not hmac.compare_digest(action["capability_hash"], capability.digest)
        ):
            return _error("UNAUTHORIZED", "Нет действующего права на запись напоминания.")
        now = await connection.fetchval("SELECT clock_timestamp()")

        if arguments.text != action["text"] or arguments.remind_at.astimezone(UTC) != action[
            "remind_at"
        ].astimezone(UTC):
            return await self._reject(
                connection,
                action,
                "ACTION_CONFLICT",
                "Параметры не совпадают с подтверждением.",
                started,
            )
        try:
            normalized = validate_local_datetime(
                arguments.remind_at.replace(tzinfo=None), action["timezone"]
            )
            if normalized.utcoffset() != arguments.remind_at.utcoffset():
                raise ReminderTimeError(
                    "INVALID_ARGUMENTS", "Смещение времени не соответствует зоне подтверждения."
                )
        except ReminderTimeError as error:
            return await self._reject(connection, action, error.code, str(error), started)

        # Authenticate and match immutable arguments before returning the receipt.
        # A completed action remains readable after TTL, time/zone/epoch changes.
        if action["state"] == "committed":
            receipt = await connection.fetchrow(
                "SELECT id, remind_at, timezone FROM labbi_reminders "
                "WHERE id=$1 AND action_id=$2 AND user_id=$3",
                action["receipt_reminder_id"],
                action["id"],
                owner,
            )
            if receipt is None:
                return await self._reject(
                    connection, action, "INVALID_RESULT", "Квитанция записи недоступна.", started
                )
            await self._audit(connection, action, "OK", "Напоминание уже сохранено.", started)
            return ReminderResult(
                ReminderOk(
                    status="ok",
                    reminder_id=receipt["id"],
                    remind_at=receipt["remind_at"].astimezone(ZoneInfo(receipt["timezone"])),
                    timezone=receipt["timezone"],
                    created=False,
                )
            )

        if action["state"] == "cancelled":
            return await self._reject(
                connection, action, "ACTION_CANCELLED", "Подтверждение отменено.", started
            )
        if action["state"] == "expired":
            return await self._reject(
                connection, action, "ACTION_EXPIRED", "Срок подтверждения истёк.", started
            )
        if action["state"] == "failed":
            return await self._reject(
                connection,
                action,
                "ACTION_CONFLICT",
                "Нужно новое подтверждение действия.",
                started,
            )
        if action["state"] != "approved":
            return await self._reject(
                connection,
                action,
                "CONFIRMATION_REQUIRED",
                "Нужно действующее подтверждение.",
                started,
            )
        if now >= action["expires_at"]:
            await connection.execute(
                "UPDATE labbi_actions SET state='expired', finished_at=$2 WHERE id=$1",
                action["id"],
                now,
            )
            return await self._reject(
                connection, action, "ACTION_EXPIRED", "Срок подтверждения истёк.", started
            )
        if user["confirmation_epoch"] != action["confirmation_epoch"]:
            return await self._reject(
                connection,
                action,
                "ACTION_CANCELLED",
                "Это подтверждение больше не действует.",
                started,
            )
        if user["timezone"] != action["timezone"]:
            return await self._reject(
                connection,
                action,
                "TIMEZONE_REQUIRED" if user["timezone"] is None else "ACTION_CONFLICT",
                "Задай часовой пояс и подготовь новое подтверждение.",
                started,
            )
        try:
            normalized = validate_reminder_time(arguments, now=now, timezone=user["timezone"])
        except ReminderTimeError as error:
            return await self._reject(connection, action, error.code, str(error), started)

        reminder_id = await connection.fetchval(
            "INSERT INTO labbi_reminders(user_id, action_id, text, remind_at, timezone, "
            "created_at) "
            "VALUES ($1, $2, $3, $4, $5, $6) RETURNING id",
            owner,
            action["id"],
            action["text"],
            action["remind_at"],
            action["timezone"],
            now,
        )
        await connection.execute(
            "UPDATE labbi_actions SET state='committed', receipt_reminder_id=$2, finished_at=$3 "
            "WHERE id=$1",
            action["id"],
            reminder_id,
            now,
        )
        await self._audit(connection, action, "OK", "Напоминание сохранено.", started)
        return ReminderResult(
            ReminderOk(
                status="ok",
                reminder_id=reminder_id,
                remind_at=normalized,
                timezone=action["timezone"],
                created=True,
            )
        )

    async def _reject(self, connection, action, code, message, started) -> ReminderResult:
        await self._audit(connection, action, "REJECTED", message, started, error_code=code)
        return _error(code, message)

    @staticmethod
    async def _audit(connection, action, status, reason, started, *, error_code=None) -> None:
        # Do not copy reminder text or capability into diagnostic explanations.
        await connection.execute(
            "INSERT INTO labbi_agent_events(user_id, request_id, action_id, decision, target, "
            "status, argument_summary, reason, error_code, mcp_calls, duration_ms) "
            "VALUES ($1, $2, $3, 'confirm', 'add_reminder', $4, $5, $6, $7, 1, $8)",
            action["user_id"],
            action["request_id"],
            action["id"],
            status,
            f"Время: {action['remind_at'].isoformat()}; зона: {action['timezone']}",
            reason,
            error_code,
            max(0, int((time.monotonic() - started) * 1000)),
        )
