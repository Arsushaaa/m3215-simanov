"""Финальный аудит агента и транзакционная запись подтверждённого действия MCP.

Подготовка карточки и Telegram callback относятся к отдельному слою. Здесь право —
token_urlsafe(32) (либо более длинный URL-safe токен до 128 символов); в БД хранится
SHA-256 ASCII-токена. Владелец всегда выводится из действия, не из аргументов.
"""

import hashlib
import hmac
import re
import secrets
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import UUID
from zoneinfo import ZoneInfo

import asyncpg
from pydantic import TypeAdapter, ValidationError

from app.confirmation_models import (
    Authorization,
    ConfirmationError,
    Outcome,
    PreparedAction,
    SafeAgentEvent,
    format_card,
)
from app.mcp.contracts import (
    ErrorCode,
    ErrorDetails,
    ErrorResult,
    ReminderArguments,
    ReminderOk,
    ReminderResult,
    ReminderTimeError,
    TimezoneName,
    validate_local_datetime,
    validate_reminder_time,
)

if TYPE_CHECKING:
    from app.repository import UserSettings

_DECISIONS = frozenset({"respond", "clarify", "call_tool", "read_resource"})
_TARGETS = frozenset({"get_weather", "get_schedule", "add_reminder", "schedule://current-week"})
_STATUSES = frozenset({"OK", "CLARIFY", "REJECTED", "FAILED", "CANCELLED", "OUTCOME_UNKNOWN"})
_TIMEZONE = TypeAdapter(TimezoneName)


def _identifier(value: object) -> bool:
    return type(value) is int and 0 < value <= 2**63 - 1


class AgentRepository:
    """Краткий финальный аудит отдельно от удаляемой истории диалога.

    Оркестратор передаёт собственные фиксированные reason-строки, а не пояснения
    модели. Сетевые вызовы выполняются до record, вне соединения и транзакции БД.
    """

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def has_processed(self, user_id: int, message_id: int) -> bool:
        if not _identifier(user_id) or not _identifier(message_id):
            raise ValueError("Некорректный идентификатор запроса.")
        return (
            await self.pool.fetchval(
                "SELECT EXISTS (SELECT 1 FROM labbi_agent_requests "
                "WHERE user_id=$1 AND telegram_message_id=$2 AND status <> 'processing')",
                user_id,
                message_id,
            )
            is True
        )

    async def timezone(self, user_id: int) -> str | None:
        if not _identifier(user_id):
            raise ValueError("Некорректный владелец настройки.")
        value = await self.pool.fetchval(
            "SELECT timezone FROM labbi_users WHERE user_id=$1", user_id
        )
        if value is None:
            return None
        try:
            return _TIMEZONE.validate_python(value, strict=True)
        except ValidationError:
            # A corrupted/manual setting must not become an assumed user timezone.
            return None

    async def record(
        self,
        user_id: int,
        message_id: int,
        *,
        decision: str,
        target: str | None,
        status: str,
        reason: str,
        error_code: str | None = None,
        llm_calls: int = 0,
        mcp_calls: int = 0,
        duration_ms: int = 0,
        argument_summary: str = "",
    ) -> None:
        if (
            not _identifier(user_id)
            or not _identifier(message_id)
            or type(decision) is not str
            or decision not in _DECISIONS
            or (target is not None and (type(target) is not str or target not in _TARGETS))
            or type(status) is not str
            or status not in _STATUSES
            or type(reason) is not str
            or not reason.strip()
            or len(reason) > 500
            or any(ord(character) < 32 or ord(character) == 127 for character in reason)
            or (
                error_code is not None
                and (
                    type(error_code) is not str
                    or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", error_code)
                )
            )
            or type(llm_calls) is not int
            or not 0 <= llm_calls <= 2
            or type(mcp_calls) is not int
            or not 0 <= mcp_calls <= 1
            or type(duration_ms) is not int
            or not 0 <= duration_ms <= 2**63 - 1
            or type(argument_summary) is not str
            or len(argument_summary) > 500
            or any(ord(character) < 32 or ord(character) == 127 for character in argument_summary)
        ):
            raise ValueError("Некорректные параметры аудита.")
        request_status = (
            "cancelled"
            if status == "CANCELLED"
            else "failed"
            if status in {"FAILED", "OUTCOME_UNKNOWN"}
            else "completed"
        )
        async with self.pool.acquire() as connection, connection.transaction():
            await connection.execute(
                "INSERT INTO labbi_users(user_id) VALUES ($1) ON CONFLICT DO NOTHING", user_id
            )
            await connection.fetchval(
                "SELECT user_id FROM labbi_users WHERE user_id=$1 FOR UPDATE", user_id
            )
            await connection.execute(
                "INSERT INTO labbi_agent_requests(user_id, telegram_message_id, telegram_chat_id) "
                "VALUES ($1, $2, $1) ON CONFLICT (user_id, telegram_message_id) DO NOTHING",
                user_id,
                message_id,
            )
            request = await connection.fetchrow(
                "SELECT id, status FROM labbi_agent_requests "
                "WHERE user_id=$1 AND telegram_message_id=$2 FOR UPDATE",
                user_id,
                message_id,
            )
            if request["status"] != "processing":
                return
            await connection.execute(
                "INSERT INTO labbi_agent_events(user_id, request_id, decision, target, status, "
                "reason, error_code, llm_calls, mcp_calls, duration_ms, argument_summary) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)",
                user_id,
                request["id"],
                decision,
                target,
                status,
                reason,
                error_code,
                llm_calls,
                mcp_calls,
                duration_ms,
                argument_summary,
            )
            await connection.execute(
                "UPDATE labbi_agent_requests SET status=$2, decision=$3, "
                "finished_at=clock_timestamp() WHERE id=$1",
                request["id"],
                request_status,
                decision,
            )


async def _confirmation_event(connection, action, decision, status, reason, *, error_code=None):
    await connection.execute(
        "INSERT INTO labbi_agent_events(user_id, request_id, action_id, decision, target, "
        "status, reason, error_code, argument_summary) "
        "VALUES ($1, $2, $3, $4, 'add_reminder', $5, $6, $7, $8)",
        action["user_id"],
        action["request_id"],
        action["id"],
        decision,
        status,
        reason,
        error_code,
        f"Время: {action['remind_at'].isoformat()}; зона: {action['timezone']}",
    )


async def revoke_actions(connection, user_id: int, reason: str) -> bool:
    """Revoke active actions after the caller has locked their owner row."""
    if (
        not _identifier(user_id)
        or not isinstance(reason, str)
        or not reason.strip()
        or len(reason) > 500
        or any(ord(character) < 32 or ord(character) == 127 for character in reason)
    ):
        raise ValueError("Некорректные параметры отмены подтверждений.")
    rows = await connection.fetch(
        "UPDATE labbi_actions SET state='cancelled', finished_at=clock_timestamp() "
        "WHERE user_id=$1 AND state IN ('pending', 'approved') RETURNING *",
        user_id,
    )
    for action in rows:
        await _confirmation_event(
            connection, action, "cancel", "CANCELLED", reason, error_code="ACTION_CANCELLED"
        )
    return bool(rows)


def _action_id(value: object) -> UUID | None:
    if isinstance(value, UUID):
        return value
    if not isinstance(value, str) or len(value) != 36:
        return None
    try:
        result = UUID(value)
    except ValueError:
        return None
    return result if str(result) == value else None


def _denied() -> Outcome:
    return Outcome("error", "UNAUTHORIZED", "Это подтверждение недоступно.")


def _unknown() -> Outcome:
    return Outcome(
        "unknown",
        "OUTCOME_UNKNOWN",
        "Результат пока неизвестен. Проверь то же подтверждение ещё раз.",
    )


class ConfirmationRepository(AgentRepository):
    """Подготовка и отзыв прав; все writers блокируют user, затем action."""

    async def now(self) -> datetime:
        return await self.pool.fetchval("SELECT clock_timestamp()")

    async def latest_event(self, user_id: int) -> SafeAgentEvent | None:
        if not _identifier(user_id):
            raise ConfirmationError("UNAUTHORIZED", "История действий недоступна.")
        row = await self.pool.fetchrow(
            "SELECT created_at, decision, target, status, reason, argument_summary, "
            "error_code, llm_calls, mcp_calls, duration_ms FROM labbi_agent_events "
            "WHERE user_id=$1 ORDER BY id DESC LIMIT 1",
            user_id,
        )
        return SafeAgentEvent(**dict(row)) if row is not None else None

    @staticmethod
    def _arguments(action) -> ReminderArguments:
        zone = _TIMEZONE.validate_python(action["timezone"], strict=True)
        return ReminderArguments(
            text=action["text"], remind_at=action["remind_at"].astimezone(ZoneInfo(zone))
        )

    @classmethod
    def _prepared(cls, action, *, replayed=False) -> PreparedAction:
        arguments = cls._arguments(action)
        return PreparedAction(
            action_id=action["id"],
            card_text=format_card(arguments, action["timezone"], action["expires_at"]),
            arguments=arguments,
            timezone=action["timezone"],
            expires_at=action["expires_at"],
            state=action["state"],
            replayed=replayed,
        )

    @staticmethod
    async def _locked(connection, user_id, action_id):
        user = await connection.fetchrow(
            "SELECT * FROM labbi_users WHERE user_id=$1 FOR UPDATE", user_id
        )
        if user is None:
            return None, None
        action = await connection.fetchrow(
            "SELECT * FROM labbi_actions WHERE user_id=$1 AND id=$2 FOR UPDATE", user_id, action_id
        )
        return user, action

    @staticmethod
    async def _receipt(connection, action) -> Outcome:
        row = await connection.fetchrow(
            "SELECT id, remind_at, timezone FROM labbi_reminders "
            "WHERE id=$1 AND action_id=$2 AND user_id=$3",
            action["receipt_reminder_id"],
            action["id"],
            action["user_id"],
        )
        if row is None:
            return Outcome("error", "INVALID_RESULT", "Квитанция записи недоступна.")
        try:
            zone = _TIMEZONE.validate_python(row["timezone"], strict=True)
        except ValidationError:
            return Outcome("error", "INVALID_RESULT", "Квитанция записи имеет некорректную зону.")
        return Outcome(
            "committed",
            None,
            "Напоминание сохранено.",
            row["id"],
            row["remind_at"].astimezone(ZoneInfo(zone)),
            zone,
        )

    @staticmethod
    async def _current(connection, user, action, now) -> Outcome | None:
        if action["state"] == "cancelled":
            return Outcome("cancelled", "ACTION_CANCELLED", "Подтверждение отменено.")
        if action["state"] == "expired":
            return Outcome("expired", "ACTION_EXPIRED", "Срок подтверждения истёк.")
        if action["state"] == "failed":
            return Outcome("error", "ACTION_CONFLICT", "Подготовь новое подтверждение.")
        if now >= action["expires_at"]:
            await connection.execute(
                "UPDATE labbi_actions SET state='expired', finished_at=$2 WHERE id=$1",
                action["id"],
                now,
            )
            await _confirmation_event(
                connection,
                action,
                "confirm",
                "REJECTED",
                "Срок подтверждения истёк.",
                error_code="ACTION_EXPIRED",
            )
            return Outcome("expired", "ACTION_EXPIRED", "Срок подтверждения истёк.")
        if (
            user["confirmation_epoch"] != action["confirmation_epoch"]
            or user["timezone"] != action["timezone"]
        ):
            await connection.execute(
                "UPDATE labbi_actions SET state='cancelled', finished_at=$2 WHERE id=$1",
                action["id"],
                now,
            )
            await _confirmation_event(
                connection,
                action,
                "cancel",
                "CANCELLED",
                "Прежнее подтверждение отозвано.",
                error_code="ACTION_CANCELLED",
            )
            return Outcome(
                "cancelled", "ACTION_CANCELLED", "Это подтверждение больше не действует."
            )
        return None

    async def prepare(
        self,
        user: "UserSettings",
        message_id: int,
        question: str,
        arguments: ReminderArguments,
        *,
        timezone: str | None = None,
    ) -> PreparedAction:
        if (
            not _identifier(getattr(user, "user_id", None))
            or not _identifier(message_id)
            or not isinstance(question, str)
            or not question.strip()
            or len(question) > 16000
            or not isinstance(arguments, ReminderArguments)
        ):
            raise ConfirmationError("INVALID_ARGUMENTS", "Параметры напоминания некорректны.")
        async with self.pool.acquire() as connection, connection.transaction():
            current = await connection.fetchrow(
                "SELECT * FROM labbi_users WHERE user_id=$1 FOR UPDATE", user.user_id
            )
            if current is None:
                raise ConfirmationError(
                    "ACTION_CANCELLED", "Диалог изменился. Отправь новый запрос."
                )
            action = await connection.fetchrow(
                "SELECT * FROM labbi_actions WHERE user_id=$1 "
                "AND telegram_message_id=$2 FOR UPDATE",
                user.user_id,
                message_id,
            )
            now = await connection.fetchval("SELECT clock_timestamp()")
            if action is not None:
                if action["state"] == "committed":
                    raise ConfirmationError("ACTION_CONFLICT", "Этот запрос уже обработан.")
                inactive = await self._current(connection, current, action, now)
                if inactive is not None:
                    # Raising rolls back housekeeping, but never revives the old action.
                    raise ConfirmationError(inactive.code, inactive.message)
                canonical = self._arguments(action)
                if arguments.text != canonical.text or arguments.remind_at != canonical.remind_at:
                    raise ConfirmationError(
                        "ACTION_CONFLICT", "Параметры прежнего запроса изменились."
                    )
                return self._prepared(action, replayed=True)
            if current["revision"] != user.revision or current["mode"] != "study":
                raise ConfirmationError(
                    "ACTION_CANCELLED", "Диалог изменился. Отправь новый запрос."
                )
            if current["timezone"] is None:
                raise ConfirmationError(
                    "TIMEZONE_REQUIRED", "Сначала задай часовой пояс через /timezone."
                )
            try:
                zone = _TIMEZONE.validate_python(current["timezone"], strict=True)
                if timezone is not None and timezone != zone:
                    raise ConfirmationError(
                        "ACTION_CONFLICT", "Часовой пояс изменился. Повтори запрос."
                    )
                canonical = ReminderArguments(
                    text=arguments.text,
                    remind_at=validate_reminder_time(arguments, now=now, timezone=zone),
                )
            except ValidationError:
                raise ConfirmationError("INVALID_ARGUMENTS", "Часовой пояс некорректен.") from None
            except ReminderTimeError as error:
                raise ConfirmationError(error.code, str(error)) from None
            if await connection.fetchval(
                "SELECT EXISTS (SELECT 1 FROM labbi_messages "
                "WHERE user_id=$1 AND telegram_message_id=$2)",
                user.user_id,
                message_id,
            ):
                raise ConfirmationError("ACTION_CONFLICT", "Этот запрос уже обработан.")
            await connection.execute(
                "INSERT INTO labbi_agent_requests(user_id, telegram_message_id, telegram_chat_id) "
                "VALUES ($1, $2, $1) ON CONFLICT (user_id, telegram_message_id) DO NOTHING",
                user.user_id,
                message_id,
            )
            request = await connection.fetchrow(
                "SELECT * FROM labbi_agent_requests WHERE user_id=$1 "
                "AND telegram_message_id=$2 FOR UPDATE",
                user.user_id,
                message_id,
            )
            if request["status"] != "processing":
                is_proposal = request["status"] == "completed" and await connection.fetchval(
                    "SELECT EXISTS (SELECT 1 FROM labbi_agent_events WHERE request_id=$1 "
                    "AND decision='call_tool' AND target='add_reminder' AND status='CLARIFY')",
                    request["id"],
                )
                if not is_proposal:
                    raise ConfirmationError("ACTION_CONFLICT", "Этот запрос уже обработан.")
            await revoke_actions(connection, user.user_id, "Прежняя карточка заменена новой.")
            epoch = await connection.fetchval(
                "UPDATE labbi_users SET confirmation_epoch=confirmation_epoch+1, "
                "revision=revision+1, "
                "updated_at=clock_timestamp() WHERE user_id=$1 RETURNING confirmation_epoch",
                user.user_id,
            )
            action = await connection.fetchrow(
                "INSERT INTO labbi_actions(user_id, request_id, telegram_message_id, "
                "confirmation_epoch, text, remind_at, timezone, prepared_at, expires_at) "
                "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9) RETURNING *",
                user.user_id,
                request["id"],
                message_id,
                epoch,
                canonical.text,
                canonical.remind_at,
                zone,
                now,
                now + timedelta(seconds=300),
            )
            prepared = self._prepared(action)
            await connection.execute(
                "INSERT INTO labbi_messages(user_id, telegram_message_id, role, content) "
                "VALUES ($1,$2,'user',$3),($1,$2,'assistant',$4)",
                user.user_id,
                message_id,
                question,
                prepared.card_text,
            )
            await connection.execute(
                "UPDATE labbi_agent_requests SET status='completed', decision='call_tool', "
                "finished_at=$2 WHERE id=$1 AND status='processing'",
                request["id"],
                now,
            )
            await _confirmation_event(
                connection, action, "call_tool", "CLARIFY", "Подготовлена карточка подтверждения."
            )
            return prepared

    async def lookup_card(self, user_id: int, message_id: int) -> PreparedAction | None:
        if not _identifier(user_id) or not _identifier(message_id):
            return None
        async with self.pool.acquire() as connection, connection.transaction():
            user = await connection.fetchrow(
                "SELECT * FROM labbi_users WHERE user_id=$1 FOR UPDATE", user_id
            )
            if user is None:
                return None
            action = await connection.fetchrow(
                "SELECT * FROM labbi_actions WHERE user_id=$1 "
                "AND telegram_message_id=$2 FOR UPDATE",
                user_id,
                message_id,
            )
            if action is None or action["state"] not in {"pending", "approved"}:
                return None
            now = await connection.fetchval("SELECT clock_timestamp()")
            if await self._current(connection, user, action, now) is not None:
                return None
            try:
                validate_reminder_time(self._arguments(action), now=now, timezone=user["timezone"])
                return self._prepared(action, replayed=True)
            except (ValidationError, ReminderTimeError):
                return None

    async def approve(self, user_id: int, action_id: UUID | str) -> Authorization | Outcome:
        identifier = _action_id(action_id)
        if not _identifier(user_id) or identifier is None:
            return _denied()
        try:
            async with self.pool.acquire() as connection, connection.transaction():
                user, action = await self._locked(connection, user_id, identifier)
                if action is None:
                    return _denied()
                if action["state"] == "committed":
                    receipt = await self._receipt(connection, action)
                    if receipt.status == "committed":
                        await _confirmation_event(
                            connection,
                            action,
                            "confirm",
                            "OK",
                            "Показана квитанция ранее сохранённого напоминания.",
                        )
                    return receipt
                now = await connection.fetchval("SELECT clock_timestamp()")
                inactive = await self._current(connection, user, action, now)
                if inactive is not None:
                    return inactive
                try:
                    arguments = self._arguments(action)
                    validate_reminder_time(arguments, now=now, timezone=user["timezone"])
                except (ValidationError, ReminderTimeError):
                    return Outcome(
                        "error", "INVALID_ARGUMENTS", "Время или зона больше не подходят."
                    )
                token = secrets.token_urlsafe(32)
                await connection.execute(
                    "UPDATE labbi_actions SET state='approved', "
                    "approved_at=COALESCE(approved_at,$2), "
                    "capability_hash=$3, capability_issued_at=$2 WHERE id=$1",
                    identifier,
                    now,
                    hashlib.sha256(token.encode("ascii")).digest(),
                )
                await _confirmation_event(
                    connection,
                    action,
                    "confirm",
                    "CLARIFY",
                    "Подтверждение принято; запись ожидается.",
                )
                return Authorization(
                    identifier, arguments, {"action_id": str(identifier), "capability": token}
                )
        except (asyncpg.PostgresConnectionError, asyncpg.InterfaceError, OSError, TimeoutError):
            return _unknown()

    async def reconcile(self, user_id: int, action_id: UUID | str) -> Outcome:
        identifier = _action_id(action_id)
        if not _identifier(user_id) or identifier is None:
            return _denied()
        try:
            async with self.pool.acquire() as connection, connection.transaction():
                user, action = await self._locked(connection, user_id, identifier)
                if action is None:
                    return _denied()
                if action["state"] == "committed":
                    return await self._receipt(connection, action)
                now = await connection.fetchval("SELECT clock_timestamp()")
                inactive = await self._current(connection, user, action, now)
                if inactive is not None:
                    return inactive
                await _confirmation_event(
                    connection,
                    action,
                    "confirm",
                    "OUTCOME_UNKNOWN",
                    "Запись ещё не подтверждена хранилищем.",
                    error_code="OUTCOME_UNKNOWN",
                )
                return _unknown()
        except (asyncpg.PostgresConnectionError, asyncpg.InterfaceError, OSError, TimeoutError):
            return _unknown()

    async def cancel(self, user_id: int, action_id: UUID | str) -> Outcome:
        identifier = _action_id(action_id)
        if not _identifier(user_id) or identifier is None:
            return _denied()
        try:
            async with self.pool.acquire() as connection, connection.transaction():
                user, action = await self._locked(connection, user_id, identifier)
                if action is None:
                    return _denied()
                if action["state"] == "committed":
                    receipt = await self._receipt(connection, action)
                    if receipt.status == "committed":
                        await _confirmation_event(
                            connection,
                            action,
                            "cancel",
                            "OK",
                            "Сохранённое напоминание не удалено; показана квитанция.",
                        )
                    return receipt
                now = await connection.fetchval("SELECT clock_timestamp()")
                inactive = await self._current(connection, user, action, now)
                if inactive is not None:
                    return inactive
                await connection.execute(
                    "UPDATE labbi_users SET confirmation_epoch=confirmation_epoch+1, "
                    "updated_at=$2 WHERE user_id=$1",
                    user_id,
                    now,
                )
                await revoke_actions(
                    connection, user_id, "Подготовленное действие отменено пользователем."
                )
                return Outcome("cancelled", "ACTION_CANCELLED", "Подтверждение отменено.")
        except (asyncpg.PostgresConnectionError, asyncpg.InterfaceError, OSError, TimeoutError):
            return _unknown()

    async def invalidate(self, user_id: int) -> bool:
        if not _identifier(user_id):
            raise ConfirmationError("UNAUTHORIZED", "Подтверждения недоступны.")
        async with self.pool.acquire() as connection, connection.transaction():
            await connection.execute(
                "INSERT INTO labbi_users(user_id) VALUES ($1) ON CONFLICT DO NOTHING", user_id
            )
            await connection.execute(
                "UPDATE labbi_users SET confirmation_epoch=confirmation_epoch+1, "
                "updated_at=clock_timestamp() WHERE user_id=$1",
                user_id,
            )
            return await revoke_actions(
                connection, user_id, "Подтверждения отозваны командой остановки."
            )

    async def set_timezone(self, user_id: int, iana: str) -> str:
        if not _identifier(user_id):
            raise ConfirmationError("UNAUTHORIZED", "Настройка недоступна.")
        try:
            zone = _TIMEZONE.validate_python(iana, strict=True)
        except ValidationError:
            raise ConfirmationError(
                "INVALID_ARGUMENTS", "Укажи существующее IANA-имя часового пояса."
            ) from None
        async with self.pool.acquire() as connection, connection.transaction():
            await connection.execute(
                "INSERT INTO labbi_users(user_id) VALUES ($1) ON CONFLICT DO NOTHING", user_id
            )
            previous = await connection.fetchval(
                "SELECT timezone FROM labbi_users WHERE user_id=$1 FOR UPDATE", user_id
            )
            if previous != zone:
                await connection.execute(
                    "UPDATE labbi_users SET timezone=$2, confirmation_epoch=confirmation_epoch+1, "
                    "revision=revision+1, updated_at=clock_timestamp() WHERE user_id=$1",
                    user_id,
                    zone,
                )
                await revoke_actions(
                    connection, user_id, "Подтверждение отозвано после смены часового пояса."
                )
        return zone


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
