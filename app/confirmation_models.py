"""DTO подтверждений и точная карточка без SQL, Telegram и MCP-транспорта."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
from uuid import UUID

from app.mcp.contracts import ErrorCode, ReminderArguments
from app.time_display import format_datetime


class ConfirmationError(Exception):
    def __init__(self, code: ErrorCode, user_message: str):
        self.code, self.user_message = code, user_message
        super().__init__(user_message)


@dataclass(frozen=True)
class PreparedAction:
    action_id: UUID
    card_text: str = field(repr=False)
    arguments: ReminderArguments = field(repr=False)
    timezone: str
    expires_at: datetime
    state: str
    replayed: bool = False


@dataclass(frozen=True)
class Authorization:
    action_id: UUID
    arguments: ReminderArguments = field(repr=False)
    context: dict[str, str] = field(repr=False)


@dataclass(frozen=True)
class Outcome:
    status: Literal["committed", "cancelled", "expired", "unknown", "error"]
    code: ErrorCode | None
    message: str
    reminder_id: UUID | None = None
    remind_at: datetime | None = None
    timezone: str | None = None


@dataclass(frozen=True)
class SafeAgentEvent:
    created_at: datetime
    decision: str
    target: str | None
    status: str
    reason: str
    argument_summary: str
    error_code: str | None
    llm_calls: int
    mcp_calls: int
    duration_ms: int


def format_card(arguments: ReminderArguments, timezone: str, expires_at: datetime) -> str:
    """Display the exact moment in the saved zone without exposing wire-format ISO."""
    if expires_at.tzinfo is None or expires_at.utcoffset() is None:
        raise ValueError("Срок подтверждения должен содержать часовой пояс.")
    moment = format_datetime(arguments.remind_at, timezone)
    return (
        f"Сохранить напоминание?\n\nТекст: {arguments.text}\nКогда: {moment}\n"
        f"Часовой пояс: {timezone}\n\n"
        "Срок подтверждения — 5 минут с подготовки.\n"
        "Напоминание ещё не сохранено. Отправка уведомлений пока не подключена."
    )
