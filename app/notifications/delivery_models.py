"""Closed internal delivery values. Attempts are reserved before Telegram HTTP."""

import math
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from aiogram.exceptions import (
    TelegramAPIError,
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramNetworkError,
    TelegramRetryAfter,
    TelegramServerError,
    TelegramUnauthorizedError,
)

from app.time_display import format_datetime


@dataclass(frozen=True)
class Lease:
    event_id: UUID
    owner: int = field(repr=False)
    token: UUID = field(repr=False)
    version: int
    body: str | None = field(default=None, repr=False)
    destination: int | None = field(default=None, repr=False)
    attempt: int = 0


@dataclass(frozen=True)
class DeliveryFailure:
    code: str
    transient: bool
    uncertain: bool = False
    retry_after: float = 0


def classify_failure(error):
    if isinstance(error, TelegramRetryAfter):
        value = error.retry_after
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            return DeliveryFailure("INVALID_RETRY_AFTER", False)
        return DeliveryFailure("RATE_LIMIT", True, retry_after=float(value))
    if isinstance(error, (TimeoutError, TelegramNetworkError, OSError)):
        return DeliveryFailure("NETWORK_UNKNOWN", True, True)
    if isinstance(error, TelegramServerError):
        return DeliveryFailure("SERVER_ERROR", True, True)
    if isinstance(error, (TelegramForbiddenError, TelegramUnauthorizedError)):
        return DeliveryFailure("CHAT_UNAVAILABLE", False)
    if isinstance(error, (TelegramBadRequest, TelegramAPIError)):
        return DeliveryFailure("REQUEST_REJECTED", False)
    return DeliveryFailure("UNEXPECTED_ERROR", True, True)


def retry_delay(attempt, retry_after=0):
    if type(attempt) is not int or not 1 <= attempt <= 3:
        raise ValueError("Invalid attempt")
    return max(30 if attempt == 1 else 120, retry_after)


def reminder_body(text: str, due: datetime, timezone: str, now: datetime):
    if type(text) is not str or not 1 <= len(text) <= 500:
        raise ValueError("Invalid reminder content")
    title = "🔔 Напоминание с задержкой" if now > due else "🔔 Напоминание"
    return f"{title}\n\n{text}\n\nПлановое время: {format_datetime(due, timezone)}"


def valid_payload(body):
    if not isinstance(body, str) or not body.strip():
        return False
    try:
        return len(body.encode("utf-16-le")) // 2 <= 4096
    except UnicodeError:
        return False
