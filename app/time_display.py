"""Читаемые даты для сообщений; ISO остаётся форматом контрактов и хранения."""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

_MONTHS = (
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
)


def format_date(value: date) -> str:
    return f"{value.day} {_MONTHS[value.month - 1]} {value.year}"


def format_utc_offset(moment: datetime) -> str:
    offset = moment.utcoffset()
    if offset is None:
        raise ValueError("Время должно содержать часовой пояс.")
    sign = "+" if offset >= timedelta(0) else "-"
    total = abs(offset // timedelta(microseconds=1))
    hours, remainder = divmod(total, 3_600_000_000)
    minutes, remainder = divmod(remainder, 60_000_000)
    seconds, micros = divmod(remainder, 1_000_000)
    result = f"UTC{sign}{hours:02}:{minutes:02}"
    if seconds or micros:
        result += f":{seconds:02}"
    if micros:
        result += f".{micros:06}"
    return result


def format_datetime(moment: datetime, timezone: str | ZoneInfo | None = None) -> str:
    """Display one exact aware moment, optionally converted to an explicit IANA zone."""
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("Время должно содержать часовой пояс.")
    if timezone is not None:
        moment = moment.astimezone(ZoneInfo(timezone) if isinstance(timezone, str) else timezone)
    clock = moment.strftime("%H:%M")
    if moment.second or moment.microsecond:
        clock += f":{moment.second:02}"
    if moment.microsecond:
        clock += f".{moment.microsecond:06}"
    return f"{format_date(moment)}, {clock} ({format_utc_offset(moment)})"
