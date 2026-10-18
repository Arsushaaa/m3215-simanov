"""Пользовательские даты сохраняют момент, точность и сезонное смещение."""

from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from app.time_display import format_date, format_datetime, format_utc_offset


@pytest.mark.parametrize(
    "raw, zone, expected",
    [
        (
            "2026-10-20T16:00:00+00:00",
            "Europe/Moscow",
            "20 октября 2026, 19:00 (UTC+03:00)",
        ),
        (
            "2026-10-20T19:00:12+03:00",
            None,
            "20 октября 2026, 19:00:12 (UTC+03:00)",
        ),
        (
            "2026-10-20T19:00:00.000001+03:00",
            None,
            "20 октября 2026, 19:00:00.000001 (UTC+03:00)",
        ),
        (
            "2026-10-20T19:00:12.123456+03:00",
            ZoneInfo("Europe/Moscow"),
            "20 октября 2026, 19:00:12.123456 (UTC+03:00)",
        ),
        (
            "2026-10-20T19:00:00+00:00",
            "Asia/Kathmandu",
            "21 октября 2026, 00:45 (UTC+05:45)",
        ),
        (
            "2026-10-20T01:00:00+00:00",
            "America/St_Johns",
            "19 октября 2026, 22:30 (UTC-02:30)",
        ),
        (
            "2026-01-01T00:00:00+00:00",
            "UTC",
            "1 января 2026, 00:00 (UTC+00:00)",
        ),
    ],
)
def test_datetime_is_readable_and_exact(raw, zone, expected):
    moment = datetime.fromisoformat(raw)
    assert format_datetime(moment, zone) == expected
    assert moment.isoformat() == raw


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2026-10-25T00:30:00+00:00", "25 октября 2026, 02:30 (UTC+02:00)"),
        ("2026-10-25T01:30:00+00:00", "25 октября 2026, 02:30 (UTC+01:00)"),
    ],
)
def test_dst_repeated_clock_is_distinguished_by_actual_offset(raw, expected):
    assert format_datetime(datetime.fromisoformat(raw), "Europe/Berlin") == expected


@pytest.mark.parametrize("formatter", [format_datetime, format_utc_offset])
def test_naive_datetime_cannot_be_displayed_as_a_known_moment(formatter):
    with pytest.raises(ValueError, match="часовой пояс"):
        formatter(datetime(2026, 10, 20, 19))


def test_utc_offset_does_not_discard_historical_seconds_or_microseconds():
    offset = timezone(-timedelta(hours=3, seconds=15, microseconds=2))
    assert format_utc_offset(datetime(1900, 1, 1, tzinfo=offset)) == "UTC-03:00:15.000002"


def test_date_uses_russian_month_without_process_locale():
    assert format_date(datetime(2026, 12, 31, tzinfo=UTC)) == "31 декабря 2026"
