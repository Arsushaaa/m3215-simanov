"""Детерминированные проверки файла и ресурса недели без внешних сервисов."""

import json
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.mcp.contracts import WeekResult
from app.mcp.schedule import MAX_SCHEDULE_BYTES, ScheduleData, ScheduleService, ScheduleSourceError

SOURCE = Path(__file__).resolve().parents[1] / "data" / "schedule.json"
NOW = datetime(2026, 10, 12, 6, tzinfo=UTC)


@pytest.fixture
def service():
    return ScheduleService.from_file(SOURCE, timezone="Europe/Moscow")


def source_data(start: date, *, timezone: str = "Europe/Moscow") -> dict:
    return {
        "timezone": timezone,
        "coverage_start": start.isoformat(),
        "coverage_end": (start + timedelta(days=13)).isoformat(),
        "days": [
            {"date": (start + timedelta(days=i)).isoformat(), "lessons": []} for i in range(14)
        ],
    }


def write_source(tmp_path, value):
    target = tmp_path / "schedule.json"
    target.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return target


def test_repository_fixture_covers_all_fourteen_days_and_has_no_personal_fields():
    source = ScheduleData.model_validate_json(SOURCE.read_bytes())
    assert source.coverage_start == date(2026, 10, 12)
    assert source.coverage_end == date(2026, 10, 25)
    assert len(source.days) == 14
    assert all(not day.lessons for day in source.days if day.date.weekday() >= 5)
    assert all(day.lessons for day in source.days if day.date.weekday() < 5)
    assert set(source.days[0].lessons[0].model_dump()) == {"start", "end", "title", "location"}


def test_schedule_returns_exact_observable_lesson_fields(service):
    result = service.get_schedule("2026-10-12").model_dump(mode="json")
    assert result == {
        "status": "ok",
        "date": "2026-10-12",
        "timezone": "Europe/Moscow",
        "source": "local_schedule",
        "lessons": [
            {
                "start": "09:00",
                "end": "10:30",
                "title": "Основы программирования",
                "location": "Учебная аудитория 101",
            },
            {
                "start": "10:40",
                "end": "12:10",
                "title": "Дискретная математика",
                "location": "Учебная аудитория 102",
            },
        ],
    }


def test_empty_day_and_outside_coverage_are_distinct(service):
    empty = service.get_schedule("2026-10-18").root
    outside = service.get_schedule("2026-10-26").root
    assert empty.status == "ok" and empty.lessons == []
    assert outside.status == "outside_coverage"
    assert outside.coverage_start == date(2026, 10, 12)
    assert outside.coverage_end == date(2026, 10, 25)
    assert "lessons" not in outside.model_dump()


def test_current_week_is_monday_to_sunday_and_json_round_trips(service):
    result = service.current_week(now=NOW)
    week = result.root
    assert week.week_start == date(2026, 10, 12)
    assert week.week_end == date(2026, 10, 18)
    assert len(week.days) == 7
    assert week.days[-1].lessons == []
    assert WeekResult.model_validate_json(result.model_dump_json()) == result


def test_week_uses_configured_zone_before_selecting_calendar_date(service):
    # В UTC ещё воскресенье, в зоне расписания уже понедельник.
    week = service.current_week(now=datetime(2026, 10, 11, 22, tzinfo=UTC)).root
    assert week.week_start == date(2026, 10, 12)
    with pytest.raises(ValueError, match="aware"):
        service.current_week(now=NOW.replace(tzinfo=None))


@pytest.mark.parametrize(
    "start,now,week_start,week_end",
    [
        (
            date(2026, 10, 26),
            datetime(2026, 11, 1, 9, tzinfo=UTC),
            date(2026, 10, 26),
            date(2026, 11, 1),
        ),
        (
            date(2026, 12, 28),
            datetime(2027, 1, 1, 9, tzinfo=UTC),
            date(2026, 12, 28),
            date(2027, 1, 3),
        ),
    ],
)
def test_week_crosses_month_and_year_without_changing_system_clock(
    tmp_path, start, now, week_start, week_end
):
    service = ScheduleService.from_file(write_source(tmp_path, source_data(start)), "Europe/Moscow")
    result = service.current_week(now=now).root
    assert result.week_start == week_start
    assert result.week_end == week_end
    assert all(day.status == "ok" for day in result.days)


def test_partial_week_preserves_unknown_days(tmp_path):
    service = ScheduleService.from_file(
        write_source(tmp_path, source_data(date(2026, 10, 14))), "Europe/Moscow"
    )
    week = service.current_week(now=NOW).root
    assert [day.status for day in week.days] == ["outside_coverage"] * 2 + ["ok"] * 5
    assert "lessons" not in week.days[0].model_dump()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(timezone="Missing/Zone"),
        lambda value: value.update(owner="secret"),
        lambda value: value.update(coverage_end="2026-10-17"),
        lambda value: value["days"].pop(),
        lambda value: value["days"].append(value["days"][-1]),
        lambda value: value["days"].reverse(),
        lambda value: value["days"][0].update(owner=42),
        lambda value: value["days"][0].update(date="2026-02-30"),
    ],
)
def test_entire_file_is_validated_before_it_can_be_read(tmp_path, mutate):
    value = source_data(date(2026, 10, 12))
    mutate(value)
    with pytest.raises(ScheduleSourceError, match="некорректный формат"):
        ScheduleService.from_file(write_source(tmp_path, value), "Europe/Moscow")


@pytest.mark.parametrize(
    "lessons",
    [
        [{"start": "24:00", "end": "25:00", "title": "Занятие", "location": "101"}],
        [{"start": "10:00", "end": "10:00", "title": "Занятие", "location": "101"}],
        [{"start": "10:00", "end": "09:00", "title": "Занятие", "location": "101"}],
        [{"start": "10:00", "end": "11:00", "title": " ", "location": "101"}],
        [
            {
                "start": "10:00",
                "end": "11:00",
                "title": "Занятие",
                "location": "101",
                "teacher": "ФИО",
            }
        ],
        [
            {"start": "11:00", "end": "12:00", "title": "Позднее", "location": "101"},
            {"start": "09:00", "end": "10:00", "title": "Раннее", "location": "101"},
        ],
    ],
)
def test_lesson_schema_values_and_order_are_validated(tmp_path, lessons):
    value = source_data(date(2026, 10, 12))
    value["days"][0]["lessons"] = lessons
    with pytest.raises(ScheduleSourceError):
        ScheduleService.from_file(write_source(tmp_path, value), "Europe/Moscow")


def test_source_timezone_must_match_configuration(tmp_path):
    target = write_source(tmp_path, source_data(date(2026, 10, 12), timezone="UTC"))
    with pytest.raises(ScheduleSourceError, match="не совпадает"):
        ScheduleService.from_file(target, "Europe/Moscow")
    with pytest.raises(ScheduleSourceError, match="неизвестная"):
        ScheduleService.from_file(target, "localtime")


def test_io_errors_do_not_reveal_path_or_file_contents(tmp_path):
    missing = tmp_path / "private-path-with-secret-name.json"
    with pytest.raises(ScheduleSourceError) as error:
        ScheduleService.from_file(missing, "Europe/Moscow")
    assert str(error.value) == "Файл расписания недоступен."
    for raw in (b"invalid-json", b"\xff", b'{"timezone":"UTC","timezone":"Europe/Moscow"}'):
        target = tmp_path / "bad.json"
        target.write_bytes(raw)
        with pytest.raises(ScheduleSourceError) as error:
            ScheduleService.from_file(target, "Europe/Moscow")
        assert str(error.value) == "Файл расписания имеет некорректный формат."


def test_source_size_is_checked_before_json_parsing(tmp_path):
    raw = SOURCE.read_bytes()
    target = tmp_path / "size.json"
    target.write_bytes(raw + b" " * (MAX_SCHEDULE_BYTES - len(raw)))
    assert (
        ScheduleService.from_file(target, "Europe/Moscow").get_schedule("2026-10-18").root.lessons
        == []
    )
    with target.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ScheduleSourceError, match="1 МиБ"):
        ScheduleService.from_file(target, "Europe/Moscow")


def test_returned_list_cannot_mutate_subsequent_results(service):
    first = service.get_schedule("2026-10-12").root
    first.lessons.clear()
    assert len(service.get_schedule("2026-10-12").root.lessons) == 2


def test_service_rejects_invalid_input_before_date_selection(service):
    with pytest.raises(ValidationError):
        service.get_schedule("tomorrow")
    with pytest.raises(ValidationError):
        service.get_schedule(True)


def test_week_output_rejects_missing_days_or_wrong_dates(service):
    payload = service.current_week(now=NOW).model_dump(mode="json")
    missing = deepcopy(payload)
    missing["days"].pop()
    with pytest.raises(ValidationError):
        WeekResult.model_validate(missing)
    payload["days"][0]["date"] = "2026-10-13"
    with pytest.raises(ValidationError):
        WeekResult.model_validate(payload)
