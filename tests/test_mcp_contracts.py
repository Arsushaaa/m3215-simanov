"""Контракты проверяются на транспортных JSON-данных, без внешних запросов."""

from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
from uuid import UUID

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaError
from mcp.types import CallToolResult, Tool
from pydantic import ValidationError

from app.mcp.contracts import (
    AgentDecision,
    ReminderArguments,
    ReminderResult,
    ReminderTimeError,
    ScheduleArguments,
    ScheduleResult,
    WeatherArguments,
    WeatherResult,
    tool_schemas,
    validate_local_datetime,
    validate_reminder_time,
)

NOW = datetime(2026, 10, 12, 6, tzinfo=UTC)
REMINDER = {"text": "отправить отчёт", "remind_at": "2026-10-13T18:30:00+03:00"}
WEATHER = {
    "status": "ok",
    "place": {"name": "Казань", "country": "Россия", "region": "Татарстан"},
    "temperature_c": 8.5,
    "condition": {"code": 3, "description": "Пасмурно"},
    "wind": {"speed": 12.0, "unit": "km/h"},
    "observed_at": "2026-10-12T09:00:00+03:00",
    "timezone": "Europe/Moscow",
    "source": "Open-Meteo",
}
SCHEDULE = {
    "status": "ok",
    "date": "2026-10-18",
    "timezone": "Europe/Moscow",
    "lessons": [],
    "source": "local_schedule",
}
RECEIPT = {
    "status": "ok",
    "reminder_id": "811c40d2-14d8-44c4-91ae-14cd1ef13ca7",
    "remind_at": "2026-10-13T18:30:00+03:00",
    "timezone": "Europe/Moscow",
    "created": True,
}


@pytest.mark.parametrize("city", ["", "  ", "a" * 101, 123, True, None])
def test_weather_rejects_empty_long_or_nonstring_city(city):
    with pytest.raises(ValidationError):
        WeatherArguments.model_validate({"city": city})


def test_arguments_trim_strings_but_do_not_accept_model_owner():
    assert WeatherArguments(city="  Казань  ").city == "Казань"
    assert ReminderArguments.model_validate({**REMINDER, "text": "  отчёт  "}).text == "отчёт"
    for model, arguments in (
        (WeatherArguments, {"city": "Казань"}),
        (ScheduleArguments, {"date": "2026-10-12"}),
        (ReminderArguments, REMINDER),
    ):
        with pytest.raises(ValidationError):
            model.model_validate({**arguments, "user_id": 42})


@pytest.mark.parametrize(
    "value",
    ["2026-2-03", "2026-02-30", "26-10-12", "1999-12-31", "2101-01-01", True, 1],
)
def test_schedule_date_is_real_iso_and_in_range(value):
    with pytest.raises(ValidationError):
        ScheduleArguments.model_validate({"date": value})


def test_schedule_accepts_leap_day_and_exact_boundary_dates():
    assert ScheduleArguments(date="2028-02-29").date == date(2028, 2, 29)
    assert ScheduleArguments(date="2000-01-01").model_dump(mode="json") == {"date": "2000-01-01"}
    assert ScheduleArguments(date=date(2100, 12, 31)).date.year == 2100


@pytest.mark.parametrize(
    "value",
    [
        "2026-10-13T18:30:00",
        "2026-10-13 18:30:00+03:00",
        "2026-02-30T18:30:00+03:00",
        "2026-10-13T18:30:00+25:00",
        "2026-10-13T18:30:00+00:60",
        "2026-10-13T18:30:00+01:99",
        "2026-10-13T18:30:00-00:60",
        "2026-10-13T24:00:00+03:00",
        "2026-10-13T18:60:00+03:00",
        "2026-10-13T18:30:60+03:00",
        1791900000,
        True,
        datetime(2026, 10, 13, 18, 30),
    ],
)
def test_reminder_requires_iso_with_explicit_valid_offset(value):
    with pytest.raises(ValidationError):
        ReminderArguments.model_validate({**REMINDER, "remind_at": value})


@pytest.mark.parametrize("text", ["", "   ", "a" * 501, False])
def test_reminder_text_limits(text):
    with pytest.raises(ValidationError):
        ReminderArguments.model_validate({**REMINDER, "text": text})


def test_reminder_future_context_is_explicit_and_normalized():
    normalized = validate_reminder_time(
        ReminderArguments.model_validate(REMINDER), now=NOW, timezone="Europe/Moscow"
    )
    assert normalized.isoformat() == REMINDER["remind_at"]
    assert normalized.tzinfo.key == "Europe/Moscow"
    for value in ("2026-10-12T09:00:00+03:00", "2026-10-11T18:30:00+03:00"):
        with pytest.raises(ReminderTimeError, match="будущем"):
            validate_reminder_time(
                ReminderArguments.model_validate({**REMINDER, "remind_at": value}),
                now=NOW,
                timezone="Europe/Moscow",
            )


@pytest.mark.parametrize(
    "timezone",
    [None, "+03:00", "Missing/Zone", "localtime", "posixrules", "posix/UTC", "right/UTC"],
)
def test_reminder_rejects_missing_or_nonportable_zone(timezone):
    with pytest.raises(ReminderTimeError) as error:
        validate_reminder_time(
            ReminderArguments.model_validate(REMINDER), now=NOW, timezone=timezone
        )
    assert error.value.code == ("TIMEZONE_REQUIRED" if timezone is None else "INVALID_ARGUMENTS")


def test_reminder_offset_must_match_saved_zone_and_clock_must_be_aware():
    arguments = ReminderArguments.model_validate({**REMINDER, "remind_at": "2026-10-13T18:30:00Z"})
    with pytest.raises(ReminderTimeError, match="Смещение"):
        validate_reminder_time(arguments, now=NOW, timezone="Europe/Moscow")
    with pytest.raises(ValueError, match="aware"):
        validate_reminder_time(arguments, now=NOW.replace(tzinfo=None), timezone="UTC")


@pytest.mark.parametrize("local", [datetime(2026, 3, 29, 2, 30), datetime(2026, 10, 25, 2, 30)])
def test_dst_gap_and_overlap_need_clarification(local):
    with pytest.raises(ReminderTimeError) as error:
        validate_local_datetime(local, "Europe/Berlin")
    assert error.value.code == "AMBIGUOUS_TIME"


def test_unambiguous_time_after_dst_is_accepted_with_correct_offset():
    local = validate_local_datetime(datetime(2026, 10, 25, 3, 30), "Europe/Berlin")
    assert local.utcoffset() == timedelta(hours=1)
    arguments = ReminderArguments.model_validate({**REMINDER, "remind_at": local.isoformat()})
    assert validate_reminder_time(arguments, now=NOW, timezone="Europe/Berlin") == local


@pytest.mark.parametrize(
    "name,arguments,output",
    [
        ("get_weather", {"city": "Казань"}, WEATHER),
        ("get_schedule", {"date": "2026-10-18"}, SCHEDULE),
        ("add_reminder", REMINDER, RECEIPT),
    ],
)
def test_schemas_interoperate_with_json_schema_and_real_mcp_sdk(name, arguments, output):
    schemas = tool_schemas(name)
    assert schemas["inputSchema"]["type"] == schemas["outputSchema"]["type"] == "object"
    for schema, value in ((schemas["inputSchema"], arguments), (schemas["outputSchema"], output)):
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        validator.validate(value)
        with pytest.raises(JsonSchemaError):
            validator.validate({**value, "owner": 42})
    tool = Tool(name=name, **schemas)
    assert tool.input_schema == schemas["inputSchema"]
    assert tool.output_schema == schemas["outputSchema"]
    assert tool.model_dump(mode="json", by_alias=True)["inputSchema"] == schemas["inputSchema"]
    result = CallToolResult(content=[], structuredContent=output, isError=False)
    assert result.model_dump(mode="json", by_alias=True)["structuredContent"] == output


def test_output_rejects_nested_extras_wrong_units_and_nonfinite_numbers():
    for field, value in (
        ("place", {**WEATHER["place"], "owner": 42}),
        ("wind", {"speed": 12.0, "unit": "mph"}),
        ("wind", {"speed": -1.0, "unit": "km/h"}),
        ("temperature_c", float("nan")),
        ("temperature_c", float("inf")),
        ("temperature_c", "8.5"),
        ("temperature_c", True),
        ("observed_at", "2026-10-12T09:00:00+02:00"),
    ):
        with pytest.raises(ValidationError):
            WeatherResult.model_validate({**WEATHER, field: value})


def test_ambiguous_cities_are_distinguishable_and_bounded():
    place = {"name": "Александровка", "country": "Россия", "region": "Первый регион"}
    payload = {"status": "ambiguous", "query": "Александровка", "source": "Open-Meteo"}
    valid = WeatherResult.model_validate(
        {**payload, "candidates": [place, {**place, "region": "Другой регион"}]}
    )
    assert valid.root.status == "ambiguous"
    for candidates in ([place], [place, place], [{**place, "region": str(i)} for i in range(6)]):
        with pytest.raises(ValidationError):
            WeatherResult.model_validate({**payload, "candidates": candidates})


@pytest.mark.parametrize(
    "code,message,retryable",
    [
        ("TIMEOUT", "Сервис не ответил вовремя.", True),
        ("AMBIGUOUS_PLACE", "Нужно уточнить одноимённый населённый пункт.", False),
    ],
)
def test_error_result_is_safe_typed_and_shared_by_all_tools(code, message, retryable):
    payload = {
        "status": "error",
        "error": {"code": code, "message": message, "retryable": retryable},
    }
    for model, name in (
        (WeatherResult, "get_weather"),
        (ScheduleResult, "get_schedule"),
        (ReminderResult, "add_reminder"),
    ):
        assert model.model_validate(payload).model_dump(mode="json") == payload
        Draft202012Validator(tool_schemas(name)["outputSchema"]).validate(payload)
        malformed = deepcopy(payload)
        malformed["error"]["retryable"] = "true"
        with pytest.raises(ValidationError):
            model.model_validate(malformed)


def test_committed_receipt_remains_valid_after_reminder_time_has_passed():
    receipt = ReminderResult.model_validate({**RECEIPT, "created": False})
    assert receipt.root.reminder_id == UUID(RECEIPT["reminder_id"])
    assert receipt.root.created is False
    # Контракт квитанции не проверяет будущее: такую проверку делают только до новой записи.
    assert receipt.root.remind_at < datetime(2027, 1, 1, tzinfo=UTC)


@pytest.mark.parametrize(
    "decision",
    [
        {"action": "respond", "text": "Список изменяемый.", "reason": "Общий учебный вопрос"},
        {"action": "clarify", "question": "Во сколько напомнить?", "reason": "Нет точного времени"},
        {
            "action": "call_tool",
            "name": "get_weather",
            "arguments": {"city": "Казань"},
            "reason": "Нужны текущие данные",
        },
        {
            "action": "call_tool",
            "name": "get_schedule",
            "arguments": {"date": "2026-10-12"},
            "reason": "Нужно расписание",
        },
        {
            "action": "call_tool",
            "name": "add_reminder",
            "arguments": REMINDER,
            "reason": "Нужно подготовить подтверждение",
        },
        {
            "action": "read_resource",
            "uri": "schedule://current-week",
            "reason": "Нужна текущая неделя",
        },
    ],
)
def test_agent_decision_is_single_typed_action(decision):
    assert AgentDecision.model_validate(decision).model_dump(mode="json") == decision


@pytest.mark.parametrize(
    "decision",
    [
        {"action": "call_tool", "name": "delete_all", "arguments": {}, "reason": "Команда"},
        {
            "action": "call_tool",
            "name": "get_weather",
            "arguments": {"date": "2026-10-12"},
            "reason": "Ошибка",
        },
        {
            "action": "call_tool",
            "name": "add_reminder",
            "arguments": {**REMINDER, "user_id": 42},
            "reason": "Ошибка",
        },
        {"action": "read_resource", "uri": "file:///etc/passwd", "reason": "Ошибка"},
        {"action": "respond", "text": "Ответ", "reason": "Причина", "calls": []},
        [{"action": "respond", "text": "Ответ", "reason": "Причина"}],
        {"action": "respond", "text": "Ответ", "reason": "a" * 301},
    ],
)
def test_agent_decision_rejects_unknown_actions_extras_and_mismatched_arguments(decision):
    with pytest.raises(ValidationError):
        AgentDecision.model_validate(decision)


def test_unknown_schema_name_is_rejected_without_dynamic_lookup():
    with pytest.raises(ValueError, match="Неизвестный инструмент"):
        tool_schemas("__import__")
