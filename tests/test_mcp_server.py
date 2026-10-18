"""Проверки настоящего MCP-протокола с независимым SDK-клиентом.

Внешний HTTP и PostgreSQL не требуются: сервисы подменены. Последняя проверка
запускает отдельный stdio-процесс и проверяет отказ записи без доверенного права.
"""

import asyncio
import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from jsonschema import Draft202012Validator
from mcp import Client
from mcp.shared.exceptions import MCPError

from app.config import MCPSettings
from app.mcp.contracts import ReminderResult, WeatherResult, WeekResult
from app.mcp.schedule import ScheduleData, ScheduleService
from app.mcp.server import create_server
from scripts.mcp_probe import probe

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 18, 7, tzinfo=UTC)


@pytest.fixture
def mcp_settings():
    return MCPSettings(
        postgres_password="test-db-password",
        schedule_file=str(ROOT / "data" / "schedule.json"),
    )


@pytest.fixture
def schedule():
    return ScheduleService.from_file(ROOT / "data" / "schedule.json", "Europe/Moscow")


@pytest.fixture
def weather():
    value = WeatherResult.model_validate(
        {
            "status": "ok",
            "place": {"name": "Казань", "country": "Россия", "region": "Татарстан"},
            "temperature_c": 7.5,
            "condition": {"code": 3, "description": "Пасмурно"},
            "wind": {"speed": 12.3, "unit": "km/h"},
            "observed_at": "2026-10-18T10:00:00+03:00",
            "timezone": "Europe/Moscow",
            "source": "Open-Meteo",
        }
    )
    return SimpleNamespace(get_weather=AsyncMock(return_value=value))


async def test_discovery_exposes_exact_contracts_and_read_only_resource(mcp_settings, schedule):
    server = create_server(mcp_settings, schedule=schedule, now=lambda: NOW)
    async with Client(server, read_timeout_seconds=3) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        resources = (await client.list_resources()).resources
    assert set(tools) == {"get_weather", "get_schedule", "add_reminder"}
    assert set(tools["get_weather"].input_schema["properties"]) == {"city"}
    assert set(tools["get_schedule"].input_schema["properties"]) == {"date"}
    assert set(tools["add_reminder"].input_schema["properties"]) == {"text", "remind_at"}
    for name, tool in tools.items():
        assert tool.description
        assert tool.input_schema["additionalProperties"] is False
        assert tool.output_schema
        Draft202012Validator.check_schema(tool.input_schema)
        Draft202012Validator.check_schema(tool.output_schema)
        assert tool.annotations.read_only_hint is (name != "add_reminder")
    assert [resource.uri for resource in resources] == ["schedule://current-week"]
    assert resources[0].mime_type == "application/json"


async def test_weather_result_is_structured_and_matches_discovered_schema(
    mcp_settings, schedule, weather
):
    server = create_server(mcp_settings, weather=weather, schedule=schedule, now=lambda: NOW)
    async with Client(server, read_timeout_seconds=3) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        result = await client.call_tool("get_weather", {"city": "  Казань  "})
    assert not result.is_error
    assert result.structured_content["temperature_c"] == 7.5
    assert result.structured_content["place"]["name"] == "Казань"
    assert result.structured_content["wind"] == {"speed": 12.3, "unit": "km/h"}
    assert result.structured_content["source"] == "Open-Meteo"
    Draft202012Validator(tools["get_weather"].output_schema).validate(result.structured_content)
    assert json.loads(result.content[0].text) == result.structured_content
    wire = result.model_dump(mode="json", by_alias=True)
    assert wire["structuredContent"] == result.structured_content
    assert wire["isError"] is False
    weather.get_weather.assert_awaited_once_with("Казань")


async def test_schedule_date_and_week_use_the_same_canonical_lessons(mcp_settings, schedule):
    server = create_server(mcp_settings, schedule=schedule, now=lambda: NOW)
    async with Client(server, read_timeout_seconds=3) as client:
        lesson_result = await client.call_tool("get_schedule", {"date": "2026-10-12"})
        free_day = await client.call_tool("get_schedule", {"date": "2026-10-18"})
        resource = await client.read_resource("schedule://current-week")
    assert not lesson_result.is_error
    assert lesson_result.structured_content == schedule.get_schedule("2026-10-12").model_dump(
        mode="json"
    )
    assert free_day.structured_content["lessons"] == []
    assert resource.contents[0].mime_type == "application/json"
    week = WeekResult.model_validate_json(resource.contents[0].text).root
    assert week.week_start == date(2026, 10, 12)
    assert week.week_end == date(2026, 10, 18)
    assert len(week.days) == 7
    assert [lesson.model_dump() for lesson in week.days[0].lessons] == [
        lesson.model_dump() for lesson in schedule.get_schedule("2026-10-12").root.lessons
    ]


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("get_weather", {"city": "   "}),
        ("get_weather", {"city": "Казань", "url": "https://private-key.example"}),
        ("get_schedule", {"date": "2026-02-30"}),
        ("get_schedule", {"date": "2026-10-12", "path": "/private-key/file"}),
        (
            "add_reminder",
            {"text": "тест", "remind_at": "2030-10-18T10:00:00+03:00", "user_id": 77},
        ),
        ("add_reminder", {"text": "тест", "remind_at": "2030-10-18T10:00:00"}),
    ],
)
async def test_invalid_arguments_do_not_reach_services(
    mcp_settings, schedule, weather, name, arguments
):
    reminders = SimpleNamespace(add_reminder=AsyncMock(side_effect=AssertionError))
    server = create_server(
        mcp_settings, schedule=schedule, weather=weather, reminders=reminders, now=lambda: NOW
    )
    async with Client(server, read_timeout_seconds=3) as client:
        result = await client.call_tool(name, arguments)
    assert result.is_error
    assert result.structured_content["error"]["code"] == "INVALID_ARGUMENTS"
    assert "private-key" not in result.model_dump_json()
    weather.get_weather.assert_not_awaited()
    reminders.add_reminder.assert_not_awaited()


async def test_unknown_tool_does_not_dispatch_any_action(mcp_settings, schedule, weather):
    server = create_server(mcp_settings, schedule=schedule, weather=weather, now=lambda: NOW)
    async with Client(server, read_timeout_seconds=3) as client:
        result = await client.call_tool("run_shell", {"command": "private-key"})
    assert result.is_error
    assert result.structured_content["error"]["code"] == "UNKNOWN_TOOL"
    assert "private-key" not in result.model_dump_json()
    weather.get_weather.assert_not_awaited()


async def test_unknown_resource_refuses_arbitrary_file_paths(mcp_settings, schedule):
    server = create_server(mcp_settings, schedule=schedule, now=lambda: NOW)
    async with Client(server, read_timeout_seconds=3) as client:
        with pytest.raises(MCPError) as exc:
            await client.read_resource("file:///private-key/schedule.json")
    assert exc.value.code == -32002
    assert "private-key" not in str(exc.value)


async def test_invalid_service_result_becomes_safe_mcp_error(mcp_settings, schedule, weather):
    weather.get_weather.return_value = {"status": "ok", "private-key": "private-result"}
    server = create_server(mcp_settings, schedule=schedule, weather=weather, now=lambda: NOW)
    async with Client(server, read_timeout_seconds=3) as client:
        result = await client.call_tool("get_weather", {"city": "Казань"})
    assert result.is_error
    assert result.structured_content["error"]["code"] == "INVALID_RESULT"
    assert "private-key" not in result.model_dump_json()
    assert "private-result" not in result.model_dump_json()


async def test_service_exception_does_not_expose_secret_details(mcp_settings, schedule, weather):
    weather.get_weather.side_effect = RuntimeError("private-key upstream connection")
    server = create_server(mcp_settings, schedule=schedule, weather=weather, now=lambda: NOW)
    async with Client(server, read_timeout_seconds=3) as client:
        result = await client.call_tool("get_weather", {"city": "Казань"})
    assert result.is_error
    assert result.structured_content["error"]["code"] == "SOURCE_UNAVAILABLE"
    assert "private-key" not in result.model_dump_json()


async def test_service_timeout_is_bounded_and_structured(mcp_settings, schedule, weather):
    async def blocked(_city):
        await asyncio.Event().wait()

    weather.get_weather.side_effect = blocked
    server = create_server(
        replace(mcp_settings, mcp_timeout_seconds=1),
        schedule=schedule,
        weather=weather,
        now=lambda: NOW,
    )
    async with Client(server, read_timeout_seconds=3) as client:
        result = await client.call_tool("get_weather", {"city": "Казань"})
    assert result.is_error
    assert result.structured_content["error"]["code"] == "TIMEOUT"


async def test_capability_stays_in_metadata_and_is_not_echoed(mcp_settings, schedule):
    reminder_id = UUID("e59e3cf1-aa7e-4c38-a642-6031327f8736")
    reminders = SimpleNamespace(
        add_reminder=AsyncMock(
            return_value=ReminderResult.model_validate(
                {
                    "status": "ok",
                    "reminder_id": str(reminder_id),
                    "remind_at": "2030-10-18T10:00:00+03:00",
                    "timezone": "Europe/Moscow",
                    "created": True,
                }
            )
        )
    )
    context = {"action_id": "322b4143-a441-4b9c-82f3-e7f1a533722a", "capability": "a" * 64}
    server = create_server(mcp_settings, schedule=schedule, reminders=reminders, now=lambda: NOW)
    async with Client(server, read_timeout_seconds=3) as client:
        result = await client.call_tool(
            "add_reminder",
            {"text": "отправить отчёт", "remind_at": "2030-10-18T10:00:00+03:00"},
            meta={"labbi/action-capability": context},
        )
    assert not result.is_error
    assert result.structured_content["reminder_id"] == str(reminder_id)
    arguments, forwarded = reminders.add_reminder.await_args.args
    assert arguments.text == "отправить отчёт"
    assert forwarded == context
    assert "capability" not in result.model_dump_json()
    assert context["capability"] not in result.model_dump_json()


async def test_mutation_timeout_reports_unknown_outcome(mcp_settings, schedule):
    async def blocked(_arguments, _context):
        await asyncio.Event().wait()

    reminders = SimpleNamespace(add_reminder=AsyncMock(side_effect=blocked))
    server = create_server(
        replace(mcp_settings, mcp_timeout_seconds=1),
        schedule=schedule,
        reminders=reminders,
        now=lambda: NOW,
    )
    async with Client(server, read_timeout_seconds=3) as client:
        result = await client.call_tool(
            "add_reminder",
            {"text": "отправить отчёт", "remind_at": "2030-10-18T10:00:00+03:00"},
            meta={
                "labbi/action-capability": {
                    "action_id": "322b4143-a441-4b9c-82f3-e7f1a533722a",
                    "capability": "a" * 64,
                }
            },
        )
    assert result.is_error
    assert result.structured_content["error"]["code"] == "OUTCOME_UNKNOWN"
    reminders.add_reminder.assert_awaited_once()


@pytest.mark.parametrize(
    "lesson_count,title_length,location_length,raw_fits",
    [(32, 200, 100, False), (12, 105, 50, True)],
    ids=["raw-payload-too-large", "resource-envelope-too-large"],
)
async def test_oversized_week_is_rejected_before_transport(
    mcp_settings, lesson_count, title_length, location_length, raw_fits
):
    monday = date(2026, 10, 12)
    data = ScheduleData.model_validate(
        {
            "timezone": "Europe/Moscow",
            "coverage_start": monday.isoformat(),
            "coverage_end": (monday + timedelta(days=13)).isoformat(),
            "days": [
                {
                    "date": (monday + timedelta(days=offset)).isoformat(),
                    "lessons": [
                        {
                            "start": "08:00",
                            "end": "08:45",
                            "title": "я" * title_length,
                            "location": "я" * location_length,
                        }
                        for _ in range(lesson_count)
                    ],
                }
                for offset in range(14)
            ],
        }
    )
    service = ScheduleService(data, "Europe/Moscow")
    raw = json.dumps(
        service.current_week(now=NOW).model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    # Второй вариант помещается сам по себе, но выходит за предел в JSON-обёртке MCP.
    assert (len(raw.encode("utf-8")) <= 32 * 1024) is raw_fits
    server = create_server(mcp_settings, schedule=service, now=lambda: NOW)
    async with Client(server, read_timeout_seconds=3) as client:
        result = await client.read_resource("schedule://current-week")
    value = json.loads(result.contents[0].text)
    assert value["status"] == "error"
    assert value["error"]["code"] == "INVALID_RESULT"
    assert len(result.contents[0].text.encode("utf-8")) <= 32 * 1024
    wire = json.dumps(
        result.model_dump(mode="json", by_alias=True, exclude_none=True), ensure_ascii=False
    )
    assert len(wire.encode("utf-8")) <= 32 * 1024


async def test_independent_stdio_probe_needs_no_database_or_telegram_keys(
    mcp_settings, monkeypatch
):
    monkeypatch.setenv("BOT_TOKEN", "invalid-telegram-token-must-be-ignored")
    monkeypatch.setenv("OPENROUTER_API_KEY", "private-key-must-not-be-used")
    settings = replace(mcp_settings, postgres_host="127.0.0.1", postgres_port=1)
    async with asyncio.timeout(20):
        report = await probe(settings)
    assert report["transport"] == "stdio"
    assert {tool["name"] for tool in report["tools"]} == {
        "get_weather",
        "get_schedule",
        "add_reminder",
    }
    assert all(tool["output_schema"] for tool in report["tools"])
    assert report["resources"] == ["schedule://current-week"]
    assert report["schedule"]["status"] == "ok"
    assert report["week"]["days"] == 7
    assert report["write_without_context"] == {"is_error": True, "code": "UNAUTHORIZED"}
    assert "private-key" not in json.dumps(report)
    assert "invalid-telegram-token" not in json.dumps(report)
