"""City confirmation, fixed point weather and bounded settings UI without external APIs."""

import asyncio
import json
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest
from mcp import Client
from test_confirmation_handlers import callback_update, sent, text_update
from test_confirmation_handlers import setup_ui as setup_ui
from test_mcp_client import setup_client as setup_client
from test_weather import FakeResponse, FakeSession, location, weather

from app.mcp.client import MCPClientError
from app.mcp.contracts import Coordinates, PointWeatherResult, WeatherToolArguments, tool_schemas
from app.mcp.server import create_server
from app.mcp.weather import WEATHER_URL, WeatherService
from app.notifications.city import EXPIRED, CityLookup, CityUI
from app.notifications.models import CityCandidate, briefing_moment, next_briefing
from app.routing.policy import ScopedMCP

POINT = Coordinates(latitude=55.79, longitude=49.12, timezone="Europe/Moscow")
CITY = CityCandidate(
    geonames_id=551487,
    latitude=55.79,
    longitude=49.12,
    timezone="Europe/Moscow",
    place={"name": "Казань", "country": "Россия", "region": "Татарстан"},
)


@pytest.mark.parametrize(
    "changes",
    [
        {"latitude": 91.0},
        {"longitude": -181.0},
        {"latitude": float("nan")},
        {"longitude": float("inf")},
        {"latitude": True},
        {"timezone": "Invalid/Zone"},
        {"actor_id": 7},
        {"label": "Москва"},
        {"destination_chat_id": 1},
    ],
)
def test_point_contract_rejects_identity_nonfinite_coordinates_and_labels(changes):
    with pytest.raises(ValueError):
        WeatherToolArguments.model_validate({"point": {**POINT.model_dump(), **changes}})


@pytest.mark.parametrize(
    "value",
    [
        {"city": "Казань", "point": POINT.model_dump()},
        {"latitude": 55.79, "longitude": 49.12},
        {"city": "Kazan"},
    ],
)
def test_mcp_weather_branches_are_closed_and_mutually_exclusive(value):
    with pytest.raises(ValueError):
        WeatherToolArguments.model_validate(value)


async def test_point_weather_calls_current_api_once_and_never_geocodes():
    session = FakeSession(FakeResponse(weather()))
    result = await WeatherService(session).get_weather_point(POINT)
    assert result.root.status == "ok" and result.root.point == POINT
    assert len(session.calls) == 1 and session.calls[0][0] == WEATHER_URL
    assert session.calls[0][1]["params"]["latitude"] == POINT.latitude
    assert "place" not in result.model_dump(mode="json")


@pytest.mark.parametrize("defect", ["timezone", "invalid", "timeout", "upstream"])
async def test_point_weather_safe_failures_have_no_retry_or_geocoding(defect):
    raw = weather(timezone="Asia/Tokyo") if defect == "timezone" else weather()
    if defect == "invalid":
        raw["current"]["temperature_2m"] = 1000.0
    session = FakeSession(
        TimeoutError()
        if defect == "timeout"
        else FakeResponse(raw, status=503 if defect == "upstream" else 200)
    )
    result = await WeatherService(session).get_weather_point(POINT)
    assert result.root.status == "error"
    assert len(session.calls) == 1


@pytest.mark.parametrize("role", ["lifestyle", "schedule", "knowledge", "router"])
async def test_user_query_scope_cannot_call_point_weather_even_though_mcp_advertises_it(role):
    mcp = SimpleNamespace(
        available=True,
        tools={
            "get_weather": {
                "name": "get_weather",
                "description": "Weather",
                **tool_schemas("get_weather"),
            }
        },
        resources={},
        call_tool=AsyncMock(),
    )
    scope = ScopedMCP(mcp, role)
    with pytest.raises(MCPClientError):
        await scope.call_tool("get_weather", {"point": POINT.model_dump()})
    mcp.call_tool.assert_not_awaited()


async def test_host_lifestyle_scope_can_read_a_public_point_without_identity_or_write_right():
    mcp = SimpleNamespace(
        available=True,
        tools={"get_weather": {"name": "get_weather"}},
        resources={},
        call_tool=AsyncMock(return_value={"status": "ok"}),
    )
    await ScopedMCP(mcp, "lifestyle", allow_point=True).call_tool(
        "get_weather", {"point": POINT.model_dump()}
    )
    mcp.call_tool.assert_awaited_once()
    with pytest.raises(ValueError):
        ScopedMCP(mcp, "schedule", allow_point=True)


@pytest.mark.parametrize(
    "now,zone,expected",
    [
        ("2026-11-01T04:59:59+00:00", "Europe/Moscow", "2026-11-01T05:00:00+00:00"),
        ("2026-11-01T05:00:00+00:00", "Europe/Moscow", "2026-11-02T05:00:00+00:00"),
        ("2026-10-24T07:00:00+00:00", "Europe/Berlin", "2026-10-25T07:00:00+00:00"),
        ("2026-03-28T09:00:00+00:00", "Europe/Berlin", "2026-03-29T06:00:00+00:00"),
    ],
)
def test_next_eight_is_computed_from_local_date_and_dst_not_a_fixed_utc_interval(
    now, zone, expected
):
    assert next_briefing(datetime.fromisoformat(now), zone) == datetime.fromisoformat(expected)


def test_skipped_calendar_day_and_naive_clock_are_not_silently_adjusted():
    with pytest.raises(ValueError):
        briefing_moment(date(2011, 12, 30), "Pacific/Apia")
    with pytest.raises(ValueError):
        next_briefing(datetime(2026, 11, 1), "Europe/Moscow")


async def test_city_lookup_keeps_homonyms_and_small_places_without_picking_by_population():
    session = FakeSession(
        FakeResponse(
            {
                "results": [
                    location(
                        id=1, name="Ростов", country="Украина", country_code="UA", population=5
                    ),
                    location(id=2, name="Ростов-на-Дону", country="Россия", population=1000000),
                ]
            }
        )
    )
    options = await CityLookup(session).search("Ростов")
    assert [o.geonames_id for o in options] == [1, 2]
    assert len(session.calls) == 1 and session.calls[0][1]["params"]["language"] == "ru"


async def test_city_lookup_accepts_country_in_another_language_without_guessing():
    session = FakeSession(
        FakeResponse(
            {
                "results": [
                    location(id=1, name="Париж", country="Франция", country_code="FR"),
                    location(id=2, name="Париж", country="США", country_code="US"),
                ]
            }
        )
    )
    options = await CityLookup(session).search("Paris, France")
    assert [o.geonames_id for o in options] == [1]


@pytest.mark.parametrize(
    "query", ["x", "file:///etc/passwd", "Москва\nРоссия", "x" * 151, "Москва,,Россия"]
)
async def test_city_query_boundaries_do_not_make_http_requests(query):
    session = FakeSession()
    with pytest.raises(ValueError):
        await CityLookup(session).search(query)
    assert session.calls == []


async def test_city_menu_preview_does_not_save_until_owner_confirms():
    token = "a" * 22
    repository = SimpleNamespace(
        menu=AsyncMock(return_value={"options": [CITY]}),
        choose=AsyncMock(return_value=CITY),
        confirm_city=AsyncMock(return_value=CITY),
    )
    ui = CityUI(repository, None)
    preview = await ui.handle(1, f"city:pick:{token}:0")
    assert "55.79" in preview.text and "Сохранить" in preview.text
    repository.confirm_city.assert_not_awaited()
    saved = await ui.handle(1, f"city:use:{token}:0")
    assert saved.dismiss_previous and not saved.rows
    repository.confirm_city.assert_awaited_once_with(1, token, 0)


@pytest.mark.parametrize(
    "data",
    [
        "city:use:" + ("a" * 22),
        "city:cancel:" + ("a" * 22) + ":0",
        "city:use:bad:0",
        "city:page:" + ("a" * 22) + ":999",
        "city:use:" + ("a" * 22) + ":0x",
    ],
)
async def test_city_callback_rejects_invalid_shapes_before_storage(data):
    repository = SimpleNamespace(confirm_city=AsyncMock())
    assert await CityUI(repository, None).handle(1, data) == EXPIRED
    repository.confirm_city.assert_not_awaited()


async def test_late_city_search_cannot_recreate_a_cancelled_menu():
    started, release = asyncio.Event(), asyncio.Event()

    async def lookup(query):
        started.set()
        await release.wait()
        return [CITY]

    repo = SimpleNamespace(
        start_menu=AsyncMock(return_value=("a" * 22, 1)),
        fill_menu=AsyncMock(return_value=False),
        menu=AsyncMock(),
    )
    task = asyncio.create_task(CityUI(repo, SimpleNamespace(search=lookup)).open(1, "Казань"))
    await started.wait()
    release.set()
    assert await task == EXPIRED
    repo.menu.assert_not_awaited()


async def test_commands_register_destination_from_actual_private_update_without_llm(setup_ui):
    dispatcher, bot, dialog = setup_ui
    service = SimpleNamespace(
        register_destination=AsyncMock(),
        status=AsyncMock(return_value="Настройки"),
        briefing=AsyncMock(return_value="Выключено"),
        city=AsyncMock(return_value="Выбери город"),
    )
    dialog.notifications = service
    await dispatcher.feed_update(bot, text_update("/notifications"), dialog=dialog)
    await dispatcher.feed_update(bot, text_update("/briefing_off", number=2), dialog=dialog)
    service.register_destination.assert_any_await(1, 1)
    service.briefing.assert_awaited_once_with(1, False)
    dialog.respond.assert_not_awaited()
    assert any(m.text == "Настройки" for m in sent(bot))


async def test_foreign_city_button_is_rejected_before_registration_or_mutation(setup_ui):
    dispatcher, bot, dialog = setup_ui
    service = SimpleNamespace(register_destination=AsyncMock(), city_callback=AsyncMock())
    dialog.notifications = service
    await dispatcher.feed_update(
        bot, callback_update("city:use:" + ("a" * 22) + ":0", owner=2), dialog=dialog
    )
    service.city_callback.assert_not_awaited()
    service.register_destination.assert_not_awaited()


async def test_sdk_point_weather_is_validated_and_does_not_access_profile_database(bot_settings):
    from app.config import MCPSettings

    session = FakeSession(FakeResponse(weather()))
    service = WeatherService(session)
    server = create_server(MCPSettings.load(None, environ={}), weather=service)
    async with Client(server, read_timeout_seconds=3) as client:
        result = await client.call_tool("get_weather", {"point": POINT.model_dump(mode="json")})
        assert not result.is_error
        assert PointWeatherResult.model_validate(result.structured_content).root.point == POINT
    assert len(session.calls) == 1


async def test_model_native_tools_keep_city_only_schema_after_mcp_point_extension(bot_settings):
    from app.agent import AgentService

    mcp = SimpleNamespace(
        tools={
            "get_weather": {
                "name": "get_weather",
                "description": "Weather",
                **tool_schemas("get_weather"),
            }
        },
        resources={},
    )
    agent = AgentService(None, mcp, None, bot_settings)
    weather_tool = next(t for t in agent._tools() if t["function"]["name"] == "get_weather")
    assert set(weather_tool["function"]["parameters"]["properties"]) == {"city"}
    assert "point" not in json.dumps(weather_tool)


async def test_mcp_client_previous_city_wire_schema_remains_supported_but_not_points(setup_client):
    from app.mcp.contracts import WeatherArguments, WeatherResult

    client, sdk, _ = setup_client
    weather_tool = next(t for t in sdk.list_tools.return_value.tools if t.name == "get_weather")
    weather_tool.input_schema = WeatherArguments.model_json_schema()
    weather_tool.output_schema = WeatherResult.model_json_schema()
    await client.start()
    try:
        assert client.available
        with pytest.raises(MCPClientError) as error:
            await client.call_tool("get_weather", {"point": POINT.model_dump()})
        assert error.value.code == "INVALID_ARGUMENTS"
        sdk.call_tool.assert_not_awaited()
    finally:
        await client.close()


async def test_mcp_client_binds_observation_to_exact_point_and_rejects_city_point_mixup(
    setup_client,
):
    from test_mcp_client import tool_result

    from app.mcp.contracts import WeatherCondition, WeatherPointOk, Wind

    client, sdk, _ = setup_client
    observation = WeatherPointOk(
        status="ok",
        point=POINT,
        temperature_c=8.5,
        condition=WeatherCondition(code=3, description="Пасмурно"),
        wind=Wind(speed=12.0, unit="km/h"),
        observed_at=datetime(2026, 11, 1, 12, tzinfo=ZoneInfo("Europe/Moscow")),
        timezone="Europe/Moscow",
        source="Open-Meteo",
    ).model_dump(mode="json")
    await client.start()
    try:
        sdk.call_tool.return_value = tool_result(observation)
        assert (await client.call_tool("get_weather", {"point": POINT.model_dump()}))[
            "point"
        ] == POINT.model_dump()
        sdk.call_tool.return_value = tool_result(
            {**observation, "point": {**observation["point"], "latitude": 1.0}}
        )
        with pytest.raises(MCPClientError) as error:
            await client.call_tool("get_weather", {"point": POINT.model_dump()})
        assert error.value.code == "INVALID_RESULT"
        sdk.call_tool.return_value = tool_result(observation)
        with pytest.raises(MCPClientError):
            await client.call_tool("get_weather", {"city": "Казань"})
    finally:
        await client.close()
