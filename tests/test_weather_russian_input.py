"""Russian-only weather input is an MCP boundary, not city intent parsing."""

import jsonschema
import pytest
from pydantic import ValidationError
from test_weather import FakeResponse, FakeSession, location, session_for, weather

from app.mcp.contracts import WeatherArguments, tool_schemas
from app.mcp.server import TOOLS
from app.mcp.weather import GEOCODING_URL, WEATHER_URL, WeatherService


@pytest.mark.parametrize(
    "city",
    [
        "Москва",
        " Орёл ",
        "Ростов-на-Дону, Россия",
        "Санкт‑Петербург, Россия",
        "Н’Джамена, Чад",
        "Сент-Джонс, Антигуа и Барбуда",
        "г. Казань, Республика Татарстан",
    ],
)
def test_russian_city_names_and_explicit_qualifiers_satisfy_discovered_contract(city):
    model = WeatherArguments(city=city)
    assert model.city == city.strip()
    jsonschema.validate(model.model_dump(mode="json"), tool_schemas("get_weather")["inputSchema"])


@pytest.mark.parametrize(
    "city",
    [
        "Moscow",
        "Paris, France",
        "Москва, Russia",
        "Москва, RU",
        "Мocква",
        "Київ",
        "Париж 2",
        "Санкт-Петербург, MS",
        "https://example.org/Москва",
        "Москва\nРоссия",
        "Москва\tРоссия",
        "Москва,",
        "Москва,,Россия",
        "Москва, ...",
        "",
    ],
)
async def test_non_russian_or_malformed_input_is_rejected_before_weather_http(city):
    with pytest.raises(ValidationError):
        WeatherArguments(city=city)
    session = FakeSession()

    result = await WeatherService(session).get_weather(city)

    assert result.root.status == "error"
    assert result.root.error.code == "INVALID_ARGUMENTS"
    assert "русском" in result.root.error.message
    assert session.calls == []


def test_discovered_schema_and_description_require_russian_before_call():
    schema = tool_schemas("get_weather")["inputSchema"]
    description = schema["properties"]["city"]["description"]
    assert "на русском" in description and "Переведи" in description
    assert "Не угадывай" in description
    assert "переведи" in TOOLS["get_weather"][2]
    for city in ("Moscow", "Москва, RU", "Мocква", "Париж 2", "Москва,", "Москва, ..."):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"city": city}, schema)


async def test_foreign_place_translated_to_russian_uses_only_russian_geocoding():
    session = session_for(
        [
            location(
                name="Париж",
                country="Франция",
                country_code="FR",
                admin1="Иль-де-Франс",
                timezone="Europe/Paris",
            )
        ],
        weather(timezone="Europe/Paris"),
    )

    result = await WeatherService(session).get_weather("Париж, Франция")

    assert result.root.status == "ok"
    assert result.root.place.name == "Париж" and result.root.place.country == "Франция"
    assert session.calls[0] == (
        GEOCODING_URL,
        {
            "params": {"name": "Париж", "count": 100, "language": "ru", "format": "json"},
            "allow_redirects": False,
            "timeout": session.calls[0][1]["timeout"],
        },
    )
    assert [url for url, _ in session.calls] == [GEOCODING_URL, WEATHER_URL]


async def test_russian_input_keeps_real_homonyms_without_population_preference():
    session = session_for(
        [
            location(id=1, name="Ростов", country="Украина", country_code="UA", population=12),
            location(id=2, name="Ростов-на-Дону", population=1_000_000),
            location(id=3, name="Ростов Великий", population=30_000),
        ]
    )

    result = await WeatherService(session).get_weather("Ростов")

    assert result.root.status == "ambiguous"
    assert [place.name for place in result.root.candidates] == [
        "Ростов",
        "Ростов-на-Дону",
        "Ростов Великий",
    ]
    assert len(session.calls) == 1


async def test_missing_country_is_localized_for_russian_disambiguation():
    session = session_for(
        [
            location(
                name="Санкт-Петербург",
                country="",
                country_code="MS",
                admin1="Parish of Saint Peter",
                timezone="America/Montserrat",
            )
        ],
        weather(timezone="America/Montserrat"),
    )

    result = await WeatherService(session).get_weather("Санкт-Петербург, Монтсеррат")

    assert result.root.status == "ok"
    assert result.root.place.country == "Монтсеррат"
    assert session.calls[0][1]["params"]["language"] == "ru"


async def test_unknown_russian_place_is_not_found_without_guessing_another_city():
    session = FakeSession(FakeResponse({"generationtime_ms": 0.1}))

    result = await WeatherService(session).get_weather("Несуществующий город")

    assert result.root.status == "not_found"
    assert result.root.query == "Несуществующий город"
    assert len(session.calls) == 1
