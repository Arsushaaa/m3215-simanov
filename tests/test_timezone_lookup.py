"""Страны из tzdata и поиск городов: подменённый HTTP, без модели и реальной сети."""

import asyncio
import json

import aiohttp
import pytest

import app.timezone_lookup as lookup_module
from app.timezone_lookup import (
    COUNTRY_LOCALES,
    GEOCODING_URL,
    MAX_HTTP_BYTES,
    MAX_OPTIONS,
    TimezoneLookup,
    TimezoneLookupError,
    TimezoneOption,
)


def location(**changes):
    return {
        "id": 498817,
        "name": "Санкт-Петербург",
        "country": "Россия",
        "country_code": "RU",
        "admin1": "Санкт-Петербург",
        "timezone": "Europe/Moscow",
        **changes,
    }


class FakeResponse:
    def __init__(self, value=None, *, raw=None, status=200, delay=0, content_length="auto"):
        self.raw = raw if raw is not None else json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.status = status
        self.delay = delay
        self.content_length = len(self.raw) if content_length == "auto" else content_length
        self.content = self

    async def __aenter__(self):
        await asyncio.sleep(self.delay)
        return self

    async def __aexit__(self, *args):
        return False

    async def iter_chunked(self, size):
        for start in range(0, len(self.raw), size):
            yield self.raw[start : start + size]


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


@pytest.mark.parametrize(
    "query",
    [
        "RU",
        "ru",
        "Russia",
        "  RUSSIA  ",
        "Россия",
        "РФ",
        "Russland",
        "Russie",
        "ロシア",
        "روسيا",
        "俄罗斯",
        "俄羅斯",
    ],
)
async def test_country_uses_packaged_zone_table_and_never_network(query):
    session = FakeSession()
    result = await TimezoneLookup(session).search(query)
    assert {"Europe/Moscow", "Asia/Vladivostok", "Asia/Kamchatka"} <= {
        option.timezone for option in result
    }
    assert len({option.timezone for option in result}) == len(result)
    assert len(result) <= MAX_OPTIONS
    assert next(option.label for option in result if option.timezone == "Europe/Moscow") == (
        "Россия — Москва"
    )
    assert session.calls == []


@pytest.mark.parametrize("query,zone", [("GB", "Europe/London"), ("Japan", "Asia/Tokyo")])
async def test_country_with_one_zone_is_still_a_suggestion(query, zone):
    result = await TimezoneLookup(FakeSession()).search(query)
    assert len(result) == 1
    assert result[0].timezone == zone


@pytest.mark.parametrize("query", ["США", "USA", "United States of America"])
async def test_common_country_aliases_are_explicit_and_do_not_choose_capital(query):
    result = await TimezoneLookup(FakeSession()).search(query)
    assert {"America/New_York", "America/Los_Angeles", "Pacific/Honolulu"} <= {
        option.timezone for option in result
    }


def test_country_locales_are_explicit_bounded_and_include_major_scripts():
    assert len(COUNTRY_LOCALES) == len(set(COUNTRY_LOCALES)) == 43
    assert {"ru", "en", "de", "ja", "ar", "zh_Hans", "zh_Hant"} <= set(COUNTRY_LOCALES)


async def test_ambiguous_translated_country_name_keeps_all_countries(monkeypatch):
    monkeypatch.setattr(
        lookup_module,
        "_countries",
        lambda: (
            {"RU": "Россия", "JP": "Япония"},
            {"неоднозначно": ("RU", "JP")},
            {"RU": ("Europe/Moscow",), "JP": ("Asia/Tokyo",)},
        ),
    )
    result = await TimezoneLookup(FakeSession()).search("Неоднозначно")
    assert {option.timezone for option in result} == {"Europe/Moscow", "Asia/Tokyo"}
    assert result[0].label.startswith("Россия")
    assert result[1].label.startswith("Япония")


async def test_country_index_is_cached_across_searches(monkeypatch):
    lookup_module._countries.cache_clear()
    calls = []
    original = lookup_module.files

    def resource_files(package):
        calls.append(package)
        return original(package)

    monkeypatch.setattr(lookup_module, "files", resource_files)
    service = TimezoneLookup(FakeSession())
    await service.search("Россия")
    await service.search("Deutschland")
    assert calls == ["tzdata"]


@pytest.mark.parametrize("zone", ["Europe/Moscow", "UTC", "GMT", "CET", "EET"])
async def test_exact_iana_needs_no_network(zone):
    session = FakeSession()
    assert await TimezoneLookup(session).search(zone) == [TimezoneOption(zone, zone)]
    assert session.calls == []


@pytest.mark.parametrize(
    "query,language", [("Санкт-Петербург", "ru"), ("Saint Petersburg", "en"), ("東京", "en")]
)
async def test_city_search_uses_only_fixed_api_and_bounded_request(query, language):
    session = FakeSession(FakeResponse({"results": [location()]}))
    result = await TimezoneLookup(session).search(query)
    assert result == [TimezoneOption("Санкт-Петербург — Россия, Санкт-Петербург", "Europe/Moscow")]
    assert len(session.calls) == 1
    url, options = session.calls[0]
    assert url == GEOCODING_URL
    assert options["params"] == {
        "name": query,
        "count": MAX_OPTIONS,
        "language": language,
        "format": "json",
    }
    assert options["allow_redirects"] is False
    assert options["timeout"].total == 15


@pytest.mark.parametrize(
    "query", ["Санкт-Петербург, RU", "Санкт-Петербург, Russia", "Санкт-Петербург, Россия"]
)
async def test_city_country_qualifier_uses_iso_filter_and_checks_it_locally(query):
    session = FakeSession(
        FakeResponse(
            {
                "results": [
                    location(),
                    location(id=2, country="Другая страна", country_code="GE"),
                ]
            }
        )
    )
    result = await TimezoneLookup(session).search(query)
    assert len(result) == 1
    assert result[0].timezone == "Europe/Moscow"
    assert session.calls[0][1]["params"]["name"] == "Санкт-Петербург"
    assert session.calls[0][1]["params"]["countryCode"] == "RU"


async def test_ambiguous_country_qualifier_filters_all_matching_countries(monkeypatch):
    monkeypatch.setattr(
        lookup_module,
        "_countries",
        lambda: ({}, {"неоднозначно": ("RU", "JP")}, {}),
    )
    session = FakeSession(
        FakeResponse(
            {
                "results": [
                    location(),
                    location(id=2, country="Япония", country_code="JP", timezone="Asia/Tokyo"),
                    location(id=3, country="Германия", country_code="DE", timezone="Europe/Berlin"),
                ]
            }
        )
    )
    result = await TimezoneLookup(session).search("Город, Неоднозначно")
    assert [option.timezone for option in result] == ["Europe/Moscow", "Asia/Tokyo"]
    assert session.calls[0][1]["params"]["name"] == "Город"
    assert "countryCode" not in session.calls[0][1]["params"]


async def test_unknown_region_qualifier_is_preserved_for_upstream_exact_matching():
    session = FakeSession(FakeResponse({"results": []}))
    assert await TimezoneLookup(session).search("Paris, Texas") == []
    assert session.calls[0][1]["params"]["name"] == "Paris, Texas"


async def test_city_ambiguity_keeps_every_different_place_without_population_choice():
    session = FakeSession(
        FakeResponse(
            {
                "results": [
                    location(population=999999999),
                    location(
                        id=3578038,
                        country="",
                        country_code="MS",
                        admin1="Parish of Saint Peter",
                        timezone="America/Montserrat",
                        population=1,
                    ),
                ]
            }
        )
    )
    result = await TimezoneLookup(session).search("Санкт-Петербург")
    assert [option.timezone for option in result] == ["Europe/Moscow", "America/Montserrat"]
    assert result[1].label == "Санкт-Петербург — MS, Parish of Saint Peter"


async def test_empty_optional_country_field_falls_back_to_source_iso_code():
    place = location()
    del place["country"]
    result = await TimezoneLookup(FakeSession(FakeResponse({"results": [place]}))).search("Казань")
    assert " — RU," in result[0].label


async def test_exact_duplicate_place_and_zone_options_are_not_repeated():
    places = [location(), location(), location(id=2)]
    result = await TimezoneLookup(FakeSession(FakeResponse({"results": places}))).search("Казань")
    assert len(result) == 1


async def test_output_is_capped_at_fifty_distinct_options():
    places = [location(id=index + 1, name=f"Город {index}") for index in range(100)]
    result = await TimezoneLookup(FakeSession(FakeResponse({"results": places}))).search("Город")
    assert len(result) == MAX_OPTIONS


@pytest.mark.parametrize("value", [{"results": []}, {"generationtime_ms": 0.2}])
async def test_not_found_is_distinct_from_upstream_unavailable(value):
    assert await TimezoneLookup(FakeSession(FakeResponse(value))).search("Неизвестный") == []


@pytest.mark.parametrize(
    "query",
    [
        "",
        " ",
        "Я",
        "a" * 151,
        "Город\nДругой",
        "Город\u0000",
        "Город\u202e",
        "Город,",
        "Город, страна, регион",
        "not/a-zone",
        "posix/Europe/Moscow",
        "right/Europe/Moscow",
        "https://example.com",
        None,
        123,
    ],
)
async def test_invalid_query_never_leaks_query_and_makes_no_request(query):
    session = FakeSession()
    with pytest.raises(TimezoneLookupError) as error:
        await TimezoneLookup(session).search(query)
    assert error.value.code == "invalid_query"
    assert "город" in error.value.safe_user_message or "IANA" in error.value.safe_user_message
    assert session.calls == []


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse({}, status=500),
        FakeResponse({}, status=429),
        FakeResponse({}, status=302),
        FakeResponse({"error": True, "reason": "sensitive query"}),
        FakeResponse({}),
        FakeResponse([]),
        FakeResponse({"results": "not a list"}),
        FakeResponse({"results": [location(timezone="Invalid/Zone")]}),
        FakeResponse({"results": [location(timezone="posix/Europe/Moscow")]}),
        FakeResponse({"results": [location(country_code="invalid")]}),
        FakeResponse({"results": [location(id=True)]}),
        FakeResponse({"results": [location(country=None)]}),
        FakeResponse({"results": [location(), location(timezone="Asia/Tokyo")]}),
        FakeResponse({"results": [location(id=i + 1) for i in range(101)]}),
        FakeResponse(raw=b"not JSON"),
        FakeResponse(raw=b"\xff"),
        FakeResponse(raw=b'{"results":[],"results":[]}'),
        FakeResponse(raw=b'{"generationtime_ms":NaN}'),
        FakeResponse(raw=b"x" * (MAX_HTTP_BYTES + 1)),
        FakeResponse(raw=b"x" * (MAX_HTTP_BYTES + 1), content_length=None),
        aiohttp.ClientConnectionError("sensitive query"),
    ],
)
async def test_upstream_failure_is_safe_and_does_not_retry(response):
    session = FakeSession(response)
    with pytest.raises(TimezoneLookupError) as error:
        await TimezoneLookup(session).search("Секретный город")
    assert error.value.code == "unavailable"
    assert str(error.value) == error.value.safe_user_message
    assert "Секретный" not in str(error.value)
    assert "sensitive" not in str(error.value)
    assert len(session.calls) == 1


async def test_timeout_does_not_retry():
    session = FakeSession(FakeResponse({"results": []}, delay=0.1))
    with pytest.raises(TimezoneLookupError) as error:
        await TimezoneLookup(session, timeout_seconds=0.01).search("Казань")
    assert error.value.code == "unavailable"
    assert len(session.calls) == 1


async def test_cancellation_propagates():
    session = FakeSession(asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await TimezoneLookup(session).search("Казань")
    assert len(session.calls) == 1


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf"), "15"])
def test_timeout_must_be_positive_finite_number(timeout):
    with pytest.raises(ValueError):
        TimezoneLookup(FakeSession(), timeout_seconds=timeout)
