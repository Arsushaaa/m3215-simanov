"""Поиск IANA-зон по стране или городу без LLM и догадок о местоположении.

Список стран и зон берётся из установленного tzdata; города — из фиксированного
Open-Meteo Geocoding API. Результат остаётся предложением для явного выбора.
"""

import asyncio
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import aiohttp
from babel import Locale
from babel.dates import get_timezone_location
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
MAX_HTTP_BYTES = 256 * 1024
MAX_OPTIONS = 50
MAX_QUERY_LENGTH = 150
# Названия стран — официальные переводы CLDR, без машинного перевода и догадок.
# Ограниченный набор не загружает сотни полных локалей Babel в память процесса.
COUNTRY_LOCALES = (
    "ru en de fr es pt it uk be pl cs sk bg sr hr ro el tr fi sv no da nl et lv lt hu "
    "ar he fa hi ja ko zh_Hans zh_Hant id vi th ka hy az kk uz"
).split()


@dataclass(frozen=True)
class TimezoneOption:
    label: str
    timezone: str


class TimezoneLookupError(Exception):
    """Безопасная ошибка: текст не содержит запрос пользователя или ответ HTTP."""

    def __init__(self, code: Literal["invalid_query", "unavailable"], safe_user_message: str):
        self.code = code
        self.safe_user_message = safe_user_message
        super().__init__(safe_user_message)


def _normalized(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold().replace("ё", "е"))
    return " ".join(
        "".join(character for character in value if not unicodedata.combining(character))
        .replace("-", " ")
        .split()
    )


def _valid_zone(value: str) -> bool:
    if value in {"localtime", "posixrules"} or value.startswith(("posix/", "right/")):
        return False
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError):
        return False
    return True


@lru_cache(maxsize=1)
def _countries() -> tuple[dict[str, str], dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    root = files("tzdata").joinpath("zoneinfo")
    names: dict[str, str] = {}
    aliases: dict[str, set[str]] = {}
    zones: dict[str, list[str]] = {}
    for line in root.joinpath("iso3166.tab").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        code, name = line.split("\t", 1)
        names[code] = name
        aliases.setdefault(_normalized(code), set()).add(code)
        aliases.setdefault(_normalized(name), set()).add(code)
    for language in COUNTRY_LOCALES:
        for code, name in Locale.parse(language).territories.items():
            if code in names:
                aliases.setdefault(_normalized(name), set()).add(code)
    for name, code in {
        "РФ": "RU",
        "Российская Федерация": "RU",
        "Russian Federation": "RU",
        "США": "US",
        "USA": "US",
        "United States of America": "US",
        "UK": "GB",
    }.items():
        aliases.setdefault(_normalized(name), set()).add(code)
    russian = Locale("ru").territories
    names = {code: russian.get(code, name) for code, name in names.items()}
    for line in root.joinpath("zone.tab").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        code, _coordinates, zone, *_comment = line.split("\t")
        if _valid_zone(zone):
            zones.setdefault(code, []).append(zone)
    return (
        names,
        {name: tuple(sorted(codes)) for name, codes in aliases.items()},
        {code: tuple(dict.fromkeys(values)) for code, values in zones.items()},
    )


class _ApiModel(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, frozen=True)


_Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
_OptionalText = Annotated[str, StringConstraints(strip_whitespace=True, max_length=100)]


class _Location(_ApiModel):
    id: int = Field(gt=0)
    name: _Text
    country_code: Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")]
    country: _OptionalText = ""
    admin1: _OptionalText = ""
    admin2: _OptionalText = ""
    timezone: _Text


class _Geocoding(_ApiModel):
    results: Annotated[list[_Location], Field(max_length=100)] = Field(default_factory=list)
    error: Literal[False] = False


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Повтор поля JSON.")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ValueError("Неконечное число JSON.")


class TimezoneLookup:
    """Вызывающий код владеет session и закрывает её при остановке приложения."""

    def __init__(self, session: aiohttp.ClientSession, timeout_seconds: float = 15):
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("Тайм-аут поиска должен быть положительным конечным числом.")
        self.session = session
        self.timeout_seconds = timeout_seconds

    async def search(self, query: str) -> list[TimezoneOption]:
        if (
            not isinstance(query, str)
            or not 2 <= len(query.strip()) <= MAX_QUERY_LENGTH
            or any(unicodedata.category(character).startswith("C") for character in query)
        ):
            raise TimezoneLookupError(
                "invalid_query", "Напиши название города или страны, например Санкт-Петербург."
            )
        query = query.strip()
        if "/" in query:
            if _valid_zone(query):
                return [TimezoneOption(query, query)]
            raise TimezoneLookupError(
                "invalid_query", "Такой IANA-зоны нет. Укажи город или страну."
            )
        names, aliases, zones = _countries()
        countries = aliases.get(_normalized(query))
        if countries is not None:
            options = []
            for country in countries:
                for zone in zones.get(country, ()):
                    city = get_timezone_location(ZoneInfo(zone), locale="ru", return_city=True)
                    options.append(TimezoneOption(f"{names[country]} — {city}", zone))
            return options[:MAX_OPTIONS]
        # Названия Japan/Turkey/Iceland бывают также IANA-ссылками. Страна выше
        # сохраняет приоритет; прочие короткие зоны всё равно явно выбираются в UI.
        if _valid_zone(query):
            return [TimezoneOption(query, query)]
        params: dict[str, str | int] = {
            "name": query,
            "count": MAX_OPTIONS,
            "language": "ru" if re.search("[А-Яа-яЁё]", query) else "en",
            "format": "json",
        }
        parts = [part.strip() for part in query.split(",")]
        if not all(parts) or len(parts) > 2:
            raise TimezoneLookupError(
                "invalid_query", "Укажи город, при необходимости через запятую страну или регион."
            )
        country_filter = aliases.get(_normalized(parts[1])) if len(parts) == 2 else None
        if country_filter:
            params["name"] = parts[0]
            if len(country_filter) == 1:
                params["countryCode"] = country_filter[0]
        try:
            async with asyncio.timeout(self.timeout_seconds):
                value = await self._json(params)
            if "results" not in value and "generationtime_ms" not in value:
                raise ValueError("Нет результата геокодирования.")
            locations = _Geocoding.model_validate(value).results
            options = []
            seen: set[tuple[str, str]] = set()
            seen_ids: dict[int, _Location] = {}
            for location in locations:
                if not _valid_zone(location.timezone):
                    raise ValueError("Неизвестная зона геокодера.")
                if location.id in seen_ids and seen_ids[location.id] != location:
                    raise ValueError("Конфликт геокодера.")
                seen_ids[location.id] = location
                # countryCode — фильтр API и дополнительная локальная проверка.
                if country_filter and location.country_code not in country_filter:
                    continue
                region = ", ".join(
                    dict.fromkeys(p for p in (location.admin1, location.admin2) if p)
                )
                label = f"{location.name} — {location.country or location.country_code}"
                if region:
                    label += f", {region}"
                key = label, location.timezone
                if key not in seen:
                    seen.add(key)
                    options.append(TimezoneOption(label, location.timezone))
            return options[:MAX_OPTIONS]
        except (aiohttp.ClientError, TimeoutError, UnicodeError, ValueError, ValidationError):
            raise TimezoneLookupError(
                "unavailable",
                "Поиск городов сейчас недоступен. Попробуй позже или укажи IANA-зону.",
            ) from None

    async def _json(self, params: dict[str, str | int]) -> dict:
        async with self.session.get(
            GEOCODING_URL,
            params=params,
            allow_redirects=False,
            timeout=aiohttp.ClientTimeout(total=self.timeout_seconds, sock_connect=5),
        ) as response:
            if response.status != 200:
                raise ValueError("Сервис поиска недоступен.")
            if response.content_length is not None and response.content_length > MAX_HTTP_BYTES:
                raise ValueError("Слишком большой ответ геокодера.")
            raw = bytearray()
            async for chunk in response.content.iter_chunked(16384):
                raw.extend(chunk)
                if len(raw) > MAX_HTTP_BYTES:
                    raise ValueError("Слишком большой ответ геокодера.")
            value = json.loads(
                raw.decode("utf-8"), object_pairs_hook=_unique_keys, parse_constant=_reject_constant
            )
            if not isinstance(value, dict):
                raise ValueError("Ожидался JSON-объект.")
            return value
