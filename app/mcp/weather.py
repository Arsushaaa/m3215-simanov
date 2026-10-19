"""Текущая погода из фиксированных Open-Meteo API, без выбора города по населению.

После названия города через запятые допускаются точные названия страны/регионов
из ответа геокодера либо ISO country_code. Регистр, диакритика и разделители
нормализуются; неизвестные переводы/сокращения не угадываются. Для русских
уточнений запрашиваются русские названия, для остальных — английские.
"""

import asyncio
import json
import logging
import math
import re
import unicodedata
from datetime import UTC, datetime
from typing import Annotated, Literal
from zoneinfo import ZoneInfo

import aiohttp
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from app.mcp.contracts import (
    ErrorCode,
    ErrorDetails,
    ErrorResult,
    Place,
    ShortText,
    TimezoneName,
    WeatherAmbiguous,
    WeatherArguments,
    WeatherCondition,
    WeatherNotFound,
    WeatherOk,
    WeatherResult,
    Wind,
)

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
MAX_HTTP_BYTES = 256 * 1024
logger = logging.getLogger(__name__)

# https://open-meteo.com/en/docs#weather_variable_documentation
WMO_CONDITIONS = {
    0: "Ясно",
    1: "Преимущественно ясно",
    2: "Переменная облачность",
    3: "Пасмурно",
    45: "Туман",
    48: "Туман с изморозью",
    51: "Слабая морось",
    53: "Умеренная морось",
    55: "Сильная морось",
    56: "Слабая переохлаждённая морось",
    57: "Сильная переохлаждённая морось",
    61: "Слабый дождь",
    63: "Умеренный дождь",
    65: "Сильный дождь",
    66: "Слабый переохлаждённый дождь",
    67: "Сильный переохлаждённый дождь",
    71: "Слабый снег",
    73: "Умеренный снег",
    75: "Сильный снег",
    77: "Снежные зёрна",
    80: "Слабый ливень",
    81: "Умеренный ливень",
    82: "Сильный ливень",
    85: "Слабый снежный ливень",
    86: "Сильный снежный ливень",
    95: "Гроза",
    96: "Гроза со слабым градом",
    97: "Сильная гроза",
    99: "Гроза с сильным градом",
}


class _ApiModel(BaseModel):
    # API может добавлять поля; известные поля проверяются без coercion.
    model_config = ConfigDict(extra="ignore", strict=True, frozen=True)


class _Location(_ApiModel):
    id: int = Field(gt=0)
    name: ShortText
    # Open-Meteo опускает пустые поля, в том числе название страны у территорий.
    # Проверенный ISO-код остаётся явным обозначением, без догадки о стране.
    country: Annotated[str, StringConstraints(strip_whitespace=True, max_length=100)] = ""
    country_code: Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")]
    admin1: Annotated[str, StringConstraints(strip_whitespace=True, max_length=100)] = ""
    admin2: Annotated[str, StringConstraints(strip_whitespace=True, max_length=100)] = ""
    admin3: Annotated[str, StringConstraints(strip_whitespace=True, max_length=100)] = ""
    admin4: Annotated[str, StringConstraints(strip_whitespace=True, max_length=100)] = ""
    latitude: float = Field(ge=-90, le=90, allow_inf_nan=False)
    longitude: float = Field(ge=-180, le=180, allow_inf_nan=False)
    timezone: TimezoneName

    def place(self) -> Place:
        regions = dict.fromkeys(
            p for p in (self.admin1, self.admin2, self.admin3, self.admin4) if p
        )
        return Place(
            name=self.name, country=self.country or self.country_code, region=", ".join(regions)
        )


class _Geocoding(_ApiModel):
    results: Annotated[list[_Location], Field(max_length=100)] = Field(default_factory=list)
    error: Literal[False] = False


class _Current(_ApiModel):
    # Защитные границы земной погоды: выброс источника не становится фактом ответа.
    temperature_2m: float = Field(ge=-100, le=100, allow_inf_nan=False)
    wind_speed_10m: float = Field(ge=0, le=500, allow_inf_nan=False)
    weather_code: int
    time: int = Field(ge=0, le=4102444800)


class _Units(_ApiModel):
    time: Literal["unixtime"]
    temperature_2m: Literal["°C"]
    wind_speed_10m: Literal["km/h"]
    weather_code: Literal["wmo code"]


class _Weather(_ApiModel):
    timezone: TimezoneName
    current: _Current
    current_units: _Units
    error: Literal[False] = False


class _FetchError(Exception):
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        retryable: bool = False,
        http_status: int | None = None,
    ):
        self.code, self.message, self.retryable = code, message, retryable
        self.http_status = http_status
        super().__init__(message)


def _log_failure(phase: str, code: ErrorCode, error: Exception) -> None:
    # Fixed fields only: URLs, HTTP bodies and exception messages contain input.
    logger.warning(
        "weather phase=%s code=%s http_status=%s error_type=%s",
        phase,
        code,
        error.http_status if isinstance(error, _FetchError) else None,
        type(error).__name__,
    )


def _normalized(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold().replace("ё", "е"))
    value = "".join(character for character in value if not unicodedata.combining(character))
    return " ".join(re.sub(r"[-‐‑‒–—]", " ", value).split())


def _locations(value: dict, name: str, qualifiers: list[str]) -> list[_Location]:
    if "results" not in value and "generationtime_ms" not in value:
        raise ValueError("Нет результата геокодирования.")
    locations = _Geocoding.model_validate(value).results
    exact = [location for location in locations if _normalized(location.name) == _normalized(name)]
    if exact:
        locations = exact
    selected = []
    seen = {}
    for location in locations:
        accepted = {
            _normalized(part)
            for part in (
                location.country,
                location.country_code,
                location.admin1,
                location.admin2,
                location.admin3,
                location.admin4,
            )
            if part
        }
        if not all(_normalized(qualifier) in accepted for qualifier in qualifiers):
            continue
        if location.id in seen:
            if seen[location.id] != location:
                raise ValueError("Геокодер вернул конфликтующие места.")
            continue
        seen[location.id] = location
        selected.append(location)
    return selected


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Повтор поля JSON.")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("Неконечное число JSON.")


class WeatherService:
    def __init__(self, session: aiohttp.ClientSession, timeout_seconds: float = 15):
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("Тайм-аут погоды должен быть положительным конечным числом.")
        self.session = session
        self.timeout_seconds = timeout_seconds

    async def _json(self, url: str, params: dict) -> dict:
        async with self.session.get(
            url,
            params=params,
            allow_redirects=False,
            timeout=aiohttp.ClientTimeout(total=self.timeout_seconds, sock_connect=5),
        ) as response:
            if response.status != 200:
                raise _FetchError(
                    "UPSTREAM_ERROR",
                    "Погодный сервис сейчас недоступен.",
                    retryable=response.status == 429 or response.status >= 500,
                    http_status=response.status,
                )
            if response.content_length is not None and response.content_length > MAX_HTTP_BYTES:
                raise _FetchError("INVALID_RESULT", "Погодный сервис вернул слишком большой ответ.")
            raw = bytearray()
            async for chunk in response.content.iter_chunked(16384):
                raw.extend(chunk)
                if len(raw) > MAX_HTTP_BYTES:
                    raise _FetchError(
                        "INVALID_RESULT", "Погодный сервис вернул слишком большой ответ."
                    )
            value = json.loads(
                raw.decode("utf-8"), object_pairs_hook=_unique_keys, parse_constant=_reject_constant
            )
            if not isinstance(value, dict):
                raise ValueError("Ожидался JSON-объект.")
            return value

    async def get_weather(self, city: str) -> WeatherResult:
        try:
            query = WeatherArguments(city=city).city
            parts = [part.strip() for part in query.split(",")]
            if not all(parts):
                raise ValueError("Пустая часть места.")
        except (ValidationError, ValueError):
            return self._error(
                "INVALID_ARGUMENTS", "Укажи город, при необходимости страну и регион."
            )
        phase = "geocoding"
        try:
            async with asyncio.timeout(self.timeout_seconds):
                name, qualifiers = parts[0], parts[1:]
                language = "ru" if re.search("[А-Яа-яЁё]", " ".join(qualifiers) or name) else "en"
                data = await self._json(
                    GEOCODING_URL,
                    {"name": name, "count": 100, "language": language, "format": "json"},
                )
                locations = _locations(data, name, qualifiers)
                if not locations:
                    return WeatherResult(
                        WeatherNotFound(status="not_found", query=query, source="Open-Meteo")
                    )
                if len(locations) > 1:
                    # Разные GeoNames ID могут иметь одинаковые доступные подписи.
                    # Объединяем только подписи для уточнения, а не сами места:
                    # выбор такого региона всё ещё обязан дать ровно одну локацию.
                    candidates = {}
                    for location in locations:
                        place = location.place()
                        key = (
                            place.name.casefold(),
                            place.country.casefold(),
                            place.region.casefold(),
                        )
                        candidates.setdefault(key, place)
                    if len(candidates) == 1:
                        return self._error(
                            "AMBIGUOUS_PLACE",
                            "Найдено несколько мест с одинаковыми названием, страной и регионом. "
                            "Геокодер не сообщает различающие районы. Уточни место; "
                            "выбрать точные координаты по этим данным нельзя.",
                        )
                    return WeatherResult(
                        WeatherAmbiguous(
                            status="ambiguous",
                            query=query,
                            candidates=list(candidates.values())[:5],
                            source="Open-Meteo",
                        )
                    )
                location = locations[0]
                phase = "current"
                data = await self._json(
                    WEATHER_URL,
                    {
                        "latitude": location.latitude,
                        "longitude": location.longitude,
                        "current": "temperature_2m,weather_code,wind_speed_10m",
                        "temperature_unit": "celsius",
                        "wind_speed_unit": "kmh",
                        "timeformat": "unixtime",
                        "timezone": location.timezone,
                        "forecast_days": 1,
                    },
                )
                weather = _Weather.model_validate(data)
                if weather.timezone != location.timezone:
                    raise ValueError("Зона ответа не соответствует запрошенной.")
                current = weather.current
                description = WMO_CONDITIONS[current.weather_code]
                observed_at = datetime.fromtimestamp(current.time, tz=UTC).astimezone(
                    ZoneInfo(weather.timezone)
                )
                return WeatherResult(
                    WeatherOk(
                        status="ok",
                        place=location.place(),
                        temperature_c=current.temperature_2m,
                        condition=WeatherCondition(
                            code=current.weather_code, description=description
                        ),
                        wind=Wind(speed=current.wind_speed_10m, unit="km/h"),
                        observed_at=observed_at,
                        timezone=weather.timezone,
                        source="Open-Meteo",
                    )
                )
        except TimeoutError as error:
            _log_failure(phase, "TIMEOUT", error)
            return self._error("TIMEOUT", "Погодный сервис не ответил вовремя.", retryable=True)
        except aiohttp.ClientError as error:
            _log_failure(phase, "UPSTREAM_ERROR", error)
            return self._error(
                "UPSTREAM_ERROR", "Не удалось связаться с погодным сервисом.", retryable=True
            )
        except _FetchError as error:
            _log_failure(phase, error.code, error)
            return self._error(error.code, error.message, retryable=error.retryable)
        except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as error:
            _log_failure(phase, "INVALID_RESULT", error)
            return self._error("INVALID_RESULT", "Погодный сервис вернул некорректные данные.")

    @staticmethod
    def _error(code: ErrorCode, message: str, *, retryable: bool = False) -> WeatherResult:
        return WeatherResult(
            ErrorResult(
                status="error", error=ErrorDetails(code=code, message=message, retryable=retryable)
            )
        )
