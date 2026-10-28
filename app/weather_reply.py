"""Keep observed weather facts exact; this never interprets a user's intent.

The LLM still produces a real tool-disabled final. Unsupported numeric/date
formatting, missing facts or truncation select a complete deterministic card
from the validated WeatherOk instead of another LLM request. This is a bounded
fact/format check, not a proof of all possible natural-language assertions.
"""

import re
from dataclasses import dataclass
from decimal import Decimal
from zoneinfo import ZoneInfo

from app.formatting import format_reply
from app.mcp.contracts import WeatherOk
from app.time_display import format_date, format_datetime, format_utc_offset

_NUMBER = r"(?<![\d.,])([+\-−]?\d+(?:[.,]\d+)?)"
_TEMPERATURE = re.compile(_NUMBER + r"\s*°\s*[cс]\b", re.IGNORECASE)
_WIND = re.compile(_NUMBER + r"\s*(?:км\s*/\s*ч|km\s*/\s*h)\b", re.IGNORECASE)
_UNCLEAR_SIGN = re.compile(r"[+\-−]\s+\d|\b(?:минус|minus|negative)\b", re.IGNORECASE)
_MARKDOWN = re.compile(r"([\\`*_{}\[\]()#+.!|><~\-])")


@dataclass(frozen=True)
class WeatherReply:
    text: str
    replaced: bool
    truncated: bool = False


def _literal(value: str) -> str:
    return _MARKDOWN.sub(r"\\\1", " ".join(value.split()))


def _number(value: float) -> str:
    # Decimal(str(...)) preserves the finite MCP value; :g would round it.
    value = format(Decimal(str(value)), "f")
    return value.rstrip("0").rstrip(".") if "." in value else value


def canonical_weather(observation: WeatherOk) -> str:
    place = observation.place
    label = ", ".join(dict.fromkeys(p for p in (place.name, place.country, place.region) if p))
    return (
        f"Место: {_literal(label)}.\n"
        f"Температура: {_number(observation.temperature_c)} °C. "
        f"{_literal(observation.condition.description)}.\n"
        f"Ветер: {_number(observation.wind.speed)} км/ч.\n\n"
        f"Наблюдение: {format_datetime(observation.observed_at, observation.timezone)}.\n"
        "Источник: Open-Meteo."
    )


def _values(pattern, text):
    return [Decimal(value.replace(",", ".").replace("−", "-")) for value in pattern.findall(text)]


def checked_weather_reply(text: str, observation: WeatherOk, *, truncated=False) -> WeatherReply:
    if not isinstance(observation, WeatherOk):
        raise TypeError("A validated WeatherOk is required.")
    valid = False
    if isinstance(text, str) and text.strip() and not truncated:
        try:
            plain = "".join(part.text for part in format_reply(text))
            moment = observation.observed_at.astimezone(ZoneInfo(observation.timezone))
            lowered = plain.casefold()
            region = " ".join(observation.place.region.split()).casefold()
            dates = (
                format_date(moment.date()),
                moment.date().isoformat(),
                moment.strftime("%d.%m.%Y"),
            )
            clock = format_datetime(moment).split(", ", 1)[1].split(" ", 1)[0]
            valid = (
                _UNCLEAR_SIGN.search(plain) is None
                and _values(_TEMPERATURE, plain) == [Decimal(str(observation.temperature_c))]
                and _values(_WIND, plain) == [Decimal(str(observation.wind.speed))]
                and any(day.casefold() in lowered for day in dates)
                and clock in plain
                and format_utc_offset(moment).casefold() in lowered
                and "open-meteo" in lowered
                and observation.place.name.casefold() in lowered
                and observation.place.country.casefold() in lowered
                and (not region or region in " ".join(lowered.split()))
                and observation.condition.description.casefold() in lowered
            )
        except (ValueError, TypeError, ArithmeticError):
            valid = False
    if valid:
        return WeatherReply(text, replaced=False)
    # This card contains the complete observed fields. A length-limited model
    # response therefore does not produce a false "continue" prompt afterward.
    return WeatherReply(canonical_weather(observation), replaced=True, truncated=False)
