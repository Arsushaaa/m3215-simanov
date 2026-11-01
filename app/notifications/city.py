"""A bounded geocoder and persisted, owner-confirmed city menu. No LLM or population cutoff."""

import re

from app.mcp.weather import GEOCODING_URL, WeatherService, _locations
from app.notifications.models import CityCandidate
from app.timezone_lookup import _countries, _normalized
from app.timezone_ui import TimezoneButton, TimezoneView

EXPIRED = "Выбор города устарел или отменён. Открой /city <название> заново."
PAGE_SIZE = 6


def label(city):
    return ", ".join(p for p in (city.place.name, city.place.country, city.place.region) if p)


class CityLookup:
    def __init__(self, session):
        self.weather = WeatherService(session)

    async def search(self, query):
        import asyncio

        if (
            not isinstance(query, str)
            or not 2 <= len(query.strip()) <= 150
            or any(ord(c) < 32 for c in query)
        ):
            raise ValueError("Invalid city query")
        parts = [part.strip() for part in query.split(",")]
        if not all(parts) or len(parts) > 3 or any(x in query for x in ("://", "/", "\\")):
            raise ValueError("City name required")
        _, aliases, _ = _countries()
        qualifiers = []
        for part in parts[1:]:
            codes = aliases.get(_normalized(part))
            qualifiers.append(codes[0] if codes and len(codes) == 1 else part)
        async with asyncio.timeout(15):
            raw = await self.weather._json(
                GEOCODING_URL, {"name": parts[0], "count": 100, "language": "ru", "format": "json"}
            )
        locations = _locations(raw, parts[0], qualifiers)
        # Keep different GeoNames IDs; coordinates distinguish identical labels.
        return [
            CityCandidate(
                geonames_id=p.id,
                place=p.place(),
                latitude=p.latitude,
                longitude=p.longitude,
                timezone=p.timezone,
            )
            for p in locations[:50]
        ]


class CityUI:
    def __init__(self, repository, lookup):
        self.repository, self.lookup = repository, lookup

    @staticmethod
    def _button(operation, token, text, index=None):
        return TimezoneButton(
            text, f"city:{operation}:{token}" + (f":{index}" if index is not None else "")
        )

    async def choices(self, owner, token, page=0):
        menu = await self.repository.menu(owner, token)
        if menu is None:
            return EXPIRED
        options = menu["options"]
        if type(page) is not int or not 0 <= page <= 8 or page * PAGE_SIZE >= len(options):
            return EXPIRED
        rows = []
        lines = ["Выбери точный город. Поиск пока ничего не меняет:"]
        start = page * PAGE_SIZE
        for i, c in enumerate(options[start : start + PAGE_SIZE], start):
            lines.append(f"\n{i + 1}. {label(c)}\n{c.latitude}, {c.longitude} · {c.timezone}")
            rows.append((self._button("pick", token, f"{i + 1}. {label(c)[:48]}", i),))
        nav = []
        if page:
            nav.append(self._button("page", token, "← Назад", page - 1))
        if start + PAGE_SIZE < len(options):
            nav.append(self._button("page", token, "Дальше →", page + 1))
        if nav:
            rows.append(tuple(nav))
        rows.append((self._button("cancel", token, "Отменить"),))
        return TimezoneView("\n".join(lines), tuple(rows))

    async def open(self, owner, query=None):
        if query is None or not query.strip():
            await self.repository.clear_menu(owner)
            profile = await self.repository.profile(owner)
            current = (
                CityCandidate.model_validate_json(profile["city"])
                if profile is not None and profile["city"] is not None
                else None
            )
            return (
                f"Город: {label(current)}.\n" if current else "Город ещё не выбран.\n"
            ) + "Напиши /city Москва или /city Paris, France. Сохранение — только по кнопке."
        token, version = await self.repository.start_menu(owner)
        try:
            choices = await self.lookup.search(query)
        except Exception:
            # No provider bodies/queries/coordinates are written to logs.
            await self.repository.cancel_menu(owner, token)
            return "Не удалось найти город. Укажи город и страну: /city Санкт-Петербург, Россия."
        if not await self.repository.fill_menu(owner, token, version, choices):
            return EXPIRED
        if not choices:
            await self.repository.cancel_menu(owner, token)
            return "Город не найден. Уточни название, страну или регион: /city <название>."
        return await self.choices(owner, token)

    async def handle(self, owner, data):
        if not isinstance(data, str) or len(data.encode()) > 64:
            return EXPIRED
        match = re.fullmatch(
            r"city:(pick|use|page|cancel):([A-Za-z0-9_-]{22})(?::([0-9]{1,2}))?", data
        )
        if match is None:
            return EXPIRED
        operation, token, raw_index = match.groups()
        if (operation != "cancel") != (raw_index is not None):
            return EXPIRED
        index = int(raw_index) if raw_index is not None else None
        if operation == "cancel":
            if not await self.repository.cancel_menu(owner, token):
                return EXPIRED
            return TimezoneView(
                "Выбор города отменён. Настройки не изменены.", (), dismiss_previous=True
            )
        if operation == "page":
            return await self.choices(owner, token, index)
        if operation == "pick":
            city = await self.repository.choose(owner, token, index)
            if city is None:
                return EXPIRED
            return TimezoneView(
                f"Сохранить этот город?\n\n{label(city)}\n"
                f"Координаты: {city.latitude}, {city.longitude}\n"
                f"Часовой пояс города: {city.timezone}.\n"
                "Сводка использует твой пояс /timezone; он не изменится.\n"
                "Источник: Open-Meteo / GeoNames. Выбор действует 5 минут от начала поиска.",
                (
                    (self._button("use", token, "Сохранить город", index),),
                    (self._button("cancel", token, "Отменить"),),
                ),
            )
        city = await self.repository.confirm_city(owner, token, index)
        if city is None:
            return EXPIRED
        return TimezoneView(
            f"Город сохранён: {label(city)}.\nЧасовой пояс города: {city.timezone}.\n"
            "Твой пояс уведомлений /timezone не изменён.",
            (),
            dismiss_previous=True,
        )
