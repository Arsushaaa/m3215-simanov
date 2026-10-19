"""Выбор часового пояса: поиск не меняет настройку, сохранение явно подтверждается."""

import re
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

from pydantic import TypeAdapter, ValidationError

from app.mcp.contracts import TimezoneName
from app.time_display import format_datetime, format_utc_offset
from app.timezone_lookup import TimezoneLookupError, TimezoneOption

_ZONE = TypeAdapter(TimezoneName)
PAGE_SIZE = 6
MENU_TTL = 600
MAX_MENUS = 1000
EXPIRED = "Это меню уже закрыто или устарело. Открой /timezone заново."
_DEFAULTS = (
    TimezoneOption("Москва / Санкт-Петербург", "Europe/Moscow"),
    TimezoneOption("Калининград", "Europe/Kaliningrad"),
    TimezoneOption("Екатеринбург", "Asia/Yekaterinburg"),
    TimezoneOption("Новосибирск", "Asia/Novosibirsk"),
    TimezoneOption("Владивосток", "Asia/Vladivostok"),
    TimezoneOption("Всемирное время", "UTC"),
)


@dataclass(frozen=True)
class TimezoneButton:
    text: str
    callback_data: str = field(repr=False)


@dataclass(frozen=True)
class TimezoneView:
    text: str = field(repr=False)
    rows: tuple[tuple[TimezoneButton, ...], ...]
    dismiss_previous: bool = False


@dataclass
class _Menu:
    token: str = field(repr=False)
    expires: float
    options: tuple[TimezoneOption, ...] = ()
    searching: bool = False
    selected: int | None = None
    search_version: int = 0


class TimezoneUI:
    def __init__(self, confirmations, lookup, *, clock=time.monotonic):
        self.confirmations, self.lookup, self.clock = confirmations, lookup, clock
        self._menus: OrderedDict[int, _Menu] = OrderedDict()

    def clear(self, user_id: int) -> None:
        self._menus.pop(user_id, None)

    @staticmethod
    def direct_timezone(query: str | None) -> str | None:
        if query is None:
            return None
        # Japan, Turkey и другие имена одновременно бывают странами и старыми
        # tz-алиасами. Название страны должно вести к выбору, а не скрытой смене.
        if "/" not in query and query.strip() not in {"UTC", "GMT"}:
            return None
        try:
            return _ZONE.validate_python(query.strip())
        except ValidationError:
            return None

    def _current(self, user_id):
        menu = self._menus.get(user_id)
        if menu is not None and menu.expires <= self.clock():
            self.clear(user_id)
            return None
        return menu

    def _new(self, user_id, options=()):
        self.clear(user_id)
        for owner, menu in list(self._menus.items()):
            if menu.expires <= self.clock():
                self.clear(owner)
        while len(self._menus) >= MAX_MENUS:
            self._menus.popitem(last=False)
        menu = _Menu(secrets.token_urlsafe(12), self.clock() + MENU_TTL, tuple(options))
        self._menus[user_id] = menu
        return menu

    @staticmethod
    def _button(menu, operation, label, index=None):
        data = f"tz:{operation}:{menu.token}"
        if index is not None:
            data += f":{index}"
        return TimezoneButton(label, data)

    def _controls(self, menu):
        return (
            (self._button(menu, "search", "🔎 Найти город или страну"),),
            (self._button(menu, "cancel", "Закрыть"),),
        )

    async def _choices(self, user_id, menu, page=0, *, heading=None):
        now = await self.confirmations.repository.now()
        if self._current(user_id) is not menu:
            return EXPIRED
        start = page * PAGE_SIZE
        lines, rows = [heading or "Выбери город или часовой пояс:"], []
        for index, option in enumerate(menu.options[start : start + PAGE_SIZE], start):
            offset = format_utc_offset(now.astimezone(ZoneInfo(option.timezone)))
            label = option.label[:200]
            lines.append(f"\n{index + 1}. {label}\n{option.timezone} · {offset}")
            rows.append((self._button(menu, "pick", f"{index + 1}. {label[:48]}", index),))
        pages = (len(menu.options) + PAGE_SIZE - 1) // PAGE_SIZE
        if pages > 1:
            lines.append(f"\nСтраница {page + 1} из {pages}.")
            nav = []
            if page:
                nav.append(self._button(menu, "page", "← Назад", page - 1))
            if page + 1 < pages:
                nav.append(self._button(menu, "page", "Дальше →", page + 1))
            rows.append(tuple(nav))
        lines.append("\nПояс изменится только после кнопки «Использовать этот пояс».")
        return TimezoneView("\n".join(lines), tuple(rows) + self._controls(menu))

    async def open(self, user_id: int, query: str | None = None):
        if query is not None and query.strip():
            query = query.strip()
            zone = self.direct_timezone(query)
            if zone is None:
                return await self._search(user_id, query)
            self.clear(user_id)
            # Полная IANA-команда уже является явным выбором, как в методичке.
            return await self.confirmations.timezone(user_id, zone)
        menu = self._new(user_id, _DEFAULTS)
        zone = await self.confirmations.repository.timezone(user_id)
        heading = (
            await self.confirmations.timezone(user_id) if zone else "Часовой пояс ещё не выбран."
        )
        return await self._choices(user_id, menu, heading=heading + "\n\nВыбери пояс или поиск:")

    async def _search(self, user_id, query, *, menu=None):
        menu = menu or self._new(user_id)
        menu.searching = True
        menu.selected, menu.options = None, ()
        menu.search_version += 1
        version = menu.search_version
        try:
            options = await self.lookup.search(query)
        except TimezoneLookupError as error:
            if self._current(user_id) is not menu or menu.search_version != version:
                return EXPIRED
            return TimezoneView(error.safe_user_message, self._controls(menu))
        if self._current(user_id) is not menu or menu.search_version != version:
            return EXPIRED
        # Новые варианты получают новые callback: индекс из прежнего списка
        # никогда не должен означать уже другой город.
        menu.token = secrets.token_urlsafe(12)
        menu.options = tuple(options[:50])
        if not menu.options:
            return TimezoneView(
                "Ничего не найдено. Уточни город или страну, например «Санкт-Петербург, "
                "Россия». Можно также указать IANA-зону: /timezone Europe/Moscow.",
                self._controls(menu),
            )
        menu.searching = False
        return await self._choices(user_id, menu)

    def accepts_input(self, user_id: int) -> bool:
        menu = self._current(user_id)
        return menu is not None and menu.searching

    def is_apply(self, user_id: int, data: str) -> bool:
        if not isinstance(data, str) or len(data.encode("utf-8")) > 64:
            return False
        match = re.fullmatch(r"tz:use:([A-Za-z0-9_-]{16}):([0-9]{1,2})", data)
        menu = self._current(user_id)
        return (
            match is not None
            and menu is not None
            and match[1] == menu.token
            and menu.selected == int(match[2])
            and int(match[2]) < len(menu.options)
        )

    async def search_input(self, user_id: int, text: str):
        if not self.accepts_input(user_id):
            return None
        # «Закрыть» под приглашением остаётся действительным во время HTTP.
        return await self._search(user_id, text, menu=self._current(user_id))

    async def handle(self, user_id: int, data: str):
        if not isinstance(data, str) or len(data.encode("utf-8")) > 64:
            return EXPIRED
        match = re.fullmatch(
            r"tz:(pick|use|page|search|cancel):([A-Za-z0-9_-]{16})(?::([0-9]{1,2}))?", data
        )
        menu = self._current(user_id)
        if match is None or menu is None or match[2] != menu.token:
            return EXPIRED
        operation, raw_index = match[1], match[3]
        if operation in {"search", "cancel"}:
            if raw_index is not None:
                return EXPIRED
            if operation == "cancel":
                self.clear(user_id)
                return TimezoneView("Меню закрыто.", (), dismiss_previous=True)
            menu.searching, menu.selected = True, None
            return TimezoneView(
                "Отправь название города или страны — например «Санкт-Петербург», "
                "«Saint Petersburg» или «Россия». Для выхода нажми «Закрыть» или /stop.",
                ((self._button(menu, "cancel", "Закрыть"),),),
            )
        if raw_index is None:
            return EXPIRED
        index = int(raw_index)
        if operation == "page":
            if index >= (len(menu.options) + PAGE_SIZE - 1) // PAGE_SIZE:
                return EXPIRED
            menu.searching, menu.selected = False, None
            return await self._choices(user_id, menu, index)
        if index >= len(menu.options):
            return EXPIRED
        option = menu.options[index]
        if operation == "pick":
            menu.searching, menu.selected = False, index
            now = await self.confirmations.repository.now()
            if self._current(user_id) is not menu:
                return EXPIRED
            return TimezoneView(
                f"Выбранное место: {option.label}\nЧасовой пояс: {option.timezone}\n"
                f"Сейчас там: {format_datetime(now.replace(microsecond=0), option.timezone)}\n\n"
                "Использовать этот пояс для напоминаний? При смене пояса ожидающее "
                "подтверждение напоминания будет отменено.",
                (
                    (self._button(menu, "use", "Использовать этот пояс", index),),
                    (self._button(menu, "page", "← К вариантам", index // PAGE_SIZE),),
                    *self._controls(menu),
                ),
            )
        if menu.selected != index:
            return EXPIRED
        # Одно нажатие принимает выбранное предложение. Повтор не запускает
        # второй writer, даже если первый ещё ожидает PostgreSQL.
        menu.selected = None
        result = await self.confirmations.timezone(user_id, option.timezone)
        if self._current(user_id) is menu:
            self.clear(user_id)
        return TimezoneView(result, (), dismiss_previous=True)
