"""Локальные границы просьбы о напоминании; ответы модели не задают его параметры."""

import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

TIME_REQUIRED = "Во сколько напомнить? Напиши точное время, например 18:30."
DATE_REQUIRED = "На какую дату подготовить напоминание? Например, завтра или 25 октября."
REMINDER_CLARIFY = (
    "Не удалось подготовить точную карточку. Укажи дату, точное время и текст напоминания. "
    "Напоминание ещё не сохранено."
)
BUTTON_REQUIRED = (
    "Для сохранения нажми кнопку «Подтвердить» под карточкой. "
    "Текстовое «да» не подтверждает запись. Напоминание ещё не сохранено."
)

_QUOTES = re.compile(r'```[\s\S]*?```|`[^`\n]*`|«[^»]*»|"[^"\n]*"|“[^”]*”')
_INTRO = r"(?:(?:пожалуйста|привет|здравствуй(?:те)?)[,!\s]+)*"
_CAN = (
    r"(?:(?:(?:ты|вы)\s+)?(?:не\s+)?"
    r"(?:можешь|можете|сможешь|сможете|мог(?:ла)?\s+бы)\s+(?:мне\s+)?)?"
)
_REMIND = re.compile(_INTRO + _CAN + r"напом(?:ни(?:те)?|нить)\b", re.I)
_CREATE = re.compile(
    _INTRO + _CAN + r"(?:создай|создать|добавь|добавить|поставь|поставить|сохрани|сохранить|"
    r"установи|установить|сделай|сделать|запланируй|запланировать)\s+"
    r"(?:мне\s+)?напоминание\b",
    re.I,
)
_WANT = re.compile(
    _INTRO + r"(?:(?:я\s+)?хочу|мне\s+(?:нужно|надо))\s+"
    r"(?:(?:создать|добавить|поставить|сохранить)\s+)?напоминание\b",
    re.I,
)
_THEORY = re.compile(
    r"^[\s,:]*(?:(?:мне|нам|пожалуйста)[\s,:]+)*"
    r"(?:что\s+такое\b|как\b|чем\b|почему\b|разниц[ау]\b|определение\b|синтаксис\b)",
    re.I,
)
_PROGRAMMING = re.compile(r"\b(?:python|питон|java|javascript|asyncio|sql)\b", re.I)
_ISO = re.compile(
    r"(?<!\d)\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?"
    r"(?:Z|[+-]\d{2}:\d{2})?(?![\d:])",
    re.I,
)
_CLOCK = re.compile(
    r"(?<![\w:])(?:[вк]\s*)?(\d{1,2}):([0-5]\d)"
    r"(?::([0-5]\d)(?:[.,](\d{1,6}))?)?(?!\d|:\d)",
    re.I,
)
_HOUR = re.compile(r"\b[вк]\s*(\d{1,2})(?![\d:.])(?:\s*час(?:а|ов)?)?\b", re.I)
_REL_DATE = re.compile(r"\b(послезавтра|завтра|сегодня)\b", re.I)
_ISO_DATE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_MONTHS = {
    name: number
    for number, name in enumerate(
        "января февраля марта апреля мая июня июля августа сентября октября ноября декабря".split(),
        1,
    )
}
_MONTH_DATE = re.compile(r"(?<!\d)(\d{1,2})\s+(" + "|".join(_MONTHS) + r")(?:\s+(\d{4}))?\b", re.I)
_NUMBERS = {
    "один": 1,
    "одну": 1,
    "два": 2,
    "две": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
    "десять": 10,
    "полтора": 1.5,
    "полторы": 1.5,
}
_NUMBER = r"(?:\d{1,6}(?!\d)|" + "|".join(_NUMBERS) + r")"
_UNIT = r"(?:секунд(?:у|ы)?|минут(?:у|ы)?|час(?:а|ов)?|день|дня|дней)"
_DURATION = re.compile(r"\bчерез\s+(?:(" + _NUMBER + r")\s+)?(" + _UNIT + r")\b", re.I)
_HALF_HOUR = re.compile(r"\bчерез\s+полчаса\b", re.I)
_FOLLOWUP_WORDS = (
    set(
        "сегодня завтра послезавтра утром днём днем вечером ночью после обеда занятий "
        "в на к через час часа часов минуту минуты минут секунду секунды секунд день дня дней "
        "лучше давай тогда нет не а пусть будет пожалуйста восемнадцать девятнадцать "
        "двадцать тридцать полчаса понедельник вторник среду четверг пятницу субботу "
        "воскресенье".split()
    )
    | set(_NUMBERS)
    | set(_MONTHS)
)


def _clean(text: str) -> str:
    return _QUOTES.sub(" ", text).strip().casefold()


def is_confirmation_reply(text: str) -> bool:
    return bool(re.fullmatch(r"\s*(?:да|давай|ага|ок|окей|okay|ok|yes)[\s.!]*", text, re.I))


def _followup(text: str) -> bool:
    cleaned = _clean(text)
    if len(cleaned) > 160 or not cleaned:
        return False
    if is_confirmation_reply(cleaned):
        return True
    # A temporal answer may be unsupported ("вечером"); it must still stay a clarification.
    words = re.findall(r"[а-яёa-z]+", _ISO.sub(" ", cleaned))
    return bool(words or re.search(r"\d", cleaned)) and all(
        word in _FOLLOWUP_WORDS for word in words
    )


def _request(text: str) -> bool:
    cleaned = _clean(text)
    match = _REMIND.match(cleaned) or _CREATE.match(cleaned) or _WANT.match(cleaned)
    if match is None:
        verb = re.search(r"\bнапом(?:ни(?:те)?|нить)\b", cleaned)
        if verb and _followup(cleaned[: verb.start()]):
            match = verb
    if match is None:
        return False
    tail = cleaned[match.end() :]
    if _THEORY.match(tail):
        return False
    if _PROGRAMMING.search(tail) and not any(
        pattern.search(cleaned)
        for pattern in (_REL_DATE, _ISO_DATE, _MONTH_DATE, _DURATION, _CLOCK)
    ):
        return False
    return True


def _reminder_response(text: str) -> bool:
    if text in {TIME_REQUIRED, DATE_REQUIRED, REMINDER_CLARIFY, BUTTON_REQUIRED}:
        return True
    cleaned = _clean(text)
    if re.search(r"\bнапомн\w*", cleaned):
        return not _PROGRAMMING.search(cleaned) or bool(
            _CLOCK.search(cleaned) or _REL_DATE.search(cleaned)
        )
    return bool(
        re.search(
            r"^во\s+сколько(?:\s+завтра|\s+сегодня)?[?.!]*$|"
            r"^в\s+какое\s+время[?.!]*$|/timezone\b",
            cleaned,
        )
    )


def requests_reminder_and_read(text: str) -> bool:
    """Две явные просьбы, связанные союзом; задача «посмотреть погоду» не является чтением."""
    cleaned = _clean(text)
    if not re.search(r"\b(?:напом(?:ни(?:те)?|нить)|напоминание|remind\s+me)\b", cleaned):
        return False
    if not re.search(r"\b(?:и|а\s+также|также|and)\b", cleaned):
        return False
    return bool(
        re.search(
            r"\b(?:какая\s+(?:сейчас\s+)?погода|сколько\s+(?:сейчас\s+)?градусов|"
            r"(?:покажи|узнай|проверь|дай)\s+(?:(?:мне|текущее|текущую)\s+)*"
            r"(?:погоду|расписание)|какие\s+(?:у\s+меня\s+)?(?:занятия|пары)|"
            r"(?:show|check|get)\s+(?:me\s+)?(?:the\s+)?(?:weather|schedule)|"
            r"what(?:'s|\s+is)\s+the\s+weather)\b",
            cleaned,
        )
    )


@dataclass(frozen=True)
class ReminderContext:
    prompt_text: str = field(repr=False)
    _clock: time | None = None
    _date: date | tuple[int, int] | int | None = None
    _instant: datetime | None = None
    _duration: timedelta | None = None
    is_orphaned: bool = False

    @property
    def has_time(self) -> bool:
        return self._clock is not None or self._instant is not None or self._duration is not None

    @property
    def has_date(self) -> bool:
        return self._date is not None or self._instant is not None or self._duration is not None

    def matches(self, moment: datetime, now: datetime, timezone: str) -> bool:
        if not self.has_time or not self.has_date:
            return False
        try:
            if any(value.tzinfo is None or value.utcoffset() is None for value in (moment, now)):
                return False
            zone = ZoneInfo(timezone)
            if self._instant is not None:
                return moment.astimezone(UTC) == self._instant.astimezone(UTC)
            if self._duration is not None:
                return moment.astimezone(UTC) == now.astimezone(UTC) + self._duration
            local = moment.astimezone(zone)
            today = now.astimezone(zone).date()
            if isinstance(self._date, int):
                expected = today + timedelta(days=self._date)
            elif isinstance(self._date, tuple):
                expected = date(today.year, *self._date)
            else:
                expected = self._date
            return local.date() == expected and local.timetz().replace(tzinfo=None) == self._clock
        except (ValueError, OverflowError, ZoneInfoNotFoundError):
            return False


def _evidence(texts: list[str]) -> ReminderContext:
    clock, day, instant, duration = None, None, None, None
    for text in texts:
        cleaned = _clean(text)
        iso = list(_ISO.finditer(cleaned))
        if iso:
            clock, day, instant, duration = None, None, None, None
            if len(iso) == 1:
                try:
                    value = datetime.fromisoformat(iso[0][0].replace("z", "Z"))
                    if value.tzinfo is not None:
                        instant = value
                    else:
                        day, clock = value.date(), value.time()
                except ValueError:
                    pass
            # ISO is one exact parameter, not separate dates and clocks found inside it.
            if len(iso) != 1 or _ISO.sub(" ", cleaned).strip(" ,.!:;") == "":
                continue
            cleaned = _ISO.sub(" ", cleaned)
        periods = list(_DURATION.finditer(cleaned))
        halves = list(_HALF_HOUR.finditer(cleaned))
        if periods or halves:
            clock, day, instant, duration = None, None, None, None
            if len(periods) + len(halves) == 1:
                match = (periods or halves)[0]
                remainder = cleaned[match.end() :]
                extra_unit = re.match(
                    r"[\s,]+(?:и\s+)?(?:полчаса|\w+\s+" + _UNIT + r")\b", remainder
                )
                outside = cleaned[: match.start()] + remainder
                has_absolute = any(
                    pattern.search(outside)
                    for pattern in (_REL_DATE, _ISO_DATE, _MONTH_DATE, _CLOCK, _HOUR)
                )
                if not extra_unit and not has_absolute:
                    if halves:
                        duration = timedelta(minutes=30)
                    else:
                        number = match[1]
                        amount = _NUMBERS.get(number, 0) if number else 1
                        if number and number.isdecimal():
                            amount = int(number)
                        unit = match[2]
                        seconds = 1 if unit.startswith("секунд") else 60
                        if unit.startswith("час"):
                            seconds = 3600
                        elif unit in {"день", "дня", "дней"}:
                            seconds = 86400
                        try:
                            duration = timedelta(seconds=amount * seconds) if amount > 0 else None
                        except OverflowError:
                            pass
            continue
        clocks = list(_CLOCK.finditer(cleaned))
        without_clocks = _CLOCK.sub(" ", cleaned)
        hours = list(_HOUR.finditer(without_clocks))
        if (
            not clocks
            and not hours
            and re.search(
                r"\b(?:утром|днём|днем|вечером|ночью|после\s+обеда)\b|"
                r"\b[вк]\s+[а-яё]+\b",
                cleaned,
            )
        ):
            clock, instant, duration = None, None, None
        if clocks or hours:
            instant, duration, clock = None, None, None
            if len(clocks) + len(hours) == 1 and not re.search(
                r"\b(?:не\s+[вк]|или|либо)\b", cleaned
            ):
                match = (clocks or hours)[0]
                try:
                    clock = (
                        time(
                            int(match[1]),
                            int(match[2]),
                            int(match[3] or 0),
                            int((match[4] or "").ljust(6, "0")),
                        )
                        if clocks
                        else time(int(match[1]))
                    )
                except ValueError:
                    pass
        dates = []
        for match in _REL_DATE.finditer(cleaned):
            dates.append({"сегодня": 0, "завтра": 1, "послезавтра": 2}[match[1]])
        for match in _ISO_DATE.finditer(cleaned):
            try:
                dates.append(date(*map(int, match.groups())))
            except ValueError:
                dates.append(None)
        for match in _MONTH_DATE.finditer(cleaned):
            month, number = _MONTHS[match[2]], int(match[1])
            try:
                dates.append(date(int(match[3]), month, number) if match[3] else (month, number))
            except ValueError:
                dates.append(None)
        if re.search(
            r"\b(?:понедельник|вторник|среду|четверг|пятницу|субботу|воскресенье)\b", cleaned
        ):
            day, instant, duration = None, None, None
        if dates:
            instant, duration = None, None
            day = (
                dates[0]
                if len(dates) == 1 and not re.search(r"\bне\s+(?:завтра|сегодня)", cleaned)
                else None
            )
    return ReminderContext("\n".join(texts), clock, day, instant, duration)


def reminder_context(text: str, history: list[dict[str, str]]) -> ReminderContext | None:
    """Текущая просьба или до восьми прошлых пар с короткими пользовательскими уточнениями."""
    if _request(text):
        return _evidence([text])
    if not _followup(text):
        return None
    last_assistant = next(
        (item.get("content", "") for item in reversed(history) if item.get("role") == "assistant"),
        None,
    )
    if last_assistant is not None and not _reminder_response(last_assistant):
        return None
    texts = [text]
    for item in reversed(history[-16:]):
        if item.get("role") != "user":
            continue
        previous = item.get("content", "")
        if _request(previous):
            return _evidence([previous, *reversed(texts)])
        if not _followup(previous):
            break
        texts.append(previous)
    return ReminderContext(text, is_orphaned=True) if last_assistant is not None else None
