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
_EN_INTRO = r"(?:(?:please|hello|hi)[,!\s]+)*(?:(?:can|could|would|will)\s+you\s+)?"
_EN_REMIND = re.compile(_EN_INTRO + r"(?:please\s+)?remind\s+me\b", re.I)
_EN_CREATE = re.compile(
    _EN_INTRO + r"(?:please\s+)?(?:create|add|set|save|schedule)\s+"
    r"(?:me\s+)?(?:a\s+)?reminder\b",
    re.I,
)
_EN_WANT = re.compile(
    _EN_INTRO + r"(?:i\s+(?:want|need)(?:\s+to\s+(?:create|add|set|save))?)\s+"
    r"(?:a\s+)?reminder\b",
    re.I,
)
_THEORY = re.compile(
    r"^[\s,:]*(?:(?:мне|нам|пожалуйста|please)[\s,:]+)*"
    r"(?:что\s+такое\b|как\b|чем\b|почему\b|разниц[ау]\b|определение\b|синтаксис\b|"
    r"what\b|how\b|why\b|the\s+(?:definition|syntax|difference)\b)",
    re.I,
)
_PROGRAMMING = re.compile(r"\b(?:python|питон|java|javascript|asyncio|sql)\b", re.I)
_ISO = re.compile(
    r"(?<!\d)\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?"
    r"(?:Z|[+-]\d{2}:\d{2})?(?![\d:])",
    re.I,
)
_CLOCK = re.compile(
    r"(?<![\w:])(?:(?:[вк]|at)\s*)?(\d{1,2}):([0-5]\d)"
    r"(?::([0-5]\d)(?:[.,](\d{1,6}))?)?(?!\d|:\d)"
    r"(?:\s*([ap])\.?m\.?(?!\w))?",
    re.I,
)
_CLOCK_CANDIDATE = re.compile(r"(?<![\w:])\d{1,2}:\d{2}(?::\d{2})?(?!\d)")
_HOUR = re.compile(
    r"(?<![\w:])(?:(?:[вк]|at)\s*|(?=\d{1,2}\s*[ap]\.?m))"
    r"(\d{1,2})(?![\d:.])(?:\s*час(?:а|ов)?)?(?:\s*([ap])\.?m\.?)?(?!\w)",
    re.I,
)
_RELATIVE_DAYS = {
    "сегодня": 0,
    "завтра": 1,
    "послезавтра": 2,
    "today": 0,
    "tomorrow": 1,
    "day after tomorrow": 2,
    "the day after tomorrow": 2,
}
_REL_DATE = re.compile(
    r"\b(послезавтра|завтра|сегодня|(?:the\s+)?day\s+after\s+tomorrow|tomorrow|today)\b",
    re.I,
)
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
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "half": 0.5,
}
_NUMBER = r"(?:\d{1,6}(?!\d)|" + "|".join(_NUMBERS) + r")"
_UNIT = (
    r"(?:секунд(?:у|ы)?|минут(?:у|ы)?|час(?:а|ов)?|день|дня|дней|seconds?|minutes?|hours?|days?)"
)
_DURATION = re.compile(r"\b(?:через|in)\s+(?:(" + _NUMBER + r")\s+)?(" + _UNIT + r")\b", re.I)
_HALF_HOUR = re.compile(r"\b(?:через\s+полчаса|in\s+half\s+an\s+hour)\b", re.I)
_WEEKDAY = re.compile(
    r"\b(?:понедельник|вторник|среду|четверг|пятницу|субботу|воскресенье|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.I,
)
_FOLLOWUP_WORDS = (
    set(
        "сегодня завтра послезавтра утром днём днем вечером ночью после обеда занятий "
        "в на к через час часа часов минуту минуты минут секунду секунды секунд день дня дней "
        "лучше давай тогда да но только нет не а или либо пусть будет пожалуйста "
        "восемнадцать девятнадцать "
        "двадцать тридцать полчаса понедельник вторник среду четверг пятницу субботу "
        "воскресенье today tomorrow the day after morning afternoon evening night "
        "at in on am pm a m p hour hours minute minutes second seconds days "
        "instead then please no not and or next monday tuesday wednesday thursday "
        "friday saturday sunday eleven twelve thirteen fourteen fifteen sixteen "
        "seventeen eighteen nineteen twenty thirty".split()
    )
    | set(_NUMBERS)
    | set(_MONTHS)
)


def _clean(text: str) -> str:
    return _QUOTES.sub(" ", text).strip().casefold()


def is_confirmation_reply(text: str) -> bool:
    if re.fullmatch(r"\s*(?:да|давай|ага|ок|окей|okay|ok|yes)[\s.!]*", text, re.I):
        return True
    cleaned = _clean(text)
    # Compound pressure to bypass the button is still textual agreement, not
    # authorization. A date/time correction must instead replace the card.
    if len(cleaned) > 500 or any(
        pattern.search(cleaned)
        for pattern in (_ISO, _REL_DATE, _ISO_DATE, _MONTH_DATE, _DURATION, _CLOCK_CANDIDATE, _HOUR)
    ):
        return False
    return bool(
        re.match(
            r"^(?:да|ага|ок|окей|okay|ok|yes|подтверждаю|согласен|согласна|"
            r"я\s+(?:владелец|подтверждаю|согласен|согласна)|вызывай\s+add_reminder)\b",
            cleaned,
        )
        and re.search(
            r"\b(?:подтвержд\w*|соглас\w*|сохран\w*|запис\w*|владелец|кнопк\w*|"
            r"add_reminder|confirm\w*|save|button\w*|owner)\b",
            cleaned,
        )
    )


def is_reminder_edit(text: str) -> bool:
    """A new request or temporal correction may invalidate an existing card.

    This only identifies intent. An active, owner-bound card must still come
    from PostgreSQL; arbitrary assistant history never supplies edit parameters.
    """
    if is_confirmation_reply(text):
        return False
    if _request(text):
        return True
    cleaned = _clean(text)
    temporal = any(
        pattern.search(cleaned)
        for pattern in (
            _ISO,
            _REL_DATE,
            _ISO_DATE,
            _MONTH_DATE,
            _DURATION,
            _CLOCK_CANDIDATE,
            _HOUR,
            _WEEKDAY,
        )
    ) or re.search(
        r"\b(?:утром|днём|днем|вечером|ночью|morning|afternoon|evening|night)\b", cleaned
    )
    if _followup(text) and temporal:
        return True
    if (
        temporal
        and len(cleaned) <= 500
        and not _PROGRAMMING.search(cleaned)
        and re.match(r"^(?:нет[,\s]+давай\s+лучше|да[,\s]+но|лучше|вместо|instead)\b", cleaned)
    ):
        return True
    return bool(
        len(cleaned) <= 500
        and not _PROGRAMMING.search(cleaned)
        and re.search(r"\b(?:измени|поменяй|перенеси|перенести|change|move|reschedule)\b", cleaned)
        and re.search(
            r"\b(?:напоминани\w*|время|дату|текст|reminder|time|date|text)\b|\d{1,2}:\d{2}",
            cleaned,
        )
    )


def is_temporal_reminder_edit(text: str) -> bool:
    """Only a short temporal reply can reuse the trusted card's existing task."""
    return not is_confirmation_reply(text) and _followup(text) and is_reminder_edit(text)


def is_reminder_request(text: str) -> bool:
    return _request(text)


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
    match = next(
        (
            match
            for pattern in (_REMIND, _CREATE, _WANT, _EN_REMIND, _EN_CREATE, _EN_WANT)
            if (match := pattern.match(cleaned)) is not None
        ),
        None,
    )
    if match is None:
        verb = re.search(r"\b(?:напом(?:ни(?:те)?|нить)|remind\s+me)\b", cleaned)
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
    if re.search(r"\b(?:напомн\w*|remind(?:er|ing)?|remind\s+you)\b", cleaned):
        return not _PROGRAMMING.search(cleaned) or bool(
            _CLOCK.search(cleaned) or _REL_DATE.search(cleaned)
        )
    return bool(
        re.search(
            r"^во\s+сколько(?:\s+завтра|\s+сегодня)?[?.!]*$|"
            r"^в\s+какое\s+время[?.!]*$|^what\s+time[?.!]*$|/timezone\b",
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


def _hour(value: int, meridiem: str | None) -> int:
    if meridiem is None:
        return value
    if not 1 <= value <= 12:
        raise ValueError("Invalid 12-hour clock")
    return value % 12 + (12 if meridiem == "p" else 0)


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
                    r"[\s,]+(?:(?:и|and)\s+)?(?:полчаса|half\s+an\s+hour|\w+\s+" + _UNIT + r")\b",
                    remainder,
                )
                outside = cleaned[: match.start()] + remainder
                has_absolute = any(
                    pattern.search(outside)
                    for pattern in (_REL_DATE, _ISO_DATE, _MONTH_DATE, _CLOCK, _HOUR, _WEEKDAY)
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
                        seconds = 1 if unit.startswith(("секунд", "second")) else 60
                        if unit.startswith(("час", "hour")):
                            seconds = 3600
                        elif unit in {"день", "дня", "дней", "day", "days"}:
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
                r"\b[вк]\s+[а-яё]+\b|\b(?:morning|afternoon|evening|night)\b|"
                r"\bat\s+[a-z]+\b",
                cleaned,
            )
        ):
            clock, instant, duration = None, None, None
        if clocks or hours:
            instant, duration, clock = None, None, None
            if len(clocks) + len(hours) == 1 and not re.search(
                r"\b(?:не\s+[вк]|или|либо|not\s+at|or)\b", cleaned
            ):
                match = (clocks or hours)[0]
                try:
                    clock = (
                        time(
                            _hour(int(match[1]), match[5]),
                            int(match[2]),
                            int(match[3] or 0),
                            int((match[4] or "").ljust(6, "0")),
                        )
                        if clocks
                        else time(_hour(int(match[1]), match[2]))
                    )
                except ValueError:
                    pass
        dates = []
        for match in _REL_DATE.finditer(cleaned):
            dates.append(_RELATIVE_DAYS[" ".join(match[1].split())])
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
        if _WEEKDAY.search(cleaned):
            day, instant, duration = None, None, None
        if dates:
            instant, duration = None, None
            day = (
                dates[0]
                if len(dates) == 1
                and not re.search(
                    r"\b(?:не\s+(?:завтра|сегодня)|not\s+(?:tomorrow|today))", cleaned
                )
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


def edited_reminder_time(
    text: str, previous: datetime, *, timezone: str, now: datetime
) -> datetime | None:
    """Apply only explicit temporal evidence to a trusted, active DB card.

    Unknown/contradictory clocks do not retain the previous clock silently.
    The caller still validates DST, future time, epoch and owner transactionally.
    """
    if not is_temporal_reminder_edit(text):
        return None
    evidence = _evidence([text])
    try:
        if any(value.tzinfo is None or value.utcoffset() is None for value in (previous, now)):
            return None
        zone = ZoneInfo(timezone)
        local = previous.astimezone(zone)
        if evidence._instant is not None:
            return evidence._instant.astimezone(zone)
        if evidence._duration is not None:
            return (now.astimezone(UTC) + evidence._duration).astimezone(zone)
        cleaned = _clean(text)
        clock_requested = bool(
            _CLOCK_CANDIDATE.search(cleaned)
            or _HOUR.search(cleaned)
            or re.search(
                r"\b(?:утром|днём|днем|вечером|ночью|morning|afternoon|evening|night)\b", cleaned
            )
        )
        date_requested = any(
            pattern.search(cleaned) for pattern in (_REL_DATE, _ISO_DATE, _MONTH_DATE, _WEEKDAY)
        )
        if (clock_requested and evidence._clock is None) or (
            date_requested and evidence._date is None
        ):
            return None
        if not clock_requested and not date_requested:
            return None
        day = local.date()
        if isinstance(evidence._date, int):
            day = now.astimezone(zone).date() + timedelta(days=evidence._date)
        elif isinstance(evidence._date, tuple):
            day = date(now.astimezone(zone).year, *evidence._date)
        elif evidence._date is not None:
            day = evidence._date
        clock = evidence._clock if clock_requested else local.time()
        return datetime.combine(day, clock, zone)
    except (ValueError, OverflowError, ZoneInfoNotFoundError):
        return None
