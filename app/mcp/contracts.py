"""Строгие контракты MCP и решений агента без зависимости от SDK и транспорта."""

import re
from datetime import UTC, date, datetime
from typing import Annotated, Literal
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    RootModel,
    StringConstraints,
    model_validator,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _date(value: object) -> date:
    if type(value) is date:
        result = value
    elif isinstance(value, str) and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        result = date.fromisoformat(value)
    else:
        raise ValueError("Нужна календарная дата YYYY-MM-DD.")
    if not 2000 <= result.year <= 2100:
        raise ValueError("Дата должна быть в диапазоне 2000–2100 годов.")
    return result


def _aware_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str) and re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T(?:[01][0-9]|2[0-3]):[0-5][0-9]"
        r"(?::[0-5][0-9](?:\.[0-9]{1,6})?)?(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])",
        value,
    ):
        result = datetime.fromisoformat(value)
    else:
        raise ValueError("Нужны ISO дата и время с T и явным смещением UTC.")
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("Дата и время должны содержать смещение UTC.")
    return result


def _timezone(value: str) -> str:
    if value in {"localtime", "posixrules"} or value.startswith(("posix/", "right/")):
        raise ValueError("Нужно переносимое IANA-имя, а не системный часовой пояс.")
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError("Нужно существующее IANA-имя часового пояса.") from None
    return value


IsoDate = Annotated[
    date,
    BeforeValidator(_date),
    Field(description="Календарная дата YYYY-MM-DD, 2000–2100 годы."),
]
AwareDateTime = Annotated[
    datetime,
    BeforeValidator(_aware_datetime),
    Field(description="ISO 8601 дата и время с явным смещением UTC."),
]
TimezoneName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=100),
    AfterValidator(_timezone),
]
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
Reason = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)]
AnswerText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=16000)
]
ReminderText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)
]
ClockTime = Annotated[str, StringConstraints(pattern=r"^(?:[01][0-9]|2[0-3]):[0-5][0-9]$")]
ErrorCode = Literal[
    "INVALID_ARGUMENTS",
    "UNKNOWN_TOOL",
    "UNAUTHORIZED",
    "CONFIRMATION_REQUIRED",
    "ACTION_CANCELLED",
    "ACTION_EXPIRED",
    "ACTION_CONFLICT",
    "TIMEZONE_REQUIRED",
    "AMBIGUOUS_TIME",
    "MCP_UNAVAILABLE",
    "TIMEOUT",
    "UPSTREAM_ERROR",
    "INVALID_RESULT",
    "SOURCE_UNAVAILABLE",
    "CALL_LIMIT",
    "OUTCOME_UNKNOWN",
]


class WeatherArguments(StrictModel):
    city: ShortText = Field(description="Название города; при уточнении также страна и регион.")


class ScheduleArguments(StrictModel):
    date: IsoDate


class ReminderArguments(StrictModel):
    text: ReminderText
    remind_at: AwareDateTime


class ErrorDetails(StrictModel):
    code: ErrorCode
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
    retryable: bool


class ErrorResult(StrictModel):
    status: Literal["error"]
    error: ErrorDetails


class Place(StrictModel):
    name: ShortText
    country: ShortText
    region: Annotated[str, StringConstraints(strip_whitespace=True, max_length=100)]


class WeatherCondition(StrictModel):
    code: int
    description: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)
    ]


class Wind(StrictModel):
    speed: float = Field(ge=0, allow_inf_nan=False, description="Скорость ветра в км/ч.")
    unit: Literal["km/h"]


class WeatherOk(StrictModel):
    status: Literal["ok"]
    place: Place
    temperature_c: float = Field(allow_inf_nan=False, description="Температура воздуха, °C.")
    condition: WeatherCondition
    wind: Wind
    observed_at: AwareDateTime
    timezone: TimezoneName
    source: Literal["Open-Meteo"]

    @model_validator(mode="after")
    def matching_timezone(self) -> "WeatherOk":
        local = self.observed_at.astimezone(ZoneInfo(self.timezone))
        if local.utcoffset() != self.observed_at.utcoffset():
            raise ValueError("Смещение наблюдения не соответствует его IANA-зоне.")
        return self


class WeatherNotFound(StrictModel):
    status: Literal["not_found"]
    query: ShortText
    source: Literal["Open-Meteo"]


class WeatherAmbiguous(StrictModel):
    status: Literal["ambiguous"]
    query: ShortText
    candidates: Annotated[list[Place], Field(min_length=2, max_length=5)]
    source: Literal["Open-Meteo"]

    @model_validator(mode="after")
    def distinct_candidates(self) -> "WeatherAmbiguous":
        keys = {
            (p.name.casefold(), p.country.casefold(), p.region.casefold()) for p in self.candidates
        }
        if len(keys) != len(self.candidates):
            raise ValueError("Варианты города должны различаться страной или регионом.")
        return self


class WeatherResult(
    RootModel[
        Annotated[
            WeatherOk | WeatherNotFound | WeatherAmbiguous | ErrorResult,
            Field(discriminator="status"),
        ]
    ]
):
    model_config = ConfigDict(strict=True, frozen=True, json_schema_extra={"type": "object"})


class Lesson(StrictModel):
    start: ClockTime
    end: ClockTime
    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
    location: ShortText

    @model_validator(mode="after")
    def chronological(self) -> "Lesson":
        if self.start >= self.end:
            raise ValueError("Начало занятия должно быть раньше конца в пределах дня.")
        return self


def _ordered_lessons(lessons: list[Lesson]) -> None:
    if lessons != sorted(lessons, key=lambda lesson: lesson.start):
        raise ValueError("Занятия должны быть упорядочены по времени начала.")


Lessons = Annotated[list[Lesson], Field(max_length=32)]


class ScheduleOk(StrictModel):
    status: Literal["ok"]
    date: IsoDate
    timezone: TimezoneName
    lessons: Lessons
    source: Literal["local_schedule"]

    @model_validator(mode="after")
    def ordered(self) -> "ScheduleOk":
        _ordered_lessons(self.lessons)
        return self


class ScheduleOutsideCoverage(StrictModel):
    status: Literal["outside_coverage"]
    date: IsoDate
    coverage_start: IsoDate
    coverage_end: IsoDate
    timezone: TimezoneName
    source: Literal["local_schedule"]

    @model_validator(mode="after")
    def outside(self) -> "ScheduleOutsideCoverage":
        if self.coverage_start > self.coverage_end:
            raise ValueError("Начало покрытия должно быть не позже конца.")
        if self.coverage_start <= self.date <= self.coverage_end:
            raise ValueError("Дата относится к покрытию; нужен результат ok.")
        return self


class ScheduleResult(
    RootModel[
        Annotated[ScheduleOk | ScheduleOutsideCoverage | ErrorResult, Field(discriminator="status")]
    ]
):
    model_config = ConfigDict(strict=True, frozen=True, json_schema_extra={"type": "object"})


class WeekDayOk(StrictModel):
    status: Literal["ok"]
    date: IsoDate
    lessons: Lessons

    @model_validator(mode="after")
    def ordered(self) -> "WeekDayOk":
        _ordered_lessons(self.lessons)
        return self


class WeekDayOutsideCoverage(StrictModel):
    status: Literal["outside_coverage"]
    date: IsoDate


WeekDay = Annotated[WeekDayOk | WeekDayOutsideCoverage, Field(discriminator="status")]


class WeekSchedule(StrictModel):
    week_start: IsoDate
    week_end: IsoDate
    timezone: TimezoneName
    source: Literal["local_schedule"]
    days: Annotated[list[WeekDay], Field(min_length=7, max_length=7)]

    @model_validator(mode="after")
    def monday_to_sunday(self) -> "WeekSchedule":
        if self.week_start.weekday() != 0 or (self.week_end - self.week_start).days != 6:
            raise ValueError("Неделя должна длиться с понедельника до воскресенья.")
        if [day.date.toordinal() for day in self.days] != list(
            range(self.week_start.toordinal(), self.week_end.toordinal() + 1)
        ):
            raise ValueError("Неделя должна содержать семь последовательных дат.")
        return self


class WeekResult(RootModel[WeekSchedule | ErrorResult]):
    model_config = ConfigDict(strict=True, frozen=True, json_schema_extra={"type": "object"})


class ReminderOk(StrictModel):
    status: Literal["ok"]
    reminder_id: UUID
    remind_at: AwareDateTime
    timezone: TimezoneName
    created: bool

    @model_validator(mode="before")
    @classmethod
    def parse_uuid(cls, value: object) -> object:
        # UUID-строка — транспортный тип, но число/bytes не преобразуются в UUID.
        if isinstance(value, dict) and isinstance(value.get("reminder_id"), str):
            value = {**value, "reminder_id": UUID(value["reminder_id"])}
        return value

    @model_validator(mode="after")
    def matching_timezone(self) -> "ReminderOk":
        normalized = self.remind_at.astimezone(ZoneInfo(self.timezone))
        if normalized.utcoffset() != self.remind_at.utcoffset():
            raise ValueError("Смещение напоминания не соответствует его IANA-зоне.")
        return self


class ReminderResult(RootModel[Annotated[ReminderOk | ErrorResult, Field(discriminator="status")]]):
    model_config = ConfigDict(strict=True, frozen=True, json_schema_extra={"type": "object"})


class RespondDecision(StrictModel):
    action: Literal["respond"]
    text: AnswerText
    reason: Reason = Field(description="Краткое объяснение выбранного действия, без рассуждений.")


class ClarifyDecision(StrictModel):
    action: Literal["clarify"]
    question: AnswerText
    reason: Reason


class WeatherDecision(StrictModel):
    action: Literal["call_tool"]
    name: Literal["get_weather"]
    arguments: WeatherArguments
    reason: Reason


class ScheduleDecision(StrictModel):
    action: Literal["call_tool"]
    name: Literal["get_schedule"]
    arguments: ScheduleArguments
    reason: Reason


class ReminderDecision(StrictModel):
    action: Literal["call_tool"]
    name: Literal["add_reminder"]
    arguments: ReminderArguments
    reason: Reason


class ReminderRoutingDecision(
    RootModel[Annotated[ReminderDecision | ClarifyDecision, Field(discriminator="action")]]
):
    """Подготовка напоминания не допускает свободного сообщения об успехе."""

    model_config = ConfigDict(strict=True, frozen=True, json_schema_extra={"type": "object"})


class ResourceDecision(StrictModel):
    action: Literal["read_resource"]
    uri: Literal["schedule://current-week"]
    reason: Reason


ToolDecision = Annotated[
    WeatherDecision | ScheduleDecision | ReminderDecision, Field(discriminator="name")
]


class AgentDecision(
    RootModel[
        Annotated[
            RespondDecision | ClarifyDecision | ToolDecision | ResourceDecision,
            Field(discriminator="action"),
        ]
    ]
):
    model_config = ConfigDict(strict=True, frozen=True, json_schema_extra={"type": "object"})


def tool_schemas(name: str) -> dict[str, dict]:
    """Канонические схемы для будущего MCP-сервера, без копий в обработчиках."""
    models = {
        "get_weather": (WeatherArguments, WeatherResult),
        "get_schedule": (ScheduleArguments, ScheduleResult),
        "add_reminder": (ReminderArguments, ReminderResult),
    }
    try:
        arguments, result = models[name]
    except KeyError:
        raise ValueError("Неизвестный инструмент.") from None
    return {
        "inputSchema": arguments.model_json_schema(),
        "outputSchema": result.model_json_schema(),
    }


class ReminderTimeError(ValueError):
    """Безопасная причина отказа, пригодная для будущего слоя подтверждений."""

    def __init__(self, code: ErrorCode, message: str):
        self.code = code
        super().__init__(message)


def validate_local_datetime(local: datetime, timezone: str | None) -> datetime:
    """Привязать локальное время к IANA-зоне, отклонив DST gap и overlap."""
    if timezone is None:
        raise ReminderTimeError("TIMEZONE_REQUIRED", "Сначала задай часовой пояс через /timezone.")
    if local.tzinfo is not None:
        raise ReminderTimeError("INVALID_ARGUMENTS", "Ожидалось локальное время без зоны.")
    try:
        zone = ZoneInfo(_timezone(timezone))
    except ValueError:
        raise ReminderTimeError(
            "INVALID_ARGUMENTS", "Неизвестное IANA-имя часового пояса."
        ) from None
    candidates = {}
    for fold in (0, 1):
        candidate = local.replace(tzinfo=zone, fold=fold)
        round_trip = candidate.astimezone(UTC).astimezone(zone)
        if round_trip.replace(tzinfo=None) == local:
            candidates.setdefault(candidate.utcoffset(), candidate)
    if len(candidates) != 1:
        raise ReminderTimeError(
            "AMBIGUOUS_TIME", "Это локальное время неоднозначно или не существует. Уточни время."
        )
    return next(iter(candidates.values()))


def validate_reminder_time(
    arguments: ReminderArguments, *, now: datetime, timezone: str | None
) -> datetime:
    """Проверить будущее время и совпадение offset с явно сохранённой IANA-зоной.

    Проверка относится только к новой записи; чтение квитанции committed-действия
    должно происходить раньше и не зависит от того, наступил ли момент напоминания.
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("Часы приложения должны возвращать aware datetime.")
    value = arguments.remind_at
    normalized = validate_local_datetime(value.replace(tzinfo=None), timezone)
    if value.utcoffset() != normalized.utcoffset():
        raise ReminderTimeError(
            "INVALID_ARGUMENTS", "Смещение времени не соответствует выбранной зоне."
        )
    if normalized.astimezone(UTC) <= now.astimezone(UTC):
        raise ReminderTimeError("INVALID_ARGUMENTS", "Время напоминания должно быть в будущем.")
    return normalized
