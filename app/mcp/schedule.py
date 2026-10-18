"""Проверенный локальный источник расписания; без HTTP, MCP-транспорта и БД."""

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Annotated
from zoneinfo import ZoneInfo

from pydantic import Field, TypeAdapter, ValidationError, model_validator

from app.mcp.contracts import (
    IsoDate,
    Lesson,
    ScheduleArguments,
    ScheduleOk,
    ScheduleOutsideCoverage,
    ScheduleResult,
    StrictModel,
    TimezoneName,
    WeekDayOk,
    WeekDayOutsideCoverage,
    WeekResult,
    WeekSchedule,
)

MAX_SCHEDULE_BYTES = 1024 * 1024


class ScheduleSourceError(ValueError):
    """Безопасная ошибка источника, преобразуемая сервером в SOURCE_UNAVAILABLE."""


class ScheduleSourceDay(StrictModel):
    date: IsoDate
    lessons: Annotated[list[Lesson], Field(max_length=32)]

    @model_validator(mode="after")
    def ordered(self) -> "ScheduleSourceDay":
        if self.lessons != sorted(self.lessons, key=lambda lesson: lesson.start):
            raise ValueError("Занятия должны быть упорядочены по времени начала.")
        return self


class ScheduleData(StrictModel):
    timezone: TimezoneName
    coverage_start: IsoDate
    coverage_end: IsoDate
    days: Annotated[list[ScheduleSourceDay], Field(min_length=14)]

    @model_validator(mode="after")
    def complete_coverage(self) -> "ScheduleData":
        if (self.coverage_end - self.coverage_start).days < 13:
            raise ValueError("Нужно покрытие минимум двух последовательных недель.")
        if [day.date.toordinal() for day in self.days] != list(
            range(self.coverage_start.toordinal(), self.coverage_end.toordinal() + 1)
        ):
            raise ValueError("Нужно явно перечислить каждый день покрытия, включая свободные.")
        return self


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Повтор ключа JSON.")
        result[key] = value
    return result


class ScheduleService:
    def __init__(self, data: ScheduleData, timezone: str):
        try:
            expected_zone = TypeAdapter(TimezoneName).validate_python(timezone)
        except ValidationError:
            raise ScheduleSourceError("Настроена неизвестная IANA-зона расписания.") from None
        if data.timezone != expected_zone:
            raise ScheduleSourceError("Зона файла не совпадает с настроенной зоной расписания.")
        self.timezone = expected_zone
        self.coverage_start, self.coverage_end = data.coverage_start, data.coverage_end
        # Результаты не разделяют изменяемый список с загруженным источником.
        self._days = MappingProxyType({day.date: tuple(day.lessons) for day in data.days})

    @classmethod
    def from_file(cls, path: Path | str, timezone: str) -> "ScheduleService":
        """Путь передаёт конфигурация; это не аргумент модели или MCP-инструмента."""
        try:
            with Path(path).open("rb") as stream:
                raw = stream.read(MAX_SCHEDULE_BYTES + 1)
            if len(raw) > MAX_SCHEDULE_BYTES:
                raise ScheduleSourceError("Файл расписания превышает 1 МиБ.")
            value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_keys)
            data = ScheduleData.model_validate(value)
        except OSError:
            raise ScheduleSourceError("Файл расписания недоступен.") from None
        except (UnicodeError, ValueError, RecursionError) as error:
            if isinstance(error, ScheduleSourceError):
                raise
            raise ScheduleSourceError("Файл расписания имеет некорректный формат.") from None
        return cls(data, timezone)

    def get_schedule(self, day: date | str) -> ScheduleResult:
        day = ScheduleArguments(date=day).date
        if day not in self._days:
            return ScheduleResult(
                ScheduleOutsideCoverage(
                    status="outside_coverage",
                    date=day,
                    coverage_start=self.coverage_start,
                    coverage_end=self.coverage_end,
                    timezone=self.timezone,
                    source="local_schedule",
                )
            )
        return ScheduleResult(
            ScheduleOk(
                status="ok",
                date=day,
                timezone=self.timezone,
                lessons=list(self._days[day]),
                source="local_schedule",
            )
        )

    def current_week(self, *, now: datetime) -> WeekResult:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("Часы приложения должны возвращать aware datetime.")
        local_day = now.astimezone(ZoneInfo(self.timezone)).date()
        monday = local_day - timedelta(days=local_day.weekday())
        days = []
        for offset in range(7):
            day = monday + timedelta(days=offset)
            if day in self._days:
                days.append(WeekDayOk(status="ok", date=day, lessons=list(self._days[day])))
            else:
                days.append(WeekDayOutsideCoverage(status="outside_coverage", date=day))
        return WeekResult(
            WeekSchedule(
                week_start=monday,
                week_end=monday + timedelta(days=6),
                timezone=self.timezone,
                source="local_schedule",
                days=days,
            )
        )
