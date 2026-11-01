"""Host-owned profile and city values, never destinations supplied by a model."""

from datetime import UTC, date, datetime, time, timedelta
from typing import Annotated

from pydantic import Field

from app.mcp.contracts import Coordinates as Coordinates
from app.mcp.contracts import Place, validate_local_datetime


class CityCandidate(Coordinates):
    geonames_id: Annotated[int, Field(strict=True, gt=0)]
    place: Place

    def coordinates(self):
        return Coordinates(latitude=self.latitude, longitude=self.longitude, timezone=self.timezone)


def briefing_moment(day: date, zone: str):
    return validate_local_datetime(datetime.combine(day, time(8)), zone).astimezone(UTC)


def next_briefing(now: datetime, zone: str):
    from zoneinfo import ZoneInfo

    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("Aware clock required")
    day = now.astimezone(ZoneInfo(zone)).date()
    moment = briefing_moment(day, zone)
    return moment if moment > now else briefing_moment(day + timedelta(days=1), zone)
