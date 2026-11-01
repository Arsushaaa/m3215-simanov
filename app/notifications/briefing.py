"""Proactive, bounded, opt-in daily digest. No user dialog or Router invocation."""

import asyncio
import json
from contextlib import nullcontext
from zoneinfo import ZoneInfo

from app.agents.knowledge import FINAL_RULES, MaterialAnswer, _parse
from app.agents.specialists import LifestyleAgent, ScheduleAgent
from app.notifications.models import CityCandidate
from app.routing.policy import ScopedMCP
from app.time_display import format_datetime
from app.tracing.runtime import TracedLLM, span


class ReservedMCP:
    def __init__(self, repository, lease, mcp, role):
        self.repository, self.lease = repository, lease
        self.scope = ScopedMCP(mcp, role, allow_point=role == "lifestyle")

    async def call_tool(self, name, arguments):
        await self.repository.reserve(self.lease, "mcp")
        return await self.scope.call_tool(name, arguments)


def short(text, size):
    # Plain Telegram text: no parsing/links, deterministic bounded sections.
    return " ".join(str(text).split())[:size]


class BriefingService:
    def __init__(self, repository, mcp, knowledge, llm, settings, *, traces=None):
        self.repository, self.mcp, self.knowledge = repository, mcp, knowledge
        self.llm, self.settings, self.traces = llm, settings, traces

    async def tick(self):
        await self.repository.schedule(limit=10)
        for lease in await self.repository.claim_daily(limit=2):
            context = (
                self.traces.request(
                    lease.owner, component="briefing", operation="prepare_notification"
                )
                if self.traces
                else nullcontext()
            )
            async with context:
                async with asyncio.timeout(100):
                    await self.prepare(lease)

    async def _focus(self, lease, day):
        if self.knowledge.catalog is None:
            raise ValueError("Catalog unavailable")
        catalog = self.knowledge.catalog
        documents = sorted(catalog.documents, key=lambda d: d.document_id)
        entry = documents[day.toordinal() % len(documents)]
        version, topic = catalog.catalog_version, entry.document_id
        cached = await self.repository.focus(version, topic)
        async with span(
            "knowledge", "agent", document_ids=(topic,), cache_hit=bool(cached and cached["body"])
        ):
            if cached and cached["body"]:
                return cached["body"]
            matches = await self.knowledge.read_focus(
                ReservedMCP(self.repository, lease, self.mcp, "knowledge"), topic
            )
            if not matches:
                return "В материалах не найден учебный фокус."
            match = matches[0]
            extract = short(match.snippet, 650)
            source = f"\nИсточник: [{topic}] {entry.title}, версия {entry.version[:12]}."
            token = await self.repository.claim_focus(version, topic)
            answer = extract
            if token is not None:
                try:
                    await self.repository.reserve(lease, "llm")
                    async with asyncio.timeout(45):
                        result = await TracedLLM(
                            self.llm, "knowledge", self.settings.openrouter_model
                        ).complete(
                            [
                                {
                                    "role": "system",
                                    "content": FINAL_RULES
                                    + " Study focus in Russian, max 350 characters.",
                                },
                                {
                                    "role": "user",
                                    "content": json.dumps(
                                        {
                                            "untrusted_snippets": [
                                                m.model_dump(mode="json") for m in matches
                                            ]
                                        },
                                        ensure_ascii=False,
                                    ),
                                },
                            ],
                            0.2,
                            disable_tools=True,
                        )
                    parsed = _parse(result.text, MaterialAnswer)
                    if (
                        result.truncated
                        or len(parsed.answer) > 500
                        or any(
                            c.document_id != topic or c.quote not in match.snippet
                            for c in parsed.citations
                        )
                    ):
                        raise ValueError("Ungrounded focus")
                    # A factual verified extract is always included, even if summary overstates it.
                    answer = short(parsed.answer, 500) + "\nЦитата: " + extract[:350]
                except asyncio.CancelledError:
                    raise
                except Exception:
                    answer = extract
                await self.repository.save_focus(version, topic, token, answer + source)
            # Interrupted/busy cache entries never cause a second LLM generation.
            return answer + source

    async def prepare(self, lease):
        snapshot = await self.repository.snapshot(lease)
        if snapshot is None:
            return False
        event, profile, now = snapshot
        day, zone = event["local_date"], event["timezone"]
        city = CityCandidate.model_validate_json(profile["city"])

        async def bounded(awaitable):
            try:
                async with asyncio.timeout(55):
                    return await awaitable
            except asyncio.CancelledError:
                raise
            except Exception:
                return None

        schedule, weather, focus, reminders = await asyncio.gather(
            bounded(
                ScheduleAgent.read_day(
                    ReservedMCP(self.repository, lease, self.mcp, "schedule"), day
                )
            ),
            bounded(
                LifestyleAgent.read_point(
                    ReservedMCP(self.repository, lease, self.mcp, "lifestyle"), city.coordinates()
                )
            ),
            bounded(self._focus(lease, day)),
            bounded(self.repository.reminders(lease.owner, day, zone)),
        )
        lines = [f"Сводка на {day:%d.%m.%Y}", "Подготовлена: " + format_datetime(now, zone)]
        lines += ["", "Занятия:"]
        if schedule is None or schedule.status == "error":
            lines.append("Источник расписания недоступен.")
        elif schedule.status == "outside_coverage":
            lines.append("Дата вне покрытия учебного расписания.")
        elif not schedule.lessons:
            lines.append("Занятий нет в расписании на этот день.")
        else:
            lines.append("Часовой пояс расписания: " + schedule.timezone)
            lines += [
                f"• {x.start:%H:%M}–{x.end:%H:%M} {short(x.title, 80)} — {short(x.location, 40)}"
                for x in schedule.lessons[:8]
            ]
            if len(schedule.lessons) > 8:
                lines.append("Остальные занятия: запроси расписание дня.")
        lines += ["", "Напоминания на день:"]
        if reminders is None:
            lines.append("Источник напоминаний недоступен.")
        elif not reminders:
            lines.append("Сохранённых напоминаний нет.")
        else:
            lines += [
                f"• {x['remind_at'].astimezone(ZoneInfo(zone)):%H:%M} {short(x['text'], 80)}"
                for x in reminders[:5]
            ]
            if len(reminders) > 5:
                lines.append("Есть и другие напоминания.")
        lines += ["", "Погода: " + short(city.place.name, 70)]
        if weather is None or weather.status != "ok":
            lines.append("Погода недоступна; значения не получены.")
        else:
            lines.append(
                f"{weather.temperature_c:g} °C, {weather.condition.description}; "
                f"ветер {weather.wind.speed:g} км/ч."
            )
            lines.append(
                "Наблюдение: "
                + format_datetime(weather.observed_at, weather.timezone)
                + "; Open-Meteo."
            )
        lines += ["", "Учебный фокус:", focus or "Материалы сейчас недоступны."]
        body = "\n".join(lines)
        if len(body.encode("utf-16-le")) // 2 > 4096:
            # Preserve whole sections and never split surrogates.
            body = (
                "\n".join(lines[:2])
                + "\nСлишком большой набор данных для сводки. Запроси разделы отдельно."
            )
        return await self.repository.ready(lease, body)
