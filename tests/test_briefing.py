"""Daily source budgets/partial failure/cache with fake transports, no real LLM."""

import json
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from test_notifications import CITY

from app.agents.knowledge import KnowledgeAgent
from app.config import Settings
from app.materials.contracts import SearchArguments
from app.materials.corpus import Corpus
from app.notifications.briefing import BriefingService
from app.notifications.delivery_models import Lease, valid_payload
from app.notifications.models import briefing_moment


class Sources:
    available = True
    tools = {name: {"name": name} for name in ("get_schedule", "get_weather", "search_materials")}
    resources = {}

    def __init__(self, *, fail=()):
        self.corpus = Corpus.load()
        self.calls = []
        self.fail = fail

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if name in self.fail:
            raise TimeoutError("private transport detail")
        if name == "search_materials":
            return self.corpus.search(SearchArguments.model_validate(arguments)).model_dump(
                mode="json"
            )
        if name == "get_schedule":
            return {
                "status": "ok",
                "date": arguments["date"],
                "timezone": "Europe/Moscow",
                "lessons": [],
                "source": "local_schedule",
            }
        return {
            "status": "ok",
            "point": arguments["point"],
            "temperature_c": 7.0,
            "condition": {"code": 0, "description": "ясно"},
            "wind": {"speed": 3.0, "unit": "km/h"},
            "observed_at": "2026-11-02T08:00:00+03:00",
            "timezone": "Europe/Moscow",
            "source": "Open-Meteo",
        }


class Memory:
    def __init__(self):
        self.used = {"mcp": 0, "llm": 0}
        self.cache = None
        self.body = None
        self.valid = True

    async def snapshot(self, lease):
        return (
            (
                {"local_date": date(2026, 11, 2), "timezone": "Europe/Moscow"},
                {"city": CITY.model_dump_json()},
                briefing_moment(date(2026, 11, 2), "Europe/Moscow"),
            )
            if self.valid
            else None
        )

    async def reserve(self, lease, kind):
        self.used[kind] += 1
        assert self.used[kind] <= (3 if kind == "mcp" else 1)

    async def reminders(self, *args):
        return []

    async def ready(self, lease, body):
        if not self.valid:
            return False
        self.body = body
        return True

    async def focus(self, *args):
        return self.cache

    async def claim_focus(self, *args):
        if self.cache is not None:
            return None
        self.cache = {"body": None}
        return uuid4()

    async def save_focus(self, *args):
        self.cache = {"body": args[-1]}


def setup(*, fail=()):
    repo = Memory()
    sources = Sources(fail=fail)
    llm = SimpleNamespace(
        complete=AsyncMock(return_value=SimpleNamespace(text="invalid", truncated=False))
    )
    svc = BriefingService(
        repo,
        sources,
        KnowledgeAgent(sources.corpus.catalog()),
        llm,
        Settings(bot_token="", postgres_password=""),
    )
    lease = Lease(uuid4(), 101, uuid4(), 1, None, 101, 0)
    return svc, repo, sources, llm, lease


@pytest.mark.parametrize(
    "fail",
    [
        (),
        ("get_weather",),
        ("get_schedule",),
        ("search_materials",),
        ("get_weather", "get_schedule", "search_materials"),
    ],
)
async def test_daily_partial_sources_still_deliver_sections_with_bounded_reserved_calls(fail):
    svc, repo, sources, llm, lease = setup(fail=fail)
    assert await svc.prepare(lease)
    assert valid_payload(repo.body) and "Учебный фокус:" in repo.body
    assert repo.used["mcp"] == 3
    assert llm.complete.await_count <= 1
    assert "private transport detail" not in repo.body
    if "get_weather" in fail:
        assert "Погода недоступна" in repo.body
    else:
        assert "7 °C" in repo.body and "Казань" in repo.body
    if "get_schedule" not in fail:
        assert "Занятий нет" in repo.body
    for name, args in sources.calls:
        assert "actor" not in json.dumps(args)
        assert name != "add_reminder"


async def test_daily_focus_cache_survives_next_preparation_without_llm_or_search():
    svc, repo, sources, llm, lease = setup()
    await svc.prepare(lease)
    assert llm.complete.await_count == 1 and repo.cache["body"]
    repo.used = {"mcp": 0, "llm": 0}
    sources.calls.clear()
    await svc.prepare(lease)
    assert repo.used == {"mcp": 2, "llm": 0}
    assert llm.complete.await_count == 1
    assert [x[0] for x in sources.calls] == ["get_schedule", "get_weather"]


async def test_interrupted_cache_entry_uses_extract_without_regenerating():
    svc, repo, _, llm, lease = setup()
    repo.cache = {"body": None}
    await svc.prepare(lease)
    assert llm.complete.await_count == 0
    assert "Источник:" in repo.body


async def test_profile_revocation_prevents_publish_during_reads():
    svc, repo, sources, llm, lease = setup()
    original = sources.call_tool

    async def revoke(name, args):
        repo.valid = False
        return await original(name, args)

    sources.call_tool = revoke
    assert not await svc.prepare(lease)
    assert repo.body is None
    repo.valid = False
    count = llm.complete.await_count
    assert not await svc.prepare(lease)
    assert llm.complete.await_count == count


async def test_daily_model_never_sees_private_reminders_or_owner():
    svc, repo, _, llm, lease = setup()
    repo.reminders = AsyncMock(
        return_value=[
            {
                "text": "PRIVATE_SENTINEL",
                "remind_at": datetime.fromisoformat("2026-11-02T11:00:00+03:00"),
            }
        ]
    )
    await svc.prepare(lease)
    assert "PRIVATE_SENTINEL" in repo.body
    args = llm.complete.call_args
    raw = json.dumps(args.args[0])
    assert "PRIVATE_SENTINEL" not in raw and "actor_id" not in raw
    assert args.kwargs == {"disable_tools": True}


@pytest.mark.parametrize("bad", ["wrong_id", "wrong_quote", "truncated", "too_long", "exception"])
async def test_daily_focus_invalid_model_keeps_verified_extract_without_retry(bad):
    svc, repo, _, llm, lease = setup()
    if bad == "exception":
        llm.complete.side_effect = TimeoutError("SECRET")
    else:
        llm.complete.return_value = SimpleNamespace(
            text=json.dumps(
                {
                    "answer": "X" * 600 if bad == "too_long" else "Совет",
                    "citations": [
                        {
                            "document_id": "unknown" if bad == "wrong_id" else "llm-context",
                            "quote": "not grounded quotation text",
                        }
                    ],
                }
            ),
            truncated=bad == "truncated",
        )
    assert await svc.prepare(lease)
    assert llm.complete.await_count == 1 and "Источник:" in repo.body and "SECRET" not in repo.body
    assert "not grounded quotation text" not in repo.body


@pytest.mark.parametrize("name", ["get_schedule", "get_weather"])
async def test_daily_specialist_rejects_well_formed_result_for_other_input(name):
    from app.agents.specialists import LifestyleAgent, ScheduleAgent
    from app.routing.policy import ScopedMCP

    sources = Sources()
    original = sources.call_tool

    async def changed(tool, args):
        raw = await original(tool, args)
        if tool == "get_schedule":
            raw["date"] = "2026-11-03"
        else:
            raw["point"] = {**raw["point"], "latitude": 1.0}
        return raw

    sources.call_tool = changed
    with pytest.raises(ValueError):
        if name == "get_schedule":
            await ScheduleAgent.read_day(ScopedMCP(sources, "schedule"), date(2026, 11, 2))
        else:
            await LifestyleAgent.read_point(
                ScopedMCP(sources, "lifestyle", allow_point=True), CITY.coordinates()
            )


async def test_two_daily_leases_begin_together_before_either_preparation_finishes():
    import asyncio

    svc, repo, _, _, first = setup()
    repo.schedule = AsyncMock()
    second = Lease(uuid4(), 202, uuid4(), 1, None, 202, 0)
    repo.claim_daily = AsyncMock(return_value=[first, second])
    entered = set()
    both = asyncio.Event()
    release = asyncio.Event()

    async def prepare(lease):
        entered.add(lease.owner)
        if entered == {101, 202}:
            both.set()
        await release.wait()
        return True

    svc.prepare = prepare
    task = asyncio.create_task(svc.tick())
    try:
        await asyncio.wait_for(both.wait(), 1)
        assert not task.done()
    finally:
        release.set()
        await task
