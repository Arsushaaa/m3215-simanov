"""Real PostgreSQL fences for edits of an unconfirmed reminder.

The shared fixture creates a disposable database/container. No production
database, Telegram account, weather endpoint or real LLM is used.
"""

import asyncio
import os
from dataclasses import replace
from datetime import datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest
from test_integration import database as database

from app.agent_repository import ConfirmationRepository, ReminderRepository
from app.confirmation_models import Authorization, ConfirmationError
from app.confirmations import ConfirmationService
from app.db import create_pool
from app.dialog import DialogService
from app.mcp.client import MCPClient
from app.mcp.contracts import ReminderArguments
from app.migration import migrate
from app.repository import DialogRepository

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1",
        reason="Включите RUN_INTEGRATION=1 для Docker-тестов",
    ),
]

ZONE = ZoneInfo("Europe/Moscow")
EDIT = "нет, давай лучше в 21:00"
ROOT = Path(__file__).resolve().parents[1]


async def seed_card(pool, owner=101):
    dialogs = DialogRepository(pool)
    confirmations = ConfirmationRepository(pool)
    await dialogs.user(owner)
    await confirmations.set_timezone(owner, ZONE.key)
    now = await confirmations.now()
    moment = datetime.combine(now.astimezone(ZONE).date() + timedelta(days=1), time(19), ZONE)
    card = await confirmations.prepare(
        await dialogs.user(owner),
        10,
        "Напомни завтра в 19:00 сдать отчёт",
        ReminderArguments(text="Сдать отчёт", remind_at=moment),
        timezone=ZONE.key,
    )
    return dialogs, confirmations, card


async def test_short_time_edit_replaces_old_card_and_commits_once_through_real_mcp(database):
    settings, _, _, _ = database
    settings = replace(
        settings,
        schedule_file=str(ROOT / "data" / "schedule.json"),
    )
    pool = await create_pool(settings)
    mcp = MCPClient(settings)
    mcp.call_tool = AsyncMock(wraps=mcp.call_tool)
    mcp.read_resource = AsyncMock(wraps=mcp.read_resource)
    try:
        await migrate(pool)
        dialogs, confirmations, old = await seed_card(pool)
        llm = SimpleNamespace(
            complete=AsyncMock(side_effect=AssertionError("Time edits require no LLM")),
            decide=AsyncMock(side_effect=AssertionError("Time edits require no LLM")),
        )
        await mcp.start()
        assert mcp.available
        service = ConfirmationService(confirmations, mcp)
        dialog = DialogService(dialogs, llm, settings, confirmations=service)

        answer = await dialog.respond(101, 11, EDIT)
        edited = await dialog.confirmation_card(101, 11)
        assert edited is not None and edited.state == "pending"
        assert edited.action_id != old.action_id
        assert edited.arguments.text == old.arguments.text
        assert edited.arguments.remind_at == old.arguments.remind_at.replace(hour=21)
        assert "21:00" in answer
        assert await dialog.respond(101, 11, EDIT) == answer
        assert (await confirmations.current_card(101)).action_id == edited.action_id
        old_state = await pool.fetchval(
            "SELECT state FROM labbi_actions WHERE id=$1", old.action_id
        )
        assert old_state == "cancelled"
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders") == 0
        llm.complete.assert_not_awaited()
        llm.decide.assert_not_awaited()
        mcp.call_tool.assert_not_awaited()

        assert "сохранено" not in await dialog.confirm(101, str(old.action_id))
        mcp.call_tool.assert_not_awaited()
        receipt = await dialog.confirm(101, str(edited.action_id))
        assert "Напоминание сохранено." in receipt and "21:00" in receipt
        assert await dialog.confirm(101, str(edited.action_id)) == receipt
        mcp.call_tool.assert_awaited_once()
        mcp.read_resource.assert_not_awaited()
        llm.complete.assert_not_awaited()
        llm.decide.assert_not_awaited()
        rows = await pool.fetch("SELECT action_id, text, remind_at FROM labbi_reminders")
        assert len(rows) == 1
        assert rows[0]["action_id"] == edited.action_id
        assert rows[0]["text"] == old.arguments.text
        assert rows[0]["remind_at"] == edited.arguments.remind_at
        history = await dialogs.history(101, 20)
        assert history[-2:] == [
            {"role": "user", "content": EDIT},
            {"role": "assistant", "content": answer},
        ]
    finally:
        await mcp.close()
        await pool.close()


async def test_stop_between_edit_snapshot_and_prepare_cannot_revive_card_across_pools(database):
    settings, _, _, _ = database
    pool = await create_pool(settings)
    other_pool = await create_pool(settings)
    paused, resume = asyncio.Event(), asyncio.Event()
    try:
        await migrate(pool)
        dialogs, confirmations, old = await seed_card(pool)
        service = ConfirmationService(confirmations, SimpleNamespace())
        stop = ConfirmationRepository(other_pool)
        captured = {}

        async def edit():
            captured["snapshot"] = await service.begin_edit(await dialogs.user(101), 11)
            paused.set()
            await resume.wait()
            source, user, epoch = captured["snapshot"]
            return await service.prepare_time_edit(user, 11, EDIT, source, epoch)

        task = asyncio.create_task(edit())
        try:
            await asyncio.wait_for(paused.wait(), timeout=5)
            source, user, epoch = captured["snapshot"]
            assert source.action_id == old.action_id
            assert (await dialogs.user(101)).revision == user.revision
            await stop.invalidate(101)
            assert (await dialogs.user(101)).revision == user.revision
            assert (
                await other_pool.fetchval(
                    "SELECT confirmation_epoch FROM labbi_users WHERE user_id=101"
                )
                > epoch
            )
            resume.set()
            with pytest.raises(ConfirmationError) as rejected:
                await task
            assert rejected.value.code == "ACTION_CANCELLED"
        finally:
            resume.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        assert await confirmations.current_card(101) is None
        assert await pool.fetchval("SELECT count(*) FROM labbi_actions") == 1
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders") == 0
        assert (await confirmations.approve(101, old.action_id)).code == "ACTION_CANCELLED"
    finally:
        await other_pool.close()
        await pool.close()


async def test_edit_revokes_already_issued_write_capability_before_replacement(database):
    settings, _, _, _ = database
    pool = await create_pool(settings)
    other_pool = await create_pool(settings)
    try:
        await migrate(pool)
        dialogs, confirmations, old = await seed_card(pool)
        authorization = await confirmations.approve(101, old.action_id)
        assert isinstance(authorization, Authorization)
        service = ConfirmationService(confirmations, SimpleNamespace())
        source, user, epoch = await service.begin_edit(await dialogs.user(101), 11)

        refused = await ReminderRepository(other_pool).add_reminder(
            authorization.arguments, authorization.context
        )
        assert refused.root.status == "error"
        assert refused.root.error.code == "ACTION_CANCELLED"
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders") == 0
        edited = await service.prepare_time_edit(user, 11, EDIT, source, epoch)
        assert edited.arguments.remind_at == old.arguments.remind_at.replace(hour=21)
        assert (await confirmations.current_card(101)).action_id == edited.action_id
        old_state = await pool.fetchval(
            "SELECT state FROM labbi_actions WHERE id=$1", old.action_id
        )
        assert old_state == "cancelled"
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders") == 0
    finally:
        await other_pool.close()
        await pool.close()


async def test_another_owner_cannot_capture_or_revoke_existing_card(database):
    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        dialogs, confirmations, old = await seed_card(pool)
        foreign = await dialogs.user(202)
        epoch = await pool.fetchval("SELECT confirmation_epoch FROM labbi_users WHERE user_id=101")

        assert await confirmations.revoke_for_edit(foreign, 11) is None

        current = await confirmations.current_card(101)
        assert current.action_id == old.action_id and current.state == "pending"
        assert current.arguments == old.arguments
        assert await confirmations.current_card(202) is None
        current_epoch = await pool.fetchval(
            "SELECT confirmation_epoch FROM labbi_users WHERE user_id=101"
        )
        assert current_epoch == epoch
        foreign_requests = await pool.fetchval(
            "SELECT count(*) FROM labbi_agent_requests WHERE user_id=202"
        )
        assert foreign_requests == 0
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders") == 0
    finally:
        await pool.close()
