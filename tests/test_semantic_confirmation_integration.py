"""Real PostgreSQL ordering between semantic analysis and owner confirmation."""

import os
from datetime import datetime, time, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from test_integration import database as database

from app.agent_repository import (
    SEMANTIC_REQUEST_WAIT,
    ConfirmationRepository,
    ReminderRepository,
)
from app.confirmation_models import Authorization, ConfirmationError
from app.confirmations import ConfirmationService
from app.db import create_pool
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


async def seed(pool):
    dialogs = DialogRepository(pool)
    repository = ConfirmationRepository(pool)
    await dialogs.user(101)
    await repository.set_timezone(101, ZONE.key)
    now = await repository.now()
    moment = datetime.combine(now.astimezone(ZONE).date() + timedelta(days=1), time(19), ZONE)
    card = await repository.prepare(
        await dialogs.user(101),
        10,
        "Напомни завтра в 19:00 сдать отчёт",
        ReminderArguments(text="Сдать отчёт", remind_at=moment),
        timezone=ZONE.key,
    )
    return dialogs, repository, card


async def finish(repository, status="OK"):
    await repository.record(
        101,
        11,
        decision="respond",
        target=None,
        status=status,
        reason="Запрос завершён без изменения напоминания.",
        llm_calls=1,
    )


@pytest.mark.parametrize("terminal_status", ["OK", "FAILED"])
async def test_unrelated_or_failed_request_temporarily_pauses_but_does_not_revoke_card(
    database, terminal_status
):
    settings, _, _, _ = database
    pool, other_pool = await create_pool(settings), await create_pool(settings)
    try:
        await migrate(pool)
        dialogs, repository, card = await seed(pool)
        other = ConfirmationRepository(other_pool)
        user = await dialogs.user(101)
        snapshot, epoch = await ConfirmationService(
            repository, SimpleNamespace()
        ).begin_semantic_request(user, 11)

        assert snapshot.action_id == card.action_id
        assert snapshot.arguments == card.arguments
        assert await dialogs.user(101) == user
        assert epoch == await pool.fetchval(
            "SELECT confirmation_epoch FROM labbi_users WHERE user_id=101"
        )
        blocked = await other.approve(101, card.action_id)
        assert blocked.code == "ACTION_CONFLICT" and blocked.message == SEMANTIC_REQUEST_WAIT
        assert (
            await pool.fetchval(
                "SELECT capability_hash FROM labbi_actions WHERE id=$1", card.action_id
            )
            is None
        )
        assert (await repository.current_card(101)).action_id == card.action_id

        await finish(repository, terminal_status)

        authorization = await other.approve(101, card.action_id)
        assert isinstance(authorization, Authorization)
        saved = await ReminderRepository(other_pool).add_reminder(
            authorization.arguments, authorization.context
        )
        assert saved.root.status == "ok"
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders") == 1
    finally:
        await other_pool.close()
        await pool.close()


async def test_issued_capability_cannot_write_during_semantics_or_after_semantic_edit(database):
    settings, _, _, _ = database
    pool, other_pool = await create_pool(settings), await create_pool(settings)
    try:
        await migrate(pool)
        dialogs, repository, card = await seed(pool)
        authorization = await repository.approve(101, card.action_id)
        assert isinstance(authorization, Authorization)
        service = ConfirmationService(repository, SimpleNamespace())
        user = await dialogs.user(101)
        snapshot, _ = await service.begin_semantic_request(user, 11)
        assert snapshot.state == "approved"

        blocked = await ReminderRepository(other_pool).add_reminder(
            authorization.arguments, authorization.context
        )
        assert blocked.root.status == "error"
        assert blocked.root.error.code == "ACTION_CONFLICT"
        assert blocked.root.error.message == SEMANTIC_REQUEST_WAIT
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders") == 0

        # The semantic decision identifies an edit: revoke before closing its marker.
        _, edited_user, epoch = await service.begin_edit(user, 11)
        await repository.record(
            101,
            11,
            decision="call_tool",
            target="add_reminder",
            status="CLARIFY",
            reason="Модель подготовила изменённые параметры напоминания.",
            llm_calls=1,
        )
        refused = await ReminderRepository(other_pool).add_reminder(
            authorization.arguments, authorization.context
        )
        assert refused.root.status == "error"
        assert refused.root.error.code == "ACTION_CANCELLED"
        replacement = await repository.prepare(
            edited_user,
            11,
            "Не в 19:00, а в 21:00",
            ReminderArguments(
                text=card.arguments.text,
                remind_at=card.arguments.remind_at.replace(hour=21),
            ),
            timezone=ZONE.key,
            expected_epoch=epoch,
        )
        assert replacement.action_id != card.action_id
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders") == 0
    finally:
        await other_pool.close()
        await pool.close()


async def test_snapshot_without_a_card_still_fences_stop_during_generation(database):
    settings, _, _, _ = database
    pool, other_pool = await create_pool(settings), await create_pool(settings)
    try:
        await migrate(pool)
        dialogs, repository = DialogRepository(pool), ConfirmationRepository(pool)
        await dialogs.user(101)
        await repository.set_timezone(101, ZONE.key)
        user = await dialogs.user(101)
        card, epoch = await ConfirmationService(
            repository, SimpleNamespace()
        ).begin_semantic_request(user, 11)
        assert card is None

        await ConfirmationRepository(other_pool).invalidate(101)
        assert await dialogs.user(101) == user
        now = await repository.now()
        with pytest.raises(ConfirmationError) as cancelled:
            await repository.prepare(
                user,
                11,
                "Не забудь напомнить в 11:00 завтра приготовить пирог",
                ReminderArguments(text="Приготовить пирог", remind_at=now + timedelta(days=1)),
                timezone=ZONE.key,
                expected_epoch=epoch,
            )
        assert cancelled.value.code == "ACTION_CANCELLED"
        assert await pool.fetchval("SELECT count(*) FROM labbi_actions") == 0
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders") == 0
    finally:
        await other_pool.close()
        await pool.close()


async def test_stale_processing_marker_does_not_pause_a_fresh_card_indefinitely(database):
    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        dialogs, repository, card = await seed(pool)
        await pool.execute(
            "UPDATE labbi_agent_requests SET created_at=clock_timestamp()-INTERVAL '400 seconds' "
            "WHERE user_id=101 AND telegram_message_id=10"
        )
        await repository.start_request(101, 11)
        await pool.execute(
            "UPDATE labbi_agent_requests SET created_at=clock_timestamp()-INTERVAL '301 seconds' "
            "WHERE user_id=101 AND telegram_message_id=11"
        )

        authorization = await repository.approve(101, card.action_id)

        assert isinstance(authorization, Authorization)
        assert (await repository.current_card(101)).action_id == card.action_id
        assert await dialogs.user(101)
    finally:
        await pool.close()


async def test_completed_receipt_is_readable_while_a_new_request_is_processing(database):
    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        dialogs, repository, card = await seed(pool)
        authorization = await repository.approve(101, card.action_id)
        saved = await ReminderRepository(pool).add_reminder(
            authorization.arguments, authorization.context
        )
        assert saved.root.status == "ok"
        snapshot, _ = await ConfirmationService(
            repository, SimpleNamespace()
        ).begin_semantic_request(await dialogs.user(101), 11)
        assert snapshot is None

        receipt = await repository.approve(101, card.action_id)

        assert receipt.status == "committed"
        assert receipt.reminder_id == saved.root.reminder_id
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders") == 1
    finally:
        await pool.close()


async def test_duplicate_update_cannot_claim_or_finish_another_request(database):
    settings, _, _, _ = database
    pool, other_pool = await create_pool(settings), await create_pool(settings)
    try:
        await migrate(pool)
        dialogs, repository, card = await seed(pool)
        user = await dialogs.user(101)
        service = ConfirmationService(repository, SimpleNamespace())
        await service.begin_semantic_request(user, 11)
        events = await pool.fetchval("SELECT count(*) FROM labbi_agent_events")

        with pytest.raises(ConfirmationError) as duplicate:
            await ConfirmationRepository(other_pool).semantic_request_context(user, 11)

        assert duplicate.value.code == "ACTION_CONFLICT"
        assert (
            await pool.fetchval(
                "SELECT status FROM labbi_agent_requests "
                "WHERE user_id=101 AND telegram_message_id=11"
            )
            == "processing"
        )
        assert await pool.fetchval("SELECT count(*) FROM labbi_agent_events") == events
        assert await pool.fetchval("SELECT count(*) FROM labbi_agent_requests") == 2
        assert (await repository.current_card(101)).action_id == card.action_id
        assert (await repository.approve(101, card.action_id)).message == SEMANTIC_REQUEST_WAIT
    finally:
        await other_pool.close()
        await pool.close()


async def test_semantic_edit_uses_epoch_when_an_unrelated_answer_changes_only_revision(database):
    settings, _, _, _ = database
    pool, other_pool = await create_pool(settings), await create_pool(settings)
    try:
        await migrate(pool)
        dialogs, repository, card = await seed(pool)
        user = await dialogs.user(101)
        service = ConfirmationService(repository, SimpleNamespace())
        _, epoch = await service.begin_semantic_request(user, 11)
        other_dialogs = DialogRepository(other_pool)
        assert await other_dialogs.save_exchange(
            user, 12, "Что такое цикл?", "Повторение действий."
        )
        assert (await dialogs.user(101)).revision == user.revision + 1
        assert (
            await pool.fetchval("SELECT confirmation_epoch FROM labbi_users WHERE user_id=101")
            == epoch
        )

        source, edited_user, new_epoch = await service.begin_edit(user, 11, expected_epoch=epoch)

        assert source.action_id == card.action_id
        assert edited_user.revision == user.revision + 2
        assert new_epoch == epoch + 1
        assert (await repository.approve(101, card.action_id)).code == "ACTION_CANCELLED"
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders") == 0
    finally:
        await other_pool.close()
        await pool.close()


async def test_old_semantic_decision_cannot_revoke_a_newer_card(database):
    settings, _, _, _ = database
    pool, other_pool = await create_pool(settings), await create_pool(settings)
    try:
        await migrate(pool)
        dialogs, repository, card = await seed(pool)
        user = await dialogs.user(101)
        service = ConfirmationService(repository, SimpleNamespace())
        _, epoch = await service.begin_semantic_request(user, 11)
        replacement = await ConfirmationRepository(other_pool).prepare(
            await DialogRepository(other_pool).user(101),
            12,
            "Напомни завтра в 21:00 купить хлеб",
            ReminderArguments(
                text="Купить хлеб", remind_at=card.arguments.remind_at.replace(hour=21)
            ),
            timezone=ZONE.key,
        )

        with pytest.raises(ConfirmationError) as obsolete:
            await service.begin_edit(user, 11, expected_epoch=epoch)

        assert obsolete.value.code == "ACTION_CANCELLED"
        assert (await repository.current_card(101)).action_id == replacement.action_id
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders") == 0
    finally:
        await other_pool.close()
        await pool.close()
