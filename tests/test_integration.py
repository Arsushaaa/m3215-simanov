"""RUN_INTEGRATION=1 python -m pytest -m integration -v.

Тест создаёт собственные контейнер и volume; пользовательскую БД не использует.
"""

import asyncio
import hashlib
import json
import os
import secrets
import socket
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4
from zoneinfo import ZoneInfo

import asyncpg
import pytest
from mcp import Client, StdioServerParameters

from app.agent import AgentService
from app.agent_repository import AgentRepository, ConfirmationRepository, ReminderRepository
from app.config import MCPSettings, Settings
from app.confirmation_models import Authorization, ConfirmationError, Outcome
from app.confirmations import ConfirmationService
from app.db import create_pool
from app.dialog import DialogService
from app.health import HealthState, health_result
from app.llm import Completion, ModelError, RoutingCompletion
from app.mcp.client import MCPClient
from app.mcp.contracts import ReminderArguments, ReminderResult
from app.migration import migrate
from app.repository import DialogRepository
from app.time_display import format_datetime
from app.timezone_lookup import TimezoneOption
from app.timezone_ui import TimezoneUI, TimezoneView
from scripts.common import CommandError, run_command

ROOT = Path(__file__).resolve().parents[1]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1",
        reason="Включите RUN_INTEGRATION=1 для Docker-тестов",
    ),
]


@pytest.fixture
def database():
    name = "itmo-test-" + secrets.token_hex(6)
    volume = name + "-data"
    password = secrets.token_urlsafe(24)

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    def docker(*args):
        return run_command(["docker", *args], label="Интеграционный Docker-тест", timeout=180)

    def wait_until_ready():
        deadline = time.monotonic() + 60
        while True:
            try:
                docker("exec", name, "pg_isready", "-h", "127.0.0.1", "-U", "bot", "-d", "bot")
                return
            except CommandError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1)

    docker("info")
    docker("volume", "create", volume)
    try:
        docker(
            "run",
            "-d",
            "--name",
            name,
            "-p",
            f"127.0.0.1:{port}:5432",
            "-e",
            "POSTGRES_DB=bot",
            "-e",
            "POSTGRES_USER=bot",
            "-e",
            f"POSTGRES_PASSWORD={password}",
            "--mount",
            f"type=volume,source={volume},target=/var/lib/postgresql/data",
            "postgres:16-bookworm",
        )
        info = json.loads(docker("inspect", name))[0]
        assert int(info["NetworkSettings"]["Ports"]["5432/tcp"][0]["HostPort"]) == port
        wait_until_ready()
        yield (
            Settings(
                bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
                postgres_password=password,
                postgres_port=port,
            ),
            docker,
            wait_until_ready,
            name,
        )
    finally:
        try:
            docker("rm", "-f", name)
        finally:
            docker("volume", "rm", volume)


async def test_real_db_password_restart_health_and_persistence(database):
    # Arrange
    settings, docker, wait_until_ready, name = database
    pool = await create_pool(settings)
    task = asyncio.create_task(asyncio.Event().wait())
    state = HealthState(pool=pool, initialized=True, polling_task=task)
    try:
        await pool.execute("CREATE TABLE integration_probe (value text NOT NULL)")
        await pool.execute("INSERT INTO integration_probe VALUES ($1)", "Сохранилось 👋")
        # Act / Assert: реальное подключение и ошибка пароля.
        assert (await health_result(state))[0] == 200
        with pytest.raises(asyncpg.InvalidPasswordError):
            await create_pool(replace(settings, postgres_password="wrong-password"))
        # Act / Assert: потеря соединения и сохранность данных после запуска.
        await asyncio.to_thread(docker, "stop", name)
        assert (await health_result(state))[0] == 503
        await asyncio.to_thread(docker, "start", name)
        await asyncio.to_thread(wait_until_ready)
        for _ in range(60):
            if (await health_result(state))[0] == 200:
                break
            await asyncio.sleep(1)
        assert (await health_result(state))[0] == 200
        assert await pool.fetchval("SELECT value FROM integration_probe") == "Сохранилось 👋"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await pool.close()


async def test_dialog_migrations_isolation_persistence_atomicity_and_revision(database):
    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        await migrate(pool)
        assert await pool.fetchval("SELECT count(*) FROM labbi_schema_migrations") == 9
        repository = DialogRepository(pool)
        alice = await repository.user(101)
        bob = await repository.user(202)
        assert (alice.mode, alice.temperature) == ("study", 0.3)
        assert await repository.save_exchange(alice, 1, "Вопрос Алисы 👋", "Ответ Алисе")
        assert await repository.save_exchange(bob, 1, "Вопрос Боба", "Ответ Бобу")
        # Обновление уже заполненной БД со старым ограничением сохраняет историю.
        await pool.execute(
            "ALTER TABLE labbi_users DROP CONSTRAINT labbi_users_temperature_check, "
            "ADD CONSTRAINT labbi_users_temperature_check "
            "CHECK (temperature IN (0.0, 0.3, 0.7, 1.0)); "
            "DELETE FROM labbi_schema_migrations WHERE version='003_temperature_range.sql'"
        )
        with pytest.raises(asyncpg.CheckViolationError):
            await repository.set_temperature(101, 0.55)
        await migrate(pool)
        await repository.set_temperature(101, 0.55)
        for value in (-0.001, 1.001, float("nan"), float("inf")):
            with pytest.raises(asyncpg.CheckViolationError):
                await repository.set_temperature(101, value)
        await pool.close()

        # Новое соединение имитирует перезапуск процесса; память Python не используется.
        pool = await create_pool(settings)
        repository = DialogRepository(pool)
        alice = await repository.user(101)
        assert alice.temperature == 0.55
        assert await repository.history(101, 20) == [
            {"role": "user", "content": "Вопрос Алисы 👋"},
            {"role": "assistant", "content": "Ответ Алисе"},
        ]
        assert await repository.answer(101, 1) == "Ответ Алисе"
        assert await repository.answer(202, 1) == "Ответ Бобу"

        await repository.change_mode(101, "quiz")
        assert await repository.history(101, 20) == []
        assert len(await repository.history(202, 20)) == 2
        alice = await repository.user(101)
        assert (alice.mode, alice.temperature) == ("quiz", 0.55)

        assert await repository.save_exchange(
            alice, 10, "Тема: циклы", "Вопрос 1/5", quiz_topic="Тема: циклы", quiz_question=1
        )
        await pool.close()
        pool = await create_pool(settings)
        repository = DialogRepository(pool)
        alice = await repository.user(101)
        assert (alice.quiz_topic, alice.quiz_question, alice.quiz_finished) == (
            "Тема: циклы",
            1,
            False,
        )
        await repository.reset(101)
        alice = await repository.user(101)
        assert alice.quiz_question == 0 and alice.quiz_topic is None

        # Ошибка второй записи откатывает всю пару, включая пользовательское сообщение.
        with pytest.raises(asyncpg.CheckViolationError):
            await repository.save_exchange(alice, 2, "Не должен сохраниться", "")
        assert await repository.history(101, 20) == []
        assert (await repository.user(101)).revision == alice.revision

        # Два процесса с одним снимком контекста не могут сохранить несовместимые ответы.
        results = await asyncio.gather(
            repository.save_exchange(alice, 3, "Первый вариант", "Ответ 1"),
            repository.save_exchange(alice, 4, "Второй вариант", "Ответ 2"),
        )
        assert sorted(results) == [False, True]
        assert len(await repository.history(101, 20)) == 2
        assert not await repository.save_exchange(alice, 5, "Устаревший запрос", "Ответ")
        await repository.reset(101)
        alice = await repository.user(101)
        assert (alice.mode, alice.temperature) == ("quiz", 0.55)
        assert await repository.history(101, 20) == []
        assert await repository.history(202, 20) == [
            {"role": "user", "content": "Вопрос Боба"},
            {"role": "assistant", "content": "Ответ Бобу"},
        ]
        assert not await repository.has_incomplete_answer(101)
        assert not await repository.has_incomplete_answer(202)
        assert await repository.save_exchange(
            alice, 20, "Эссе", "Неполный текст для продолжения", is_complete=False
        )
        await pool.close()
        pool = await create_pool(settings)
        repository = DialogRepository(pool)
        assert await repository.has_incomplete_answer(101)
        assert not await repository.has_incomplete_answer(202)
        assert await repository.answer(101, 20) == "Неполный текст для продолжения"
        flags = await pool.fetch(
            "SELECT role, is_complete FROM labbi_messages WHERE user_id=101 ORDER BY id"
        )
        assert [(row["role"], row["is_complete"]) for row in flags] == [
            ("user", True),
            ("assistant", False),
        ]
        alice = await repository.user(101)
        assert await repository.save_exchange(alice, 21, "Продолжи", "Конец эссе")
        assert not await repository.has_incomplete_answer(101)
    finally:
        await pool.close()


async def _insert_agent_request(pool, user_id, message_id):
    return await pool.fetchval(
        "INSERT INTO labbi_agent_requests(user_id, telegram_message_id, telegram_chat_id) "
        "VALUES ($1, $2, $1) RETURNING id",
        user_id,
        message_id,
    )


async def _insert_agent_action(pool, user_id, message_id, request_id, **changes):
    prepared_at = changes.get("prepared_at", datetime.now(UTC))
    values = {
        "user_id": user_id,
        "request_id": request_id,
        "telegram_message_id": message_id,
        "confirmation_epoch": 0,
        "text": "Отправить отчёт",
        "remind_at": prepared_at + timedelta(days=1),
        "timezone": "Europe/Moscow",
        "prepared_at": prepared_at,
        "expires_at": prepared_at + timedelta(seconds=300),
    }
    values.update(changes)
    columns = ", ".join(values)
    parameters = ", ".join(f"${index}" for index in range(1, len(values) + 1))
    return await pool.fetchrow(
        f"INSERT INTO labbi_actions({columns}) VALUES ({parameters}) RETURNING *",
        *values.values(),
    )


async def test_agent_migration_upgrades_lab1_and_preserves_history_and_registry(database):
    settings, _, _, _ = database
    pool = await create_pool(settings)
    migrations = Path(__file__).parents[1] / "app" / "migrations"
    old_paths = sorted(migrations.glob("00[1-5]_*.sql"))
    old_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in old_paths}
    assert len(old_paths) == 5
    try:
        # Build the actual LR1 schema before the new migration exists in its registry.
        async with pool.acquire() as connection, connection.transaction():
            await connection.execute(
                "CREATE TABLE labbi_schema_migrations "
                "(version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )
            for path in old_paths:
                await connection.execute(path.read_text(encoding="utf-8"))
                await connection.execute(
                    "INSERT INTO labbi_schema_migrations(version) VALUES ($1)", path.name
                )
        repository = DialogRepository(pool)
        alice = await repository.user(101)
        assert await repository.save_exchange(alice, 1, "Старый вопрос", "Старый ответ")
        applied_before = await pool.fetch(
            "SELECT version, applied_at FROM labbi_schema_migrations ORDER BY version"
        )
        await migrate(pool)
        await migrate(pool)
        assert await pool.fetchval("SELECT count(*) FROM labbi_schema_migrations") == 9
        assert (
            await pool.fetch(
                "SELECT version, applied_at FROM labbi_schema_migrations "
                "WHERE version < '006_agent_state.sql' ORDER BY version"
            )
            == applied_before
        )
        assert old_hashes == {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in old_paths
        }
        settings_row = await pool.fetchrow(
            "SELECT timezone, revision, confirmation_epoch FROM labbi_users WHERE user_id=101"
        )
        assert dict(settings_row) == {"timezone": None, "revision": 1, "confirmation_epoch": 0}
        alice = await repository.user(101)
        assert await repository.save_exchange(alice, 2, "Новый вопрос", "Новый ответ")
        assert (
            await pool.fetchval("SELECT confirmation_epoch FROM labbi_users WHERE user_id=101") == 0
        )

        await repository.user(202)
        request_id = await _insert_agent_request(pool, 202, 9)
        action = await _insert_agent_action(pool, 202, 9, request_id)
        await pool.execute(
            "UPDATE labbi_users SET timezone='Europe/Moscow', confirmation_epoch=1 "
            "WHERE user_id=202"
        )
        # Reset/mode preserve the ledger and zone while revoking the confirmation epoch.
        await repository.reset(202)
        await repository.change_mode(202, "translate")
        await pool.close()
        pool = await create_pool(settings)
        repository = DialogRepository(pool)
        assert await repository.history(101, 20) == [
            {"role": "user", "content": "Старый вопрос"},
            {"role": "assistant", "content": "Старый ответ"},
            {"role": "user", "content": "Новый вопрос"},
            {"role": "assistant", "content": "Новый ответ"},
        ]
        assert await pool.fetchval("SELECT id FROM labbi_agent_requests WHERE id=$1", request_id)
        assert await pool.fetchval("SELECT id FROM labbi_actions WHERE id=$1", action["id"])
        assert dict(
            await pool.fetchrow(
                "SELECT timezone, confirmation_epoch FROM labbi_users WHERE user_id=202"
            )
        ) == {"timezone": "Europe/Moscow", "confirmation_epoch": 3}
        assert (
            await pool.fetchval("SELECT state FROM labbi_actions WHERE id=$1", action["id"])
            == "cancelled"
        )
    finally:
        await pool.close()


async def test_agent_database_ownership_concurrent_uniqueness_and_atomic_receipt(database):
    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        repository = DialogRepository(pool)
        await repository.user(101)
        await repository.user(202)
        first_request = await _insert_agent_request(pool, 101, 1)
        second_request = await _insert_agent_request(pool, 101, 2)
        with pytest.raises(asyncpg.UniqueViolationError):
            await _insert_agent_request(pool, 101, 1)
        results = await asyncio.gather(
            _insert_agent_action(pool, 101, 1, first_request),
            _insert_agent_action(pool, 101, 2, second_request),
            return_exceptions=True,
        )
        assert sum(isinstance(result, asyncpg.UniqueViolationError) for result in results) == 1
        action = next(result for result in results if isinstance(result, asyncpg.Record))
        assert await pool.fetchval("SELECT count(*) FROM labbi_actions WHERE user_id=101") == 1
        remaining_message = 2 if action["telegram_message_id"] == 1 else 1
        remaining_request = second_request if remaining_message == 2 else first_request

        # A valid owner cannot claim a different owner's source request.
        with pytest.raises(asyncpg.ForeignKeyViolationError):
            await _insert_agent_action(pool, 202, remaining_message, remaining_request)
        with pytest.raises(asyncpg.CheckViolationError):
            await pool.execute(
                "INSERT INTO labbi_agent_requests(user_id, telegram_message_id, telegram_chat_id) "
                "VALUES (202, 20, 101)"
            )
        reminder_sql = (
            "INSERT INTO labbi_reminders(user_id, action_id, text, remind_at, timezone) "
            "VALUES ($1, $2, $3, $4, $5) RETURNING id"
        )
        canonical = [101, action["id"], action["text"], action["remind_at"], action["timezone"]]
        with pytest.raises(asyncpg.ForeignKeyViolationError):
            await pool.fetchval(reminder_sql, 202, *canonical[1:])
        with pytest.raises(asyncpg.ForeignKeyViolationError):
            await pool.fetchval(reminder_sql, *canonical[:2], "Подменённый текст", *canonical[3:])
        # Even valid parameters cannot leave a reminder behind for an uncommitted action.
        with pytest.raises(asyncpg.ForeignKeyViolationError):
            await pool.fetchval(reminder_sql, *canonical)
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders") == 0
        with pytest.raises(asyncpg.CheckViolationError):
            await pool.execute(
                "UPDATE labbi_actions SET state='committed' WHERE id=$1", action["id"]
            )

        await pool.execute(
            "UPDATE labbi_actions SET state='approved', approved_at=now(), "
            "capability_hash=$2, capability_issued_at=now() WHERE id=$1",
            action["id"],
            hashlib.sha256(b"isolated integration test capability").digest(),
        )
        # Receipt, reminder and final action state become visible together.
        async with pool.acquire() as connection, connection.transaction():
            reminder_id = await connection.fetchval(reminder_sql, *canonical)
            await connection.execute(
                "UPDATE labbi_actions SET state='committed', receipt_reminder_id=$2, "
                "finished_at=now() WHERE id=$1",
                action["id"],
                reminder_id,
            )
        repeated = await asyncio.gather(
            pool.fetchval(reminder_sql, *canonical),
            pool.fetchval(reminder_sql, *canonical),
            return_exceptions=True,
        )
        assert all(isinstance(result, asyncpg.UniqueViolationError) for result in repeated)
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders") == 1
        assert (
            await pool.fetchval(
                "SELECT receipt_reminder_id FROM labbi_actions WHERE id=$1", action["id"]
            )
            == reminder_id
        )
        with pytest.raises(asyncpg.ForeignKeyViolationError):
            await pool.execute(
                "UPDATE labbi_actions SET receipt_reminder_id=$2 WHERE id=$1", action["id"], uuid4()
            )
        # The terminal slot is released, while the old source cannot prepare anew.
        with pytest.raises(asyncpg.UniqueViolationError):
            await _insert_agent_action(
                pool, 101, action["telegram_message_id"], action["request_id"]
            )
        await _insert_agent_action(pool, 101, remaining_message, remaining_request)
        event_sql = (
            "INSERT INTO labbi_agent_events(user_id, request_id, action_id, decision, "
            "target, status, reason) VALUES ($1, $2, $3, 'confirm', 'add_reminder', 'OK', $4)"
        )
        with pytest.raises(asyncpg.ForeignKeyViolationError):
            await pool.execute(event_sql, 202, action["request_id"], action["id"], "Чужая запись")
        await pool.execute(
            event_sql, 101, action["request_id"], action["id"], "Напоминание сохранено"
        )
        await repository.reset(101)
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders WHERE user_id=101") == 1
        assert await pool.fetchval("SELECT count(*) FROM labbi_agent_events WHERE user_id=101") == 2
        assert await pool.fetchval("SELECT count(*) FROM labbi_agent_events WHERE user_id=202") == 0
    finally:
        await pool.close()


async def test_agent_database_rejects_inconsistent_action_and_audit_data(database):
    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        prepared_at = datetime.now(UTC)
        invalid_changes = [
            {"kind": "run_command"},
            {"state": "invented"},
            {"confirmation_epoch": -1},
            {"text": "   "},
            {"text": "x" * 501},
            {"timezone": ""},
            {"expires_at": prepared_at},
            {"expires_at": prepared_at + timedelta(seconds=301)},
            {"capability_hash": b"short"},
            {"capability_hash": b"x" * 32, "capability_issued_at": prepared_at},
            {"state": "approved"},
            {"state": "cancelled"},
            {
                "state": "approved",
                "approved_at": prepared_at,
                "capability_hash": b"x" * 32,
                "capability_issued_at": prepared_at + timedelta(seconds=300),
            },
        ]
        repository = DialogRepository(pool)
        for index, changes in enumerate(invalid_changes, start=1001):
            await repository.user(index)
            request_id = await _insert_agent_request(pool, index, 1)
            with pytest.raises(asyncpg.CheckViolationError):
                await _insert_agent_action(
                    pool, index, 1, request_id, prepared_at=prepared_at, **changes
                )
        await repository.user(101)
        for changes in (
            {"decision": "arbitrary"},
            {"status": "UNKNOWN"},
            {"reason": " "},
            {"reason": "x" * 501},
            {"argument_summary": "x" * 501},
            {"error_code": "unsafe text"},
            {"llm_calls": 3},
            {"mcp_calls": 2},
            {"duration_ms": -1},
            {"action_id": uuid4()},
        ):
            values = {"user_id": 101, "decision": "respond", "status": "OK", "reason": "Ответ"}
            values.update(changes)
            columns = ", ".join(values)
            parameters = ", ".join(f"${index}" for index in range(1, len(values) + 1))
            with pytest.raises(asyncpg.CheckViolationError):
                await pool.execute(
                    f"INSERT INTO labbi_agent_events({columns}) VALUES ({parameters})",
                    *values.values(),
                )
        with pytest.raises(asyncpg.CheckViolationError):
            await pool.execute("UPDATE labbi_users SET confirmation_epoch=-1 WHERE user_id=101")
        columns = await pool.fetch(
            "SELECT column_name FROM information_schema.columns WHERE table_name='labbi_actions'"
        )
        assert "capability_hash" in {row["column_name"] for row in columns}
        assert "capability" not in {row["column_name"] for row in columns}
    finally:
        await pool.close()


async def _confirmed_reminder(pool, user_id, *, timezone="Europe/Moscow", **changes):
    await DialogRepository(pool).user(user_id)
    await pool.execute("UPDATE labbi_users SET timezone=$2 WHERE user_id=$1", user_id, timezone)
    request_id = await _insert_agent_request(pool, user_id, 1)
    now = await pool.fetchval("SELECT clock_timestamp()")
    prepared_at = changes.pop("prepared_at", now)
    token = secrets.token_urlsafe(32)
    values = {
        "state": "approved",
        "prepared_at": prepared_at,
        "approved_at": prepared_at,
        "capability_issued_at": prepared_at,
        "capability_hash": hashlib.sha256(token.encode("ascii")).digest(),
        "timezone": timezone,
        "remind_at": now + timedelta(days=1),
    }
    values.update(changes)
    if values["state"] in {"cancelled", "expired", "failed"}:
        values.setdefault("finished_at", now)
    action = await _insert_agent_action(pool, user_id, 1, request_id, **values)
    arguments = ReminderArguments(
        text=action["text"],
        remind_at=action["remind_at"].astimezone(ZoneInfo(timezone)),
    )
    return action, arguments, {"action_id": str(action["id"]), "capability": token}


async def test_reminder_repository_concurrent_confirmation_and_restart_receipt(database):
    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        action, arguments, context = await _confirmed_reminder(pool, 101)
        repository = ReminderRepository(pool)
        first, second = await asyncio.gather(
            repository.add_reminder(arguments, context), repository.add_reminder(arguments, context)
        )
        assert first.root.status == second.root.status == "ok"
        assert first.root.reminder_id == second.root.reminder_id
        assert sorted([first.root.created, second.root.created]) == [False, True]
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders") == 1
        assert await pool.fetchval("SELECT count(*) FROM labbi_agent_events WHERE status='OK'") == 2
        assert (
            await pool.fetchval(
                "SELECT receipt_reminder_id FROM labbi_actions WHERE id=$1", action["id"]
            )
            == first.root.reminder_id
        )
        # A completed receipt survives expired TTL and a changed zone/epoch.
        await pool.execute(
            "UPDATE labbi_actions SET prepared_at=prepared_at-INTERVAL '1 hour', "
            "expires_at=expires_at-INTERVAL '1 hour', approved_at=approved_at-INTERVAL '1 hour', "
            "capability_issued_at=capability_issued_at-INTERVAL '1 hour' WHERE id=$1",
            action["id"],
        )
        await pool.execute(
            "UPDATE labbi_users SET timezone='Asia/Tokyo', confirmation_epoch=9 WHERE user_id=101"
        )
        await pool.close()
        pool = await create_pool(settings)
        repeated = await ReminderRepository(pool).add_reminder(arguments, context)
        assert repeated.root.status == "ok" and not repeated.root.created
        assert repeated.root.reminder_id == first.root.reminder_id
        assert repeated.root.timezone == "Europe/Moscow"
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders") == 1
        audit = await pool.fetch("SELECT argument_summary, reason FROM labbi_agent_events")
        assert all(context["capability"] not in str(dict(row)) for row in audit)
        assert all(arguments.text not in str(dict(row)) for row in audit)
        denied = await ReminderRepository(pool).add_reminder(
            arguments, {**context, "capability": secrets.token_urlsafe(32)}
        )
        assert denied.root.error.code == "UNAUTHORIZED"

        # A historical committed result is also readable after remind_at itself.
        now = await pool.fetchval("SELECT clock_timestamp()")
        old_action, old_arguments, old_context = await _confirmed_reminder(
            pool,
            202,
            prepared_at=now - timedelta(hours=2),
            remind_at=now - timedelta(hours=1),
        )
        async with pool.acquire() as connection, connection.transaction():
            old_receipt = await connection.fetchval(
                "INSERT INTO labbi_reminders(user_id, action_id, text, remind_at, timezone) "
                "VALUES ($1, $2, $3, $4, $5) RETURNING id",
                old_action["user_id"],
                old_action["id"],
                old_action["text"],
                old_action["remind_at"],
                old_action["timezone"],
            )
            await connection.execute(
                "UPDATE labbi_actions SET state='committed', receipt_reminder_id=$2, "
                "finished_at=$3 WHERE id=$1",
                old_action["id"],
                old_receipt,
                old_action["prepared_at"] + timedelta(seconds=1),
            )
        old_result = await ReminderRepository(pool).add_reminder(old_arguments, old_context)
        assert old_result.root.status == "ok" and not old_result.root.created
        assert old_result.root.reminder_id == old_receipt
    finally:
        await pool.close()


async def test_reminder_repository_rejects_unauthorized_stale_or_changed_actions(database):
    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        repository = ReminderRepository(pool)
        now = await pool.fetchval("SELECT clock_timestamp()")
        action, arguments, context = await _confirmed_reminder(pool, 101)
        checks = [
            (arguments, None, "UNAUTHORIZED"),
            (arguments, {**context, "capability": secrets.token_urlsafe(32)}, "UNAUTHORIZED"),
            (arguments, {**context, "action_id": str(uuid4())}, "UNAUTHORIZED"),
            (arguments, {**context, "user_id": 202}, "UNAUTHORIZED"),
            (
                ReminderArguments(text="Другой текст", remind_at=arguments.remind_at),
                context,
                "ACTION_CONFLICT",
            ),
            (
                ReminderArguments(
                    text=arguments.text, remind_at=arguments.remind_at + timedelta(hours=1)
                ),
                context,
                "ACTION_CONFLICT",
            ),
            (
                ReminderArguments(
                    text=arguments.text, remind_at=arguments.remind_at.astimezone(UTC)
                ),
                context,
                "INVALID_ARGUMENTS",
            ),
        ]
        for requested, supplied, expected_code in checks:
            result = await repository.add_reminder(requested, supplied)
            assert result.root.status == "error" and result.root.error.code == expected_code
        await pool.execute("UPDATE labbi_users SET confirmation_epoch=1 WHERE user_id=101")
        result = await repository.add_reminder(arguments, context)
        assert result.root.error.code == "ACTION_CANCELLED"
        await pool.execute(
            "UPDATE labbi_users SET confirmation_epoch=0, timezone=NULL WHERE user_id=101"
        )
        result = await repository.add_reminder(arguments, context)
        assert result.root.error.code == "TIMEZONE_REQUIRED"
        await pool.execute("UPDATE labbi_users SET timezone='Asia/Tokyo' WHERE user_id=101")
        result = await repository.add_reminder(arguments, context)
        assert result.root.error.code == "ACTION_CONFLICT"

        for user_id, state, expected_code in (
            (202, "cancelled", "ACTION_CANCELLED"),
            (303, "expired", "ACTION_EXPIRED"),
            (404, "failed", "ACTION_CONFLICT"),
        ):
            _, requested, supplied = await _confirmed_reminder(pool, user_id, state=state)
            result = await repository.add_reminder(requested, supplied)
            assert result.root.error.code == expected_code
        expired, requested, supplied = await _confirmed_reminder(
            pool, 505, prepared_at=now - timedelta(minutes=10)
        )
        result = await repository.add_reminder(requested, supplied)
        assert result.root.error.code == "ACTION_EXPIRED"
        assert (
            await pool.fetchval("SELECT state FROM labbi_actions WHERE id=$1", expired["id"])
            == "expired"
        )
        _, requested, supplied = await _confirmed_reminder(
            pool, 606, remind_at=now - timedelta(seconds=1)
        )
        result = await repository.add_reminder(requested, supplied)
        assert result.root.error.code == "INVALID_ARGUMENTS"
        await pool.execute(
            "UPDATE labbi_actions SET timezone='Unknown/Zone' WHERE id=$1", action["id"]
        )
        result = await repository.add_reminder(arguments, context)
        assert result.root.error.code == "INVALID_ARGUMENTS"
        _, requested, supplied = await _confirmed_reminder(
            pool,
            707,
            state="pending",
            approved_at=None,
            capability_issued_at=None,
            capability_hash=None,
        )
        result = await repository.add_reminder(requested, supplied)
        assert result.root.error.code == "UNAUTHORIZED"
        _, requested, supplied = await _confirmed_reminder(pool, 808)
        result = await repository.add_reminder(
            requested, {**supplied, "capability": context["capability"]}
        )
        assert result.root.error.code == "UNAUTHORIZED"
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders") == 0
    finally:
        await pool.close()


async def test_reminder_repository_unknown_commit_ack_recovers_same_action(database):
    settings, _, _, _ = database
    pool = await create_pool(settings)

    class LostCommitConnection:
        def __init__(self, connection):
            self.connection = connection

        def __getattr__(self, name):
            return getattr(self.connection, name)

        @asynccontextmanager
        async def transaction(self):
            async with self.connection.transaction():
                yield
            # A real commit succeeded, but its acknowledgement was lost in transport.
            raise asyncpg.ConnectionDoesNotExistError("private lost commit acknowledgement")

    class LostCommitPool:
        @asynccontextmanager
        async def acquire(self):
            async with pool.acquire() as connection:
                yield LostCommitConnection(connection)

    try:
        await migrate(pool)
        action, arguments, context = await _confirmed_reminder(pool, 101)
        result = await ReminderRepository(LostCommitPool()).add_reminder(arguments, context)
        assert result.root.status == "error" and result.root.error.code == "OUTCOME_UNKNOWN"
        assert "private" not in result.model_dump_json()
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders") == 1
        receipt = await pool.fetchval(
            "SELECT receipt_reminder_id FROM labbi_actions WHERE id=$1", action["id"]
        )
        recovered = await ReminderRepository(pool).add_reminder(arguments, context)
        assert recovered.root.status == "ok" and not recovered.root.created
        assert recovered.root.reminder_id == receipt
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders") == 1
    finally:
        await pool.close()


async def test_independent_stdio_client_commits_confirmed_reminder_in_real_postgres(database):
    settings, _, _, _ = database
    pool = await create_pool(settings)
    mcp_settings = MCPSettings(
        postgres_host=settings.postgres_host,
        postgres_port=settings.postgres_port,
        postgres_db=settings.postgres_db,
        postgres_user=settings.postgres_user,
        postgres_password=settings.postgres_password,
        schedule_file=str(ROOT / "data" / "schedule.json"),
    )
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp.server"],
        cwd=ROOT,
        env=mcp_settings.environment(),
    )

    async def database_pids():
        return {
            row["pid"]
            for row in await pool.fetch(
                "SELECT pid FROM pg_stat_activity WHERE datname=$1", settings.postgres_db
            )
        }

    try:
        await migrate(pool)
        action, arguments, context = await _confirmed_reminder(pool, 101)
        initial_pids = await database_pids()
        async with asyncio.timeout(25):
            async with Client(parameters, read_timeout_seconds=10) as client:
                tools = {tool.name for tool in (await client.list_tools()).tools}
                resources = {resource.uri for resource in (await client.list_resources()).resources}
                assert tools == {"get_weather", "get_schedule", "add_reminder", "search_materials"}
                assert resources == {"schedule://current-week", "materials://catalog"}
                raw_arguments = arguments.model_dump(mode="json")
                denied = await client.call_tool("add_reminder", raw_arguments)
                assert denied.is_error
                assert denied.structured_content["error"]["code"] == "UNAUTHORIZED"
                assert await pool.fetchval("SELECT count(*) FROM labbi_reminders") == 0
                first = await client.call_tool(
                    "add_reminder", raw_arguments, meta={"labbi/action-capability": context}
                )
                second = await client.call_tool(
                    "add_reminder", raw_arguments, meta={"labbi/action-capability": context}
                )
                assert not first.is_error and not second.is_error
                created = ReminderResult.model_validate(first.structured_content).root
                repeated = ReminderResult.model_validate(second.structured_content).root
                assert created.status == repeated.status == "ok"
                assert created.created and not repeated.created
                assert created.reminder_id == repeated.reminder_id
                assert context["capability"] not in first.model_dump_json()
                child_pids = await database_pids() - initial_pids
                assert child_pids
        # Closing the independent stdio process releases its own database pool.
        assert child_pids.isdisjoint(await database_pids())
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders") == 1
        assert (
            await pool.fetchval(
                "SELECT receipt_reminder_id FROM labbi_actions WHERE id=$1", action["id"]
            )
            == created.reminder_id
        )
        assert await pool.fetchval("SELECT count(*) FROM labbi_agent_events WHERE status='OK'") == 2
    finally:
        await pool.close()


async def test_agent_audit_is_concurrent_idempotent_isolated_and_survives_reset(database):
    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        repository = AgentRepository(pool)
        assert await repository.timezone(101) is None
        assert not await repository.has_processed(101, 50)
        dialog = DialogRepository(pool)
        alice = await dialog.user(101)
        assert await dialog.save_exchange(alice, 1, "История до сброса", "Сохранённый ответ")
        await pool.execute("UPDATE labbi_users SET timezone='Europe/Moscow' WHERE user_id=101")
        await asyncio.gather(
            *(
                repository.record(
                    101,
                    50,
                    decision="call_tool",
                    target="get_schedule",
                    status="OK",
                    reason="Расписание получено из источника.",
                    llm_calls=2,
                    mcp_calls=1,
                    duration_ms=50,
                )
                for _ in range(10)
            )
        )
        assert await pool.fetchval("SELECT count(*) FROM labbi_agent_requests") == 1
        assert await pool.fetchval("SELECT count(*) FROM labbi_agent_events") == 1
        assert await repository.has_processed(101, 50)
        assert not await repository.has_processed(202, 50)
        first_event = dict(
            await pool.fetchrow("SELECT * FROM labbi_agent_events WHERE user_id=101")
        )
        assert first_event["target"] == "get_schedule"
        assert first_event["llm_calls"] == 2 and first_event["mcp_calls"] == 1
        assert first_event["duration_ms"] == 50
        assert first_event["argument_summary"] == ""
        # A repeated final record cannot replace the first result or its counters.
        await repository.record(
            101,
            50,
            decision="respond",
            target=None,
            status="FAILED",
            reason="Повторное завершение отклонено.",
        )
        assert (
            dict(await pool.fetchrow("SELECT * FROM labbi_agent_events WHERE user_id=101"))
            == first_event
        )
        # Identical Telegram message numbers belong to separate private users.
        await repository.record(
            202,
            50,
            decision="clarify",
            target=None,
            status="CLARIFY",
            reason="Запрошено уточнение.",
        )
        await dialog.reset(101)
        await dialog.change_mode(101, "translate")
        assert await dialog.history(101, 20) == []
        await pool.close()
        pool = await create_pool(settings)
        repository = AgentRepository(pool)
        assert await repository.timezone(101) == "Europe/Moscow"
        assert await repository.timezone(202) is None
        assert await repository.has_processed(101, 50)
        assert await repository.has_processed(202, 50)
        rows = await pool.fetch(
            "SELECT user_id, telegram_message_id, status FROM labbi_agent_requests ORDER BY user_id"
        )
        assert [tuple(row) for row in rows] == [(101, 50, "completed"), (202, 50, "completed")]
        assert await pool.fetchval("SELECT count(*) FROM labbi_agent_events WHERE user_id=101") == 1
        assert await pool.fetchval("SELECT count(*) FROM labbi_agent_events WHERE user_id=202") == 1
        assert (
            dict(await pool.fetchrow("SELECT * FROM labbi_agent_events WHERE user_id=101"))
            == first_event
        )
    finally:
        await pool.close()


async def test_agent_audit_records_terminal_status_without_overwriting_prior_result(database):
    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        repository = AgentRepository(pool)
        for message_id, (status, expected_state) in enumerate(
            [
                ("OK", "completed"),
                ("CLARIFY", "completed"),
                ("REJECTED", "completed"),
                ("FAILED", "failed"),
                ("CANCELLED", "cancelled"),
                ("OUTCOME_UNKNOWN", "failed"),
            ],
            start=1,
        ):
            await repository.record(
                101,
                message_id,
                decision="respond",
                target=None,
                status=status,
                reason="Обработка запроса завершена.",
                error_code="TIMEOUT" if status == "FAILED" else None,
            )
            row = await pool.fetchrow(
                "SELECT status, decision, finished_at FROM labbi_agent_requests "
                "WHERE user_id=101 AND telegram_message_id=$1",
                message_id,
            )
            assert row["status"] == expected_state
            assert row["decision"] == "respond" and row["finished_at"] is not None
            await repository.record(
                101,
                message_id,
                decision="clarify",
                target=None,
                status="CLARIFY",
                reason="Поздний повтор не заменяет результат.",
            )
            assert (
                await pool.fetchval(
                    "SELECT status FROM labbi_agent_requests "
                    "WHERE user_id=101 AND telegram_message_id=$1",
                    message_id,
                )
                == expected_state
            )
        assert await pool.fetchval("SELECT count(*) FROM labbi_agent_requests") == 6
        assert await pool.fetchval("SELECT count(*) FROM labbi_agent_events") == 6
        await DialogRepository(pool).user(202)
        await _insert_agent_request(pool, 202, 1)
        assert not await repository.has_processed(202, 1)
    finally:
        await pool.close()


async def test_agent_audit_database_failure_rolls_back_request_and_event(database):
    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        await pool.execute(
            "CREATE FUNCTION reject_agent_audit() RETURNS trigger LANGUAGE plpgsql AS $$ "
            "BEGIN RAISE EXCEPTION 'injected database failure'; END $$; "
            "CREATE TRIGGER reject_agent_audit BEFORE INSERT ON labbi_agent_events "
            "FOR EACH ROW EXECUTE FUNCTION reject_agent_audit()"
        )
        repository = AgentRepository(pool)
        with pytest.raises(asyncpg.RaiseError):
            await repository.record(
                101,
                1,
                decision="respond",
                target=None,
                status="OK",
                reason="Подготовлен обычный ответ.",
            )
        assert await pool.fetchval("SELECT count(*) FROM labbi_agent_requests") == 0
        assert await pool.fetchval("SELECT count(*) FROM labbi_agent_events") == 0
        await pool.execute("DROP TRIGGER reject_agent_audit ON labbi_agent_events")
        await repository.record(
            101,
            1,
            decision="respond",
            target=None,
            status="OK",
            reason="Подготовлен обычный ответ.",
        )
        assert await pool.fetchval("SELECT count(*) FROM labbi_agent_requests") == 1
        assert await pool.fetchval("SELECT count(*) FROM labbi_agent_events") == 1
    finally:
        await pool.close()


async def test_started_request_survives_failed_audit_without_exposing_stale_why(database):
    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        repository = ConfirmationRepository(pool)
        await repository.record(
            101,
            1,
            decision="call_tool",
            target="get_weather",
            status="OK",
            reason="Погода получена из источника.",
            argument_summary="Город: Москва",
        )
        # Concurrent starts share the registry identity and do not manufacture events.
        await asyncio.gather(*(repository.start_request(101, 2) for _ in range(10)))
        assert await pool.fetchval("SELECT count(*) FROM labbi_agent_requests") == 2
        assert await pool.fetchval("SELECT count(*) FROM labbi_agent_events") == 1
        await pool.execute(
            "CREATE FUNCTION reject_new_audit() RETURNS trigger LANGUAGE plpgsql AS $$ "
            "BEGIN PERFORM pg_sleep(1); RETURN NEW; END $$; "
            "CREATE TRIGGER reject_new_audit BEFORE INSERT ON labbi_agent_events "
            "FOR EACH ROW EXECUTE FUNCTION reject_new_audit()"
        )
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.1):
                await repository.record(
                    101,
                    2,
                    decision="call_tool",
                    target="get_weather",
                    status="OK",
                    reason="Погода получена из источника.",
                    argument_summary="Город: Санкт-Петербург",
                )
        assert not await repository.has_processed(101, 2)
        assert await pool.fetchval("SELECT count(*) FROM labbi_agent_events") == 1
        with pytest.raises(ConfirmationError, match="Итог последнего запроса ещё не сохранён"):
            await repository.latest_event(101)
        # The missing-audit marker survives process restart and a history reset.
        await DialogRepository(pool).reset(101)
        await pool.close()
        pool = await create_pool(settings)
        repository = ConfirmationRepository(pool)
        with pytest.raises(ConfirmationError, match="Итог последнего запроса ещё не сохранён"):
            await repository.latest_event(101)
        assert await repository.latest_event(202) is None
        await pool.execute("DROP TRIGGER reject_new_audit ON labbi_agent_events")
        await repository.record(
            101,
            2,
            decision="call_tool",
            target="get_weather",
            status="OK",
            reason="Погода получена из источника.",
            argument_summary="Город: Санкт-Петербург",
            llm_calls=2,
            mcp_calls=1,
        )
        event = await repository.latest_event(101)
        assert event.argument_summary == "Город: Санкт-Петербург"
        assert (event.llm_calls, event.mcp_calls) == (2, 1)
        assert await pool.fetchval("SELECT count(*) FROM labbi_agent_events") == 2
        await repository.start_request(101, 2)
        assert await repository.has_processed(101, 2)
        assert await repository.latest_event(101) == event
        # An older unfinished marker must not hide a newer, successfully audited request.
        await repository.start_request(101, 3)
        await repository.record(
            101,
            4,
            decision="clarify",
            target=None,
            status="CLARIFY",
            reason="Нужно уточнить запрос.",
        )
        assert (await repository.latest_event(101)).status == "CLARIFY"
        # Starts overlap: A starts, B starts, A completes, then B's audit times out.
        # A's event timestamp is newer than B's start, but it still describes A.
        await repository.start_request(101, 5)
        await repository.start_request(101, 6)
        await repository.record(
            101,
            5,
            decision="call_tool",
            target="get_weather",
            status="OK",
            reason="Погода получена из источника.",
            argument_summary="Город: Москва",
        )
        await pool.execute(
            "CREATE TRIGGER reject_new_audit BEFORE INSERT ON labbi_agent_events "
            "FOR EACH ROW EXECUTE FUNCTION reject_new_audit()"
        )
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.1):
                await repository.record(
                    101,
                    6,
                    decision="call_tool",
                    target="get_weather",
                    status="OK",
                    reason="Погода получена из источника.",
                    argument_summary="Город: Санкт-Петербург",
                )
        with pytest.raises(ConfirmationError, match="Итог последнего запроса ещё не сохранён"):
            await repository.latest_event(101)
    finally:
        await pool.close()


async def test_agent_dialog_real_stdio_and_postgres_keep_bounded_calls_and_plain_history(database):
    settings, _, _, _ = database
    settings = replace(
        settings, schedule_file=str(ROOT / "data" / "schedule.json"), llm_tool_mode="structured"
    )
    pool = await create_pool(settings)
    mcp = MCPClient(settings)
    mcp.call_tool = AsyncMock(wraps=mcp.call_tool)
    mcp.read_resource = AsyncMock(wraps=mcp.read_resource)
    decisions = [
        {
            "action": "call_tool",
            "name": "get_schedule",
            "arguments": {"date": "2026-10-12"},
            "reason": "private-model-reason-do-not-store",
        },
        {
            "action": "read_resource",
            "uri": "schedule://current-week",
            "reason": "private-model-reason-do-not-store",
        },
    ]

    async def compose(messages, _temperature, *, disable_tools):
        assert disable_tools is True
        value = json.loads(messages[-1]["content"])["untrusted_mcp_result"]
        assert value["source"] == "local_schedule"
        assert value["timezone"] == "Europe/Moscow"
        if "date" in value:
            assert value["status"] == "ok" and value["lessons"]
            return Completion(f"Занятия на {value['date']}. Источник: учебное расписание.")
        assert len(value["days"]) == 7
        return Completion(
            f"Неделя с {value['week_start']} по {value['week_end']}. Источник: учебное расписание."
        )

    llm = SimpleNamespace(
        decide=AsyncMock(
            side_effect=[
                RoutingCompletion(text=json.dumps(decision), tool_calls=[])
                for decision in decisions
            ]
        ),
        complete=AsyncMock(side_effect=compose),
    )
    try:
        await migrate(pool)
        await mcp.start()
        assert mcp.available
        repository = DialogRepository(pool)
        audit = AgentRepository(pool)
        agent = AgentService(
            llm, mcp, audit, settings, now=lambda: datetime(2026, 10, 18, 9, tzinfo=UTC)
        )
        dialog = DialogService(repository, llm, settings, agent=agent)
        first = await dialog.respond(101, 1, "Покажи занятия 12 октября")
        assert first.startswith("Занятия на 2026-10-12.")
        assert await dialog.respond(101, 1, "Покажи занятия 12 октября") == first
        assert llm.decide.await_count == llm.complete.await_count == 1
        second = await dialog.respond(101, 2, "Покажи расписание текущей недели")
        assert second.startswith("Неделя с ")
        assert llm.decide.await_count == llm.complete.await_count == 2
        mcp.call_tool.assert_awaited_once_with("get_schedule", {"date": "2026-10-12"})
        mcp.read_resource.assert_awaited_once_with("schedule://current-week")
        history = await repository.history(101, 20)
        assert history == [
            {"role": "user", "content": "Покажи занятия 12 октября"},
            {"role": "assistant", "content": first},
            {"role": "user", "content": "Покажи расписание текущей недели"},
            {"role": "assistant", "content": second},
        ]
        events = await pool.fetch(
            "SELECT target, llm_calls, mcp_calls, reason FROM labbi_agent_events "
            "WHERE user_id=101 ORDER BY id"
        )
        assert [(row["target"], row["llm_calls"], row["mcp_calls"]) for row in events] == [
            ("get_schedule", 2, 1),
            ("schedule://current-week", 2, 1),
        ]
        assert all("private-model-reason" not in row["reason"] for row in events)
        await repository.reset(101)
        with pytest.raises(ModelError, match="уже обработан"):
            await dialog.respond(101, 1, "Покажи занятия 12 октября")
        assert llm.decide.await_count == llm.complete.await_count == 2
        assert await pool.fetchval("SELECT count(*) FROM labbi_agent_events") == 2
        assert await repository.history(101, 20) == []
    finally:
        await mcp.close()
        await pool.close()


async def test_reminder_repository_rolls_back_write_when_audit_fails(database):
    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        action, arguments, context = await _confirmed_reminder(pool, 101)
        await pool.execute(
            "CREATE FUNCTION reject_reminder_audit() RETURNS trigger LANGUAGE plpgsql AS $$ "
            "BEGIN RAISE EXCEPTION 'private injected diagnostics'; END $$; "
            "CREATE TRIGGER reject_reminder_audit BEFORE INSERT ON labbi_agent_events "
            "FOR EACH ROW EXECUTE FUNCTION reject_reminder_audit()"
        )
        result = await ReminderRepository(pool).add_reminder(arguments, context)
        assert result.root.status == "error" and result.root.error.code == "SOURCE_UNAVAILABLE"
        assert "private" not in result.model_dump_json()
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders") == 0
        row = await pool.fetchrow(
            "SELECT state, receipt_reminder_id FROM labbi_actions WHERE id=$1", action["id"]
        )
        assert dict(row) == {"state": "approved", "receipt_reminder_id": None}
        assert await pool.fetchval("SELECT count(*) FROM labbi_agent_events") == 0
    finally:
        await pool.close()


async def _prepare_confirmation(pool, user_id, message_id=1, *, timezone="Europe/Moscow"):
    dialog = DialogRepository(pool)
    confirmations = ConfirmationRepository(pool)
    await dialog.user(user_id)
    await confirmations.set_timezone(user_id, timezone)
    user = await dialog.user(user_id)
    now = await confirmations.now()
    arguments = ReminderArguments(
        text="Отправить личный отчёт",
        remind_at=(now + timedelta(days=1)).astimezone(ZoneInfo(timezone)),
    )
    return await confirmations.prepare(
        user, message_id, "Напомни отправить личный отчёт", arguments, timezone=timezone
    )


async def _age_confirmation(pool, action_id):
    # Advance the same persisted card past its TTL without waiting five minutes.
    # All relative timestamps move together to retain actual schema constraints.
    await pool.execute(
        "UPDATE labbi_actions SET prepared_at=prepared_at-INTERVAL '1 hour', "
        "expires_at=expires_at-INTERVAL '1 hour', approved_at=approved_at-INTERVAL '1 hour', "
        "capability_issued_at=capability_issued_at-INTERVAL '1 hour' WHERE id=$1",
        action_id,
    )


async def test_confirmation_prepare_is_concurrent_atomic_and_separate_from_dialog_revision(
    database,
):
    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        dialog = DialogRepository(pool)
        confirmations = ConfirmationRepository(pool)
        await dialog.user(101)
        await confirmations.set_timezone(101, "Europe/Moscow")
        user = await dialog.user(101)
        arguments = ReminderArguments(
            text="Сдать отчёт",
            remind_at=(await confirmations.now() + timedelta(days=1)).astimezone(
                ZoneInfo("Europe/Moscow")
            ),
        )
        cards = await asyncio.gather(
            *(confirmations.prepare(user, 10, "Напомни сдать отчёт", arguments) for _ in range(8))
        )
        assert len({card.action_id for card in cards}) == 1
        assert sum(not card.replayed for card in cards) == 1
        card = cards[0]
        row = await pool.fetchrow("SELECT * FROM labbi_actions WHERE id=$1", card.action_id)
        assert row["expires_at"] - row["prepared_at"] == timedelta(minutes=5)
        assert row["state"] == "pending" and row["capability_hash"] is None
        assert await pool.fetchval("SELECT count(*) FROM labbi_actions") == 1
        assert await pool.fetchval("SELECT count(*) FROM labbi_agent_requests") == 1
        assert await pool.fetchval("SELECT count(*) FROM labbi_agent_events") == 1
        assert await dialog.history(101, 20) == [
            {"role": "user", "content": "Напомни сдать отчёт"},
            {"role": "assistant", "content": card.card_text},
        ]
        assert (await dialog.user(101)).revision == user.revision + 1
        restored = await confirmations.lookup_card(101, 10)
        assert restored.action_id == card.action_id and restored.replayed
        assert restored.card_text == card.card_text
        assert await confirmations.lookup_card(202, 10) is None
        assert (await confirmations.approve(202, card.action_id)).code == "UNAUTHORIZED"
        assert (await confirmations.cancel(202, card.action_id)).code == "UNAUTHORIZED"
        with pytest.raises(ConfirmationError) as conflict:
            await confirmations.prepare(
                user,
                10,
                "Подмена исходного запроса",
                ReminderArguments(text="Подменённый отчёт", remind_at=arguments.remind_at),
            )
        assert conflict.value.code == "ACTION_CONFLICT"

        epoch = await pool.fetchval("SELECT confirmation_epoch FROM labbi_users WHERE user_id=101")
        current = await dialog.user(101)
        assert await dialog.save_exchange(current, 11, "Привет", "Привет!")
        assert (await dialog.user(101)).revision == current.revision + 1
        assert (
            await pool.fetchval("SELECT confirmation_epoch FROM labbi_users WHERE user_id=101")
            == epoch
        )
        assert (await confirmations.lookup_card(101, 10)).action_id == card.action_id
        assert isinstance(await confirmations.approve(101, card.action_id), Authorization)
        with pytest.raises(ConfirmationError) as stale:
            await confirmations.prepare(current, 12, "Устаревший запрос", arguments)
        assert stale.value.code == "ACTION_CANCELLED"
        assert await pool.fetchval("SELECT count(*) FROM labbi_actions") == 1
    finally:
        await pool.close()


async def test_confirmation_preparation_checks_saved_zone_db_time_and_dst(database):
    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        dialog = DialogRepository(pool)
        confirmations = ConfirmationRepository(pool)
        user = await dialog.user(101)
        now = await confirmations.now()
        future = ReminderArguments(
            text="Отчёт", remind_at=(now + timedelta(days=1)).astimezone(ZoneInfo("Europe/Moscow"))
        )
        with pytest.raises(ConfirmationError) as missing:
            await confirmations.prepare(user, 1, "Напомни об отчёте", future)
        assert missing.value.code == "TIMEZONE_REQUIRED"
        await confirmations.set_timezone(101, "Europe/Moscow")
        user = await dialog.user(101)
        cases = [
            (
                ReminderArguments(
                    text="Отчёт", remind_at=now.astimezone(ZoneInfo("Europe/Moscow"))
                ),
                None,
                "INVALID_ARGUMENTS",
            ),
            (
                ReminderArguments(text="Отчёт", remind_at=future.remind_at.astimezone(UTC)),
                None,
                "INVALID_ARGUMENTS",
            ),
            (future, "Asia/Tokyo", "ACTION_CONFLICT"),
        ]
        for message_id, (arguments, zone, code) in enumerate(cases, start=2):
            with pytest.raises(ConfirmationError) as rejected:
                await confirmations.prepare(
                    user, message_id, "Напомни об отчёте", arguments, timezone=zone
                )
            assert rejected.value.code == code

        await confirmations.set_timezone(101, "Europe/Berlin")
        user = await dialog.user(101)
        # The last Sunday in March/October next year is always in the future.
        for message_id, month in [(10, 3), (11, 10)]:
            last_day = datetime(now.year + 1, month, 31, 2, 30)
            sunday = last_day - timedelta(days=(last_day.weekday() + 1) % 7)
            arguments = ReminderArguments(
                text="Перевод часов", remind_at=sunday.replace(tzinfo=ZoneInfo("Europe/Berlin"))
            )
            with pytest.raises(ConfirmationError) as ambiguous:
                await confirmations.prepare(user, message_id, "Напомни", arguments)
            assert ambiguous.value.code == "AMBIGUOUS_TIME"
        assert await pool.fetchval("SELECT count(*) FROM labbi_actions") == 0
        assert await dialog.history(101, 20) == []
        assert await pool.fetchval("SELECT count(*) FROM labbi_agent_requests") == 0
    finally:
        await pool.close()


async def test_confirmation_capability_rotates_after_restart_and_receipt_survives_expiry(database):
    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        card = await _prepare_confirmation(pool, 101)
        confirmations = ConfirmationRepository(pool)
        first = await confirmations.approve(101, card.action_id)
        assert isinstance(first, Authorization)
        token = first.context["capability"]
        # 32 random bytes are encoded without padding; only the SHA-256 digest persists.
        assert len(token) == 43
        first_row = await pool.fetchrow("SELECT * FROM labbi_actions WHERE id=$1", card.action_id)
        assert first_row["capability_hash"] == hashlib.sha256(token.encode("ascii")).digest()
        assert token not in str(dict(first_row)) and token not in card.card_text
        assert (await confirmations.reconcile(202, card.action_id)).code == "UNAUTHORIZED"
        unknown = await confirmations.reconcile(101, card.action_id)
        assert unknown.status == "unknown" and unknown.code == "OUTCOME_UNKNOWN"
        event = await confirmations.latest_event(101)
        assert event.status == "OUTCOME_UNKNOWN" and event.error_code == "OUTCOME_UNKNOWN"
        assert await confirmations.latest_event(202) is None
        explanations = ConfirmationService(confirmations, SimpleNamespace())
        why = await explanations.why(101)
        assert "результат пока неизвестен" in why and "add_reminder" in why
        assert "Europe/Moscow" in why and card.arguments.text not in why and token not in why
        assert await explanations.why(202) == "Пока нет запросов с решением агента."

        await pool.close()
        pool = await create_pool(settings)
        confirmations = ConfirmationRepository(pool)
        resumed = await confirmations.lookup_card(101, 1)
        assert resumed.action_id == card.action_id and resumed.state == "approved"
        second = await confirmations.approve(101, card.action_id)
        assert isinstance(second, Authorization)
        assert second.context["capability"] != token
        second_row = await pool.fetchrow("SELECT * FROM labbi_actions WHERE id=$1", card.action_id)
        assert second_row["expires_at"] == first_row["expires_at"]
        assert second_row["approved_at"] == first_row["approved_at"]
        assert second_row["capability_issued_at"] >= first_row["capability_issued_at"]
        stale = await ReminderRepository(pool).add_reminder(first.arguments, first.context)
        assert stale.root.error.code == "UNAUTHORIZED"
        results = await asyncio.gather(
            *(
                ReminderRepository(pool).add_reminder(second.arguments, second.context)
                for _ in range(6)
            )
        )
        assert all(result.root.status == "ok" for result in results)
        assert sum(result.root.created for result in results) == 1
        receipt = results[0].root.reminder_id
        assert {result.root.reminder_id for result in results} == {receipt}
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders") == 1
        await _age_confirmation(pool, card.action_id)
        await confirmations.set_timezone(101, "Asia/Tokyo")
        await DialogRepository(pool).reset(101)
        for method in (confirmations.approve, confirmations.reconcile, confirmations.cancel):
            outcome = await method(101, card.action_id)
            assert isinstance(outcome, Outcome) and outcome.status == "committed"
            assert outcome.reminder_id == receipt and outcome.timezone == "Europe/Moscow"
        assert await confirmations.lookup_card(101, 1) is None
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders") == 1
        audit = await pool.fetch("SELECT reason, argument_summary FROM labbi_agent_events")
        assert all(token not in str(dict(row)) for row in audit)
        assert all(second.context["capability"] not in str(dict(row)) for row in audit)
        assert all(card.arguments.text not in str(dict(row)) for row in audit)
    finally:
        await pool.close()


async def test_confirmation_revocation_covers_pending_and_approved_cards_and_expiry(database):
    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        confirmations = ConfirmationRepository(pool)
        dialog = DialogRepository(pool)
        user_id = 1000
        for approved in (False, True):
            for operation in ("cancel", "stop", "reset", "mode", "zone", "new_card", "expiry"):
                user_id += 1
                card = await _prepare_confirmation(pool, user_id)
                authorization = (
                    await confirmations.approve(user_id, card.action_id) if approved else None
                )
                before = await pool.fetchrow(
                    "SELECT revision, confirmation_epoch FROM labbi_users WHERE user_id=$1", user_id
                )
                if operation == "cancel":
                    assert (
                        await confirmations.cancel(user_id, card.action_id)
                    ).status == "cancelled"
                elif operation == "stop":
                    assert await confirmations.invalidate(user_id)
                elif operation == "reset":
                    await dialog.reset(user_id)
                elif operation == "mode":
                    await dialog.change_mode(user_id, "translate")
                elif operation == "zone":
                    await confirmations.set_timezone(user_id, "Asia/Tokyo")
                elif operation == "new_card":
                    replacement = await _prepare_confirmation(pool, user_id, 2)
                    assert replacement.action_id != card.action_id
                else:
                    await _age_confirmation(pool, card.action_id)
                outcome = await confirmations.approve(user_id, card.action_id)
                assert outcome.code == (
                    "ACTION_EXPIRED" if operation == "expiry" else "ACTION_CANCELLED"
                )
                assert await confirmations.lookup_card(user_id, 1) is None
                after = await pool.fetchrow(
                    "SELECT revision, confirmation_epoch FROM labbi_users WHERE user_id=$1", user_id
                )
                assert after["confirmation_epoch"] == before["confirmation_epoch"] + (
                    operation != "expiry"
                )
                if authorization is not None:
                    rejected = await ReminderRepository(pool).add_reminder(
                        authorization.arguments, authorization.context
                    )
                    assert rejected.root.error.code == outcome.code
                assert await pool.fetchval(
                    "SELECT EXISTS (SELECT 1 FROM labbi_agent_events WHERE action_id=$1 "
                    "AND status IN ('CANCELLED', 'REJECTED'))",
                    card.action_id,
                )
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders") == 0
    finally:
        await pool.close()


async def test_reminder_dialog_confirmation_real_stdio_postgres_and_no_extra_model_calls(database):
    settings, _, _, _ = database
    settings = replace(
        settings, schedule_file=str(ROOT / "data" / "schedule.json"), llm_tool_mode="structured"
    )
    pool = await create_pool(settings)
    mcp = MCPClient(settings)
    mcp.call_tool = AsyncMock(wraps=mcp.call_tool)
    mcp.read_resource = AsyncMock(wraps=mcp.read_resource)
    try:
        await migrate(pool)
        now = await pool.fetchval("SELECT clock_timestamp()")
        moment = (now + timedelta(days=1)).astimezone(ZoneInfo("Europe/Moscow"))
        private_reason = "private-reminder-model-reason-do-not-store"
        private_text = "Личный отчёт для лабораторной"
        question = f"Напомни: {private_text} {moment.isoformat()}"

        def source(quote):
            return {"source": "user", "message_index": 0, "quote": quote}

        decisions = [
            {
                "action": "call_tool",
                "name": "add_reminder",
                "arguments": {"text": private_text, "remind_at": moment.isoformat()},
                "evidence": {
                    "request": source("Напомни"),
                    "text": source(private_text),
                    "date": source(moment.isoformat()),
                    "time": source(moment.isoformat()),
                },
                "reason": private_reason,
            },
            {"action": "use_confirmation", "reason": private_reason},
            {"action": "respond", "text": "Привет!", "reason": private_reason},
        ]
        llm = SimpleNamespace(
            decide=AsyncMock(
                side_effect=[
                    RoutingCompletion(text=json.dumps(decision), tool_calls=[])
                    for decision in decisions
                ]
            ),
            complete=AsyncMock(
                side_effect=AssertionError("No final LLM call for reminder or callback")
            ),
        )
        await mcp.start()
        assert mcp.available
        repository = DialogRepository(pool)
        confirmations = ConfirmationRepository(pool)
        service = ConfirmationService(confirmations, mcp)
        agent = AgentService(llm, mcp, AgentRepository(pool), settings, now=lambda: now)
        dialog = DialogService(repository, llm, settings, agent=agent, confirmations=service)
        assert "Europe/Moscow" in await dialog.timezone(101, "Europe/Moscow")
        card_text = await dialog.respond(101, 1, question)
        card = await dialog.confirmation_card(101, 1)
        assert card is not None and card.state == "pending"
        assert card.card_text == card_text
        assert format_datetime(moment, "Europe/Moscow") in card_text and private_text in card_text
        assert "ещё не сохранено" in card_text and "Europe/Moscow" in card_text
        assert await dialog.respond(101, 1, question) == card_text
        assert (await dialog.confirmation_card(101, 1)).action_id == card.action_id
        assert llm.decide.await_count == 1
        llm.complete.assert_not_awaited()
        mcp.call_tool.assert_not_awaited()
        mcp.read_resource.assert_not_awaited()
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders") == 0
        epoch = await pool.fetchval("SELECT confirmation_epoch FROM labbi_users WHERE user_id=101")
        assert (await confirmations.current_card(101)).action_id == card.action_id
        assert await confirmations.current_card(202) is None
        button_reply = await dialog.respond(101, 3, "давай")
        assert "Подтвердить" in button_reply
        assert "не сохранено" in button_reply
        assert await dialog.respond(101, 3, "давай") == button_reply
        assert (await confirmations.current_card(101)).action_id == card.action_id
        assert llm.decide.await_count == 2
        mcp.call_tool.assert_not_awaited()
        assert await pool.fetchval("SELECT count(*) FROM labbi_actions") == 1
        assert await dialog.respond(101, 2, "Привет") == "Привет!"
        assert (
            await pool.fetchval("SELECT confirmation_epoch FROM labbi_users WHERE user_id=101")
            == epoch
        )
        assert "недоступно" in await dialog.confirm(202, str(card.action_id))
        mcp.call_tool.assert_not_awaited()

        answer = await dialog.confirm(101, str(card.action_id))
        assert "Напоминание сохранено." in answer
        assert format_datetime(moment, "Europe/Moscow") in answer and "Europe/Moscow" in answer
        assert "Доставка выполняется из очереди" in answer
        assert llm.decide.await_count == 3
        llm.complete.assert_not_awaited()
        mcp.call_tool.assert_awaited_once()
        name, arguments = mcp.call_tool.await_args.args
        context = mcp.call_tool.await_args.kwargs["action_context"]
        assert name == "add_reminder" and arguments == {
            "text": private_text,
            "remind_at": moment.isoformat(),
        }
        assert set(context) == {"action_id", "capability"}
        assert context["action_id"] == str(card.action_id) and len(context["capability"]) >= 43
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders") == 1
        assert await dialog.confirm(101, str(card.action_id)) == answer
        mcp.call_tool.assert_awaited_once()
        mcp.read_resource.assert_not_awaited()
        assert await repository.history(101, 20) == [
            {"role": "user", "content": question},
            {"role": "assistant", "content": card_text},
            {"role": "user", "content": "давай"},
            {"role": "assistant", "content": button_reply},
            {"role": "user", "content": "Привет"},
            {"role": "assistant", "content": "Привет!"},
        ]
        audit = await pool.fetch(
            "SELECT decision, target, status, reason, argument_summary, llm_calls, mcp_calls "
            "FROM labbi_agent_events WHERE user_id=101 ORDER BY id"
        )
        assert sum(row["llm_calls"] for row in audit) == 3
        assert sum(row["mcp_calls"] for row in audit) == 1
        assert all(private_reason not in str(dict(row)) for row in audit)
        assert all(private_text not in str(dict(row)) for row in audit)
        assert all(context["capability"] not in str(dict(row)) for row in audit)
        why = await dialog.why(101)
        assert "add_reminder" in why and "успешно" in why
        assert private_reason not in why and private_text not in why
        assert context["capability"] not in why
        assert await dialog.why(202) == "Пока нет запросов с решением агента."

        await _age_confirmation(pool, card.action_id)
        await dialog.invalidate_confirmations(101)
        await dialog.reset(101)
        assert await dialog.confirm(101, str(card.action_id)) == answer
        assert await dialog.confirmation_card(101, 1) is None
        assert await confirmations.current_card(101) is None
        assert await repository.history(101, 20) == []
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders") == 1
        mcp.call_tool.assert_awaited_once()
        assert llm.decide.await_count == 3
        llm.complete.assert_not_awaited()
    finally:
        await mcp.close()
        await pool.close()


async def test_confirmation_prepare_rolls_back_history_action_and_revocation_together(database):
    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        first = await _prepare_confirmation(pool, 101)
        dialog = DialogRepository(pool)
        before_user = await pool.fetchrow("SELECT * FROM labbi_users WHERE user_id=101")
        before_history = await dialog.history(101, 20)
        await pool.execute(
            "CREATE FUNCTION reject_confirmation_history() RETURNS trigger LANGUAGE plpgsql AS $$ "
            "BEGIN RAISE EXCEPTION 'injected failure after action preparation'; END $$; "
            "CREATE TRIGGER reject_confirmation_history BEFORE INSERT ON labbi_messages "
            "FOR EACH ROW WHEN (NEW.role = 'assistant') "
            "EXECUTE FUNCTION reject_confirmation_history()"
        )
        with pytest.raises(asyncpg.RaiseError):
            await _prepare_confirmation(pool, 101, 2)
        assert await pool.fetchrow("SELECT * FROM labbi_users WHERE user_id=101") == before_user
        assert await dialog.history(101, 20) == before_history
        assert await pool.fetchval("SELECT count(*) FROM labbi_actions") == 1
        assert await pool.fetchval("SELECT count(*) FROM labbi_agent_requests") == 1
        assert await pool.fetchval("SELECT count(*) FROM labbi_agent_events") == 1
        assert (
            await pool.fetchval("SELECT state FROM labbi_actions WHERE id=$1", first.action_id)
            == "pending"
        )
        await pool.execute("DROP TRIGGER reject_confirmation_history ON labbi_messages")
        replacement = await _prepare_confirmation(pool, 101, 2)
        assert replacement.action_id != first.action_id
        assert await pool.fetchval("SELECT count(*) FROM labbi_actions") == 2
        assert len(await dialog.history(101, 20)) == 4
    finally:
        await pool.close()


async def test_timezone_menu_only_applies_after_owner_confirmation_and_persists(database):
    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        card = await _prepare_confirmation(pool, 101)
        repository = DialogRepository(pool)
        confirmations = ConfirmationRepository(pool)
        mcp = SimpleNamespace(call_tool=AsyncMock())
        service = ConfirmationService(confirmations, mcp)
        lookup = SimpleNamespace(
            search=AsyncMock(return_value=[TimezoneOption("Япония — Токио", "Asia/Tokyo")])
        )
        llm = SimpleNamespace(complete=AsyncMock(), decide=AsyncMock())
        ui = TimezoneUI(service, lookup)
        dialog = DialogService(repository, llm, settings, confirmations=service, timezone_ui=ui)
        original_user = await pool.fetchrow("SELECT * FROM labbi_users WHERE user_id=101")
        original_history = await repository.history(101, 20)

        choices = await dialog.timezone_menu(101, "Япония")
        assert isinstance(choices, TimezoneView)
        pick = choices.rows[0][0].callback_data
        preview = await dialog.timezone_callback(101, pick)
        assert "Asia/Tokyo" in preview.text and "UTC+09:00" in preview.text
        use = preview.rows[0][0].callback_data
        assert dialog.timezone_callback_is_apply(101, use)
        assert not dialog.timezone_callback_is_apply(202, use)
        assert isinstance(await dialog.timezone_callback(202, use), str)
        assert await pool.fetchrow("SELECT * FROM labbi_users WHERE user_id=101") == original_user
        assert (await confirmations.current_card(101)).action_id == card.action_id
        assert await repository.history(101, 20) == original_history

        receipt = await dialog.timezone_callback(101, use)
        assert receipt.dismiss_previous and "Asia/Tokyo" in receipt.text
        assert await ConfirmationRepository(pool).timezone(101) == "Asia/Tokyo"
        assert (
            await pool.fetchval("SELECT state FROM labbi_actions WHERE id=$1", card.action_id)
            == "cancelled"
        )
        assert await confirmations.current_card(101) is None
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders") == 0
        assert await repository.history(101, 20) == original_history
        epoch = await pool.fetchval("SELECT confirmation_epoch FROM labbi_users WHERE user_id=101")
        assert epoch == original_user["confirmation_epoch"] + 1
        assert isinstance(await dialog.timezone_callback(101, use), str)
        assert "Asia/Tokyo" in await dialog.timezone_menu(101, "Asia/Tokyo")
        assert (
            await pool.fetchval("SELECT confirmation_epoch FROM labbi_users WHERE user_id=101")
            == epoch
        )
        await dialog.timezone_menu(101)
        await dialog.invalidate_confirmations(101)
        assert isinstance(await dialog.timezone_callback(101, pick), str)
        llm.complete.assert_not_awaited()
        llm.decide.assert_not_awaited()
        mcp.call_tool.assert_not_awaited()
    finally:
        await pool.close()


async def test_confirmation_services_race_across_pools_commits_once_and_replays_without_mcp(
    database,
):
    settings, _, _, _ = database
    pools = [await create_pool(settings), await create_pool(settings)]
    try:
        await migrate(pools[0])
        card = await _prepare_confirmation(pools[0], 101)
        barrier = asyncio.Barrier(6)
        results = []

        async def call_tool(pool, name, arguments, *, action_context):
            assert name == "add_reminder"
            # Every callback obtains its own capability before any server write.
            # Independent pools model two app processes without a shared asyncio.Lock.
            await barrier.wait()
            result = await ReminderRepository(pool).add_reminder(
                ReminderArguments.model_validate(arguments), action_context
            )
            results.append(result)
            return result.model_dump(mode="json")

        mcps = [SimpleNamespace(call_tool=AsyncMock()) for _ in pools]
        for mcp, pool in zip(mcps, pools, strict=True):

            async def invoke(name, arguments, *, action_context, pool=pool):
                return await call_tool(pool, name, arguments, action_context=action_context)

            mcp.call_tool.side_effect = invoke
        services = [
            ConfirmationService(ConfirmationRepository(pool), mcp)
            for pool, mcp in zip(pools, mcps, strict=True)
        ]
        replies = await asyncio.wait_for(
            asyncio.gather(
                *(services[index % 2].confirm(101, card.action_id) for index in range(6))
            ),
            timeout=15,
        )
        assert len(results) == 6
        assert sum(result.root.status == "ok" for result in results) == 1
        assert sum(result.root.status == "error" for result in results) == 5
        assert {result.root.error.code for result in results if result.root.status == "error"} == {
            "UNAUTHORIZED"
        }
        assert sum(mcp.call_tool.await_count for mcp in mcps) == 6
        assert await pools[0].fetchval("SELECT count(*) FROM labbi_actions") == 1
        assert await pools[0].fetchval("SELECT count(*) FROM labbi_reminders") == 1
        receipt = await pools[0].fetchrow("SELECT * FROM labbi_reminders")
        assert receipt["action_id"] == card.action_id and receipt["user_id"] == 101
        assert all(
            "Напоминание сохранено." in reply or "Результат записи пока неизвестен" in reply
            for reply in replies
        )
        # Recover each callback from the same durable receipt; no new server call.
        recovered = await asyncio.gather(
            *(services[index % 2].confirm(101, card.action_id) for index in range(6))
        )
        assert len(set(recovered)) == 1
        assert str(receipt["id"]) in recovered[0] and "Напоминание сохранено." in recovered[0]
        assert sum(mcp.call_tool.await_count for mcp in mcps) == 6
        assert await pools[1].fetchval("SELECT count(*) FROM labbi_reminders") == 1
    finally:
        await asyncio.gather(*(pool.close() for pool in pools))


async def test_confirmation_database_failure_rolls_back_permission_and_every_revocation(database):
    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        confirmations = ConfirmationRepository(pool)
        dialog = DialogRepository(pool)
        cases = []
        for approved in (False, True):
            for operation in ("approve", "cancel", "stop", "reset", "mode", "zone"):
                user_id = 1000 + len(cases)
                card = await _prepare_confirmation(pool, user_id)
                if approved:
                    assert isinstance(
                        await confirmations.approve(user_id, card.action_id), Authorization
                    )
                cases.append((user_id, card, operation))

        async def snapshot(user_id, card):
            return (
                await pool.fetchrow("SELECT * FROM labbi_users WHERE user_id=$1", user_id),
                await pool.fetchrow("SELECT * FROM labbi_actions WHERE id=$1", card.action_id),
                await dialog.history(user_id, 20),
                await pool.fetch(
                    "SELECT * FROM labbi_agent_requests WHERE user_id=$1 ORDER BY id", user_id
                ),
                await pool.fetch(
                    "SELECT * FROM labbi_agent_events WHERE user_id=$1 ORDER BY id", user_id
                ),
            )

        async def apply(user_id, card, operation):
            if operation == "approve":
                return await confirmations.approve(user_id, card.action_id)
            if operation == "cancel":
                return await confirmations.cancel(user_id, card.action_id)
            if operation == "stop":
                return await confirmations.invalidate(user_id)
            if operation == "reset":
                return await dialog.reset(user_id)
            if operation == "mode":
                return await dialog.change_mode(user_id, "translate")
            return await confirmations.set_timezone(user_id, "Asia/Tokyo")

        await pool.execute(
            "CREATE FUNCTION reject_stage6_confirmation_audit() RETURNS trigger "
            "LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'private injected storage failure'; "
            "END $$; CREATE TRIGGER reject_stage6_confirmation_audit "
            "BEFORE INSERT ON labbi_agent_events FOR EACH ROW "
            "EXECUTE FUNCTION reject_stage6_confirmation_audit()"
        )
        for user_id, card, operation in cases:
            before = await snapshot(user_id, card)
            with pytest.raises(asyncpg.RaiseError):
                await apply(user_id, card, operation)
            assert await snapshot(user_id, card) == before
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders") == 0
        await pool.execute("DROP TRIGGER reject_stage6_confirmation_audit ON labbi_agent_events")
        for user_id, card, operation in cases:
            result = await apply(user_id, card, operation)
            state = await pool.fetchval(
                "SELECT state FROM labbi_actions WHERE id=$1", card.action_id
            )
            if operation == "approve":
                assert isinstance(result, Authorization) and state == "approved"
            else:
                assert state == "cancelled"
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders") == 0
    finally:
        await pool.close()


async def test_confirmation_database_restart_preserves_zone_actions_receipts_and_private_audit(
    database,
):
    settings, docker, wait_until_ready, name = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        confirmations = ConfirmationRepository(pool)
        cards = {}
        zones = {101: "Europe/Moscow", 202: "Asia/Tokyo", 303: "Europe/Berlin"}
        for user_id, zone in zones.items():
            cards[user_id] = await _prepare_confirmation(pool, user_id, timezone=zone)
        old_authorization = await confirmations.approve(202, cards[202].action_id)
        committed = await confirmations.approve(303, cards[303].action_id)
        result = await ReminderRepository(pool).add_reminder(committed.arguments, committed.context)
        assert result.root.status == "ok" and result.root.created
        receipt = result.root.reminder_id
        before_events = {user_id: await confirmations.latest_event(user_id) for user_id in zones}
        before_why = {
            user_id: await ConfirmationService(confirmations, SimpleNamespace()).why(user_id)
            for user_id in zones
        }
        await pool.close()
        await asyncio.to_thread(docker, "stop", name)
        await asyncio.to_thread(docker, "start", name)
        await asyncio.to_thread(wait_until_ready)
        pool = await create_pool(settings)
        confirmations = ConfirmationRepository(pool)
        mcp = SimpleNamespace(call_tool=AsyncMock())
        service = ConfirmationService(confirmations, mcp)
        for user_id, zone in zones.items():
            assert await confirmations.timezone(user_id) == zone
            assert await confirmations.latest_event(user_id) == before_events[user_id]
            assert await service.why(user_id) == before_why[user_id]
            history = await DialogRepository(pool).history(user_id, 20)
            assert len(history) == 2 and history[-1]["content"] == cards[user_id].card_text
        for user_id, state in ((101, "pending"), (202, "approved")):
            restored = await confirmations.current_card(user_id)
            assert restored.action_id == cards[user_id].action_id and restored.state == state
            assert restored.expires_at == cards[user_id].expires_at
        assert await confirmations.current_card(303) is None
        restored_authorization = await confirmations.approve(202, cards[202].action_id)
        assert isinstance(restored_authorization, Authorization)
        assert restored_authorization.context != old_authorization.context
        rejected = await ReminderRepository(pool).add_reminder(
            old_authorization.arguments, old_authorization.context
        )
        assert rejected.root.error.code == "UNAUTHORIZED"
        repeated = await service.confirm(303, cards[303].action_id)
        assert "Напоминание сохранено." in repeated and str(receipt) in repeated
        mcp.call_tool.assert_not_awaited()
        assert await confirmations.timezone(404) is None
        assert await confirmations.current_card(404) is None
        assert await confirmations.lookup_card(404, 1) is None
        assert await confirmations.latest_event(404) is None
        assert await service.why(404) == "Пока нет запросов с решением агента."
        for method in (confirmations.approve, confirmations.cancel, confirmations.reconcile):
            for card in cards.values():
                assert (await method(404, card.action_id)).code == "UNAUTHORIZED"
        assert await pool.fetchval("SELECT count(*) FROM labbi_actions") == 3
        assert await pool.fetchval("SELECT count(*) FROM labbi_agent_requests") == 3
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders") == 1
        assert await pool.fetchval("SELECT count(*) FROM labbi_messages") == 6
    finally:
        await pool.close()


async def test_quiz_structured_transitions_persist_across_pools_and_isolate_users(database):
    settings, _, _, _ = database
    pool = await create_pool(settings)
    llm = SimpleNamespace(decide=AsyncMock(), complete=AsyncMock())

    def turn(kind, text, *, feedback=""):
        return RoutingCompletion(
            json.dumps({"kind": kind, "feedback": feedback, "text": text}, ensure_ascii=False),
            [],
        )

    try:
        await migrate(pool)
        repository = DialogRepository(pool)
        dialog = DialogService(repository, llm, settings)
        await dialog.mode(101, "quiz")
        await dialog.mode(202, "quiz")

        llm.decide.return_value = turn("clarify", "Назови тему по программированию.")
        personal = await dialog.respond(101, 1, "мое имя")
        assert personal == "Уточнение для тренировки\n\nНазови тему по программированию."
        alice = await repository.user(101)
        assert (alice.quiz_question, alice.quiz_topic, alice.quiz_finished) == (0, None, False)

        llm.decide.return_value = turn("question", "Что делает SELECT?")
        await dialog.respond(202, 1, "SQL")
        bob = await repository.user(202)
        bob_history = await repository.history(202, 20)
        await pool.close()
        pool = await create_pool(settings)
        repository = DialogRepository(pool)
        dialog = DialogService(repository, llm, settings)
        assert await repository.user(101) == alice
        assert await repository.answer(101, 1) == personal

        llm.decide.return_value = turn("question", "Вопрос 1/5\n\nЧто делает len?")
        first = await dialog.respond(101, 2, "Python")
        assert first == "Вопрос 1/5\n\nЧто делает len?"
        alice = await repository.user(101)
        assert (alice.quiz_question, alice.quiz_topic, alice.quiz_finished) == (1, "Python", False)
        calls = llm.decide.await_count
        assert await dialog.respond(101, 2, "Python") == first
        assert llm.decide.await_count == calls
        assert await repository.user(101) == alice

        for message_id, kind, request in (
            (3, "clarify", "А что за объект?"),
            (4, "hint", "Подскажи"),
        ):
            llm.decide.return_value = turn(kind, "Вспомни последовательности.")
            await dialog.respond(101, message_id, request)
            alice = await repository.user(101)
            assert (alice.quiz_question, alice.quiz_topic, alice.quiz_finished) == (
                1,
                "Python",
                False,
            )

        history = await repository.history(101, 20)
        before = await pool.fetchrow("SELECT * FROM labbi_users WHERE user_id=101")
        for message_id, result in (
            (5, turn("finished", "Этот итог пока недопустим.")),
            (6, RoutingCompletion('{"kind":"question","feedback":"","text":"Обрыв', [])),
        ):
            llm.decide.return_value = result
            with pytest.raises(ModelError):
                await dialog.respond(101, message_id, "Ответ")
            assert await pool.fetchrow("SELECT * FROM labbi_users WHERE user_id=101") == before
            assert await repository.history(101, 20) == history
            assert await repository.answer(101, message_id) is None

        assert await repository.user(202) == bob
        assert await repository.history(202, 20) == bob_history
        assert await pool.fetchval("SELECT count(*) FROM labbi_messages WHERE user_id=101") == 8
        assert await pool.fetchval("SELECT count(*) FROM labbi_messages WHERE user_id=202") == 2
        assert llm.decide.await_count == 7
        llm.complete.assert_not_awaited()
        for call in llm.decide.call_args_list:
            assert call.kwargs["decision_schema"]["additionalProperties"] is False
    finally:
        await pool.close()


async def test_lab3_traces_survive_pool_restart_isolate_owners_and_reject_foreign_parents(database):
    from app.routing.contracts import TraceSpan
    from app.tracing.repository import TraceRepository
    from app.tracing.runtime import TraceManager, span

    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        repository = TraceRepository(pool)
        async with TraceManager(repository).request(101) as first:
            async with span("schedule", "agent", target="schedule"):
                async with span("schedule", "tool_call", tool_name="get_schedule"):
                    pass
        async with TraceManager(repository).request(202) as foreign:
            pass
        assert await repository.read(202, first.trace_id) is None
        assert await repository.read(101, foreign.trace_id) is None
        row, spans = await repository.read(101, first.trace_id)
        assert row["status"] == "ok" and len(spans) == 3
        parent = next(s for s in spans if s.parent_span_id is None)
        forged = TraceSpan.model_validate(
            {
                **parent.model_dump(),
                "span_id": uuid4(),
                "trace_id": foreign.trace_id,
                "request_id": foreign.request_id,
                "parent_span_id": parent.span_id,
            }
        )
        with pytest.raises(asyncpg.ForeignKeyViolationError):
            await repository.span(forged)
        await pool.close()
        pool = await create_pool(settings)
        repository = TraceRepository(pool)
        assert (await repository.read(101))[0]["id"] == first.trace_id
        output = await repository.display(101, str(first.trace_id))
        assert "schedule/tool_call get_schedule: ok" in output
        assert "не найдена" in await repository.display(202, str(first.trace_id))
        assert "Используй" in await repository.display(101, "invalid")
        abandoned, request = uuid4(), uuid4()
        await repository.start(101, abandoned, request, datetime.now(UTC) - timedelta(minutes=11))
        assert (await repository.read(101, abandoned))[0]["status"] == "interrupted"
        metadata = await pool.fetch("SELECT data FROM labbi_trace_spans")
        assert all("actor_id" not in r["data"] and "capability" not in r["data"] for r in metadata)
    finally:
        await pool.close()


async def test_lab3_supervisor_keeps_confirmation_guard_during_router_and_callback_has_no_llm(
    database,
):
    from app.mcp.contracts import tool_schemas
    from app.routing.policy import ScopedMCP
    from app.routing.supervisor import Supervisor
    from app.tracing.repository import TraceRepository
    from app.tracing.runtime import TraceManager

    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        repository = DialogRepository(pool)
        await repository.user(101)
        confirmations_repo = ConfirmationRepository(pool)
        await confirmations_repo.set_timezone(101, "Europe/Moscow")
        now = (await pool.fetchval("SELECT clock_timestamp()")).replace(microsecond=0)
        when = (now.astimezone(ZoneInfo("Europe/Moscow")) + timedelta(days=1)).replace(
            hour=11, minute=0, second=0
        )
        text = f"Напомни {when.date()} в 11:00 сдать отчёт"
        route = RoutingCompletion(
            json.dumps({"target": "schedule", "reason": "schedule_request", "confidence": 0.9}), []
        )
        reminder = RoutingCompletion(
            json.dumps(
                {
                    "action": "call_tool",
                    "name": "add_reminder",
                    "reason": "Prepare",
                    "arguments": {"text": "Сдать отчёт", "remind_at": when.isoformat()},
                    "evidence": {
                        field: {"source": "user", "message_index": 0, "quote": text}
                        for field in ("request", "text", "date", "time")
                    },
                }
            ),
            [],
        )
        llm = SimpleNamespace(decide=AsyncMock(side_effect=[route, reminder]), complete=AsyncMock())
        writer = ReminderRepository(pool)

        async def actual_write(name, arguments, *, action_context):
            assert name == "add_reminder"
            result = await writer.add_reminder(
                ReminderArguments.model_validate(arguments), action_context
            )
            return result.model_dump(mode="json")

        mcp = SimpleNamespace(
            available=True,
            tools={
                name: {
                    "name": name,
                    "description": name,
                    **tool_schemas(name),
                }
                for name in ("get_weather", "get_schedule", "add_reminder")
            },
            resources={},
            call_tool=AsyncMock(side_effect=actual_write),
            read_resource=AsyncMock(),
        )
        confirmation = ConfirmationService(
            confirmations_repo, ScopedMCP(mcp, "schedule", confirmation=True)
        )
        traces = TraceManager(TraceRepository(pool))
        dialog = DialogService(
            repository,
            llm,
            settings,
            agent=Supervisor(llm, mcp, AgentRepository(pool), settings, now=lambda: now),
            confirmations=confirmation,
            traces=traces,
        )
        async with traces.request(101):
            answer = await dialog.respond(101, 1, text)
        assert "Подтверждения" not in answer
        card = await confirmations_repo.current_card(101)
        assert card is not None and card.arguments.remind_at == when
        assert llm.decide.await_count == 2 and llm.complete.await_count == 0
        mcp.call_tool.assert_not_awaited()
        started, release = asyncio.Event(), asyncio.Event()

        async def delayed_router(*args, **kwargs):
            started.set()
            await release.wait()
            return RoutingCompletion(
                json.dumps(
                    {"target": "legacy_study", "reason": "general_study", "confidence": 0.8}
                ),
                [],
            )

        llm.decide.side_effect = delayed_router
        llm.complete.return_value = Completion("Привет!")
        task = asyncio.create_task(dialog.respond(101, 2, "Привет"))
        try:
            await asyncio.wait_for(started.wait(), 2)
            denied = await confirmations_repo.approve(101, card.action_id)
            assert isinstance(denied, Outcome) and denied.code == "ACTION_CONFLICT"
            mcp.call_tool.assert_not_awaited()
        finally:
            release.set()
            await task
        assert (await confirmations_repo.current_card(101)).action_id == card.action_id
        counts = (llm.decide.await_count, llm.complete.await_count)
        async with traces.request(101):
            receipt = await dialog.confirm(101, str(card.action_id))
        assert "Напоминание сохранено" in receipt
        assert (llm.decide.await_count, llm.complete.await_count) == counts
        mcp.call_tool.assert_awaited_once()
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders") == 1
        again = await dialog.confirm(101, str(card.action_id))
        assert "Напоминание сохранено" in again
        mcp.call_tool.assert_awaited_once()
        _, spans = await traces.repository.read(101)
        assert [(s.tool_name, s.operation) for s in spans if s.tool_name] == [
            ("add_reminder", "tool_call")
        ]
    finally:
        await pool.close()


async def test_lab3_knowledge_real_stdio_and_postgres_persist_grounded_answer_and_trace(database):
    from app.agents.knowledge import KnowledgeAgent
    from app.materials.contracts import SearchArguments
    from app.materials.corpus import Corpus
    from app.routing.supervisor import Supervisor
    from app.tracing.repository import TraceRepository
    from app.tracing.runtime import TraceManager

    settings, _, _, _ = database
    pool = await create_pool(settings)
    mcp = MCPClient(settings)
    try:
        await migrate(pool)
        await mcp.start()
        assert mcp.available
        knowledge = await KnowledgeAgent.initialize(mcp)
        assert knowledge.catalog is not None
        query = "контекстное окно"
        match = Corpus.load().search(SearchArguments(query=query)).matches[0]
        llm = SimpleNamespace(
            decide=AsyncMock(
                side_effect=[
                    RoutingCompletion(
                        json.dumps(
                            {
                                "target": "knowledge",
                                "reason": "materials_request",
                                "confidence": 0.9,
                            }
                        ),
                        [],
                    ),
                    RoutingCompletion(
                        json.dumps({"action": "search_materials", "arguments": {"query": query}}),
                        [],
                    ),
                ]
            ),
            complete=AsyncMock(
                return_value=Completion(
                    json.dumps(
                        {
                            "answer": "Контекстное окно ограничено токенами.",
                            "citations": [
                                {"document_id": match.document_id, "quote": match.snippet[:90]}
                            ],
                        }
                    )
                )
            ),
        )
        traces = TraceManager(TraceRepository(pool))
        dialog = DialogService(
            DialogRepository(pool),
            llm,
            settings,
            agent=Supervisor(llm, mcp, AgentRepository(pool), settings, knowledge=knowledge),
            confirmations=ConfirmationService(ConfirmationRepository(pool), mcp),
            traces=traces,
        )
        async with traces.request(101) as trace:
            answer = await dialog.respond(101, 1, "По материалам объясни контекстное окно")
        assert "[llm-context]" in answer and "Источники:" in answer
        assert llm.decide.await_count == 2 and llm.complete.await_count == 1
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders") == 0
        assert await pool.fetchval("SELECT count(*) FROM labbi_actions") == 0
        assert (await AgentRepository(pool).has_processed(101, 1)) is True
        await pool.close()
        pool = await create_pool(settings)
        assert await DialogRepository(pool).answer(101, 1) == answer
        assert await DialogRepository(pool).answer(202, 1) is None
        row, spans = await TraceRepository(pool).read(101, trace.trace_id)
        assert row["status"] == "ok"
        assert [s.tool_name for s in spans if s.tool_name] == ["search_materials"]
        assert sum(s.operation.startswith("model_") for s in spans) == 3
        assert any("llm-context" in s.document_ids for s in spans)
        assert await TraceRepository(pool).read(202, trace.trace_id) is None
    finally:
        await mcp.close()
        await pool.close()


async def test_lab3_city_menu_ownership_restart_expiry_and_no_connection_during_lookup(
    database,
):
    from test_notifications import CITY

    from app.notifications.city import EXPIRED, CityUI
    from app.notifications.repository import NotificationRepository

    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        await pool.close()
        pool = await asyncpg.create_pool(
            host=settings.postgres_host,
            port=settings.postgres_port,
            database=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
            min_size=1,
            max_size=1,
            command_timeout=5,
        )
        repo = NotificationRepository(pool)
        await DialogRepository(pool).user(101)
        assert await repo.profile(101) is None  # Migration never guesses a private destination.
        with pytest.raises(ValueError):
            await repo.register_destination(101, 202)
        await repo.register_destination(101, 101)
        await ConfirmationRepository(pool).set_timezone(101, "Europe/Moscow")
        token, version = await repo.start_menu(101)
        assert await repo.fill_menu(101, token, version, [CITY])
        assert (await repo.profile(101))["city"] is None
        assert await repo.choose(101, token, 0) == CITY
        assert (await repo.profile(101))["city"] is None
        assert await repo.confirm_city(202, token, 0) is None
        assert await asyncio.wait_for(repo.confirm_city(101, token, 0), 2) == CITY
        assert await repo.confirm_city(101, token, 0) is None
        profile = await repo.profile(101)
        assert json.loads(profile["city"]) == CITY.model_dump(mode="json")
        assert profile["timezone"] == "Europe/Moscow" and not profile["briefing_enabled"]
        assert profile["version"] > version
        await pool.close()
        pool = await create_pool(settings)
        repo = NotificationRepository(pool)
        assert (await repo.profile(101))["city"] == profile["city"]
        old, v = await repo.start_menu(101)
        assert await repo.fill_menu(101, old, v, [CITY])
        await repo.choose(101, old, 0)
        await ConfirmationRepository(pool).set_timezone(101, "Asia/Tokyo")
        assert await repo.confirm_city(101, old, 0) is None
        expired, v = await repo.start_menu(101)
        await repo.fill_menu(101, expired, v, [CITY])
        await repo.choose(101, expired, 0)
        await pool.execute(
            "UPDATE labbi_city_menus SET expires_at=clock_timestamp()-INTERVAL '1 second' "
            "WHERE user_id=101"
        )
        assert await repo.confirm_city(101, expired, 0) is None
        assert not await repo.fill_menu(101, expired, v, [CITY])
        started, release = asyncio.Event(), asyncio.Event()

        async def lookup(query):
            started.set()
            await release.wait()
            return [CITY]

        task = asyncio.create_task(CityUI(repo, SimpleNamespace(search=lookup)).open(101, "Казань"))
        await asyncio.wait_for(started.wait(), 2)
        assert pool.get_idle_size() == pool.get_size()
        await repo.clear_menu(101)
        release.set()
        assert await task == EXPIRED
        assert json.loads((await repo.profile(101))["city"]) == CITY.model_dump(mode="json")
    finally:
        await pool.close()


async def test_lab3_briefing_opt_in_requires_confirmed_profile_and_outbox_is_unique_and_revocable(
    database,
):
    from test_notifications import CITY

    from app.notifications.repository import NotificationRepository

    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        repo = NotificationRepository(pool)
        await repo.register_destination(101, 101)
        with pytest.raises(ConfirmationError) as missing:
            await repo.set_briefing(101, True)
        assert missing.value.code == "TIMEZONE_REQUIRED"
        await ConfirmationRepository(pool).set_timezone(101, "Europe/Moscow")
        with pytest.raises(ConfirmationError):
            await repo.set_briefing(101, True)
        token, v = await repo.start_menu(101)
        await repo.fill_menu(101, token, v, [CITY])
        assert await repo.confirm_city(101, token, 0) is None  # Preview selection is mandatory.
        await repo.choose(101, token, 0)
        await repo.confirm_city(101, token, 0)
        moment = await repo.set_briefing(101, True)
        profile = await repo.profile(101)
        assert moment > profile["consented_at"]
        await repo.set_briefing(101, True)
        assert (await repo.profile(101))["version"] == profile["version"]
        assert await pool.fetchval("SELECT count(*) FROM labbi_outbox") == 0
        day = moment.astimezone(ZoneInfo(profile["timezone"])).date()
        events = await asyncio.gather(
            *(
                repo.enqueue_briefing(101, day, expected_version=profile["version"])
                for _ in range(12)
            )
        )
        assert len(set(events)) == 1 and events[0] is not None
        assert (
            await repo.enqueue_briefing(101, day, expected_version=profile["version"]) == events[0]
        )
        assert await pool.fetchval("SELECT count(*) FROM labbi_outbox") == 1
        event = await repo.event(101, events[0])
        assert event["state"] == "pending" and event["body"] is None and event["attempts"] == 0
        assert event["deadline_at"] - event["due_at"] == timedelta(hours=1)
        assert await repo.event(202, events[0]) is None
        assert (
            await repo.enqueue_briefing(101, day, expected_version=profile["version"] - 1) is None
        )
        # Turning off cancels an in-progress preparation and fences its old lease.
        await pool.execute(
            "UPDATE labbi_outbox SET state='preparing',"
            "lease_until=clock_timestamp()+INTERVAL '150 seconds',"
            "lease_token=$2,llm_reserved=1 WHERE id=$1",
            events[0],
            uuid4(),
        )
        await repo.set_briefing(101, False)
        cancelled = await repo.event(101, events[0])
        assert cancelled["state"] == "cancelled" and cancelled["lease_token"] is None
        assert cancelled["llm_reserved"] == 1
        assert await repo.enqueue_briefing(101, day, expected_version=profile["version"]) is None
        await repo.set_briefing(101, True)
        updated = await repo.profile(101)
        assert (
            await repo.enqueue_briefing(101, day, expected_version=updated["version"]) == events[0]
        )
        assert (await repo.event(101, events[0]))[
            "state"
        ] == "cancelled"  # A spent budget is never reset.
    finally:
        await pool.close()


async def test_lab3_reminder_outbox_is_atomic_idempotent_and_legacy_addresses_are_not_guessed(
    database,
):
    from app.notifications.repository import NotificationRepository

    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        repo = NotificationRepository(pool)
        action, arguments, capability = await _confirmed_reminder(pool, 101)
        writer = ReminderRepository(pool)
        receipt = (await writer.add_reminder(arguments, capability)).root
        assert receipt.status == "ok"
        assert await pool.fetchval("SELECT count(*) FROM labbi_outbox") == 0
        await repo.register_destination(101, 101)
        await repo.register_destination(101, 101)
        again = (await writer.add_reminder(arguments, capability)).root
        assert again.reminder_id == receipt.reminder_id and not again.created
        assert await pool.fetchval("SELECT count(*) FROM labbi_outbox") == 1
        row = await pool.fetchrow("SELECT * FROM labbi_outbox")
        assert row["destination_chat_id"] == 101 and row["reminder_id"] == receipt.reminder_id
        assert row["deadline_at"] - row["due_at"] == timedelta(minutes=15)
        await repo.register_destination(303, 303)
        action2, args2, cap2 = await _confirmed_reminder(pool, 303)
        await pool.execute(
            "ALTER TABLE labbi_outbox ADD CONSTRAINT simulate_outbox_failure "
            "CHECK(kind<>'reminder') NOT VALID"
        )
        denied = (await writer.add_reminder(args2, cap2)).root
        assert denied.status == "error" and denied.error.code == "SOURCE_UNAVAILABLE"
        assert await pool.fetchval("SELECT count(*) FROM labbi_reminders WHERE user_id=303") == 0
        assert (
            await pool.fetchval("SELECT state FROM labbi_actions WHERE id=$1", action2["id"])
            == "approved"
        )
        await pool.execute("ALTER TABLE labbi_outbox DROP CONSTRAINT simulate_outbox_failure")
        assert (await writer.add_reminder(args2, cap2)).root.status == "ok"
        assert await pool.fetchval("SELECT count(*) FROM labbi_outbox WHERE user_id=303") == 1
    finally:
        await pool.close()


async def test_lab3_city_cancel_confirmation_race_and_profile_change_fence_old_menus(database):
    from test_notifications import CITY

    from app.notifications.repository import NotificationRepository

    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        repo = NotificationRepository(pool)
        await repo.register_destination(101, 101)
        for _ in range(8):
            token, v = await repo.start_menu(101)
            await repo.fill_menu(101, token, v, [CITY])
            await repo.choose(101, token, 0)
            saved, cancelled = await asyncio.wait_for(
                asyncio.gather(repo.confirm_city(101, token, 0), repo.cancel_menu(101, token)), 3
            )
            assert (saved == CITY and not cancelled) or (saved is None and cancelled)
        token, v = await repo.start_menu(101)
        await repo.fill_menu(101, token, v, [CITY])
        await repo.choose(101, token, 0)
        await ConfirmationRepository(pool).set_timezone(101, "Asia/Tokyo")
        assert await repo.confirm_city(101, token, 0) is None
        assert not await repo.fill_menu(101, token, v, [CITY])
        for change in (
            DialogRepository(pool).reset,
            lambda owner: DialogRepository(pool).change_mode(owner, "translate"),
        ):
            token, v = await repo.start_menu(101)
            await repo.fill_menu(101, token, v, [CITY])
            await repo.choose(101, token, 0)
            await change(101)
            assert await repo.confirm_city(101, token, 0) is None
    finally:
        await pool.close()


async def test_lab3_profile_change_cancels_prepared_briefing_but_not_started_send(
    database,
):
    from test_notifications import CITY

    from app.notifications.repository import NotificationRepository

    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        repo = NotificationRepository(pool)

        async def setup(owner):
            await repo.register_destination(owner, owner)
            await ConfirmationRepository(pool).set_timezone(owner, "Europe/Moscow")
            token, v = await repo.start_menu(owner)
            await repo.fill_menu(owner, token, v, [CITY])
            await repo.choose(owner, token, 0)
            await repo.confirm_city(owner, token, 0)
            moment = await repo.set_briefing(owner, True)
            profile = await repo.profile(owner)
            day = moment.astimezone(ZoneInfo(profile["timezone"])).date()
            return await repo.enqueue_briefing(owner, day, expected_version=profile["version"])

        first = await setup(101)
        await pool.execute(
            "UPDATE labbi_outbox SET state='ready',body='Already prepared' WHERE id=$1", first
        )
        token, v = await repo.start_menu(101)
        changed = CITY.model_copy(update={"latitude": CITY.latitude + 1.0})
        await repo.fill_menu(101, token, v, [changed])
        await repo.choose(101, token, 0)
        await repo.confirm_city(101, token, 0)
        assert (await repo.event(101, first))["state"] == "cancelled"
        assert (await repo.event(101, first))["body"] == "Already prepared"
        assert (await repo.profile(101))["briefing_enabled"]
        second = await setup(202)
        await pool.execute(
            "UPDATE labbi_outbox SET state='sending',body='Sending',attempts=1,"
            "lease_token=$2,lease_until=clock_timestamp()+INTERVAL '60 seconds' WHERE id=$1",
            second,
            uuid4(),
        )
        await repo.set_briefing(202, False)
        event = await repo.event(202, second)
        assert event["state"] == "sending"  # An external send may already be in flight.
        assert event["profile_version"] < (await repo.profile(202))["version"]
        for statement in (
            "UPDATE labbi_outbox SET attempts=4 WHERE id=$1",
            "UPDATE labbi_outbox SET lease_token=NULL WHERE id=$1",
            "UPDATE labbi_outbox SET body=NULL WHERE id=$1",
            "UPDATE labbi_outbox SET deadline_at=due_at+INTERVAL '1 minute' WHERE id=$1",
            "UPDATE labbi_outbox SET next_attempt_at=due_at-INTERVAL '1 second' WHERE id=$1",
        ):
            with pytest.raises(asyncpg.CheckViolationError):
                await pool.execute(statement, second)
    finally:
        await pool.close()


class DeliveryClock:
    def __init__(self, now):
        self.value = now

    def __call__(self):
        return self.value


async def _delivery_seed(pool, owner=101, *, due=None):
    from app.notifications.repository import NotificationRepository

    due = due or ((await pool.fetchval("SELECT clock_timestamp()")) + timedelta(minutes=2))
    await NotificationRepository(pool).register_destination(owner, owner)
    _, arguments, context = await _confirmed_reminder(pool, owner, remind_at=due)
    result = (await ReminderRepository(pool).add_reminder(arguments, context)).root
    assert result.status == "ok"
    row = await pool.fetchrow("SELECT * FROM labbi_outbox WHERE reminder_id=$1", result.reminder_id)
    return row, DeliveryClock(due)


async def test_reminder_delivery_real_db_once_no_connection_during_http_and_durable_receipt(
    database,
):
    from app.notifications.delivery_repository import DeliveryRepository
    from app.notifications.worker import ReminderWorker
    from app.tracing.repository import TraceRepository
    from app.tracing.runtime import TraceManager

    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        row, clock = await _delivery_seed(pool)

        async def send(**kwargs):
            assert pool.get_idle_size() == pool.get_size()
            assert kwargs["chat_id"] == row["destination_chat_id"]
            assert kwargs["parse_mode"] is None
            return SimpleNamespace(message_id=99)

        bot = SimpleNamespace(send_message=AsyncMock(side_effect=send))
        worker = ReminderWorker(
            DeliveryRepository(pool, clock=clock), bot, traces=TraceManager(TraceRepository(pool))
        )
        await worker.tick()
        await worker.tick()
        bot.send_message.assert_awaited_once()
        saved = await pool.fetchrow("SELECT * FROM labbi_outbox WHERE id=$1", row["id"])
        assert (
            saved["state"] == "sent"
            and saved["attempts"] == 1
            and saved["telegram_message_id"] == 99
        )
        assert saved["body"] == bot.send_message.call_args.kwargs["text"]
        assert saved["llm_reserved"] == saved["mcp_reserved"] == 0
        attempts = await pool.fetch(
            "SELECT * FROM labbi_delivery_attempts WHERE event_id=$1", row["id"]
        )
        assert len(attempts) == 1 and attempts[0]["outcome"] == "sent"
        assert attempts[0]["trace_id"] is not None
        await pool.close()
        pool = await create_pool(settings)
        await ReminderWorker(DeliveryRepository(pool, clock=clock), bot).tick()
        bot.send_message.assert_awaited_once()
        _, spans = await TraceRepository(pool).read(101, attempts[0]["trace_id"])
        assert any(s.operation == "send_message" and s.status == "ok" for s in spans)
        assert not any(s.operation.startswith("model_") for s in spans)
        assert await TraceRepository(pool).read(202, attempts[0]["trace_id"]) is None
    finally:
        await pool.close()


@pytest.mark.parametrize(
    "microseconds,expected",
    [(0, "pending"), (1, "sent"), (900_000_000, "sent"), (900_000_001, "missed")],
)
async def test_reminder_delivery_never_early_and_fifteen_minute_deadline_is_inclusive(
    database, microseconds, expected
):
    from app.notifications.delivery_repository import DeliveryRepository
    from app.notifications.worker import ReminderWorker

    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        row, clock = await _delivery_seed(pool)
        clock.value = (
            row["due_at"] - timedelta(microseconds=1)
            if microseconds == 0
            else row["due_at"] + timedelta(microseconds=microseconds)
        )
        bot = SimpleNamespace(send_message=AsyncMock(return_value=SimpleNamespace(message_id=1)))
        await ReminderWorker(DeliveryRepository(pool, clock=clock), bot).tick()
        saved = await pool.fetchrow("SELECT * FROM labbi_outbox WHERE id=$1", row["id"])
        assert saved["state"] == expected
        assert bot.send_message.await_count == (1 if expected == "sent" else 0)
        if expected == "sent":
            assert "с задержкой" in saved["body"]
        else:
            assert saved["attempts"] == 0
    finally:
        await pool.close()


async def test_reminder_delivery_two_workers_claim_one_send_and_stale_preparation_is_fenced(
    database,
):
    from app.notifications.delivery_repository import DeliveryRepository
    from app.notifications.worker import ReminderWorker

    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        row, clock = await _delivery_seed(pool)
        repo = DeliveryRepository(pool, clock=clock)
        first = (await repo.claim_preparations())[0]
        clock.value += timedelta(seconds=151)
        await repo.recover()
        second = (await repo.claim_preparations())[0]
        assert second.version > first.version
        assert not await repo.prepare(first)
        assert await repo.prepare(second)
        started, release = asyncio.Event(), asyncio.Event()

        async def send(**kwargs):
            started.set()
            await release.wait()
            return SimpleNamespace(message_id=8)

        bot = SimpleNamespace(send_message=AsyncMock(side_effect=send))
        a = ReminderWorker(repo, bot)
        b = ReminderWorker(DeliveryRepository(pool, clock=clock), bot)
        tasks = [asyncio.create_task(w.tick()) for w in (a, b)]
        try:
            await asyncio.wait_for(started.wait(), 2)
            await asyncio.sleep(0.02)
            assert bot.send_message.await_count == 1
        finally:
            release.set()
            await asyncio.gather(*tasks)
        assert await pool.fetchval("SELECT attempts FROM labbi_outbox WHERE id=$1", row["id"]) == 1
    finally:
        await pool.close()


async def test_reminder_delivery_network_failures_wait_thirty_then_one_twenty_and_stop_at_three(
    database,
):
    from app.notifications.delivery_repository import DeliveryRepository
    from app.notifications.worker import ReminderWorker

    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        row, clock = await _delivery_seed(pool)
        bot = SimpleNamespace(send_message=AsyncMock(side_effect=TimeoutError()))
        await ReminderWorker(DeliveryRepository(pool, clock=clock), bot).tick()
        first = await pool.fetchrow("SELECT * FROM labbi_outbox WHERE id=$1", row["id"])
        assert first["state"] == "retry_wait" and first["uncertain"]
        assert first["next_attempt_at"] - clock.value == timedelta(seconds=30)
        payload = first["body"]
        await pool.close()
        pool = await create_pool(settings)
        worker = ReminderWorker(DeliveryRepository(pool, clock=clock), bot)
        clock.value = first["next_attempt_at"] - timedelta(microseconds=1)
        await worker.tick()
        assert bot.send_message.await_count == 1
        clock.value = first["next_attempt_at"]
        await worker.tick()
        second = await pool.fetchrow("SELECT * FROM labbi_outbox WHERE id=$1", row["id"])
        assert second["next_attempt_at"] - clock.value == timedelta(seconds=120)
        assert second["body"] == payload
        clock.value = second["next_attempt_at"]
        await worker.tick()
        await worker.tick()
        saved = await pool.fetchrow("SELECT * FROM labbi_outbox WHERE id=$1", row["id"])
        assert (
            saved["state"] == "failed"
            and saved["attempts"] == 3
            and saved["error_code"] == "ATTEMPTS_EXHAUSTED"
        )
        assert bot.send_message.await_count == 3
        assert all(c.kwargs["text"] == payload for c in bot.send_message.call_args_list)
        assert (
            await pool.fetchval(
                "SELECT count(*) FROM labbi_delivery_attempts WHERE event_id=$1", row["id"]
            )
            == 3
        )
    finally:
        await pool.close()


async def test_reminder_delivery_http_policy_handles_rate_limit_deadline_and_permanent_errors(
    database,
):
    from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError, TelegramRetryAfter
    from aiogram.methods import SendMessage

    from app.notifications.delivery_repository import DeliveryRepository
    from app.notifications.worker import ReminderWorker

    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        due = (await pool.fetchval("SELECT clock_timestamp()")) + timedelta(minutes=2)
        rows = {}
        for owner in (101, 102, 103, 104):
            rows[owner], clock = await _delivery_seed(pool, owner, due=due)
        method = SendMessage(chat_id=101, text="private")
        errors = {
            101: TelegramRetryAfter(method, "rate", 90),
            102: TelegramRetryAfter(method, "rate", 1000),
            103: TelegramForbiddenError(method, "blocked"),
            104: TelegramBadRequest(method, "bad"),
        }

        async def send(**kwargs):
            raise errors[kwargs["chat_id"]]

        bot = SimpleNamespace(send_message=AsyncMock(side_effect=send))
        await ReminderWorker(DeliveryRepository(pool, clock=clock), bot).tick()
        for owner, expected in (
            (101, "retry_wait"),
            (102, "missed"),
            (103, "failed"),
            (104, "failed"),
        ):
            saved = await pool.fetchrow("SELECT * FROM labbi_outbox WHERE id=$1", rows[owner]["id"])
            assert saved["state"] == expected and saved["attempts"] == 1 and not saved["uncertain"]
            if owner == 101:
                assert saved["next_attempt_at"] - clock.value == timedelta(seconds=90)
        assert bot.send_message.await_count == 4
    finally:
        await pool.close()


async def test_reminder_delivery_late_receipt_cannot_mutate_new_lease_and_prevents_known_duplicate(
    database,
):
    from app.notifications.delivery_repository import DeliveryRepository

    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        row, clock = await _delivery_seed(pool)
        repo = DeliveryRepository(pool, clock=clock)
        await repo.prepare((await repo.claim_preparations())[0])
        old = await repo.begin_send(row["id"], 101)
        clock.value += timedelta(seconds=61)
        await repo.recover()
        interrupted = await pool.fetchrow("SELECT * FROM labbi_outbox WHERE id=$1", row["id"])
        assert interrupted["uncertain"] and interrupted["state"] == "retry_wait"
        clock.value = interrupted["next_attempt_at"]
        newer = await repo.begin_send(row["id"], 101)
        assert newer.version > old.version and newer.attempt == 2
        assert not await repo.sent(old, 777)
        unchanged = await pool.fetchrow("SELECT * FROM labbi_outbox WHERE id=$1", row["id"])
        assert unchanged["state"] == "sending" and unchanged["lease_token"] == newer.token
        assert not await repo.preflight(newer)  # Earlier successful ACK is now durable.
        saved = await pool.fetchrow("SELECT * FROM labbi_outbox WHERE id=$1", row["id"])
        assert saved["state"] == "sent" and saved["telegram_message_id"] == 777
        assert (
            await pool.fetchval(
                "SELECT outcome FROM labbi_delivery_attempts WHERE event_id=$1 AND attempt=1",
                row["id"],
            )
            == "sent"
        )
        assert not await repo.sent(old, 888)
        assert (
            await pool.fetchval(
                "SELECT telegram_message_id FROM labbi_delivery_attempts "
                "WHERE event_id=$1 AND attempt=1",
                row["id"],
            )
            == 777
        )
    finally:
        await pool.close()


async def test_reminder_delivery_unknown_commit_may_duplicate_but_keeps_payload(
    database,
):
    from app.notifications.delivery_repository import DeliveryRepository
    from app.notifications.worker import ReminderWorker

    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        row, clock = await _delivery_seed(pool)
        repo = DeliveryRepository(pool, clock=clock)
        repo.sent = AsyncMock(side_effect=ConnectionError("ACK store offline"))
        bot = SimpleNamespace(send_message=AsyncMock(return_value=SimpleNamespace(message_id=1)))
        await ReminderWorker(repo, bot).tick()
        assert bot.send_message.await_count == 1
        first = await pool.fetchrow("SELECT * FROM labbi_outbox WHERE id=$1", row["id"])
        assert first["state"] == "sending"
        await pool.close()
        pool = await create_pool(settings)
        clock.value += timedelta(seconds=61)
        repo = DeliveryRepository(pool, clock=clock)
        await repo.recover()
        pending = await pool.fetchrow("SELECT * FROM labbi_outbox WHERE id=$1", row["id"])
        clock.value = pending["next_attempt_at"]
        bot.send_message.return_value = SimpleNamespace(message_id=2)
        await ReminderWorker(repo, bot).tick()
        saved = await pool.fetchrow("SELECT * FROM labbi_outbox WHERE id=$1", row["id"])
        assert saved["state"] == "sent" and saved["attempts"] == 2 and saved["uncertain"]
        assert bot.send_message.await_count == 2
        assert saved["body"] == first["body"]
        assert (
            bot.send_message.call_args_list[0].kwargs["text"]
            == bot.send_message.call_args_list[1].kwargs["text"]
        )
    finally:
        await pool.close()


async def test_reminder_delivery_ready_survives_reset_and_restart_without_sending_briefings(
    database,
):
    from app.notifications.delivery_repository import DeliveryRepository
    from app.notifications.worker import ReminderWorker

    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        row, clock = await _delivery_seed(pool)
        repo = DeliveryRepository(pool, clock=clock)
        await repo.prepare((await repo.claim_preparations())[0])
        payload = await pool.fetchval("SELECT body FROM labbi_outbox WHERE id=$1", row["id"])
        await DialogRepository(pool).reset(101)
        await DialogRepository(pool).change_mode(101, "translate")
        await pool.execute(
            "INSERT INTO labbi_outbox(user_id,kind,local_date,destination_chat_id,"
            "profile_version,timezone,due_at,deadline_at,next_attempt_at) "
            "VALUES(101,'briefing',$1,101,0,'Europe/Moscow',$2::timestamptz,"
            "$2::timestamptz+INTERVAL '60 minutes',$2::timestamptz)",
            clock.value.date(),
            clock.value,
        )
        await pool.close()
        pool = await create_pool(settings)
        bot = SimpleNamespace(send_message=AsyncMock(return_value=SimpleNamespace(message_id=1)))
        await ReminderWorker(DeliveryRepository(pool, clock=clock), bot).tick()
        bot.send_message.assert_awaited_once()
        assert bot.send_message.call_args.kwargs["text"] == payload
        assert (
            await pool.fetchval("SELECT state FROM labbi_outbox WHERE kind='briefing'") == "pending"
        )
    finally:
        await pool.close()


async def test_reminder_delivery_expired_backlog_does_not_starve_current_reminder(database):
    from app.notifications.delivery_repository import DeliveryRepository
    from app.notifications.worker import ReminderWorker

    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        due = (await pool.fetchval("SELECT clock_timestamp()")) + timedelta(minutes=2)
        for owner in range(301, 326):
            await _delivery_seed(pool, owner, due=due)
        clock = DeliveryClock(due)
        repo = DeliveryRepository(pool, clock=clock)
        for _ in range(3):
            for lease in await repo.claim_preparations():
                await repo.prepare(lease)
        newer, clock = await _delivery_seed(pool, 999, due=due + timedelta(days=1))
        bot = SimpleNamespace(send_message=AsyncMock(return_value=SimpleNamespace(message_id=5)))
        await ReminderWorker(DeliveryRepository(pool, clock=clock), bot).tick()
        bot.send_message.assert_awaited_once()
        assert bot.send_message.call_args.kwargs["chat_id"] == 999
        assert (
            await pool.fetchval("SELECT state FROM labbi_outbox WHERE id=$1", newer["id"]) == "sent"
        )
    finally:
        await pool.close()


async def test_reminder_delivery_preflight_checks_current_destination_lease_and_trace_owner(
    database,
):
    from app.notifications.delivery_repository import DeliveryRepository
    from app.tracing.repository import TraceRepository
    from app.tracing.runtime import TraceManager

    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        row, clock = await _delivery_seed(pool)
        repo = DeliveryRepository(pool, clock=clock)
        await repo.prepare((await repo.claim_preparations())[0])
        async with TraceManager(TraceRepository(pool)).request(202) as foreign:
            pass
        with pytest.raises(ValueError, match="Foreign"):
            await repo.begin_send(row["id"], 101, trace_id=foreign.trace_id)
        assert await pool.fetchval("SELECT attempts FROM labbi_outbox WHERE id=$1", row["id"]) == 0
        lease = await repo.begin_send(row["id"], 101)
        clock.value += timedelta(seconds=61)
        assert not await repo.preflight(lease)
        await repo.recover()
        saved = await pool.fetchrow("SELECT * FROM labbi_outbox WHERE id=$1", row["id"])
        clock.value = saved["next_attempt_at"]
        newer = await repo.begin_send(row["id"], 101)
        await pool.execute(
            "UPDATE labbi_notification_profiles SET destination_chat_id=999 WHERE user_id=101"
        )
        assert not await repo.preflight(newer)
        assert (
            await pool.fetchval("SELECT state FROM labbi_outbox WHERE id=$1", row["id"])
            == "cancelled"
        )
    finally:
        await pool.close()


async def test_reminder_delivery_success_ack_after_deadline_is_recorded_without_another_send(
    database,
):
    from app.notifications.delivery_repository import DeliveryRepository
    from app.notifications.repository import NotificationRepository
    from app.notifications.worker import ReminderWorker

    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        await migrate(pool)
        row, clock = await _delivery_seed(pool)
        clock.value += timedelta(seconds=899)

        async def send(**kwargs):
            clock.value += timedelta(seconds=2)
            return SimpleNamespace(message_id=50)

        bot = SimpleNamespace(send_message=AsyncMock(side_effect=send))
        worker = ReminderWorker(DeliveryRepository(pool, clock=clock), bot)
        await worker.tick()
        await worker.tick()
        bot.send_message.assert_awaited_once()
        assert (
            await pool.fetchval("SELECT state FROM labbi_outbox WHERE id=$1", row["id"]) == "sent"
        )
        counts, uncertain = await NotificationRepository(pool).delivery_status(101)
        assert counts == {"sent": 1} and uncertain == 0
        assert await NotificationRepository(pool).delivery_status(202) == ({}, 0)
    finally:
        await pool.close()
