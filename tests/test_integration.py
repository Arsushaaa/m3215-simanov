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
from app.agent_repository import AgentRepository, ReminderRepository
from app.config import MCPSettings, Settings
from app.db import create_pool
from app.dialog import DialogService
from app.health import HealthState, health_result
from app.llm import Completion, ModelError, RoutingCompletion
from app.mcp.client import MCPClient
from app.mcp.contracts import ReminderArguments, ReminderResult
from app.migration import migrate
from app.repository import DialogRepository
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
        assert await pool.fetchval("SELECT count(*) FROM labbi_schema_migrations") == 6
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
        assert await pool.fetchval("SELECT count(*) FROM labbi_schema_migrations") == 6
        assert (
            await pool.fetch(
                "SELECT version, applied_at FROM labbi_schema_migrations "
                "WHERE version <> '006_agent_state.sql' ORDER BY version"
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
        # LR1 reset/mode SQL deletes only history. Ledger and settings survive it.
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
        ) == {"timezone": "Europe/Moscow", "confirmation_epoch": 1}
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
        assert await pool.fetchval("SELECT count(*) FROM labbi_agent_events WHERE user_id=101") == 1
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
                assert tools == {"get_weather", "get_schedule", "add_reminder"}
                assert resources == {"schedule://current-week"}
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
