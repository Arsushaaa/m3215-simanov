"""RUN_INTEGRATION=1 python -m pytest -m integration -v.

Тест создаёт собственные контейнер и volume; пользовательскую БД не использует.
"""

import asyncio
import hashlib
import json
import os
import secrets
import socket
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest

from app.config import Settings
from app.db import create_pool
from app.health import HealthState, health_result
from app.migration import migrate
from app.repository import DialogRepository
from scripts.common import CommandError, run_command

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
