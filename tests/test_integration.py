"""RUN_INTEGRATION=1 python -m pytest -m integration -v.

Тест создаёт собственные контейнер и volume; пользовательскую БД не использует.
"""

import asyncio
import json
import os
import secrets
import socket
import time
from dataclasses import replace

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
        assert await pool.fetchval("SELECT count(*) FROM labbi_schema_migrations") == 2
        repository = DialogRepository(pool)
        alice = await repository.user(101)
        bob = await repository.user(202)
        assert (alice.mode, alice.temperature) == ("study", 0.3)
        assert await repository.save_exchange(alice, 1, "Вопрос Алисы 👋", "Ответ Алисе")
        assert await repository.save_exchange(bob, 1, "Вопрос Боба", "Ответ Бобу")
        await repository.set_temperature(101, 0.7)
        await pool.close()

        # Новое соединение имитирует перезапуск процесса; память Python не используется.
        pool = await create_pool(settings)
        repository = DialogRepository(pool)
        alice = await repository.user(101)
        assert alice.temperature == 0.7
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
        assert (alice.mode, alice.temperature) == ("quiz", 0.7)

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
        assert (alice.mode, alice.temperature) == ("quiz", 0.7)
        assert await repository.history(101, 20) == []
        assert await repository.history(202, 20) == [
            {"role": "user", "content": "Вопрос Боба"},
            {"role": "assistant", "content": "Ответ Бобу"},
        ]
    finally:
        await pool.close()
