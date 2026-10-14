"""Транзакционные миграции; один запуск применяет каждую версию только один раз."""

import logging
from pathlib import Path

import asyncpg

logger = logging.getLogger(__name__)


async def migrate(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as connection, connection.transaction():
        # Блокировка общая для всех экземпляров Лабби в одной БД.
        await connection.execute("SELECT pg_advisory_xact_lock(734825019)")
        await connection.execute(
            "CREATE TABLE IF NOT EXISTS labbi_schema_migrations "
            "(version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        applied = {
            row["version"]
            for row in await connection.fetch("SELECT version FROM labbi_schema_migrations")
        }
        for path in sorted((Path(__file__).parent / "migrations").glob("*.sql")):
            if path.name not in applied:
                await connection.execute(path.read_text(encoding="utf-8"))
                await connection.execute(
                    "INSERT INTO labbi_schema_migrations(version) VALUES ($1)", path.name
                )
                logger.info("Миграция %s применена.", path.name)
