"""Хранение настроек и полных пар диалога; SQL не попадает в обработчики."""

import json
from dataclasses import dataclass, field

import asyncpg


@dataclass(frozen=True)
class UserSettings:
    user_id: int
    mode: str
    temperature: float
    revision: int
    quiz_topic: str | None = field(default=None, repr=False)
    quiz_question: int = 0
    quiz_finished: bool = False


@dataclass(frozen=True)
class StoredReply:
    content: str = field(repr=False)
    parts: list[str] = field(repr=False)
    is_complete: bool


class DialogRepository:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def user(self, user_id: int) -> UserSettings:
        await self.pool.execute(
            "INSERT INTO labbi_users(user_id) VALUES ($1) ON CONFLICT DO NOTHING", user_id
        )
        row = await self.pool.fetchrow(
            "SELECT user_id, mode, temperature, revision, quiz_topic, quiz_question, quiz_finished "
            "FROM labbi_users WHERE user_id=$1",
            user_id,
        )
        return UserSettings(**dict(row))

    async def history(self, user_id: int, max_messages: int) -> list[dict[str, str]]:
        rows = await self.pool.fetch(
            "SELECT role, content FROM ("
            "SELECT id, role, content FROM labbi_messages WHERE user_id=$1 "
            "ORDER BY id DESC LIMIT $2) AS recent ORDER BY id",
            user_id,
            max_messages,
        )
        return [dict(row) for row in rows]

    async def answer(self, user_id: int, message_id: int) -> str | None:
        return await self.pool.fetchval(
            "SELECT content FROM labbi_messages "
            "WHERE user_id=$1 AND telegram_message_id=$2 AND role='assistant'",
            user_id,
            message_id,
        )

    async def has_incomplete_answer(self, user_id: int) -> bool:
        return (
            await self.pool.fetchval(
                "SELECT NOT is_complete FROM labbi_messages "
                "WHERE user_id=$1 AND role='assistant' ORDER BY id DESC LIMIT 1",
                user_id,
            )
            is True
        )

    async def reply(self, user_id: int, message_id: int) -> StoredReply | None:
        row = await self.pool.fetchrow(
            "SELECT content, parts, is_complete FROM labbi_messages "
            "WHERE user_id=$1 AND telegram_message_id=$2 AND role='assistant'",
            user_id,
            message_id,
        )
        if row is None:
            return None
        parts = json.loads(row["parts"]) or [row["content"]]
        return StoredReply(row["content"], parts, row["is_complete"])

    async def save_exchange(
        self,
        user: UserSettings,
        message_id: int,
        question: str,
        answer: str,
        *,
        is_complete: bool = True,
        parts: list[str] | None = None,
        quiz_question: int | None = None,
        quiz_topic: str | None = None,
        quiz_finished: bool = False,
    ) -> bool:
        async with self.pool.acquire() as connection, connection.transaction():
            revision = await connection.fetchval(
                "SELECT revision FROM labbi_users WHERE user_id=$1 FOR UPDATE", user.user_id
            )
            if revision != user.revision:
                return False
            await connection.execute(
                "INSERT INTO labbi_messages(user_id, telegram_message_id, role, content, "
                "is_complete, parts) "
                "VALUES ($1, $2, 'user', $3, true, '[]'::jsonb), "
                "($1, $2, 'assistant', $4, $5, $6::jsonb)",
                user.user_id,
                message_id,
                question,
                answer,
                is_complete,
                json.dumps(parts or [answer], ensure_ascii=False),
            )
            await self._update_user(
                connection, user.user_id, quiz_question, quiz_topic, quiz_finished
            )
        return True

    async def update_exchange(
        self,
        user: UserSettings,
        message_id: int,
        answer: str,
        *,
        is_complete: bool,
        parts: list[str],
        quiz_question: int | None = None,
        quiz_topic: str | None = None,
        quiz_finished: bool = False,
    ) -> bool:
        async with self.pool.acquire() as connection, connection.transaction():
            revision = await connection.fetchval(
                "SELECT revision FROM labbi_users WHERE user_id=$1 FOR UPDATE", user.user_id
            )
            if revision != user.revision:
                return False
            updated = await connection.fetchval(
                "UPDATE labbi_messages SET content=$3, is_complete=$4, parts=$5::jsonb "
                "WHERE user_id=$1 AND telegram_message_id=$2 AND role='assistant' RETURNING id",
                user.user_id,
                message_id,
                answer,
                is_complete,
                json.dumps(parts, ensure_ascii=False),
            )
            if updated is None:
                return False
            await self._update_user(
                connection, user.user_id, quiz_question, quiz_topic, quiz_finished
            )
        return True

    @staticmethod
    async def _update_user(connection, user_id, quiz_question, quiz_topic, quiz_finished):
        await connection.execute(
            "UPDATE labbi_users SET revision=revision+1, updated_at=now(), "
            "quiz_question=COALESCE($2, quiz_question), "
            "quiz_topic=CASE WHEN $2::integer IS NULL THEN quiz_topic ELSE $3 END, "
            "quiz_finished=CASE WHEN $2::integer IS NULL THEN quiz_finished ELSE $4 END "
            "WHERE user_id=$1",
            user_id,
            quiz_question,
            quiz_topic,
            quiz_finished,
        )

    async def change_mode(self, user_id: int, mode: str) -> None:
        async with self.pool.acquire() as connection, connection.transaction():
            await connection.execute(
                "UPDATE labbi_users SET mode=$2, revision=revision+1, updated_at=now(), "
                "quiz_topic=NULL, quiz_question=0, quiz_finished=false "
                "WHERE user_id=$1",
                user_id,
                mode,
            )
            await connection.execute("DELETE FROM labbi_messages WHERE user_id=$1", user_id)

    async def reset(self, user_id: int) -> None:
        async with self.pool.acquire() as connection, connection.transaction():
            await connection.execute(
                "UPDATE labbi_users SET revision=revision+1, updated_at=now(), "
                "quiz_topic=NULL, quiz_question=0, quiz_finished=false WHERE user_id=$1",
                user_id,
            )
            await connection.execute("DELETE FROM labbi_messages WHERE user_id=$1", user_id)

    async def set_temperature(self, user_id: int, temperature: float) -> None:
        await self.pool.execute(
            "UPDATE labbi_users SET temperature=$2, revision=revision+1, updated_at=now() "
            "WHERE user_id=$1",
            user_id,
            temperature,
        )
