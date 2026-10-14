"""Хранение настроек и полных пар диалога; SQL не попадает в обработчики."""

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

    async def save_exchange(
        self,
        user: UserSettings,
        message_id: int,
        question: str,
        answer: str,
        *,
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
                "INSERT INTO labbi_messages(user_id, telegram_message_id, role, content) "
                "VALUES ($1, $2, 'user', $3), ($1, $2, 'assistant', $4)",
                user.user_id,
                message_id,
                question,
                answer,
            )
            await connection.execute(
                "UPDATE labbi_users SET revision=revision+1, updated_at=now(), "
                "quiz_question=COALESCE($2, quiz_question), "
                "quiz_topic=CASE WHEN $2::integer IS NULL THEN quiz_topic ELSE $3 END, "
                "quiz_finished=CASE WHEN $2::integer IS NULL THEN quiz_finished ELSE $4 END "
                "WHERE user_id=$1",
                user.user_id,
                quiz_question,
                quiz_topic,
                quiz_finished,
            )
        return True

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
