from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from app.config import Settings
from app.dialog import DialogService
from app.repository import UserSettings


class FakeRepository:
    """Граница хранения для тестов сервиса; SQL проверяется на настоящей БД."""

    def __init__(self):
        self.users = {}
        self.exchanges = {}

    async def user(self, user_id):
        return self.users.setdefault(user_id, UserSettings(user_id, "study", 0.3, 0))

    async def history(self, user_id, max_messages):
        messages = []
        for (owner, _), (question, answer) in self.exchanges.items():
            if owner == user_id:
                messages.extend(
                    [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ]
                )
        return messages[-max_messages:]

    async def answer(self, user_id, message_id):
        pair = self.exchanges.get((user_id, message_id))
        return pair[1] if pair else None

    async def save_exchange(self, user, message_id, question, answer, **quiz):
        if self.users[user.user_id].revision != user.revision:
            return False
        self.exchanges[user.user_id, message_id] = (question, answer)
        self.users[user.user_id] = replace(user, revision=user.revision + 1, **quiz)
        return True

    async def change_mode(self, user_id, mode):
        await self.reset(user_id)
        self.users[user_id] = replace(self.users[user_id], mode=mode)

    async def reset(self, user_id):
        self.users[user_id] = replace(
            self.users[user_id],
            revision=self.users[user_id].revision + 1,
            quiz_question=0,
            quiz_topic=None,
            quiz_finished=False,
        )
        self.exchanges = {key: value for key, value in self.exchanges.items() if key[0] != user_id}

    async def set_temperature(self, user_id, temperature):
        user = self.users[user_id]
        self.users[user_id] = replace(user, temperature=temperature, revision=user.revision + 1)


@pytest.fixture
def bot_settings():
    return Settings(
        bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
        postgres_password="fake-db",
        openrouter_api_key="fake-openrouter",
    )


@pytest.fixture
def dialog_setup(bot_settings):
    repo = FakeRepository()
    llm = AsyncMock()
    llm.complete.return_value = "Ответ модели."
    return DialogService(repo, llm, bot_settings), repo, llm
