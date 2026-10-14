import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot, Dispatcher
from aiogram.methods import SendMessage
from aiogram.types import Chat, Message, PhotoSize, Update, User

from app.handlers.assistant import DATABASE_ERROR, START_TEXT, create_router, split_text
from app.llm import ModelError


def update(text=None, *, user_id=1, message_id=1, chat_type="private", photo=None):
    return Update(
        update_id=message_id,
        message=Message(
            message_id=message_id,
            date=datetime.now(UTC),
            chat=Chat(id=user_id, type=chat_type),
            from_user=User(id=user_id, is_bot=False, first_name="Студент"),
            text=text,
            photo=photo,
        ),
    )


@pytest.fixture
def dispatcher_setup(bot_settings):
    bot = Bot(bot_settings.bot_token)
    bot.session = AsyncMock()
    dispatcher = Dispatcher()
    dispatcher.include_router(create_router())
    return dispatcher, bot


def sent_texts(bot):
    return [
        call.args[1].text
        for call in bot.session.call_args_list
        if isinstance(call.args[1], SendMessage)
    ]


async def test_start_and_commands_do_not_call_model(dispatcher_setup, dialog_setup):
    dispatcher, bot = dispatcher_setup
    service, _, llm = dialog_setup
    await dispatcher.feed_update(bot, update("/start"), dialog=service)
    assert sent_texts(bot) == [
        START_TEXT.replace("Сейчас по умолчанию используется /study.", "Текущий режим: /study.")
    ]
    await dispatcher.feed_update(bot, update("/quiz", message_id=2), dialog=service)
    await dispatcher.feed_update(bot, update("/settings 0.7", message_id=3), dialog=service)
    await dispatcher.feed_update(bot, update("/reset", message_id=4), dialog=service)
    assert "Режим: Проверка знаний" in sent_texts(bot)[1]
    assert "Температура: 0.7" in sent_texts(bot)[2]
    assert sent_texts(bot)[3] == "История очищена. Режим и температура сохранены."
    llm.complete.assert_not_awaited()


async def test_regular_message_calls_model_and_sends_plain_answer(dispatcher_setup, dialog_setup):
    dispatcher, bot = dispatcher_setup
    service, repo, llm = dialog_setup
    llm.complete.return_value = "<b>буквальный текст</b> 👋"
    await dispatcher.feed_update(bot, update("Что такое список?"), dialog=service)
    assert sent_texts(bot) == ["<b>буквальный текст</b> 👋"]
    sent = next(
        call.args[1] for call in bot.session.call_args_list if isinstance(call.args[1], SendMessage)
    )
    assert sent.parse_mode is None
    assert len(await repo.history(1, 20)) == 2


@pytest.mark.parametrize(
    "message",
    [
        update("Групповой вопрос", chat_type="group"),
        update(photo=[PhotoSize(file_id="a", file_unique_id="b", width=1, height=1)]),
    ],
)
async def test_groups_and_nontext_are_ignored(dispatcher_setup, dialog_setup, message):
    dispatcher, bot = dispatcher_setup
    service, repo, llm = dialog_setup
    await dispatcher.feed_update(bot, message, dialog=service)
    llm.complete.assert_not_awaited()
    assert repo.users == {}
    assert sent_texts(bot) == []


async def test_unknown_command_is_not_sent_to_model(dispatcher_setup, dialog_setup):
    dispatcher, bot = dispatcher_setup
    service, _, llm = dialog_setup
    await dispatcher.feed_update(bot, update("/missing"), dialog=service)
    assert sent_texts(bot) == ["Неизвестная команда. Список команд — /start."]
    llm.complete.assert_not_awaited()


async def test_safe_error_and_database_error_do_not_expose_details(dispatcher_setup, dialog_setup):
    dispatcher, bot = dispatcher_setup
    service, repo, llm = dialog_setup
    llm.complete.side_effect = ModelError("Лимит запросов к модели исчерпан. Попробуй позже.")
    await dispatcher.feed_update(bot, update("Вопрос"), dialog=service)
    assert sent_texts(bot) == ["Лимит запросов к модели исчерпан. Попробуй позже."]
    assert await repo.history(1, 20) == []
    repo.user = AsyncMock(side_effect=RuntimeError("private SQL and user data"))
    await dispatcher.feed_update(bot, update("/start", message_id=2), dialog=service)
    assert sent_texts(bot)[-1] == DATABASE_ERROR


async def test_long_answer_is_sent_in_order_without_text_loss(dispatcher_setup, dialog_setup):
    dispatcher, bot = dispatcher_setup
    service, _, llm = dialog_setup
    original = "я" * 4095 + "😀" * 100 + "\nПоследняя строка"
    llm.complete.return_value = original
    await dispatcher.feed_update(bot, update("Длинный ответ"), dialog=service)
    parts = sent_texts(bot)
    assert len(parts) == 2
    assert "".join(parts) == original
    assert all(len(part.encode("utf-16-le")) // 2 <= 4096 for part in parts)


def test_split_preserves_unicode_and_boundaries():
    assert split_text("я" * 4096) == ["я" * 4096]
    assert split_text("😀" * 2049) == ["😀" * 2048, "😀"]


async def test_same_user_requests_are_serialized_with_replies(dispatcher_setup, dialog_setup):
    dispatcher, bot = dispatcher_setup
    service, _, llm = dialog_setup
    entered, release = asyncio.Event(), asyncio.Event()

    async def complete(messages, temperature):
        if messages[-1]["content"] == "Первый":
            entered.set()
            await release.wait()
        return "Ответ " + messages[-1]["content"]

    llm.complete.side_effect = complete
    first = asyncio.create_task(dispatcher.feed_update(bot, update("Первый"), dialog=service))
    await asyncio.wait_for(entered.wait(), timeout=2)
    second = asyncio.create_task(
        dispatcher.feed_update(bot, update("Второй", message_id=2), dialog=service)
    )
    release.set()
    await asyncio.gather(first, second)
    assert sent_texts(bot) == ["Ответ Первый", "Ответ Второй"]
    messages = llm.complete.call_args.args[0]
    assert [m["content"] for m in messages[1:]] == ["Первый", "Ответ Первый", "Второй"]
