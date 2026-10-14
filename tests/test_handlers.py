import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot, Dispatcher
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.methods import SendMessage
from aiogram.types import Chat, Message, PhotoSize, Update, User

from app.dialog import TRUNCATED_NOTICE
from app.essay import ESSAY_TOPIC_QUESTION
from app.formatting import format_reply
from app.handlers.assistant import DATABASE_ERROR, START_TEXT, create_router, split_text
from app.llm import Completion, ModelError


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
    llm.complete.return_value = Completion("<b>буквальный текст</b> 👋")
    await dispatcher.feed_update(bot, update("Что такое список?"), dialog=service)
    assert sent_texts(bot) == ["<b>буквальный текст</b> 👋"]
    sent = next(
        call.args[1] for call in bot.session.call_args_list if isinstance(call.args[1], SendMessage)
    )
    assert sent.parse_mode is None
    assert len(await repo.history(1, 20)) == 2


async def test_markdown_is_sent_as_telegram_entities(dispatcher_setup, dialog_setup):
    dispatcher, bot = dispatcher_setup
    service, repo, llm = dialog_setup
    source = "### Как это работает\n\n**Дробь** `0.7`\n\n```python\nx = 0.7\nprint(x)\n```"
    llm.complete.return_value = Completion(source)
    await dispatcher.feed_update(bot, update("Объясни 0.7"), dialog=service)
    sent = next(
        call.args[1] for call in bot.session.call_args_list if isinstance(call.args[1], SendMessage)
    )
    assert sent.text == "Как это работает\n\nДробь 0.7\n\nx = 0.7\nprint(x)"
    assert {e.type for e in sent.entities} == {"bold", "code", "pre"}
    assert sent.parse_mode is None
    assert (await repo.history(1, 20))[-1]["content"] == source


async def test_long_code_keeps_formatting_across_messages(dispatcher_setup, dialog_setup):
    dispatcher, bot = dispatcher_setup
    service, _, llm = dialog_setup
    source = "```python\n" + "print('👋')\n" * 900 + "```"
    llm.complete.return_value = Completion(source)
    await dispatcher.feed_update(bot, update("Покажи код"), dialog=service)
    sent = [
        call.args[1] for call in bot.session.call_args_list if isinstance(call.args[1], SendMessage)
    ]
    assert len(sent) > 2
    assert "".join(part.text for part in sent) == "".join(
        part.text for part in format_reply(source)
    )
    assert all(part.entities[0].type == "pre" for part in sent)
    assert all(part.entities[0].language == "python" for part in sent)
    assert all(len(part.text.encode("utf-16-le")) // 2 <= 4096 for part in sent)


async def test_telegram_entity_failure_falls_back_without_losing_answer(
    dispatcher_setup, dialog_setup
):
    dispatcher, bot = dispatcher_setup
    service, _, llm = dialog_setup
    llm.complete.return_value = Completion("**Готово**")

    async def send(_bot, method, **kwargs):
        if isinstance(method, SendMessage) and method.entities:
            raise TelegramBadRequest(method=method, message="can't parse entities")

    bot.session.side_effect = send
    await dispatcher.feed_update(bot, update("Вопрос"), dialog=service)
    attempts = [
        call.args[1] for call in bot.session.call_args_list if isinstance(call.args[1], SendMessage)
    ]
    assert [part.text for part in attempts] == ["Готово", "Готово"]
    assert attempts[0].entities and not attempts[1].entities


async def test_telegram_rate_limit_retries_same_part_before_next(
    dispatcher_setup, dialog_setup, monkeypatch
):
    dispatcher, bot = dispatcher_setup
    service, _, llm = dialog_setup
    llm.complete.return_value = Completion("я" * 5000)
    attempts = []
    sleep = AsyncMock()
    monkeypatch.setattr("app.handlers.assistant.asyncio.sleep", sleep)

    async def send(_bot, method, **kwargs):
        if isinstance(method, SendMessage):
            attempts.append(method.text)
            if len(attempts) == 1:
                raise TelegramRetryAfter(method=method, message="Too Many Requests", retry_after=1)

    bot.session.side_effect = send
    await dispatcher.feed_update(bot, update("Вопрос"), dialog=service)
    assert attempts == ["я" * 4096, "я" * 4096, "я" * 904]
    sleep.assert_any_await(1)


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
    llm.complete.return_value = Completion(original)
    await dispatcher.feed_update(bot, update("Длинный ответ"), dialog=service)
    parts = sent_texts(bot)
    assert len(parts) == 2
    assert "".join(parts) == original
    assert all(len(part.encode("utf-16-le")) // 2 <= 4096 for part in parts)


async def test_truncated_answer_is_delivered_in_parts_instead_of_refusal(
    dispatcher_setup, dialog_setup
):
    dispatcher, bot = dispatcher_setup
    service, repo, llm = dialog_setup
    body = "Текст эссе. " * 1000 + "НЕЗАВЕРШЁННЫЙ КОНЕЦ"
    llm.complete.return_value = Completion(body, truncated=True)
    await dispatcher.feed_update(bot, update("Напиши эссе о Python"), dialog=service)
    parts = sent_texts(bot)
    assert len(parts) > 2
    assert "".join(parts) == TRUNCATED_NOTICE + "\n\n" + body
    assert all(len(part.encode("utf-16-le")) // 2 <= 4096 for part in parts)
    assert await repo.has_incomplete_answer(1)
    llm.complete.assert_awaited_once()


async def test_unbounded_request_warns_immediately_without_model(dispatcher_setup, dialog_setup):
    dispatcher, bot = dispatcher_setup
    service, _, llm = dialog_setup
    await dispatcher.feed_update(
        bot, update("напиши мне огромное эссе, насколько большое ты можешь"), dialog=service
    )
    assert sent_texts(bot)[0] == ESSAY_TOPIC_QUESTION
    llm.complete.assert_not_awaited()


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
        return Completion("Ответ " + messages[-1]["content"])

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


async def test_stop_cancels_active_and_queued_requests_and_new_request_still_works(
    dispatcher_setup, dialog_setup
):
    dispatcher, bot = dispatcher_setup
    service, repo, llm = dialog_setup
    entered, queued, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
    lock = service.lock
    locks = 0

    def observe_lock(user_id):
        nonlocal locks
        locks += 1
        if locks == 2:
            queued.set()
        return lock(user_id)

    service.lock = observe_lock

    async def slow_model(*args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    llm.complete.side_effect = slow_model
    first = asyncio.create_task(
        dispatcher.feed_update(bot, update("Первый", message_id=1), dialog=service)
    )
    await asyncio.wait_for(entered.wait(), 2)
    second = asyncio.create_task(
        dispatcher.feed_update(bot, update("Второй", message_id=2), dialog=service)
    )
    await asyncio.wait_for(queued.wait(), 2)
    await asyncio.wait_for(
        dispatcher.feed_update(bot, update("/stop", message_id=3), dialog=service), 2
    )
    await asyncio.gather(first, second)
    assert cancelled.is_set()
    assert sent_texts(bot) == ["Ответ и ожидающие запросы остановлены."]
    llm.complete.assert_awaited_once()
    assert await repo.history(1, 20) == []
    llm.complete.side_effect = None
    llm.complete.return_value = Completion("Новый ответ")
    await dispatcher.feed_update(bot, update("Третий", message_id=4), dialog=service)
    assert sent_texts(bot)[-1] == "Новый ответ"


async def test_reset_interrupts_model_immediately_and_preserves_other_users(
    dispatcher_setup, dialog_setup
):
    dispatcher, bot = dispatcher_setup
    service, repo, llm = dialog_setup
    await service.respond(2, 1, "Чужой вопрос")
    before = await repo.history(2, 20)
    entered = asyncio.Event()

    async def slow_model(*args):
        entered.set()
        await asyncio.Event().wait()

    llm.complete.side_effect = slow_model
    task = asyncio.create_task(
        dispatcher.feed_update(bot, update("Ждём", message_id=1), dialog=service)
    )
    await asyncio.wait_for(entered.wait(), 2)
    await asyncio.wait_for(
        dispatcher.feed_update(bot, update("/reset", message_id=2), dialog=service), 2
    )
    await task
    assert sent_texts(bot) == ["История очищена. Режим и температура сохранены."]
    assert await repo.history(1, 20) == []
    assert await repo.history(2, 20) == before


async def test_stop_when_idle_does_not_call_model(dispatcher_setup, dialog_setup):
    dispatcher, bot = dispatcher_setup
    service, _, llm = dialog_setup
    await dispatcher.feed_update(bot, update("/stop"), dialog=service)
    assert sent_texts(bot) == ["Сейчас нет активного ответа."]
    llm.complete.assert_not_awaited()


async def test_shutdown_cancels_active_and_queued_generations_before_closing_resources(
    dispatcher_setup, dialog_setup
):
    dispatcher, bot = dispatcher_setup
    service, repo, llm = dialog_setup
    entered, queued = asyncio.Event(), asyncio.Event()
    lock = service.lock
    calls = 0

    def observe_lock(user_id):
        nonlocal calls
        calls += 1
        if calls == 2:
            queued.set()
        return lock(user_id)

    service.lock = observe_lock

    async def slow_model(*args):
        entered.set()
        await asyncio.Event().wait()

    llm.complete.side_effect = slow_model
    first = asyncio.create_task(
        dispatcher.feed_update(bot, update("Первый", message_id=1), dialog=service)
    )
    await asyncio.wait_for(entered.wait(), 2)
    second = asyncio.create_task(
        dispatcher.feed_update(bot, update("Второй", message_id=2), dialog=service)
    )
    await asyncio.wait_for(queued.wait(), 2)
    await asyncio.wait_for(dispatcher.emit_shutdown(bot=bot), 2)
    await asyncio.gather(first, second)
    await dispatcher.feed_update(bot, update("После выключения", message_id=3), dialog=service)
    llm.complete.assert_awaited_once()
    assert sent_texts(bot) == []
    assert await repo.history(1, 20) == []


async def test_stop_interrupts_delivery_of_long_answer(dispatcher_setup, dialog_setup):
    dispatcher, bot = dispatcher_setup
    service, _, llm = dialog_setup
    entered = asyncio.Event()
    llm.complete.return_value = Completion("я" * 5000)

    async def send(_bot, method, **kwargs):
        if isinstance(method, SendMessage) and method.text == "я" * 4096:
            entered.set()
            await asyncio.Event().wait()

    bot.session.side_effect = send
    task = asyncio.create_task(
        dispatcher.feed_update(bot, update("Длинный ответ", message_id=1), dialog=service)
    )
    await asyncio.wait_for(entered.wait(), 2)
    await asyncio.wait_for(
        dispatcher.feed_update(bot, update("/stop", message_id=2), dialog=service), 2
    )
    await task
    assert sent_texts(bot) == ["я" * 4096, "Ответ и ожидающие запросы остановлены."]
