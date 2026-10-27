import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot, Dispatcher
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.methods import AnswerCallbackQuery, SendMessage
from aiogram.types import CallbackQuery, Chat, Message, PhotoSize, Update, User

from app.dialog import TRUNCATED_NOTICE
from app.essay import ESSAY_TOPIC_QUESTION
from app.formatting import format_reply
from app.handlers.assistant import DATABASE_ERROR, START_TEXT, create_router, split_text
from app.llm import Completion, ModelError, RoutingCompletion
from app.request_limits import RequestLimits


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
def dispatcher_setup(bot_settings, monkeypatch):
    monkeypatch.setattr(
        "app.handlers.assistant.RequestLimits", lambda: RequestLimits(clock=lambda: 0.0)
    )
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


def reminder_click(number, operation="confirm"):
    return Update(
        update_id=number,
        callback_query=CallbackQuery(
            id=f"fake-click-{number}",
            from_user=User(id=1, is_bot=False, first_name="Студент"),
            chat_instance="fake-chat",
            data=f"rem:{operation}:b79320a8-a64c-4da1-838f-e2ec93e0d5c1",
            message=Message(
                message_id=1,
                date=datetime.now(UTC),
                chat=Chat(id=1, type="private"),
                from_user=User(id=123456789, is_bot=True, first_name="Лабби"),
                text="Карточка",
            ),
        ),
    )


async def test_callback_flood_is_coalesced_before_blocking_ack(dispatcher_setup, dialog_setup):
    dispatcher, bot = dispatcher_setup
    service, _, _ = dialog_setup
    entered, release = asyncio.Event(), asyncio.Event()
    service.confirm = AsyncMock(return_value="Подтверждено")

    async def send(_bot, method, **_kwargs):
        if isinstance(method, AnswerCallbackQuery):
            entered.set()
            await release.wait()

    bot.session.side_effect = send
    first = asyncio.create_task(dispatcher.feed_update(bot, reminder_click(1), dialog=service))
    await asyncio.wait_for(entered.wait(), 2)
    rest = [
        asyncio.create_task(dispatcher.feed_update(bot, reminder_click(i), dialog=service))
        for i in range(2, 32)
    ]
    try:
        # The first duplicate may send one busy ACK; the rest return immediately.
        await asyncio.wait_for(asyncio.gather(*rest[1:]), 2)
        acks = [c for c in bot.session.call_args_list if isinstance(c.args[1], AnswerCallbackQuery)]
        assert len(acks) == 2
        service.confirm.assert_not_awaited()
    finally:
        release.set()
        await asyncio.gather(first, *rest)
    service.confirm.assert_awaited_once()


async def test_cancel_callback_uses_a_separate_lane_from_busy_confirm(
    dispatcher_setup, dialog_setup
):
    dispatcher, bot = dispatcher_setup
    service, _, _ = dialog_setup
    entered, release = asyncio.Event(), asyncio.Event()

    async def confirm(*_args):
        entered.set()
        await release.wait()
        return "Подтверждено"

    service.confirm = AsyncMock(side_effect=confirm)
    service.cancel_confirmation = AsyncMock(return_value="Подтверждение отменено.")
    first = asyncio.create_task(dispatcher.feed_update(bot, reminder_click(1), dialog=service))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        await asyncio.wait_for(
            dispatcher.feed_update(bot, reminder_click(2, "cancel"), dialog=service), 2
        )
        service.cancel_confirmation.assert_awaited_once()
        assert not first.done()
    finally:
        release.set()
        await asyncio.wait_for(first, 2)


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


@pytest.mark.parametrize(
    "question",
    [
        "напиши мне эссе по теме c++ на огромное количество слов",
        "напиши мне эссе по теме c++ на 10000 слов",
        "Напиши очень длинное эссе о C++",
    ],
)
async def test_large_essay_from_screenshot_is_refused_without_model(
    dispatcher_setup, dialog_setup, question
):
    dispatcher, bot = dispatcher_setup
    service, repo, llm = dialog_setup
    # Даже зависающий endpoint не должен задержать ответ на такой запрос.
    llm.complete.side_effect = AssertionError("Модель не должна вызываться")
    await dispatcher.feed_update(bot, update(question), dialog=service)
    assert len(sent_texts(bot)) == 1
    assert "не поместится в один ответ" in sent_texts(bot)[0]
    llm.complete.assert_not_awaited()
    assert await repo.history(1, 20) == []


async def test_mixed_quote_and_inline_code_reaches_telegram_as_readable_text(
    dispatcher_setup, dialog_setup
):
    dispatcher, bot = dispatcher_setup
    service, repo, llm = dialog_setup
    source = (
        "### Размер массива\n\n"
        "> ❌ `int` может быть слишком мал: `sizeof(массив)` может превышать `INT_MAX`.\n\n"
        "Используй `size_t`."
    )
    llm.complete.return_value = Completion(source)
    await dispatcher.feed_update(bot, update("Почему нужен size_t?"), dialog=service)
    sent = [
        call.args[1] for call in bot.session.call_args_list if isinstance(call.args[1], SendMessage)
    ]
    assert len(sent) == 1
    assert sent[0].text == (
        "Размер массива\n\n"
        "❌ int может быть слишком мал: sizeof(массив) может превышать INT_MAX.\n\n"
        "Используй size_t."
    )
    assert {e.type for e in sent[0].entities} == {"bold", "code"}
    assert (await repo.history(1, 20))[-1]["content"] == source
    llm.complete.assert_awaited_once()


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


async def test_burst_cannot_build_an_unbounded_generation_or_rejection_reply_queue(
    dispatcher_setup, dialog_setup
):
    dispatcher, bot = dispatcher_setup
    service, repo, llm = dialog_setup
    entered, release = asyncio.Event(), asyncio.Event()

    async def complete(*_args):
        entered.set()
        await release.wait()
        return Completion("Ответ")

    llm.complete.side_effect = complete
    first = asyncio.create_task(dispatcher.feed_update(bot, update("Первый"), dialog=service))
    await asyncio.wait_for(entered.wait(), 2)
    rest = [
        asyncio.create_task(
            dispatcher.feed_update(bot, update("Ещё вопрос", message_id=i), dialog=service)
        )
        for i in range(2, 31)
    ]
    try:
        # One request may wait; the other 28 must be rejected while LLM is blocked.
        await asyncio.wait_for(asyncio.gather(*rest[1:]), 2)
        assert not rest[0].done()
        llm.complete.assert_awaited_once()
        assert len(sent_texts(bot)) == 1
        assert "/stop" in sent_texts(bot)[0]
    finally:
        release.set()
        await asyncio.gather(first, *rest)
    assert llm.complete.await_count == 2
    assert len(await repo.history(1, 100)) == 4


async def test_same_owner_waiter_does_not_occupy_a_global_execution_slot(
    dispatcher_setup, dialog_setup
):
    dispatcher, bot = dispatcher_setup
    service, _, llm = dialog_setup
    entered, queued, release, full = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    active = 0
    owner_lock = service.lock
    same_owner_locks = 0

    def observe(owner):
        nonlocal same_owner_locks
        if owner == 1:
            same_owner_locks += 1
            if same_owner_locks == 2:
                queued.set()
        return owner_lock(owner)

    service.lock = observe

    async def complete(*_args):
        nonlocal active
        active += 1
        entered.set()
        if active == 4:
            full.set()
        try:
            await release.wait()
            return Completion("Ответ")
        finally:
            active -= 1

    llm.complete.side_effect = complete
    tasks = [asyncio.create_task(dispatcher.feed_update(bot, update("Первый"), dialog=service))]
    try:
        await asyncio.wait_for(entered.wait(), 2)
        tasks.append(
            asyncio.create_task(
                dispatcher.feed_update(bot, update("Второй", message_id=2), dialog=service)
            )
        )
        await asyncio.wait_for(queued.wait(), 2)
        tasks.extend(
            asyncio.create_task(
                dispatcher.feed_update(bot, update("Вопрос", user_id=i), dialog=service)
            )
            for i in range(2, 5)
        )
        await asyncio.wait_for(full.wait(), 2)
        assert llm.complete.await_count == 4
    finally:
        release.set()
        await asyncio.gather(*tasks)


@pytest.mark.parametrize("command", ["/stop", "/reset", "/translate"])
async def test_control_commands_bypass_exhausted_generation_rate(
    dispatcher_setup, dialog_setup, command
):
    from app.request_limits import REQUESTS_PER_MINUTE

    dispatcher, bot = dispatcher_setup
    service, _, llm = dialog_setup
    for i in range(REQUESTS_PER_MINUTE):
        await dispatcher.feed_update(bot, update("Вопрос", message_id=i + 1), dialog=service)
    await dispatcher.feed_update(bot, update("Лишний", message_id=100), dialog=service)
    assert "Слишком много запросов" in sent_texts(bot)[-1]
    assert llm.complete.await_count == REQUESTS_PER_MINUTE
    await dispatcher.feed_update(bot, update(command, message_id=101), dialog=service)
    assert "Слишком много запросов" not in sent_texts(bot)[-1]
    assert llm.complete.await_count == REQUESTS_PER_MINUTE


async def test_read_command_flood_is_bounded_before_user_lock_and_stop_clears_it(
    dispatcher_setup, dialog_setup
):
    dispatcher, bot = dispatcher_setup
    service, _, llm = dialog_setup
    entered = asyncio.Event()

    async def blocked(*_args):
        entered.set()
        await asyncio.Event().wait()

    llm.complete.side_effect = blocked
    service.preferences = AsyncMock(wraps=service.preferences)
    generation = asyncio.create_task(
        dispatcher.feed_update(bot, update("Вопрос", message_id=1), dialog=service)
    )
    await asyncio.wait_for(entered.wait(), 2)
    tasks = [
        asyncio.create_task(
            dispatcher.feed_update(bot, update("/settings", message_id=i), dialog=service)
        )
        for i in range(2, 32)
    ]
    try:
        await asyncio.wait_for(asyncio.gather(*tasks[2:]), 2)
        assert all(not task.done() for task in tasks[:2])
        service.preferences.assert_not_awaited()
        assert len(sent_texts(bot)) == 1
        assert "Команда уже ожидает" in sent_texts(bot)[0]
        await asyncio.wait_for(
            dispatcher.feed_update(bot, update("/stop", message_id=33), dialog=service), 2
        )
        await asyncio.wait_for(asyncio.gather(generation, *tasks), 2)
    finally:
        for task in [generation, *tasks]:
            task.cancel()
        await asyncio.gather(generation, *tasks, return_exceptions=True)
    assert all(task.done() for task in [generation, *tasks])
    service.preferences.assert_not_awaited()
    llm.complete.assert_awaited_once()


async def test_stop_flood_coalesces_before_database_and_blocks_new_generation(
    dispatcher_setup, dialog_setup
):
    dispatcher, bot = dispatcher_setup
    service, _, llm = dialog_setup
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked(_owner):
        entered.set()
        await release.wait()
        return False

    service.invalidate_confirmations = AsyncMock(side_effect=blocked)
    first = asyncio.create_task(
        dispatcher.feed_update(bot, update("/stop", message_id=1), dialog=service)
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        await asyncio.wait_for(
            asyncio.gather(
                *[
                    dispatcher.feed_update(bot, update("/stop", message_id=i), dialog=service)
                    for i in range(2, 32)
                ]
            ),
            2,
        )
        await dispatcher.feed_update(bot, update("Новый вопрос", message_id=33), dialog=service)
        service.invalidate_confirmations.assert_awaited_once()
        llm.complete.assert_not_awaited()
    finally:
        release.set()
        await asyncio.wait_for(first, 2)


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


@pytest.mark.parametrize("command", ["/stop", "/reset", "/translate"])
@pytest.mark.parametrize("phase", ["decision", "final"])
async def test_commands_cancel_agent_without_a_late_reply(
    dispatcher_setup, dialog_setup, command, phase
):
    from app.agent import AgentService
    from app.mcp.contracts import tool_schemas

    dispatcher, bot = dispatcher_setup
    service, repository, llm = dialog_setup
    entered = asyncio.Event()

    async def waiting(*_args, **_kwargs):
        entered.set()
        await asyncio.Event().wait()

    llm.decide.return_value = RoutingCompletion(
        text=json.dumps(
            {
                "action": "call_tool",
                "name": "get_schedule",
                "arguments": {"date": "2026-10-12"},
                "reason": "Чтение.",
            }
        ),
        tool_calls=[],
    )
    if phase == "decision":
        llm.decide.side_effect = waiting
    else:
        llm.complete.side_effect = waiting
    mcp = SimpleNamespace(
        available=True,
        tools={
            "get_schedule": {
                "name": "get_schedule",
                "description": "Расписание",
                **tool_schemas("get_schedule"),
            }
        },
        resources={},
        call_tool=AsyncMock(return_value={"status": "ok", "source": "local_schedule"}),
    )
    audit = SimpleNamespace(
        has_processed=AsyncMock(return_value=False),
        start_request=AsyncMock(),
        timezone=AsyncMock(return_value=None),
        record=AsyncMock(),
    )
    service.agent = AgentService(llm, mcp, audit, service.settings)
    answering = asyncio.create_task(
        dispatcher.feed_update(bot, update("Расписание", message_id=1), dialog=service)
    )
    await asyncio.wait_for(entered.wait(), 2)
    await dispatcher.feed_update(bot, update(command, message_id=2), dialog=service)
    await asyncio.wait_for(answering, 2)
    assert len(sent_texts(bot)) == 1
    assert await repository.history(1, 20) == []
    assert audit.record.call_args.kwargs["status"] == "CANCELLED"
    assert mcp.call_tool.await_count == (phase == "final")
    assert llm.decide.await_count == 1
    assert llm.complete.await_count == (phase == "final")
