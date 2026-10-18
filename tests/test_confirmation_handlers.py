"""Telegram UI подтверждений: доверенный владелец, ранний ACK и отмена без LLM."""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import pytest
from aiogram import Bot, Dispatcher
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.methods import AnswerCallbackQuery, SendMessage
from aiogram.types import CallbackQuery, Chat, InaccessibleMessage, Message, Update, User

from app.confirmation_models import ConfirmationError, PreparedAction
from app.handlers.assistant import create_router
from app.mcp.contracts import ReminderArguments

ACTION = UUID("b79320a8-a64c-4da1-838f-e2ec93e0d5c1")
BOT_ID = 123456789


def text_update(text, *, owner=1, number=1):
    return Update(
        update_id=number,
        message=Message(
            message_id=number,
            date=datetime.now(UTC),
            chat=Chat(id=owner, type="private"),
            from_user=User(id=owner, is_bot=False, first_name="Студент"),
            text=text,
        ),
    )


def callback_update(
    data=None,
    *,
    owner=1,
    chat_id=1,
    chat_type="private",
    number=10,
    inaccessible=False,
    inline=False,
):
    message = Message(
        message_id=20,
        date=datetime.now(UTC),
        chat=Chat(id=chat_id, type=chat_type),
        from_user=User(id=BOT_ID, is_bot=True, first_name="Лабби"),
        text="Карточка напоминания",
    )
    if inaccessible:
        message = InaccessibleMessage(chat=message.chat, message_id=20, date=0)
    return Update(
        update_id=number,
        callback_query=CallbackQuery(
            id=f"callback-{number}",
            from_user=User(id=owner, is_bot=False, first_name="Студент"),
            chat_instance="test-chat",
            data=data or f"rem:confirm:{ACTION}",
            message=None if inline else message,
            inline_message_id="inline-message" if inline else None,
        ),
    )


class FakeDialog:
    def __init__(self):
        self.locks = {}
        self.lock = Mock(side_effect=lambda user_id: self.locks.setdefault(user_id, asyncio.Lock()))
        self.repository = SimpleNamespace(
            user=AsyncMock(return_value=SimpleNamespace(mode="study"))
        )
        self.respond = AsyncMock(return_value="Ответ")
        self.confirmation_card = AsyncMock(return_value=None)
        self.confirm = AsyncMock(return_value="Напоминание сохранено. ID: test-reminder.")
        self.cancel_confirmation = AsyncMock(return_value="Подтверждение отменено.")
        self.invalidate_confirmations = AsyncMock(return_value=False)
        self.tools = AsyncMock(return_value="Доступны get_weather, get_schedule и add_reminder.")
        self.why = AsyncMock(return_value="Последнее действие: чтение расписания.")
        self.timezone = AsyncMock(return_value="Часовой пояс: Europe/Moscow.")
        self.reset = AsyncMock(return_value="История очищена.")
        self.mode = AsyncMock(return_value="Режим изменён.")
        self.llm = SimpleNamespace(complete=AsyncMock(), decide=AsyncMock())


@pytest.fixture
def setup_ui(bot_settings):
    bot = Bot(bot_settings.bot_token)
    bot.session = AsyncMock()
    dispatcher = Dispatcher()
    dispatcher.include_router(create_router())
    return dispatcher, bot, FakeDialog()


def sent(bot):
    return [
        call.args[1] for call in bot.session.call_args_list if isinstance(call.args[1], SendMessage)
    ]


def acknowledgements(bot):
    return [
        call.args[1]
        for call in bot.session.call_args_list
        if isinstance(call.args[1], AnswerCallbackQuery)
    ]


def prepared(text="Текст: *x* и `code`\nКогда: 2030-10-18T10:00:00+03:00"):
    return PreparedAction(
        action_id=ACTION,
        card_text=text,
        arguments=ReminderArguments(text="*x* и `code`", remind_at="2030-10-18T10:00:00+03:00"),
        timezone="Europe/Moscow",
        expires_at=datetime(2030, 10, 17, tzinfo=UTC),
        state="pending",
    )


async def test_prepared_card_is_literal_and_contains_only_action_uuid(setup_ui):
    dispatcher, bot, dialog = setup_ui
    card = prepared()
    dialog.respond.return_value = "Служебный ответ не должен заменить карточку из БД"
    dialog.confirmation_card.return_value = card
    await dispatcher.feed_update(bot, text_update("Напомни"), dialog=dialog)
    assert len(sent(bot)) == 1
    answer = sent(bot)[0]
    assert answer.text == card.card_text
    assert not answer.entities
    assert answer.parse_mode is None
    buttons = answer.reply_markup.inline_keyboard[0]
    assert [button.text for button in buttons] == ["Подтвердить", "Отменить"]
    assert [button.callback_data for button in buttons] == [
        f"rem:confirm:{ACTION}",
        f"rem:cancel:{ACTION}",
    ]
    assert all(len(button.callback_data.encode()) <= 64 for button in buttons)
    assert "capability" not in answer.reply_markup.model_dump_json()
    dialog.confirmation_card.assert_awaited_once_with(1, 1)
    dialog.llm.complete.assert_not_awaited()


async def test_synthetic_long_card_has_keyboard_on_final_part_and_same_retry(setup_ui):
    dispatcher, bot, dialog = setup_ui
    dialog.confirmation_card.return_value = prepared("я" * 5000)
    throttled = False

    async def send(_bot, method, **_kwargs):
        nonlocal throttled
        if isinstance(method, SendMessage) and method.reply_markup and not throttled:
            throttled = True
            raise TelegramRetryAfter(method=method, message="Too Many Requests", retry_after=0)

    bot.session.side_effect = send
    await dispatcher.feed_update(bot, text_update("Напомни"), dialog=dialog)
    attempts = sent(bot)
    assert [part.text for part in attempts] == ["я" * 4096, "я" * 904, "я" * 904]
    assert attempts[0].reply_markup is None
    assert attempts[1].reply_markup == attempts[2].reply_markup
    assert attempts[1].reply_markup is not None
    dialog.respond.assert_awaited_once()


async def test_callback_ack_precedes_lock_and_owner_comes_from_clicker(setup_ui):
    dispatcher, bot, dialog = setup_ui
    lock = dialog.lock(1)
    await lock.acquire()
    acked = asyncio.Event()

    async def send(_bot, method, **_kwargs):
        if isinstance(method, AnswerCallbackQuery):
            acked.set()

    bot.session.side_effect = send
    task = asyncio.create_task(dispatcher.feed_update(bot, callback_update(), dialog=dialog))
    try:
        await asyncio.wait_for(acked.wait(), 2)
        dialog.confirm.assert_not_awaited()
    finally:
        lock.release()
    await asyncio.wait_for(task, 2)
    dialog.confirm.assert_awaited_once_with(1, ACTION)
    assert len(acknowledgements(bot)) == 1
    assert sent(bot)[0].chat_id == 1
    assert all(call.args[0] == 1 for call in dialog.lock.call_args_list)
    dialog.respond.assert_not_awaited()
    dialog.llm.complete.assert_not_awaited()
    dialog.llm.decide.assert_not_awaited()


@pytest.mark.parametrize(
    "options",
    [
        {"owner": 2, "chat_id": 1},
        {"chat_type": "group"},
        {"inaccessible": True},
        {"inline": True},
    ],
)
async def test_callback_from_foreign_or_inaccessible_chat_is_rejected(setup_ui, options):
    dispatcher, bot, dialog = setup_ui
    await dispatcher.feed_update(bot, callback_update(**options), dialog=dialog)
    assert len(acknowledgements(bot)) == 1
    assert acknowledgements(bot)[0].show_alert
    assert sent(bot) == []
    dialog.confirm.assert_not_awaited()
    dialog.cancel_confirmation.assert_not_awaited()
    dialog.lock.assert_not_called()


@pytest.mark.parametrize(
    "data",
    [
        "rem:confirm:invalid-private-key",
        f"rem:confirm:{str(ACTION).upper()}",
        f"rem:confirm:{ACTION.hex}",
        f"rem:confirm:{{{ACTION}}}",
        f"rem:confirm:{ACTION}:extra",
        f"rem:delete:{ACTION}",
    ],
)
async def test_callback_payload_requires_canonical_uuid_and_known_operation(setup_ui, data):
    dispatcher, bot, dialog = setup_ui
    await dispatcher.feed_update(bot, callback_update(data), dialog=dialog)
    assert len(acknowledgements(bot)) == 1
    assert acknowledgements(bot)[0].show_alert
    assert "private-key" not in acknowledgements(bot)[0].text
    assert sent(bot) == []
    dialog.confirm.assert_not_awaited()


async def test_failed_ack_is_not_repeated_and_confirmation_still_runs(setup_ui, caplog):
    dispatcher, bot, dialog = setup_ui

    async def send(_bot, method, **_kwargs):
        if isinstance(method, AnswerCallbackQuery):
            raise TelegramBadRequest(method=method, message="private-key expired callback query")

    bot.session.side_effect = send
    await dispatcher.feed_update(bot, callback_update(), dialog=dialog)
    assert len(acknowledgements(bot)) == 1
    dialog.confirm.assert_awaited_once_with(1, ACTION)
    assert len(sent(bot)) == 1
    assert "private-key" not in caplog.text


@pytest.mark.parametrize("code", ["ACTION_EXPIRED", "ACTION_CANCELLED", "UNAUTHORIZED"])
async def test_service_refusal_of_old_card_is_shown_without_llm(setup_ui, code):
    dispatcher, bot, dialog = setup_ui
    dialog.confirm.side_effect = ConfirmationError(code, "Это подтверждение недействительно.")
    await dispatcher.feed_update(bot, callback_update(), dialog=dialog)
    assert [answer.text for answer in sent(bot)] == ["Это подтверждение недействительно."]
    assert len(acknowledgements(bot)) == 1
    dialog.llm.complete.assert_not_awaited()
    dialog.llm.decide.assert_not_awaited()


async def test_cancel_callback_can_revoke_while_confirmation_holds_user_lock(setup_ui):
    dispatcher, bot, dialog = setup_ui
    entered, release = asyncio.Event(), asyncio.Event()

    async def confirm(_owner, _action):
        entered.set()
        await release.wait()
        return "Подтверждение отменено."

    async def cancel(_owner, _action):
        release.set()
        return "Подтверждение отменено."

    dialog.confirm.side_effect, dialog.cancel_confirmation.side_effect = confirm, cancel
    confirming = asyncio.create_task(dispatcher.feed_update(bot, callback_update(), dialog=dialog))
    await asyncio.wait_for(entered.wait(), 2)
    await asyncio.wait_for(
        dispatcher.feed_update(
            bot, callback_update(f"rem:cancel:{ACTION}", number=11), dialog=dialog
        ),
        2,
    )
    await asyncio.wait_for(confirming, 2)
    dialog.cancel_confirmation.assert_awaited_once_with(1, ACTION)
    assert len(acknowledgements(bot)) == 2
    assert len(dialog.lock.call_args_list) == 1


async def test_cancel_callback_stops_active_text_before_cancelling_card(setup_ui):
    dispatcher, bot, dialog = setup_ui
    entered, stopped = asyncio.Event(), asyncio.Event()

    async def answer(*_args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    async def cancel(*_args):
        assert stopped.is_set()
        return "Подтверждение отменено."

    dialog.respond.side_effect, dialog.cancel_confirmation.side_effect = answer, cancel
    task = asyncio.create_task(dispatcher.feed_update(bot, text_update("Вопрос"), dialog=dialog))
    await asyncio.wait_for(entered.wait(), 2)
    await dispatcher.feed_update(bot, callback_update(f"rem:cancel:{ACTION}"), dialog=dialog)
    await asyncio.wait_for(task, 2)
    assert [answer.text for answer in sent(bot)] == ["Подтверждение отменено."]


async def test_idle_stop_revokes_confirmation_outside_user_lock(setup_ui):
    dispatcher, bot, dialog = setup_ui
    dialog.invalidate_confirmations.return_value = True
    lock = dialog.lock(1)
    await lock.acquire()
    try:
        await asyncio.wait_for(dispatcher.feed_update(bot, text_update("/stop"), dialog=dialog), 2)
    finally:
        lock.release()
    dialog.invalidate_confirmations.assert_awaited_once_with(1)
    assert "подтверждение напоминания отменено" in sent(bot)[0].text
    dialog.llm.complete.assert_not_awaited()


@pytest.mark.parametrize(
    "command,method,arguments",
    [
        ("/tools", "tools", ()),
        ("/why", "why", (1,)),
        ("/timezone", "timezone", (1, None)),
    ],
)
async def test_new_read_commands_use_services_without_model_or_revocation(
    setup_ui, command, method, arguments
):
    dispatcher, bot, dialog = setup_ui
    await dispatcher.feed_update(bot, text_update(command), dialog=dialog)
    getattr(dialog, method).assert_awaited_once_with(*arguments)
    assert len(sent(bot)) == 1
    dialog.invalidate_confirmations.assert_not_awaited()
    dialog.respond.assert_not_awaited()
    dialog.llm.complete.assert_not_awaited()


async def test_timezone_set_is_outside_lock_and_service_validates_before_revocation(setup_ui):
    dispatcher, bot, dialog = setup_ui
    dialog.timezone.return_value = "Неизвестная IANA-зона."
    lock = dialog.lock(1)
    await lock.acquire()
    try:
        await asyncio.wait_for(
            dispatcher.feed_update(bot, text_update("/timezone wrong/zone"), dialog=dialog), 2
        )
    finally:
        lock.release()
    dialog.timezone.assert_awaited_once_with(1, "wrong/zone")
    dialog.invalidate_confirmations.assert_not_awaited()
    assert sent(bot)[0].text == "Неизвестная IANA-зона."


@pytest.mark.parametrize("command", ["/reset", "/translate"])
async def test_reset_and_mode_revoke_before_waiting_for_user_lock(setup_ui, command):
    dispatcher, bot, dialog = setup_ui
    lock = dialog.lock(1)
    await lock.acquire()
    invalidated = asyncio.Event()

    async def invalidate(_owner):
        invalidated.set()
        return False

    dialog.invalidate_confirmations.side_effect = invalidate
    task = asyncio.create_task(dispatcher.feed_update(bot, text_update(command), dialog=dialog))
    try:
        await asyncio.wait_for(invalidated.wait(), 2)
        assert not task.done()
    finally:
        lock.release()
    await asyncio.wait_for(task, 2)
    assert len(sent(bot)) == 1


async def test_shutdown_waits_for_callback_cancellation_before_closing_resources(setup_ui):
    dispatcher, bot, dialog = setup_ui
    entered, reconciled = asyncio.Event(), asyncio.Event()

    async def confirm(_owner, _action):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # Подмена отмены сервиса: shutdown должен дождаться чтения итогового состояния.
            await asyncio.sleep(0)
            reconciled.set()
            raise

    dialog.confirm.side_effect = confirm
    task = asyncio.create_task(dispatcher.feed_update(bot, callback_update(), dialog=dialog))
    await asyncio.wait_for(entered.wait(), 2)
    await asyncio.wait_for(dispatcher.emit_shutdown(bot=bot), 2)
    await asyncio.gather(task, return_exceptions=True)
    assert task.done()
    assert reconciled.is_set()
    assert len(acknowledgements(bot)) == 1
    assert sent(bot) == []
    dialog.invalidate_confirmations.reset_mock()
    await dispatcher.feed_update(bot, text_update("/reset", number=22), dialog=dialog)
    dialog.invalidate_confirmations.assert_not_awaited()
    assert sent(bot) == []
