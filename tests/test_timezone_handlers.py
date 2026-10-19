"""Telegram-меню часового пояса: выбор, владельцы, отмена поиска и доставка."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram import Bot, Dispatcher
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import (
    AnswerCallbackQuery,
    DeleteMessage,
    EditMessageReplyMarkup,
    EditMessageText,
    SendMessage,
)
from aiogram.types import CallbackQuery, Chat, InaccessibleMessage, Message, Update, User

from app.handlers.assistant import create_router
from app.timezone_ui import TimezoneButton, TimezoneView

TOKEN = "abcdefghijklmnop"


def view(text="Выбери **город** или страну.", *, dismiss=False):
    return TimezoneView(
        text,
        () if dismiss else ((TimezoneButton("Найти", f"tz:search:{TOKEN}"),),),
        dismiss_previous=dismiss,
    )


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
    data=None, *, owner=1, chat_id=1, chat_type="private", inaccessible=False, inline=False
):
    message = Message(
        message_id=20,
        date=datetime.now(UTC),
        chat=Chat(id=chat_id, type=chat_type),
        from_user=User(id=123456789, is_bot=True, first_name="Лабби"),
        text="Выбор часового пояса",
    )
    if inaccessible:
        message = InaccessibleMessage(chat=message.chat, message_id=20, date=0)
    return Update(
        update_id=10,
        callback_query=CallbackQuery(
            id="timezone-callback",
            from_user=User(id=owner, is_bot=False, first_name="Студент"),
            chat_instance="test-chat",
            data=data if data is not None else f"tz:pick:{TOKEN}:0",
            message=None if inline else message,
            inline_message_id="inline-message" if inline else None,
        ),
    )


class FakeDialog:
    def __init__(self):
        self.locks = {}
        self.lock = Mock(side_effect=lambda owner: self.locks.setdefault(owner, asyncio.Lock()))
        self.timezone_menu = AsyncMock(return_value=view())
        self.timezone_is_direct = Mock(return_value=False)
        self.timezone_callback = AsyncMock(return_value=view("Сейчас там: 20 октября, 19:00"))
        self.timezone_callback_is_apply = Mock(return_value=False)
        self.timezone_search_active = Mock(return_value=False)
        self.timezone_input = AsyncMock(return_value=view())
        self.respond = AsyncMock(return_value="Ответ на обычный вопрос")
        self.confirmation_card = AsyncMock(return_value=None)
        self.invalidate_confirmations = AsyncMock(return_value=False)
        self.reset = AsyncMock(return_value="История очищена.")
        self.mode = AsyncMock(return_value="Режим изменён.")


@pytest.fixture
def setup_ui(bot_settings):
    bot = Bot(bot_settings.bot_token)
    bot.session = AsyncMock()
    dispatcher = Dispatcher()
    dispatcher.include_router(create_router())
    return dispatcher, bot, FakeDialog()


def methods(bot, kind=None):
    result = [call.args[1] for call in bot.session.call_args_list]
    return result if kind is None else [method for method in result if isinstance(method, kind)]


async def test_timezone_menu_uses_literal_text_and_inline_keyboard_without_model(setup_ui):
    dispatcher, bot, dialog = setup_ui
    await dispatcher.feed_update(bot, text_update("/timezone"), dialog=dialog)
    dialog.timezone_menu.assert_awaited_once_with(1, None)
    assert len(methods(bot, SendMessage)) == 1
    answer = methods(bot, SendMessage)[0]
    assert answer.text == "Выбери **город** или страну."
    assert not answer.entities and answer.parse_mode is None
    assert answer.reply_markup.inline_keyboard[0][0].callback_data == f"tz:search:{TOKEN}"
    dialog.respond.assert_not_awaited()
    dialog.confirmation_card.assert_not_awaited()
    dialog.invalidate_confirmations.assert_not_awaited()


@pytest.mark.parametrize("command", ["/timezone Россия", "/timezone Санкт-Петербург"])
async def test_city_or_country_lookup_does_not_cancel_an_active_answer(setup_ui, command):
    dispatcher, bot, dialog = setup_ui
    entered, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def respond(*_args):
        entered.set()
        try:
            await release.wait()
            return "Ответ"
        except asyncio.CancelledError:
            cancelled.set()
            raise

    dialog.respond.side_effect = respond
    task = asyncio.create_task(dispatcher.feed_update(bot, text_update("Вопрос"), dialog=dialog))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        await asyncio.wait_for(
            dispatcher.feed_update(bot, text_update(command, number=2), dialog=dialog), 2
        )
        assert not cancelled.is_set()
        assert not task.done()
    finally:
        release.set()
        await asyncio.wait_for(task, 2)
    dialog.timezone_menu.assert_awaited_once_with(1, command.partition(" ")[2])


async def test_explicit_iana_command_cancels_active_answer_before_setting(setup_ui):
    dispatcher, bot, dialog = setup_ui
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def respond(*_args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def menu(*_args):
        assert cancelled.is_set()
        return "Часовой пояс: Europe/Moscow."

    dialog.respond.side_effect = respond
    dialog.timezone_is_direct.return_value = True
    dialog.timezone_menu.side_effect = menu
    task = asyncio.create_task(dispatcher.feed_update(bot, text_update("Вопрос"), dialog=dialog))
    await asyncio.wait_for(entered.wait(), 2)
    await dispatcher.feed_update(
        bot, text_update("/timezone Europe/Moscow", number=2), dialog=dialog
    )
    await asyncio.wait_for(task, 2)
    assert [method.text for method in methods(bot, SendMessage)] == ["Часовой пояс: Europe/Moscow."]


@pytest.mark.parametrize("operation", ["pick", "page", "search"])
async def test_nonterminal_timezone_callback_acknowledges_then_edits_same_message(
    setup_ui, operation
):
    dispatcher, bot, dialog = setup_ui
    data = f"tz:{operation}:{TOKEN}" + ("" if operation == "search" else ":0")
    await dispatcher.feed_update(bot, callback_update(data), dialog=dialog)
    dialog.timezone_callback.assert_awaited_once_with(1, data)
    assert [type(method) for method in methods(bot)] == [AnswerCallbackQuery, EditMessageText]
    edit = methods(bot, EditMessageText)[0]
    assert edit.chat_id == 1 and edit.message_id == 20
    assert edit.parse_mode is None and not edit.entities
    assert edit.reply_markup.inline_keyboard[0][0].callback_data == f"tz:search:{TOKEN}"
    dialog.respond.assert_not_awaited()
    dialog.confirmation_card.assert_not_awaited()


@pytest.mark.parametrize(
    "options",
    [{"owner": 2}, {"chat_type": "group"}, {"inaccessible": True}, {"inline": True}],
)
async def test_timezone_callbacks_reject_foreign_or_inaccessible_chat(setup_ui, options):
    dispatcher, bot, dialog = setup_ui
    await dispatcher.feed_update(bot, callback_update(**options), dialog=dialog)
    assert len(methods(bot, AnswerCallbackQuery)) == 1
    assert methods(bot, AnswerCallbackQuery)[0].show_alert
    assert methods(bot, SendMessage) == []
    assert methods(bot, EditMessageText) == []
    dialog.timezone_callback.assert_not_awaited()
    dialog.timezone_callback_is_apply.assert_not_called()
    dialog.lock.assert_not_called()


@pytest.mark.parametrize(
    "data",
    [
        "tz:use:private-query:0",
        f"tz:unknown:{TOKEN}:0",
        f"tz:pick:{TOKEN}:100",
        f"tz:pick:{TOKEN}:-1",
        f"tz:pick:{TOKEN}:0:extra",
        f"tz:pick:{TOKEN}",
        f"tz:use:{TOKEN}",
        f"tz:search:{TOKEN}:0",
        f"tz:cancel:{TOKEN}:0",
        "tz:" + "x" * 65,
    ],
)
async def test_malformed_timezone_callbacks_do_not_reach_service_or_cancel_answers(setup_ui, data):
    dispatcher, bot, dialog = setup_ui
    await dispatcher.feed_update(bot, callback_update(data), dialog=dialog)
    assert methods(bot, AnswerCallbackQuery)[0].show_alert
    assert "private-query" not in methods(bot, AnswerCallbackQuery)[0].text
    dialog.timezone_callback.assert_not_awaited()
    dialog.timezone_callback_is_apply.assert_not_called()


async def test_stale_use_callback_does_not_cancel_active_answer(setup_ui):
    dispatcher, bot, dialog = setup_ui
    entered, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def respond(*_args):
        entered.set()
        try:
            await release.wait()
            return "Ответ"
        except asyncio.CancelledError:
            cancelled.set()
            raise

    dialog.respond.side_effect = respond
    dialog.timezone_callback.return_value = "Меню устарело."
    task = asyncio.create_task(dispatcher.feed_update(bot, text_update("Вопрос"), dialog=dialog))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        await dispatcher.feed_update(bot, callback_update(f"tz:use:{TOKEN}:0"), dialog=dialog)
        assert not cancelled.is_set()
        assert not task.done()
    finally:
        release.set()
        await asyncio.wait_for(task, 2)
    assert methods(bot, DeleteMessage) == []


async def test_valid_apply_cancels_active_answer_before_changing_timezone(setup_ui):
    dispatcher, bot, dialog = setup_ui
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def respond(*_args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def apply(_owner, _data):
        assert cancelled.is_set()
        return view("Часовой пояс сохранён.", dismiss=True)

    dialog.respond.side_effect = respond
    dialog.timezone_callback_is_apply.return_value = True
    dialog.timezone_callback.side_effect = apply
    task = asyncio.create_task(dispatcher.feed_update(bot, text_update("Вопрос"), dialog=dialog))
    await asyncio.wait_for(entered.wait(), 2)
    await dispatcher.feed_update(bot, callback_update(f"tz:use:{TOKEN}:0"), dialog=dialog)
    await asyncio.wait_for(task, 2)
    assert [method.text for method in methods(bot, SendMessage)] == ["Часовой пояс сохранён."]
    assert methods(bot, DeleteMessage)[0].message_id == 20


async def test_search_input_goes_to_timezone_service_without_llm_or_reminder_lookup(setup_ui):
    dispatcher, bot, dialog = setup_ui
    dialog.timezone_search_active.return_value = True
    dialog.timezone_input.return_value = view("Найден город **буквально**.")
    await dispatcher.feed_update(bot, text_update("Санкт-Петербург"), dialog=dialog)
    dialog.timezone_input.assert_awaited_once_with(1, "Санкт-Петербург")
    dialog.respond.assert_not_awaited()
    dialog.confirmation_card.assert_not_awaited()
    answer = methods(bot, SendMessage)[0]
    assert answer.text == "Найден город **буквально**."
    assert not answer.entities


async def test_closed_search_does_not_send_city_query_to_model(setup_ui):
    dispatcher, bot, dialog = setup_ui
    dialog.timezone_search_active.return_value = True
    dialog.timezone_input.return_value = None
    await dispatcher.feed_update(bot, text_update("Санкт-Петербург"), dialog=dialog)
    dialog.respond.assert_not_awaited()
    assert methods(bot, SendMessage)[0].text == "Меню закрыто. Открой /timezone заново."


@pytest.mark.parametrize("source", ["/timezone Санкт-Петербург", "Санкт-Петербург"])
async def test_stop_cancels_pending_city_lookup_and_clears_menu(setup_ui, source):
    dispatcher, bot, dialog = setup_ui
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def search(*_args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    dialog.timezone_search_active.return_value = True
    dialog.timezone_menu.side_effect = search
    dialog.timezone_input.side_effect = search
    task = asyncio.create_task(dispatcher.feed_update(bot, text_update(source), dialog=dialog))
    await asyncio.wait_for(entered.wait(), 2)
    await dispatcher.feed_update(bot, text_update("/stop", number=2), dialog=dialog)
    await asyncio.wait_for(task, 2)
    assert cancelled.is_set()
    dialog.invalidate_confirmations.assert_awaited_once_with(1)
    assert [method.text for method in methods(bot, SendMessage)] == [
        "Ответ и ожидающие запросы остановлены."
    ]


async def test_menu_edit_failure_falls_back_to_literal_new_message_without_data_logs(
    setup_ui, caplog
):
    dispatcher, bot, dialog = setup_ui

    async def send(_bot, method, **_kwargs):
        if isinstance(method, EditMessageText):
            raise TelegramBadRequest(method=method, message="private-city cannot edit")

    bot.session.side_effect = send
    await dispatcher.feed_update(bot, callback_update(), dialog=dialog)
    assert len(methods(bot, EditMessageText)) == len(methods(bot, SendMessage)) == 1
    assert methods(bot, SendMessage)[0].reply_markup is not None
    assert "private-city" not in caplog.text


async def test_repeated_menu_edit_with_identical_text_does_not_duplicate_message(setup_ui):
    dispatcher, bot, dialog = setup_ui

    async def send(_bot, method, **_kwargs):
        if isinstance(method, EditMessageText):
            raise TelegramBadRequest(method=method, message="Bad Request: message is not modified")

    bot.session.side_effect = send
    await dispatcher.feed_update(bot, callback_update(), dialog=dialog)
    assert len(methods(bot, EditMessageText)) == 1
    assert methods(bot, SendMessage) == []


@pytest.mark.parametrize("operation", ["cancel", "use"])
async def test_terminal_menu_sends_result_before_delete(setup_ui, operation):
    dispatcher, bot, dialog = setup_ui
    dialog.timezone_callback.return_value = view("Меню закрыто.", dismiss=True)
    data = f"tz:{operation}:{TOKEN}" + (":0" if operation == "use" else "")
    await dispatcher.feed_update(bot, callback_update(data), dialog=dialog)
    assert [type(method) for method in methods(bot)] == [
        AnswerCallbackQuery,
        SendMessage,
        DeleteMessage,
    ]


async def test_menu_delete_failure_removes_buttons_and_does_not_repeat_setting(setup_ui, caplog):
    dispatcher, bot, dialog = setup_ui
    dialog.timezone_callback.return_value = view("Сохранено.", dismiss=True)

    async def send(_bot, method, **_kwargs):
        if isinstance(method, (DeleteMessage, EditMessageReplyMarkup)):
            raise TelegramBadRequest(method=method, message="private-city Telegram unavailable")

    bot.session.side_effect = send
    await dispatcher.feed_update(bot, callback_update(f"tz:use:{TOKEN}:0"), dialog=dialog)
    assert len(methods(bot, SendMessage)) == len(methods(bot, DeleteMessage)) == 1
    assert len(methods(bot, EditMessageReplyMarkup)) == 1
    assert methods(bot, EditMessageReplyMarkup)[0].reply_markup is None
    dialog.timezone_callback.assert_awaited_once()
    assert "private-city" not in caplog.text


async def test_terminal_receipt_delivery_failure_does_not_delete_menu(setup_ui):
    dispatcher, bot, dialog = setup_ui
    dialog.timezone_callback.return_value = view("Сохранено.", dismiss=True)

    async def send(_bot, method, **_kwargs):
        if isinstance(method, SendMessage):
            raise TelegramBadRequest(method=method, message="cannot send")

    bot.session.side_effect = send
    with pytest.raises(TelegramBadRequest):
        await dispatcher.feed_update(bot, callback_update(f"tz:use:{TOKEN}:0"), dialog=dialog)
    assert methods(bot, DeleteMessage) == methods(bot, EditMessageReplyMarkup) == []


async def test_shutdown_cancels_callback_before_session_cleanup_and_rejects_new_clicks(setup_ui):
    dispatcher, bot, dialog = setup_ui
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def callback(*_args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    dialog.timezone_callback.side_effect = callback
    task = asyncio.create_task(dispatcher.feed_update(bot, callback_update(), dialog=dialog))
    await asyncio.wait_for(entered.wait(), 2)
    await asyncio.wait_for(dispatcher.emit_shutdown(bot=bot), 2)
    await asyncio.gather(task, return_exceptions=True)
    assert task.done() and cancelled.is_set()
    assert methods(bot, SendMessage) == []
    assert methods(bot, EditMessageText) == []
    dialog.timezone_callback.reset_mock()
    await dispatcher.feed_update(bot, callback_update(), dialog=dialog)
    dialog.timezone_callback.assert_not_awaited()
    assert methods(bot, AnswerCallbackQuery)[-1].show_alert
