import asyncio
import logging
import re
from contextlib import nullcontext, suppress
from functools import wraps
from uuid import UUID

from aiogram import F, Router
from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from app.confirmation_models import ConfirmationError, ConfirmationText, PreparedAction
from app.dialog import DialogService
from app.formatting import FormattedText, format_reply, split_formatted
from app.llm import ModelError
from app.request_limits import CommandSlots, ControlSlots, RequestLimitError, RequestLimits
from app.timezone_ui import TimezoneView

logger = logging.getLogger(__name__)
START_TEXT = (
    "Привет! Я Лабби — AI-помощник студента.\n"
    "/study — объяснения по программированию\n"
    "/translate — перевод текста\n"
    "/quiz — проверка знаний\n"
    "/settings — режим, модель и температура\n"
    "/tools — доступные действия через MCP\n"
    "/timezone — часовой пояс напоминаний\n"
    "/why — объяснение последнего действия\n"
    "/stop — остановить ответ и ожидающие запросы\n"
    "/reset — очистить историю\n\n"
    "Сейчас по умолчанию используется /study. Отправь вопрос или выбери режим. "
    "Напоминания сохраняются после подтверждения; отправки уведомлений по времени пока нет. "
    "Могу ошибаться: проверяй важные ответы по учебным материалам."
)
DATABASE_ERROR = "Не удалось сохранить или прочитать диалог. Попробуй позже."


def split_text(text: str, limit: int = 4096) -> list[str]:
    return [part.text for part in split_formatted(FormattedText(text), limit)]


async def send_part(
    message: Message, part: FormattedText, reply_markup: InlineKeyboardMarkup | None = None
) -> None:
    entities = part.entities
    retries = 0
    while True:
        try:
            kwargs = {"reply_markup": reply_markup} if reply_markup is not None else {}
            await message.answer(part.text, entities=entities, parse_mode=None, **kwargs)
            return
        except TelegramRetryAfter as exc:
            if retries == 2:
                raise
            retries += 1
            await asyncio.sleep(exc.retry_after)
        except TelegramBadRequest as exc:
            if not entities or "entit" not in exc.message.casefold():
                raise
            logger.warning("Telegram отклонил разметку; отправляем ту же часть обычным текстом.")
            entities = []


async def typing(message: Message) -> None:
    while True:
        with suppress(Exception):
            await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
        await asyncio.sleep(4)


def create_router() -> Router:
    router = Router(name="assistant")
    router.message.filter(F.chat.type == "private", F.from_user, F.text)
    pending: dict[int, set[asyncio.Task]] = {}
    readonly_pending: dict[int, set[asyncio.Task]] = {}
    callbacks: set[asyncio.Task] = set()
    commands: set[asyncio.Task] = set()
    closing = False
    limits = RequestLimits()
    command_slots, control_slots = CommandSlots(), ControlSlots()
    controls: set[asyncio.Task] = set()

    async def shutdown():
        nonlocal closing
        closing = True
        tasks = (
            {task for tasks in pending.values() for task in tasks}
            | callbacks.copy()
            | commands.copy()
            | controls.copy()
        )
        tasks.discard(asyncio.current_task())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    router.shutdown.register(shutdown)

    async def cancel_answers(user_id: int) -> int:
        tasks = list(pending.get(user_id, set()) | readonly_pending.get(user_id, set()))
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return len(tasks)

    async def reject(event, text):
        if limits.notify_rejection(event.from_user.id):
            if isinstance(event, CallbackQuery):
                await acknowledge(event, text)
            else:
                await event.answer(text, parse_mode=None)

    def bounded_control(lane):
        def decorate(handler):
            @wraps(handler)
            async def wrapped(event, *args, **kwargs):
                owner = event.from_user.id
                kind = lane(event) if callable(lane) else lane
                if kind == "stop":
                    # Every stop cancels its current snapshot, even when its
                    # database/reply work is coalesced into an earlier stop.
                    for task in pending.get(owner, set()) | readonly_pending.get(owner, set()):
                        task.cancel()
                if closing:
                    if isinstance(event, CallbackQuery):
                        await reject(event, "Бот останавливается. Попробуй позже.")
                    return
                admission = control_slots.admit(owner, kind, stop_reserve=bool(pending.get(owner)))
                if admission is None:
                    await reject(event, "Предыдущая команда ещё выполняется. Подожди немного.")
                    return
                task = asyncio.current_task()
                controls.add(task)
                try:
                    return await handler(event, *args, **kwargs)
                finally:
                    controls.discard(task)
                    admission.release()

            return wrapped

        return decorate

    def callback_lane(query):
        data = query.data or ""
        if data.startswith(("rem:cancel:", "tz:cancel:")):
            return "cancel"
        return "callback"

    async def execute(
        message: Message,
        dialog: DialogService,
        action,
        *,
        generating=False,
        user_id: int | None = None,
        serialized=True,
        literal=False,
        check_confirmation=True,
        edit_view=False,
        dismiss_confirmation=False,
    ):
        if closing:
            return
        owner = user_id if user_id is not None else message.from_user.id

        async def work():
            async with (
                dialog.lock(owner) if serialized else nullcontext(),
                limits.running() if generating else nullcontext(),
            ):
                formatted = generating and not literal
                markup = None
                dismiss_previous = False
                view = False
                indicator = asyncio.create_task(typing(message)) if generating else None
                try:
                    try:
                        answer = await action()
                        if isinstance(answer, TimezoneView):
                            view = True
                            dismiss_previous = answer.dismiss_previous
                            if answer.rows:
                                markup = InlineKeyboardMarkup(
                                    inline_keyboard=[
                                        [
                                            InlineKeyboardButton(
                                                text=button.text, callback_data=button.callback_data
                                            )
                                            for button in row
                                        ]
                                        for row in answer.rows
                                    ]
                                )
                            answer = answer.text
                            formatted = False
                        elif dismiss_confirmation and isinstance(answer, ConfirmationText):
                            dismiss_previous = answer.dismiss_card
                        if generating and check_confirmation and not view:
                            card = await dialog.confirmation_card(owner, message.message_id)
                            if isinstance(card, PreparedAction) and card.state in {
                                "pending",
                                "approved",
                            }:
                                answer = card.card_text
                                markup = InlineKeyboardMarkup(
                                    inline_keyboard=[
                                        [
                                            InlineKeyboardButton(
                                                text="Подтвердить",
                                                callback_data=f"rem:confirm:{card.action_id}",
                                            ),
                                            InlineKeyboardButton(
                                                text="Отменить",
                                                callback_data=f"rem:cancel:{card.action_id}",
                                            ),
                                        ]
                                    ]
                                )
                                # Параметры подтверждения отображаются буквально.
                                formatted = False
                    except (ModelError, ConfirmationError) as exc:
                        answer = exc.user_message
                        formatted = False
                    except Exception as exc:
                        logger.error("Диалог: ошибка %s", type(exc).__name__)
                        answer = DATABASE_ERROR
                        formatted = False
                    parts = (
                        format_reply(answer)
                        if formatted
                        else split_formatted(FormattedText(answer))
                    )
                    if edit_view and view and not dismiss_previous and len(parts) == 1:
                        try:
                            await message.edit_text(
                                parts[0].text,
                                entities=parts[0].entities,
                                parse_mode=None,
                                reply_markup=markup,
                            )
                            return
                        except TelegramBadRequest as exc:
                            if "message is not modified" in exc.message.casefold():
                                return
                            logger.warning("Меню: ошибка редактирования %s", type(exc).__name__)
                        except Exception as exc:
                            logger.warning("Меню: ошибка редактирования %s", type(exc).__name__)
                    for number, part in enumerate(parts):
                        await send_part(message, part, markup if number == len(parts) - 1 else None)
                    # Сначала доставляем результат; сбой Telegram не отменяет запись в БД.
                    if dismiss_previous:
                        await dismiss_message(message)
                finally:
                    if indicator:
                        indicator.cancel()
                        await asyncio.gather(indicator, return_exceptions=True)

        if not generating:
            if asyncio.current_task() not in controls:
                try:
                    admission = command_slots.admit(owner)
                except RequestLimitError as error:
                    await reject(message, error.user_message)
                    return
                task = asyncio.create_task(work())
                commands.add(task)
                readonly_pending.setdefault(owner, set()).add(task)
                try:
                    await task
                except asyncio.CancelledError:
                    if asyncio.current_task().cancelling():
                        raise
                finally:
                    admission.release()
                    commands.discard(task)
                    readonly_pending[owner].discard(task)
                    if not readonly_pending[owner]:
                        readonly_pending.pop(owner)
                return
            task = asyncio.current_task()
            commands.add(task)
            try:
                await work()
            finally:
                commands.discard(task)
            return
        if control_slots.active(owner, "stop") or control_slots.active(owner, "change"):
            await reject(
                message, "Команда остановки или смены режима ещё выполняется. Подожди немного."
            )
            return
        try:
            admission = limits.admit(owner)
        except RequestLimitError as error:
            if limits.notify_rejection(owner):
                await message.answer(error.user_message, parse_mode=None)
            return
        task = asyncio.create_task(work())
        pending.setdefault(owner, set()).add(task)
        try:
            await task
        except asyncio.CancelledError:
            # Пользователь отменяет только нашу задачу. Завершение приложения передаём выше.
            if asyncio.current_task().cancelling():
                raise
        finally:
            admission.release()
            pending[owner].discard(task)
            if not pending[owner]:
                pending.pop(owner)

    async def invalidate(message: Message, dialog: DialogService) -> bool | None:
        if closing:
            return None
        task = asyncio.current_task()
        commands.add(task)
        try:
            return await dialog.invalidate_confirmations(message.from_user.id) is True
        except Exception as exc:
            logger.error("Подтверждение: ошибка %s", type(exc).__name__)
            await message.answer(DATABASE_ERROR, parse_mode=None)
            return None
        finally:
            commands.discard(task)

    async def acknowledge(query: CallbackQuery, text: str | None = None):
        try:
            await query.answer(text=text, show_alert=text is not None)
        except Exception as exc:
            # Update уже получен; сбой ACK не отменяет подтверждённое действие.
            logger.warning("Callback ACK: ошибка %s", type(exc).__name__)

    async def dismiss_message(message: Message):
        try:
            await message.delete()
            return
        except Exception as exc:
            logger.warning("Карточка: ошибка удаления %s", type(exc).__name__)
        try:
            await message.edit_reply_markup(reply_markup=None)
        except Exception as exc:
            logger.warning("Карточка: ошибка удаления кнопок %s", type(exc).__name__)

    @router.message(CommandStart())
    async def start(message: Message, dialog: DialogService):
        async def initialize():
            user = await dialog.repository.user(message.from_user.id)
            return START_TEXT.replace(
                "Сейчас по умолчанию используется /study.", f"Текущий режим: /{user.mode}."
            )

        await execute(message, dialog, initialize)

    @router.message(Command("study", "translate", "quiz"))
    @bounded_control("change")
    async def mode(message: Message, command: CommandObject, dialog: DialogService):
        await cancel_answers(message.from_user.id)
        if await invalidate(message, dialog) is None:
            return
        await execute(message, dialog, lambda: dialog.mode(message.from_user.id, command.command))

    @router.message(Command("settings"))
    async def settings(message: Message, command: CommandObject, dialog: DialogService):
        await execute(
            message, dialog, lambda: dialog.preferences(message.from_user.id, command.args)
        )

    @router.message(Command("reset"))
    @bounded_control("change")
    async def reset(message: Message, dialog: DialogService):
        await cancel_answers(message.from_user.id)
        if await invalidate(message, dialog) is None:
            return
        await execute(message, dialog, lambda: dialog.reset(message.from_user.id))

    @router.message(Command("stop"))
    @bounded_control("stop")
    async def stop(message: Message, dialog: DialogService):
        if closing:
            return
        count = await cancel_answers(message.from_user.id)
        revoked = await invalidate(message, dialog)
        if revoked is None:
            return
        answer = (
            "Ответ и ожидающие запросы остановлены." if count else "Сейчас нет активного ответа."
        )
        if revoked:
            answer += " Ожидающее подтверждение напоминания отменено."
        await message.answer(
            answer,
            parse_mode=None,
        )

    @router.message(Command("tools"))
    async def tools(message: Message, dialog: DialogService):
        await execute(message, dialog, dialog.tools)

    @router.message(Command("why"))
    async def why(message: Message, dialog: DialogService):
        await execute(message, dialog, lambda: dialog.why(message.from_user.id))

    @router.message(Command("timezone"))
    @bounded_control("timezone")
    async def timezone(message: Message, command: CommandObject, dialog: DialogService):
        setting = dialog.timezone_is_direct(command.args)
        if setting:
            await cancel_answers(message.from_user.id)
        await execute(
            message,
            dialog,
            lambda: dialog.timezone_menu(message.from_user.id, command.args),
            generating=command.args is not None and not setting,
            serialized=command.args is None,
            literal=True,
            check_confirmation=False,
        )

    @router.callback_query(F.data.startswith("tz:"))
    @bounded_control(callback_lane)
    async def timezone_callback(query: CallbackQuery, dialog: DialogService):
        message = query.message
        if (
            not isinstance(message, Message)
            or message.chat.type != "private"
            or message.chat.id != query.from_user.id
        ):
            await acknowledge(query, "Выбор пояса доступен только в личном чате владельца.")
            return
        data = query.data or ""
        match = re.fullmatch(
            r"tz:(pick|use|page|search|cancel):[A-Za-z0-9_-]{16}(?::([0-9]{1,2}))?", data
        )
        if (
            len(data.encode("utf-8")) > 64
            or match is None
            or (match[1] in {"pick", "use", "page"}) != (match[2] is not None)
        ):
            await acknowledge(query, "Кнопка выбора пояса недействительна.")
            return
        if closing:
            await acknowledge(query, "Бот останавливается. Попробуй позже.")
            return
        await acknowledge(query)
        owner = query.from_user.id
        if dialog.timezone_callback_is_apply(owner, data):
            await cancel_answers(owner)
        task = asyncio.current_task()
        callbacks.add(task)
        try:
            await execute(
                message,
                dialog,
                lambda: dialog.timezone_callback(owner, data),
                user_id=owner,
                serialized=False,
                edit_view=True,
                check_confirmation=False,
            )
        finally:
            callbacks.discard(task)

    @router.callback_query(F.data.startswith("rem:"))
    @bounded_control(callback_lane)
    async def reminder_callback(query: CallbackQuery, dialog: DialogService):
        message = query.message
        if (
            not isinstance(message, Message)
            or message.chat.type != "private"
            or message.chat.id != query.from_user.id
        ):
            await acknowledge(query, "Подтверждение доступно только в личном чате владельца.")
            return
        data = query.data or ""
        try:
            prefix, operation, raw_id = data.split(":", 2)
            action_id = UUID(raw_id)
            if (
                prefix != "rem"
                or operation not in {"confirm", "cancel"}
                or str(action_id) != raw_id
                or len(data.encode("utf-8")) > 64
            ):
                raise ValueError
        except (ValueError, TypeError):
            await acknowledge(query, "Кнопка подтверждения недействительна.")
            return
        if closing:
            await acknowledge(query, "Бот останавливается. Попробуй позже.")
            return
        await acknowledge(query)
        owner = query.from_user.id
        if operation == "cancel":
            await cancel_answers(owner)
        action = dialog.confirm if operation == "confirm" else dialog.cancel_confirmation
        task = asyncio.current_task()
        callbacks.add(task)
        try:
            await execute(
                message,
                dialog,
                lambda: action(owner, action_id),
                user_id=owner,
                serialized=operation == "confirm",
                dismiss_confirmation=True,
            )
        finally:
            callbacks.discard(task)

    @router.message(F.text.startswith("/"))
    async def unknown(message: Message):
        await reject(message, "Неизвестная команда. Список команд — /start.")

    @router.message()
    async def text(message: Message, dialog: DialogService):
        if dialog.timezone_search_active(message.from_user.id):

            async def search():
                result = await dialog.timezone_input(message.from_user.id, message.text)
                return result if result is not None else "Меню закрыто. Открой /timezone заново."

            await execute(
                message,
                dialog,
                search,
                generating=True,
                literal=True,
                check_confirmation=False,
            )
            return
        await execute(
            message,
            dialog,
            lambda: dialog.respond(message.from_user.id, message.message_id, message.text),
            generating=True,
        )

    return router
