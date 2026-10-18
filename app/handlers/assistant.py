import asyncio
import logging
from contextlib import nullcontext, suppress
from uuid import UUID

from aiogram import F, Router
from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from app.confirmation_models import ConfirmationError, PreparedAction
from app.dialog import DialogService
from app.formatting import FormattedText, format_reply, split_formatted
from app.llm import ModelError

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
    callbacks: set[asyncio.Task] = set()
    closing = False

    async def shutdown():
        nonlocal closing
        closing = True
        tasks = {task for tasks in pending.values() for task in tasks} | callbacks.copy()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    router.shutdown.register(shutdown)

    async def cancel_answers(user_id: int) -> int:
        tasks = list(pending.get(user_id, ()))
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return len(tasks)

    async def execute(
        message: Message,
        dialog: DialogService,
        action,
        *,
        generating=False,
        user_id: int | None = None,
        serialized=True,
    ):
        if closing:
            return
        owner = user_id if user_id is not None else message.from_user.id

        async def work():
            async with dialog.lock(owner) if serialized else nullcontext():
                formatted = generating
                markup = None
                indicator = asyncio.create_task(typing(message)) if generating else None
                try:
                    try:
                        answer = await action()
                        if generating:
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
                    for number, part in enumerate(parts):
                        await send_part(message, part, markup if number == len(parts) - 1 else None)
                finally:
                    if indicator:
                        indicator.cancel()
                        await asyncio.gather(indicator, return_exceptions=True)

        if not generating:
            await work()
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
            pending[owner].discard(task)
            if not pending[owner]:
                pending.pop(owner)

    async def invalidate(message: Message, dialog: DialogService) -> bool | None:
        if closing:
            return None
        try:
            return await dialog.invalidate_confirmations(message.from_user.id) is True
        except Exception as exc:
            logger.error("Подтверждение: ошибка %s", type(exc).__name__)
            await message.answer(DATABASE_ERROR, parse_mode=None)
            return None

    async def acknowledge(query: CallbackQuery, text: str | None = None):
        try:
            await query.answer(text=text, show_alert=text is not None)
        except Exception as exc:
            # Update уже получен; сбой ACK не отменяет подтверждённое действие.
            logger.warning("Callback ACK: ошибка %s", type(exc).__name__)

    @router.message(CommandStart())
    async def start(message: Message, dialog: DialogService):
        async def initialize():
            user = await dialog.repository.user(message.from_user.id)
            return START_TEXT.replace(
                "Сейчас по умолчанию используется /study.", f"Текущий режим: /{user.mode}."
            )

        await execute(message, dialog, initialize)

    @router.message(Command("study", "translate", "quiz"))
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
    async def reset(message: Message, dialog: DialogService):
        await cancel_answers(message.from_user.id)
        if await invalidate(message, dialog) is None:
            return
        await execute(message, dialog, lambda: dialog.reset(message.from_user.id))

    @router.message(Command("stop"))
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
    async def timezone(message: Message, command: CommandObject, dialog: DialogService):
        setting = command.args is not None
        if setting:
            await cancel_answers(message.from_user.id)
        await execute(
            message,
            dialog,
            lambda: dialog.timezone(message.from_user.id, command.args),
            serialized=not setting,
        )

    @router.callback_query(F.data.startswith("rem:"))
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
            )
        finally:
            callbacks.discard(task)

    @router.message(F.text.startswith("/"))
    async def unknown(message: Message):
        await message.answer("Неизвестная команда. Список команд — /start.", parse_mode=None)

    @router.message()
    async def text(message: Message, dialog: DialogService):
        await execute(
            message,
            dialog,
            lambda: dialog.respond(message.from_user.id, message.message_id, message.text),
            generating=True,
        )

    return router
