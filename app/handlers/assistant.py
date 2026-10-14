import asyncio
import logging
from contextlib import suppress

from aiogram import F, Router
from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import Message

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
    "/reset — очистить историю\n\n"
    "Сейчас по умолчанию используется /study. Отправь вопрос или выбери режим. "
    "Могу ошибаться: проверяй важные ответы по учебным материалам."
)
DATABASE_ERROR = "Не удалось сохранить или прочитать диалог. Попробуй позже."


def split_text(text: str, limit: int = 4096) -> list[str]:
    return [part.text for part in split_formatted(FormattedText(text), limit)]


async def send_part(message: Message, part: FormattedText) -> None:
    entities = part.entities
    retries = 0
    while True:
        try:
            await message.answer(part.text, entities=entities, parse_mode=None)
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

    async def execute(message: Message, dialog: DialogService, action, *, generating=False):
        async with dialog.lock(message.from_user.id):
            formatted = generating
            indicator = asyncio.create_task(typing(message)) if generating else None
            try:
                try:
                    answer = await action()
                except ModelError as exc:
                    answer = exc.user_message
                    formatted = False
                except Exception as exc:
                    # Не журналируем исключения с текстом запросов или SQL-параметрами.
                    logger.error("Диалог: ошибка %s", type(exc).__name__)
                    answer = DATABASE_ERROR
                    formatted = False
            finally:
                if indicator:
                    indicator.cancel()
                    await asyncio.gather(indicator, return_exceptions=True)
            parts = format_reply(answer) if formatted else split_formatted(FormattedText(answer))
            for part in parts:
                await send_part(message, part)

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
        await execute(message, dialog, lambda: dialog.mode(message.from_user.id, command.command))

    @router.message(Command("settings"))
    async def settings(message: Message, command: CommandObject, dialog: DialogService):
        await execute(
            message, dialog, lambda: dialog.preferences(message.from_user.id, command.args)
        )

    @router.message(Command("reset"))
    async def reset(message: Message, dialog: DialogService):
        await execute(message, dialog, lambda: dialog.reset(message.from_user.id))

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
