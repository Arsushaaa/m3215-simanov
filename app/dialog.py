"""Сервис диалога: порядок запросов, лимиты контекста и настройки пользователя."""

import asyncio
import json
import re
from decimal import Decimal, InvalidOperation
from weakref import WeakValueDictionary

from app.config import Settings
from app.confirmation_models import ConfirmationError
from app.essay import ESSAY_TOPIC_QUESTION, needs_essay_topic
from app.llm import ModelError, OpenRouterClient
from app.output_limits import check_output_size
from app.prompts import MODE_NAMES, PROMPTS
from app.quiz import INVALID_QUIZ, QuizState
from app.reminder_intent import (
    is_confirmation_reply,
    is_reminder_edit,
    is_reminder_request,
)
from app.repository import DialogRepository

TRUNCATED_NOTICE = (
    "Ответ достиг ограничения длины. Ниже — весь полученный текст, но он ещё не закончен. "
    "Чтобы получить следующую часть, напиши «Продолжи»."
)
CONTINUE = re.compile(
    r"^\s*(?:пожалуйста[,\s]+)?(?:продолжи|продолжай|дальше|continue)\b", re.IGNORECASE
)


def estimate_tokens(messages: list[dict[str, str]]) -> int:
    # Оценка по UTF-8 учитывает кириллицу; это бюджет, а не точный токенизатор.
    return sum((len(m["content"].encode("utf-8")) + 2) // 3 + 4 for m in messages)


def build_context(
    prompt: str,
    history: list[dict[str, str]],
    text: str,
    settings: Settings,
    *,
    pinned: list[dict[str, str]] | None = None,
    preserve_last: bool = False,
    schema: dict | None = None,
) -> list[dict[str, str]]:
    system, current = {"role": "system", "content": prompt}, {"role": "user", "content": text}
    pinned = pinned or []
    schema_tokens = (
        (len(json.dumps(schema, ensure_ascii=False).encode("utf-8")) + 2) // 3 + 4 if schema else 0
    )
    if estimate_tokens([system, *pinned, current]) + schema_tokens > settings.history_max_tokens:
        raise ModelError("Сообщение слишком длинное для контекста. Сократи его и попробуй снова.")
    # Удаляем самые старые полные пары. В БД архив остаётся до /reset или смены режима.
    recent = list(history)
    while recent and (
        len(pinned) + len(recent) + 1 > settings.history_max_messages
        or estimate_tokens([system, *pinned, *recent, current]) + schema_tokens
        > settings.history_max_tokens
    ):
        if preserve_last and len(recent) == 2:
            # Для продолжения большого ответа сохраняем его конец, даже если вся пара не влезает.
            tail = recent[-1]["content"]
            if len(tail) <= 128:
                raise ModelError(
                    "Контекст не вмещает продолжение. Укажи тему и попроси следующую часть."
                )
            recent[-1] = {"role": "assistant", "content": tail[len(tail) // 4 :]}
        else:
            del recent[:2]
    return [system, *pinned, *recent, current]


class DialogService:
    def __init__(
        self,
        repository: DialogRepository,
        llm: OpenRouterClient,
        settings: Settings,
        *,
        agent=None,
        confirmations=None,
        timezone_ui=None,
    ):
        self.repository, self.llm, self.settings = repository, llm, settings
        self.agent = agent
        self.confirmations = confirmations
        self.timezone_ui = timezone_ui
        self.locks: WeakValueDictionary[int, asyncio.Lock] = WeakValueDictionary()

    def lock(self, user_id: int) -> asyncio.Lock:
        lock = self.locks.get(user_id)
        if lock is None:
            lock = asyncio.Lock()
            self.locks[user_id] = lock
        return lock

    async def respond(self, user_id: int, message_id: int, text: str) -> str:
        if not text.strip():
            raise ModelError("Отправь непустое текстовое сообщение.")
        cached = await self.repository.answer(user_id, message_id)
        if cached is not None:
            return cached
        user = await self.repository.user(user_id)
        history = await self.repository.history(user_id, self.settings.history_max_messages)
        edit_epoch = None
        if user.mode == "study" and self.confirmations is not None and is_confirmation_reply(text):
            try:
                answer = await self.confirmations.button_reply(user_id, message_id)
            except ConfirmationError as error:
                raise ModelError(error.user_message) from None
            if answer is not None:
                if not await self.repository.save_exchange(user, message_id, text, answer):
                    raise ModelError("Диалог изменился во время ответа. Отправь вопрос ещё раз.")
                return answer
        if user.mode == "study" and self.confirmations is not None and is_reminder_edit(text):
            try:
                editing = await self.confirmations.begin_edit(user, message_id)
                if editing is not None:
                    source, user, edit_epoch = editing
                    if not is_reminder_request(text):
                        edited = await self.confirmations.prepare_time_edit(
                            user, message_id, text, source, edit_epoch
                        )
                        if isinstance(edited, str):
                            if not await self.repository.save_exchange(
                                user, message_id, text, edited
                            ):
                                raise ModelError(
                                    "Диалог изменился во время ответа. Отправь вопрос ещё раз."
                                )
                            return edited
                        return edited.card_text
            except ConfirmationError as error:
                raise ModelError(error.user_message) from None
        if user.mode == "study" and needs_essay_topic(text, history):
            if not await self.repository.save_exchange(
                user, message_id, text, ESSAY_TOPIC_QUESTION
            ):
                raise ModelError("Диалог изменился во время ответа. Отправь вопрос ещё раз.")
            return ESSAY_TOPIC_QUESTION
        previous = history[-1]["content"] if history else None
        check_output_size(
            text,
            self.settings.llm_max_output_tokens,
            previous,
            output_followup=user.mode == "study" and previous == ESSAY_TOPIC_QUESTION,
            implicit_output=user.mode == "study",
        )
        continuing = bool(CONTINUE.search(text)) and await self.repository.has_incomplete_answer(
            user_id
        )
        prompt, pinned, quiz = PROMPTS[user.mode], [], {}
        quiz_state, schema = None, None
        if user.mode == "quiz":
            if user.quiz_finished:
                raise ModelError("Тренировка завершена. Для новой тренировки отправь /quiz.")
            quiz_state = QuizState(user.quiz_topic, user.quiz_question, text, continuing, previous)
            schema = quiz_state.schema()
            prompt += quiz_state.instruction()
            if user.quiz_topic:
                # Тема является данными пользователя, поэтому не вставляется в system.
                pinned = [{"role": "user", "content": user.quiz_topic}]
        prompt += (
            f"\nБюджет текущего ответа — {self.settings.llm_max_output_tokens} токенов. "
            "Выдай законченную часть, уложившись в этот бюджет; не обещай неограниченный объём."
        )
        if continuing:
            prompt += (
                "\nПоследний ответ оборван. Продолжи его с места остановки без повторения "
                "уже отправленного текста. Заверши текущую часть; если нужен код, открой новый "
                "блок кода. Не считай слово «Продолжи» ответом на вопрос тренировки."
            )
        messages = build_context(
            prompt,
            history,
            text,
            self.settings,
            pinned=pinned,
            preserve_last=continuing,
            schema=schema,
        )
        if user.mode == "study" and self.agent is not None:
            result = await self.agent.respond(
                user, message_id, text, prompt=prompt, history=history, continuing=continuing
            )
        elif quiz_state is not None:
            result = await self.llm.decide(messages, user.temperature, decision_schema=schema)
            if result.tool_calls:
                raise ModelError(INVALID_QUIZ)
        else:
            result = await self.llm.complete(messages, user.temperature)
        if getattr(result, "reminder", None) is not None and self.confirmations is not None:
            try:
                card = await self.confirmations.prepare(
                    user,
                    message_id,
                    text,
                    result.reminder,
                    **({"expected_epoch": edit_epoch} if edit_epoch is not None else {}),
                )
            except ConfirmationError as error:
                raise ModelError(error.user_message) from None
            # prepare saves the action and exact user/card pair in one transaction.
            return card.card_text
        answer = result.text
        if quiz_state is not None:
            answer, quiz = quiz_state.render(result.text, truncated=result.truncated)
        if result.truncated:
            answer = TRUNCATED_NOTICE + "\n\n" + answer
        if not await self.repository.save_exchange(
            user, message_id, text, answer, is_complete=not result.truncated, **quiz
        ):
            raise ModelError("Диалог изменился во время ответа. Отправь вопрос ещё раз.")
        return answer

    async def confirmation_card(self, user_id: int, message_id: int):
        if self.confirmations is None:
            return None
        return await self.confirmations.repository.lookup_card(user_id, message_id)

    async def confirm(self, user_id: int, action_id: str) -> str:
        if self.confirmations is None:
            return "Подтверждение напоминаний сейчас недоступно."
        return await self.confirmations.confirm(user_id, action_id)

    async def cancel_confirmation(self, user_id: int, action_id: str) -> str:
        if self.confirmations is None:
            return "Подтверждение напоминаний сейчас недоступно."
        return await self.confirmations.cancel(user_id, action_id)

    async def invalidate_confirmations(self, user_id: int) -> bool:
        if self.timezone_ui is not None:
            self.timezone_ui.clear(user_id)
        if self.confirmations is None:
            return False
        return await self.confirmations.invalidate(user_id)

    async def timezone(self, user_id: int, argument: str | None = None) -> str:
        if self.confirmations is None:
            return "Настройка часового пояса сейчас недоступна."
        await self.repository.user(user_id)
        if argument is not None and self.timezone_ui is not None:
            self.timezone_ui.clear(user_id)
        return await self.confirmations.timezone(user_id, argument)

    @staticmethod
    def timezone_is_direct(argument: str | None) -> bool:
        from app.timezone_ui import TimezoneUI

        return TimezoneUI.direct_timezone(argument) is not None

    async def timezone_menu(self, user_id: int, argument: str | None = None):
        if self.timezone_ui is None:
            return await self.timezone(user_id, argument)
        await self.repository.user(user_id)
        return await self.timezone_ui.open(user_id, argument)

    async def timezone_callback(self, user_id: int, data: str):
        if self.timezone_ui is None:
            return "Меню недоступно. Используй /timezone Europe/Moscow."
        return await self.timezone_ui.handle(user_id, data)

    def timezone_callback_is_apply(self, user_id: int, data: str) -> bool:
        return self.timezone_ui is not None and self.timezone_ui.is_apply(user_id, data)

    def timezone_search_active(self, user_id: int) -> bool:
        return self.timezone_ui is not None and self.timezone_ui.accepts_input(user_id)

    async def timezone_input(self, user_id: int, text: str):
        if self.timezone_ui is None:
            return None
        return await self.timezone_ui.search_input(user_id, text)

    async def why(self, user_id: int) -> str:
        if self.confirmations is None:
            return "Решений для объяснения пока нет."
        return await self.confirmations.why(user_id)

    async def tools(self) -> str:
        mcp = self.agent.mcp if self.agent is not None else None
        if mcp is None or not mcp.available:
            return "Инструменты сейчас недоступны. Обычные вопросы по-прежнему доступны."
        lines = ["Возможности режима /study:"]
        for name, metadata in mcp.tools.items():
            kind = (
                "чтение"
                if metadata.get("annotations", {}).get("readOnlyHint") is True
                else "изменение после подтверждения"
            )
            lines.append(f"{name} ({kind}) — {metadata['description']}")
        for uri, metadata in mcp.resources.items():
            lines.append(f"{uri} (чтение) — {metadata['description']}")
        lines.append("Напоминания сохраняются; отправка по времени пока не выполняется.")
        return "\n".join(lines)

    async def mode(self, user_id: int, mode: str) -> str:
        await self.repository.user(user_id)
        await self.repository.change_mode(user_id, mode)
        hints = {
            "study": "Отправь вопрос по программированию.",
            "translate": "Укажи язык перевода и отправь текст.",
            "quiz": "Назови тему по программированию — начнём тренировку из пяти вопросов.",
        }
        return f"Режим: {MODE_NAMES[mode]}. История очищена. {hints[mode]}"

    async def reset(self, user_id: int) -> str:
        await self.repository.user(user_id)
        await self.repository.reset(user_id)
        return "История очищена. Режим и температура сохранены."

    async def preferences(self, user_id: int, argument: str | None = None) -> str:
        if argument:
            try:
                temperature = Decimal(argument.strip().replace(",", "."))
                if not temperature.is_finite() or not Decimal(0) <= temperature <= Decimal(1):
                    raise ValueError
            except (InvalidOperation, ValueError):
                return "Укажи число от 0 до 1 включительно, например /settings 0.55."
            await self.repository.user(user_id)
            await self.repository.set_temperature(user_id, float(temperature))
        user = await self.repository.user(user_id)
        return (
            f"Режим: {MODE_NAMES[user.mode]}\nМодель: {self.settings.openrouter_model}\n"
            f"Температура: {user.temperature}\n"
            "Изменить: /settings <число от 0 до 1>, например /settings 0.55."
        )
