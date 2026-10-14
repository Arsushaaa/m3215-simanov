"""Сервис диалога: порядок запросов, лимиты контекста и настройки пользователя."""

import asyncio
import re
from decimal import Decimal, InvalidOperation
from weakref import WeakValueDictionary

from app.config import Settings
from app.llm import ModelError, OpenRouterClient
from app.prompts import MODE_NAMES, PROMPTS
from app.repository import DialogRepository

LARGE_OUTPUT = (
    "Такой объём не поместится в один ответ модели. Попроси текст по частям: "
    "например, «Напиши первую часть эссе на 500 слов», затем следующую часть. "
    "Готовый ответ я разделю на сообщения для Telegram."
)
OUTPUT_SIZE = re.compile(
    r"\b(?:напиши|написать|дай|составь|создай|сделай|write|generate|produce)\b"
    r".{0,100}?\b(?:на|из|объёмом|объемом|длиной|of|in)\s+"
    r"(\d[\d\s_]*(?:[.,]\d+)?\s*[кk]?)\s*"
    r"(слов(?:а|о)?|символ(?:ов|а)?|знак(?:ов|а)?|words?|characters?|chars?)\b",
    re.IGNORECASE,
)


def check_output_size(text: str, token_budget: int) -> None:
    # Предварительная оценка только явного запроса на объём ответа, не размера входа.
    for match in OUTPUT_SIZE.finditer(text):
        amount, unit = match.groups()
        amount = re.sub(r"[\s_]", "", amount).lower().replace(",", ".")
        multiplier = 1000 if amount.endswith(("к", "k")) else 1
        count = Decimal(amount.rstrip("кk")) * multiplier
        estimate = count * (3 if unit.lower().startswith(("слов", "word")) else Decimal("0.67"))
        if estimate > token_budget:
            raise ModelError(LARGE_OUTPUT)


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
) -> list[dict[str, str]]:
    system, current = {"role": "system", "content": prompt}, {"role": "user", "content": text}
    pinned = pinned or []
    if estimate_tokens([system, *pinned, current]) > settings.history_max_tokens:
        raise ModelError("Сообщение слишком длинное для контекста. Сократи его и попробуй снова.")
    # Удаляем самые старые полные пары. В БД архив остаётся до /reset или смены режима.
    recent = list(history)
    while recent and (
        len(pinned) + len(recent) + 1 > settings.history_max_messages
        or estimate_tokens([system, *pinned, *recent, current]) > settings.history_max_tokens
    ):
        del recent[:2]
    return [system, *pinned, *recent, current]


class DialogService:
    def __init__(self, repository: DialogRepository, llm: OpenRouterClient, settings: Settings):
        self.repository, self.llm, self.settings = repository, llm, settings
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
        check_output_size(text, self.settings.llm_max_output_tokens)
        user = await self.repository.user(user_id)
        history = await self.repository.history(user_id, self.settings.history_max_messages)
        prompt, pinned, quiz = PROMPTS[user.mode], [], {}
        if user.mode == "quiz":
            if user.quiz_finished:
                raise ModelError("Тренировка завершена. Для новой тренировки отправь /quiz.")
            hint = user.quiz_question > 0 and text.strip().casefold().startswith(
                ("подсказ", "дай подсказ", "нужна подсказ", "можно подсказ", "подскажи")
            )
            finished = user.quiz_question == 5 and not hint
            number = user.quiz_question if hint else min(user.quiz_question + 1, 5)
            quiz = {
                "quiz_question": number,
                "quiz_topic": user.quiz_topic or text,
                "quiz_finished": finished,
            }
            if user.quiz_topic:
                # Тема является данными пользователя, поэтому не вставляется в system.
                pinned = [{"role": "user", "content": user.quiz_topic}]
            if hint:
                prompt += (
                    f"\nПользователь просит подсказку к вопросу {number} из пяти. "
                    "Дай только подсказку, не раскрывая ответа и не задавая новый вопрос."
                )
            elif finished:
                prompt += (
                    "\nСейчас пользователь отвечает на пятый вопрос. Оцени ответ и подведи "
                    "итог тренировки по доступной истории. Новых вопросов не задавай. "
                    "Если часть истории отсутствует, не выдумывай результаты старых ответов."
                )
            elif user.quiz_question == 0:
                prompt += "\nСейчас пользователь задаёт тему. Сформулируй первый вопрос из пяти."
            else:
                prompt += (
                    f"\nСейчас пользователь отвечает на вопрос {user.quiz_question} из пяти. "
                    f"Дай обратную связь и сформулируй только вопрос {number} из пяти. "
                )
        messages = build_context(prompt, history, text, self.settings, pinned=pinned)
        answer = await self.llm.complete(messages, user.temperature)
        if quiz:
            prefix = "Тренировка завершена." if quiz["quiz_finished"] else f"Вопрос {number}/5"
            if hint:
                prefix = f"Подсказка к вопросу {number}/5"
            answer = prefix + "\n\n" + answer
        if not await self.repository.save_exchange(user, message_id, text, answer, **quiz):
            raise ModelError("Диалог изменился во время ответа. Отправь вопрос ещё раз.")
        return answer

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
