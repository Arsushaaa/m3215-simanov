"""Переходы викторины: тип ответа модели отделён от счётчика и отображения."""

import json
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.llm import ModelError

INVALID_QUIZ = (
    "Не удалось получить корректный ответ тренировки. Повтори сообщение; прогресс сохранён."
)
CLARIFY_HEADING = "Уточнение для тренировки"
FINISHED_HEADING = "Тренировка завершена."


class QuizTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    kind: Literal["question", "clarify", "hint", "finished"]
    feedback: str = Field(max_length=8000)
    text: str = Field(min_length=1, max_length=12000)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate field")
        result[key] = value
    return result


def parse_turn(raw):
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > 32 * 1024:
        raise ModelError(INVALID_QUIZ)
    try:
        return QuizTurn.model_validate(json.loads(raw, object_pairs_hook=_object))
    except (ValidationError, ValueError, TypeError, RecursionError):
        # Незакрытый JSON не является пользовательским текстом: не показываем его
        # и не сохраняем частичную операцию. Автоматической перегенерации нет.
        raise ModelError(INVALID_QUIZ) from None


_NUMBERED = re.compile(
    r"^\s*(?:\#{1,6}\s*)?(?P<emphasis>\*\*|__)?"
    r"(?:Подсказка к вопросу|Вопрос|Question)\s+№?\s*\d+"
    r"(?:\s*(?:/|из|of)\s*\d+)?\s*[:.)—–-]?\s*(?P=emphasis)?\s*[:.)—–-]?\s*",
    re.IGNORECASE,
)
_CONTROL = re.compile(
    r"^\s*(?:\#{1,6}\s*)?(?P<emphasis>\*\*|__)?"
    r"(?:Уточнение для тренировки|Тренировка завершена)"
    r"[.!:]?\s*(?P=emphasis)?\s*[.!:]?\s*",
    re.IGNORECASE,
)
_QUESTION = re.compile(r"Вопрос ([1-5])/5")
_HINT = re.compile(r"Подсказка к вопросу ([1-5])/5")


def _fence(line, current):
    marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
    if marker is None:
        return current, False
    value = marker[1]
    if current is None:
        return value, True
    if value[0] == current[0] and len(value) >= len(current):
        return None, True
    return current, True


def _prose_lines(text):
    """Не принимаем строку внутри блока/отступа кода за управляющий заголовок."""
    fence = None
    for line in text.splitlines():
        fence, delimiter = _fence(line, fence)
        if delimiter:
            continue
        if fence is None and not line.startswith("    "):
            yield line


def _close_feedback_fence(text):
    # Свой заголовок должен оставаться вне кода даже у частичной обратной связи.
    # Содержимое блока не меняется: добавляется только закрывающий разделитель.
    fence = None
    for line in text.splitlines():
        fence, _ = _fence(line, fence)
    return text + "\n" + fence if fence else text


def clean_body(text):
    # Модель не владеет номером. Удаляем только известные ведущие заголовки,
    # сохраняя смысловые числа, inline-код и блоки программного кода.
    while True:
        cleaned = _CONTROL.sub("", _NUMBERED.sub("", text, count=1), count=1).strip()
        if cleaned == text:
            break
        text = cleaned
    if any(_NUMBERED.match(line) or _CONTROL.match(line) for line in _prose_lines(text)):
        raise ModelError(INVALID_QUIZ)
    return text


@dataclass(frozen=True)
class QuizState:
    topic: str | None
    question: int
    request: str
    continuing: bool = False
    previous: str | None = None

    def pending_kind(self):
        if not self.continuing:
            return None
        markers = []
        for line in _prose_lines(self.previous or ""):
            value = line.strip()
            if match := _QUESTION.fullmatch(value):
                if self.question >= 5 or int(match[1]) != self.question + 1:
                    raise ModelError(INVALID_QUIZ)
                markers.append("question")
            elif match := _HINT.fullmatch(value):
                if self.question == 0 or int(match[1]) != self.question:
                    raise ModelError(INVALID_QUIZ)
                markers.append("hint")
            elif value == FINISHED_HEADING:
                if self.question != 5:
                    raise ModelError(INVALID_QUIZ)
                markers.append("finished")
            elif value == CLARIFY_HEADING:
                markers.append("clarify")
        if len(markers) != 1:
            raise ModelError(INVALID_QUIZ)
        return markers[0]

    def allowed_kinds(self):
        if pending := self.pending_kind():
            return [pending]
        if self.question == 0:
            return ["question", "clarify"]
        hint = (
            self.request.strip()
            .casefold()
            .startswith(("подсказ", "дай подсказ", "нужна подсказ", "можно подсказ", "подскажи"))
        )
        if hint:
            return ["hint", "clarify"]
        if self.question == 5:
            return ["hint", "clarify", "finished"]
        return ["question", "hint", "clarify"]

    def schema(self):
        schema = QuizTurn.model_json_schema()
        schema["properties"]["kind"]["enum"] = self.allowed_kinds()
        return schema

    def instruction(self):
        rules = (
            "\nВерни один JSON-объект по схеме тренировки: kind, feedback и text. "
            "kind=clarify — уточнение темы/сообщения без нового вопроса; kind=question — "
            "ровно один настоящий учебный вопрос по программированию в text; "
            "kind=hint — только подсказка; kind=finished — итог после ответа на пятый вопрос. "
            "feedback — оценка предыдущего ответа, если она нужна, иначе пустая строка. "
            "text не повторяет feedback. Не пиши собственные номера и заголовки вопросов: "
            "их добавляет программа. Отказ, личный вопрос или просьба назвать неизвестное "
            "имя не являются учебной темой или вопросом викторины. "
            "Нельзя считать любое сообщение ответом: при уточнении, просьбе о подсказке "
            "или смене темы не переходи дальше. Для смены уже принятой темы предложи /quiz. "
        )
        if pending := self.pending_kind():
            return rules + (
                f"Продолжи незаконченный ответ типа {pending}, сохрани этот kind. "
                "Не оценивай «Продолжи» как ответ и не начинай другой вопрос."
            )
        if self.question == 0:
            return rules + (
                "Сейчас пользователь задаёт тему. Если сообщение не задаёт понятную тему "
                "по программированию, верни clarify и попроси тему без номера вопроса. "
                "Если тема понятна, верни question — первый вопрос из пяти."
            )
        if "question" in self.allowed_kinds():
            return rules + (
                f"Сейчас пользователь отвечает на вопрос {self.question} из пяти. "
                "Если это ответ, дай обратную связь и сформулируй только "
                f"вопрос {self.question + 1} из пяти. Если это не ответ, уточни сообщение "
                "или дай запрошенную подсказку без нового вопроса."
            )
        if self.question == 5 and "finished" in self.allowed_kinds():
            return rules + (
                "Сейчас пользователь отвечает на пятый вопрос. Только после его ответа "
                "оцени результат и подведи итог; новых вопросов не задавай. Если это "
                "уточнение/подсказка, сохрани текущий шаг. Если часть истории отсутствует, "
                "не выдумывай результаты старых ответов."
            )
        return rules + (
            f"Пользователь просит подсказку к вопросу {self.question} из пяти. "
            "Дай только подсказку, не раскрывая ответа и не задавая новый вопрос."
        )

    def render(self, raw, *, truncated=False):
        turn = parse_turn(raw)
        if turn.kind not in self.allowed_kinds():
            raise ModelError(INVALID_QUIZ)
        text = clean_body(turn.text)
        feedback = _close_feedback_fence(clean_body(turn.feedback))
        if not text:
            raise ModelError(INVALID_QUIZ)
        number = self.question + 1 if turn.kind == "question" else self.question
        headings = {
            "question": f"Вопрос {number}/5",
            "hint": f"Подсказка к вопросу {number}/5",
            "clarify": CLARIFY_HEADING,
            "finished": FINISHED_HEADING,
        }
        answer = "\n\n".join(part for part in (feedback, headings[turn.kind], text) if part)
        return answer, {
            "quiz_question": number if not truncated else self.question,
            "quiz_topic": (self.topic or self.request) if turn.kind == "question" else self.topic,
            "quiz_finished": turn.kind == "finished" and not truncated,
        }
