"""Узкая проверка просьбы написать эссе без темы, без вызова модели."""

import re

ESSAY_TOPIC_QUESTION = (
    "На какую тему написать эссе? Укажи тему и желаемый объём. "
    "Большой текст можно запросить отдельными частями."
)

REQUEST = re.compile(
    r"^\s*(?:пожалуйста[,\s]+)?(?:напиши|написать|дай|составь|создай|сделай|write|generate|produce)\b",
    re.IGNORECASE,
)
ESSAY = re.compile(r"\b(?:эссе|essay)\b", re.IGNORECASE)
SIZE = re.compile(
    r"(?<!\w)(?:(?:на|из|объёмом|объемом|длиной|of|in)\s*)?"
    r"\d[\d\s_]*(?:[.,]\d+)?\s*[кk]?\s*"
    r"(?:слов(?:а|о)?|символ(?:ов|а)?|знак(?:ов|а)?|words?|characters?|chars?)\b",
    re.IGNORECASE,
)
LANGUAGE = re.compile(
    r"\b(?:на|in)\s+(?:русском|английском|russian|english)(?:\s+языке)?\b",
    re.IGNORECASE,
)
RELATIVE_SIZE = re.compile(
    r"\b(?:в\s+)?\d+(?:[.,]\d+)?\s+раз(?:а)?\s+(?:длинн|больш|подробн)\w*\b",
    re.IGNORECASE,
)
CONTROLS = re.compile(
    r"\b(?:мне|нам|пожалуйста|an?|the|first|part|long|short|huge|"
    r"перв\w*|следующ\w*|часть|коротк\w*|небольш\w*|огромн\w*|гигантск\w*|"
    r"больш\w*|длинн\w*|подробн\w*|академическ\w*|формальн\w*|научн\w*|стил\w*|"
    r"максимально|насколько|как|можно|"
    r"очень|сам\w*|ты|можешь|сможешь|можете|сможете|только|"
    r"такой|такое|такую|же|ответ|его|это|и|а|в|раза?|"
    r"два|двое|три|трое|четыре|пять|десять|люб\w*|какую|угодно|тему|выбери|придумай)\b",
    re.IGNORECASE,
)
PUNCTUATION = re.compile(r"[\s.,!?;:—–\-()]+")


def _only_controls(text: str) -> bool:
    text = SIZE.sub("", text)
    text = LANGUAGE.sub("", text)
    text = RELATIVE_SIZE.sub("", text)
    text = CONTROLS.sub("", text)
    return not PUNCTUATION.sub("", text)


def _essay_body(text: str) -> str | None:
    request = REQUEST.match(text)
    if not request:
        return None
    essay = ESSAY.search(text, request.end())
    if not essay or not _only_controls(text[request.end() : essay.start()]):
        # «Напиши функцию, которая печатает слово эссе» не является просьбой об эссе.
        return None
    return text[essay.end() :]


def _previous_topic(history: list[dict[str, str]]) -> bool:
    users = [index for index, item in enumerate(history) if item["role"] == "user"]
    if not users:
        return False
    index = users[-1]
    text = history[index]["content"]
    body = _essay_body(text)
    if body is not None:
        return not _only_controls(body)
    return (
        index > 0
        and history[index - 1]["role"] == "assistant"
        and history[index - 1]["content"] == ESSAY_TOPIC_QUESTION
        and not _only_controls(text)
    )


def needs_essay_topic(text: str, history: list[dict[str, str]]) -> bool:
    """Не выбирает тему по несвязанной истории; не применяется к переводам и quiz."""
    pending = bool(
        history
        and history[-1]["role"] == "assistant"
        and history[-1]["content"] == ESSAY_TOPIC_QUESTION
    )
    body = _essay_body(text)
    if body is None:
        return pending and (_only_controls(text) or bool(re.fullmatch(r"\s*\d+\s*", text)))
    if not _only_controls(body):
        # Содержательная тема, в том числе после «о», двоеточия или в кавычках.
        return False
    return pending or not _previous_topic(history)
