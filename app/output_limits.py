"""Предварительная проверка запрошенного объёма, без вызовов модели."""

import re
from decimal import Decimal

from app.llm import ModelError

LARGE_OUTPUT = (
    "Такой объём не поместится в один ответ модели. Попроси текст по частям: "
    "например, «Напиши первую часть эссе на 500 слов», затем следующую часть. "
    "Готовый ответ я разделю на сообщения для Telegram."
)
# Цитаты и код являются входным материалом, а не указанием размера ответа.
QUOTED_INPUT = re.compile(r'```[\s\S]*?```|`[^`\n]*`|«[^»]*»|“[^”]*”|"[^"\n]*"|\'[^\'\n]*\'')
OUTPUT_REQUEST = re.compile(
    r"^\s*(?:(?:пожалуйста|можешь|можете|можно|ты\s+можешь|can\s+you|could\s+you|please)[,\s]+)*"
    r"(?:напиши|написать|дай|составь|создай|сделай|write|generate|produce|compose|create|give)\b",
    re.IGNORECASE,
)
OUTPUT_OBJECT = re.compile(
    r"\b(?:эссе|текст\w*|ответ\w*|рассказ\w*|сообщени\w*|сочинени\w*|истори\w*|стать\w*|"
    r"essay|text|answer|story|response|article|message)\b",
    re.IGNORECASE,
)
OBJECT_CONTROLS = re.compile(
    r"\b(?:мне|нам|пожалуйста|такой|такое|такую|же|его|это|an?|the|me|us|first|part|"
    r"перв\w*|следующ\w*|часть|коротк\w*|кратк\w*|небольш\w*|подробн\w*|академическ\w*|"
    r"формальн\w*|научн\w*|short|detailed|academic|formal|long|huge|enormous|massive|"
    r"gigantic|endless|unlimited|very|as|possible|огромн\w*|гигантск\w*|колоссальн\w*|"
    r"бесконечн\w*|безразмерн\w*|длинн\w*|больш\w*|максимально|максимальн\w*|"
    r"сам\w*|очень|невероятно|как|можно|много|больше|длиннее)\b",
    re.IGNORECASE,
)
SIZE_UNITS = r"(?:слов(?:а|о)?|символ(?:ов|а)?|знак(?:ов|а)?|words?|characters?|chars?)"
OUTPUT_SIZE = re.compile(
    r"\b(?:на|из|объёмом|объемом|длиной|размером|of|in|with)\s*"
    r"(\d[\d\s_]*(?:[.,]\d+)?\s*[кk]?)\s*(" + SIZE_UNITS + r")\b",
    re.IGNORECASE,
)
# После «изложения текста ...» или «эссе о тексте ...» размер относится к источнику.
INPUT_OBJECT = re.compile(
    r"\b(?:текста|строки|материала|документа|статьи|исходника|входа)\b|"
    r"\b(?:о|об|про|из|для|по|на\s+основе|of|about|from)\s+"
    r"(?:(?:эт\w*|данн\w*|следующ\w*|входн\w*|огромн\w*|больш\w*|"
    r"гигантск\w*|the|a|an|input|following|large|huge)\s+){0,3}"
    r"(?:текст\w*|строк\w*|материал\w*|документ\w*|стать\w*|эссе|text|essay|article|document)\b",
    re.IGNORECASE,
)
UNBOUNDED_ADJECTIVE = (
    r"(?:огромн\w*|гигантск\w*|колоссальн\w*|бесконечн\w*|безразмерн\w*|"
    r"(?:максимально|сам\w*|очень|невероятно)\s+(?:длинн|больш)\w*|"
    r"huge|enormous|massive|gigantic|endless|unlimited|very\s+long)"
)
UNBOUNDED_AMOUNT = re.compile(
    r"\b(?:" + UNBOUNDED_ADJECTIVE + r"\s+(?:(?:количеств|числ|объ[её]м)\w*\s+)?|"
    r"(?:очень\s+много|как\s+можно\s+больше|максимум)\s+|"
    r"(?:a\s+)?(?:huge|enormous|massive|unlimited)\s+(?:number|amount)\s+of\s+)"
    r"(?:" + SIZE_UNITS + r"|текст\w*|text)\b|"
    r"\b(?:как\s+можно\s+большее|максимальн\w*)\s+количество\s+" + SIZE_UNITS + r"\b|"
    r"\bas\s+many\s+" + SIZE_UNITS + r"\s+as\s+(?:you\s+can|possible)\b",
    re.IGNORECASE,
)
UNBOUNDED_SIZE = re.compile(
    r"\b" + UNBOUNDED_ADJECTIVE + r"\s+" + OUTPUT_OBJECT.pattern + r"|"
    r"\b(?:насколько\s+(?:больш|длинн)\w*.{0,40}?мож|"
    r"(?:максимально|как\s+можно)\s+(?:больш|длинн)\w*|"
    r"as\s+long\s+as\s+(?:you\s+can|possible)|without\s+(?:a\s+)?(?:length\s+)?limit|"
    r"без\s+ограничени\w*\s+(?:по\s+)?(?:объ[её]м\w*|длин\w*))",
    re.IGNORECASE,
)
TOPIC_PREFIX = re.compile(
    r"\b(?:о|об|про|на\s+тему|about|on\s+the\s+topic\s+of)\s+$", re.IGNORECASE
)
RELATIVE_SIZE = re.compile(
    r"\b(?:в\s+)?(два|двое|три|трое|четыре|пять|десять|\d+(?:[.,]\d+)?)\s+"
    r"раз(?:а)?\s+(?:длинн|больш|объёмн|объемн|подробн)\w*",
    re.IGNORECASE,
)
MULTIPLIERS = {"два": 2, "двое": 2, "три": 3, "трое": 3, "четыре": 4, "пять": 5, "десять": 10}


def _output_body(text: str, output_followup: bool) -> str | None:
    if output_followup:
        return text
    request = OUTPUT_REQUEST.match(text)
    if not request:
        return None
    body = text[request.end() :]
    for artifact in OUTPUT_OBJECT.finditer(body):
        prefix = OBJECT_CONTROLS.sub("", body[: artifact.start()])
        if not re.sub(r"[\s.,:;—–\-]", "", prefix):
            # Сохраняем прилагательное перед объектом, но не включаем сам объект
            # в поиск входного материала: «текст» здесь — запрошенный ответ.
            source = INPUT_OBJECT.search(body, artifact.end())
            return body[: source.start()] if source else body
    # Допускаем прямую просьбу об объёме без названия жанра: «напиши на 10000 слов».
    if OUTPUT_SIZE.match(body.strip()) or UNBOUNDED_AMOUNT.match(body.strip()):
        return body
    return None


def check_output_size(
    text: str,
    token_budget: int,
    previous_answer: str | None = None,
    *,
    output_followup: bool = False,
) -> None:
    text = QUOTED_INPUT.sub(" ", text)
    body = _output_body(text, output_followup)
    if body is not None:
        for match in OUTPUT_SIZE.finditer(body):
            amount, unit = match.groups()
            amount = re.sub(r"[\s_]", "", amount).lower().replace(",", ".")
            multiplier = 1000 if amount.endswith(("к", "k")) else 1
            count = Decimal(amount.rstrip("кk")) * multiplier
            estimate = count * (3 if unit.lower().startswith(("слов", "word")) else Decimal("0.67"))
            if estimate > token_budget:
                raise ModelError(LARGE_OUTPUT)
        for pattern in (UNBOUNDED_SIZE, UNBOUNDED_AMOUNT):
            for match in pattern.finditer(body):
                if not TOPIC_PREFIX.search(body[: match.start()]):
                    raise ModelError(LARGE_OUTPUT)
    if previous_answer and (
        OUTPUT_REQUEST.match(text) or output_followup or RELATIVE_SIZE.match(text.strip())
    ):
        for match in RELATIVE_SIZE.finditer(text):
            amount = match.group(1).lower()
            multiplier = (
                Decimal(MULTIPLIERS[amount])
                if amount in MULTIPLIERS
                else Decimal(amount.replace(",", "."))
            )
            estimate = (len(previous_answer.encode("utf-8")) + 2) // 3
            if estimate * multiplier > token_budget:
                raise ModelError(LARGE_OUTPUT)
