"""Предварительная проверка запрошенного объёма, без вызовов модели."""

import re
from decimal import Decimal

from app.llm import ModelError

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
OUTPUT_REQUEST = re.compile(
    r"^\s*(?:пожалуйста[,\s]+)?(?:напиши|дай|составь|создай|сделай|write|generate|produce)\b",
    re.IGNORECASE,
)
UNBOUNDED_SIZE = re.compile(
    r"\b(?:огромн\w*|гигантск\w*|бесконечн\w*|(?:максимально|сам\w*|очень)\s+(?:длинн|больш)\w*)\s+"
    r"(?:эссе|текст|ответ|рассказ|сообщени\w*|сочинени\w*)\b"
    r"|\b(?:эссе|текст|ответ|рассказ|сообщени\w*|сочинени\w*)\b.{0,70}?"
    r"(?:насколько\s+(?:больш|длинн)\w*.{0,40}?мож|(?:максимально|как\s+можно)\s+(?:больш|длинн)\w*)",
    re.IGNORECASE,
)
RELATIVE_SIZE = re.compile(
    r"\b(?:в\s+)?(два|двое|три|трое|четыре|пять|десять|\d+(?:[.,]\d+)?)\s+"
    r"раз(?:а)?\s+(?:длинн|больш|объёмн|объемн|подробн)\w*",
    re.IGNORECASE,
)
MULTIPLIERS = {"два": 2, "двое": 2, "три": 3, "трое": 3, "четыре": 4, "пять": 5, "десять": 10}


def check_output_size(text: str, token_budget: int, previous_answer: str | None = None) -> None:
    # Размер входного материала не считается просьбой написать столько же текста.
    for match in OUTPUT_SIZE.finditer(text):
        amount, unit = match.groups()
        amount = re.sub(r"[\s_]", "", amount).lower().replace(",", ".")
        multiplier = 1000 if amount.endswith(("к", "k")) else 1
        count = Decimal(amount.rstrip("кk")) * multiplier
        estimate = count * (3 if unit.lower().startswith(("слов", "word")) else Decimal("0.67"))
        if estimate > token_budget:
            raise ModelError(LARGE_OUTPUT)
    if OUTPUT_REQUEST.search(text) and UNBOUNDED_SIZE.search(text):
        raise ModelError(LARGE_OUTPUT)
    if previous_answer:
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
