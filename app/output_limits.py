"""Оценка общего объёма ответа, который может состоять из нескольких генераций."""

import re
from dataclasses import dataclass
from decimal import Decimal

from app.llm import ModelError

OUTPUT_SIZE = re.compile(
    r"\b(?:напиши|написать|дай|составь|создай|сделай|write|generate|produce)\b"
    r".{0,100}?\b(?:на|из|объёмом|объемом|длиной|of|in)\s+"
    r"(\d[\d\s_]*(?:[.,]\d+)?\s*[кk]?)\s*"
    r"(слов(?:а|о)?|символ(?:ов|а)?|знак(?:ов|а)?|words?|characters?|chars?)\b",
    re.IGNORECASE,
)
RELATIVE_SIZE = re.compile(
    r"\b(?:в\s+)?(два|двое|три|трое|четыре|пять|десять|\d+(?:[.,]\d+)?)\s+"
    r"раз(?:а)?\s+(?:длинн|больш|объёмн|объемн|подробн)\w*",
    re.IGNORECASE,
)
MULTIPLIERS = {"два": 2, "двое": 2, "три": 3, "трое": 3, "четыре": 4, "пять": 5, "десять": 10}


@dataclass(frozen=True)
class OutputTarget:
    amount: int
    unit: str

    @property
    def estimated_tokens(self) -> int:
        return int(self.amount * (3 if self.unit == "слов" else Decimal("0.67")))

    def count(self, text: str) -> int:
        return len(re.findall(r"\w+(?:[-’']\w+)*", text)) if self.unit == "слов" else len(text)


def output_target(
    text: str, total_budget: int, previous_answer: str | None = None
) -> OutputTarget | None:
    candidates = []
    for match in OUTPUT_SIZE.finditer(text):
        amount, unit = match.groups()
        amount = re.sub(r"[\s_]", "", amount).lower().replace(",", ".")
        multiplier = 1000 if amount.endswith(("к", "k")) else 1
        candidates.append(
            OutputTarget(
                int(Decimal(amount.rstrip("кk")) * multiplier),
                "слов" if unit.lower().startswith(("слов", "word")) else "символов",
            )
        )
    if previous_answer:
        for match in RELATIVE_SIZE.finditer(text):
            amount = match.group(1).lower()
            multiplier = (
                Decimal(MULTIPLIERS[amount])
                if amount in MULTIPLIERS
                else Decimal(amount.replace(",", "."))
            )
            candidates.append(OutputTarget(int(len(previous_answer) * multiplier), "символов"))
    target = max(candidates, key=lambda t: t.estimated_tokens, default=None)
    if target and target.estimated_tokens > total_budget:
        raise ModelError("Запрошенный объём превышает предел одного задания. Укажи меньший объём.")
    return target
