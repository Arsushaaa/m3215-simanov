"""Bounded literal prompt-echo defense, independent of user intent classification.

Prompts must contain no secrets. This catches long literal copies, not all semantic
paraphrases; model instructions and capability isolation remain separate controls.
"""

import json

from app.llm import ModelError


def prevent_prompt_echo(text, messages):
    if not isinstance(text, str):
        return
    normalized = " ".join(text.casefold().split())
    width = 80
    if len(normalized) < width:
        return
    output = {normalized[i : i + width] for i in range(len(normalized) - width + 1)}
    for message in messages:
        if message.get("role") != "system" or not isinstance(message.get("content"), str):
            continue
        instruction = " ".join(message["content"].casefold().split())
        if any(instruction[i : i + width] in output for i in range(len(instruction) - width + 1)):
            raise ModelError(
                "Не могу показать внутренние инструкции. Могу объяснить возможности бота."
            )


def prevent_decision_echo(result, messages):
    """Inspect decoded text fields too: JSON escaping must not bypass the guard."""
    candidates = [result.text]
    candidates.extend(call["function"]["arguments"] for call in result.tool_calls)
    for raw in candidates:
        prevent_prompt_echo(raw, messages)
        try:
            decoded = json.loads(raw)
        except (ValueError, TypeError, RecursionError):
            # Invalid decisions are rejected by their existing schema/parser.
            continue
        pending = [decoded]
        while pending:
            value = pending.pop()
            if isinstance(value, str):
                prevent_prompt_echo(value, messages)
            elif isinstance(value, dict):
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
