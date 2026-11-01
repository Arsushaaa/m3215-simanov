"""Bounded literal prompt-echo defense, independent of user intent classification.

Prompts must contain no secrets. This catches long literal copies, not all semantic
paraphrases; model instructions and capability isolation remain separate controls.
"""

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
