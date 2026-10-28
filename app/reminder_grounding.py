"""Validate source references, not the meaning of natural-language phrases.

The LLM interprets language and computes a proposed moment. Code verifies exact
user quotations, retained trusted card fields and calendar constraints; the
owner reviews the resulting card before any write is authorized.
"""

import json
from zoneinfo import ZoneInfo

from app.mcp.contracts import PendingCardReference, ReminderEvidence, UserQuote


def user_sources(history, text):
    previous = [item["content"] for item in history if item["role"] == "user"]
    return {0: text, **{i: value for i, value in enumerate(reversed(previous), 1)}}


def indexed_message(text, index=0):
    """Label the original user text without letting it supply its own index."""
    return json.dumps({"message_index": index, "text": text}, ensure_ascii=False)


def indexed_history(history):
    result = [dict(item) for item in history]
    index = 0
    for item in reversed(result):
        if item["role"] == "user":
            index += 1
            item["content"] = indexed_message(item["content"], index)
    return result


def validate_evidence(evidence: ReminderEvidence, arguments, sources, pending, timezone):
    for ref in (evidence.request, evidence.text, evidence.date, evidence.time):
        if isinstance(ref, UserQuote):
            source = sources.get(ref.message_index)
            if source is None or ref.quote not in source:
                raise ValueError("Unverified user source")
        elif pending is None:
            raise ValueError("No trusted pending card")
    if pending is None:
        return
    zone = ZoneInfo(timezone)
    old = pending.arguments.remind_at.astimezone(zone)
    new = arguments.remind_at.astimezone(zone)
    retained = (
        (evidence.text, arguments.text == pending.arguments.text),
        (evidence.date, new.date() == old.date()),
        (evidence.time, new.timetz().replace(tzinfo=None) == old.timetz().replace(tzinfo=None)),
    )
    if any(isinstance(ref, PendingCardReference) and not matches for ref, matches in retained):
        raise ValueError("Retained field changed")
