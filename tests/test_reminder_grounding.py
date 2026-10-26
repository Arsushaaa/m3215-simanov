"""Validate source integrity without interpreting natural-language time."""

from copy import deepcopy
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.mcp.contracts import ReminderArguments, ReminderEvidence
from app.reminder_grounding import indexed_history, user_sources, validate_evidence

QUESTION = "Не забудь мне напомнить в 11:00 завтра приготовить пирог"
ARGUMENTS = ReminderArguments(text="Приготовить пирог", remind_at="2030-01-02T11:00:00+03:00")


def quote(text=QUESTION, index=0):
    return {"source": "user", "message_index": index, "quote": text}


def evidence(**changes):
    return ReminderEvidence.model_validate(
        {
            **{name: quote() for name in ("request", "text", "date", "time")},
            **changes,
        }
    )


def test_source_indexing_omits_assistant_text_and_keeps_current_at_zero():
    history = [
        {"role": "user", "content": "Старая просьба"},
        {"role": "assistant", "content": "Придуманные параметры в 18:00"},
        {"role": "user", "content": "Последняя просьба"},
    ]
    before = deepcopy(history)
    assert user_sources(history, QUESTION) == {
        0: QUESTION,
        1: "Последняя просьба",
        2: "Старая просьба",
    }
    indexed = indexed_history(history)
    assert indexed[1] == history[1]
    assert '"message_index": 2' in indexed[0]["content"]
    assert '"message_index": 1' in indexed[2]["content"]
    assert history == before


def test_exact_user_quotes_are_valid_without_a_phrase_dictionary():
    validate_evidence(evidence(), ARGUMENTS, {0: QUESTION}, None, "Europe/Moscow")


@pytest.mark.parametrize(
    "bad_ref",
    [
        quote("В 18:00"),
        quote(QUESTION, 20),
        quote("не забудь"),
        quote("11.00"),
    ],
)
def test_forged_or_not_exact_quote_is_rejected(bad_ref):
    with pytest.raises(ValueError, match="Unverified"):
        validate_evidence(evidence(time=bad_ref), ARGUMENTS, {0: QUESTION}, None, "Europe/Moscow")


def test_assistant_quote_cannot_be_forged_as_a_user_source():
    sources = user_sources(
        [{"role": "assistant", "content": "Завтра в 11:00 приготовить пирог"}],
        "да",
    )
    with pytest.raises(ValueError, match="Unverified"):
        validate_evidence(
            evidence(time=quote("11:00", 1)), ARGUMENTS, sources, None, "Europe/Moscow"
        )


@pytest.mark.parametrize("field", ["request", "text", "date", "time"])
def test_evidence_never_accepts_an_assistant_source_tag(field):
    with pytest.raises(ValidationError):
        evidence(**{field: {"source": "assistant", "message_index": 1, "quote": "11:00"}})


def test_request_itself_cannot_be_inherited_from_a_pending_card():
    with pytest.raises(ValidationError):
        evidence(request={"source": "pending_card"})


@pytest.mark.parametrize("field", ["text", "date", "time"])
def test_pending_field_requires_a_real_pending_snapshot(field):
    with pytest.raises(ValueError, match="No trusted"):
        validate_evidence(
            evidence(**{field: {"source": "pending_card"}}),
            ARGUMENTS,
            {0: QUESTION},
            None,
            "Europe/Moscow",
        )


def test_date_and_task_can_be_retained_while_time_comes_from_new_user_turn():
    changed = ARGUMENTS.model_copy(update={"remind_at": ARGUMENTS.remind_at.replace(hour=21)})
    refs = evidence(
        request=quote("Лучше в 21:00"),
        text={"source": "pending_card"},
        date={"source": "pending_card"},
        time=quote("21:00"),
    )
    validate_evidence(
        refs,
        changed,
        {0: "Лучше в 21:00"},
        SimpleNamespace(arguments=ARGUMENTS),
        "Europe/Moscow",
    )


@pytest.mark.parametrize("field", ["text", "date", "time"])
def test_a_field_declared_retained_cannot_change(field):
    updates = {
        "text": {"text": "Другая задача"},
        "date": {"remind_at": ARGUMENTS.remind_at + timedelta(days=1)},
        "time": {"remind_at": ARGUMENTS.remind_at.replace(hour=21)},
    }
    with pytest.raises(ValueError, match="Retained field changed"):
        validate_evidence(
            evidence(**{field: {"source": "pending_card"}}),
            ARGUMENTS.model_copy(update=updates[field]),
            {0: QUESTION},
            SimpleNamespace(arguments=ARGUMENTS),
            "Europe/Moscow",
        )


def test_retained_fields_compare_in_saved_timezone_not_iso_spelling():
    utc = ARGUMENTS.model_copy(
        update={"remind_at": datetime.fromisoformat("2030-01-02T08:00:00+00:00")},
    )
    refs = evidence(**{field: {"source": "pending_card"} for field in ("text", "date", "time")})
    validate_evidence(
        refs,
        utc,
        {0: QUESTION},
        SimpleNamespace(arguments=ARGUMENTS),
        "Europe/Moscow",
    )
