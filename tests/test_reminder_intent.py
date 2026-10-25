from datetime import UTC, datetime

import pytest

from app.reminder_intent import (
    TIME_REQUIRED,
    is_confirmation_reply,
    reminder_context,
    requests_reminder_and_read,
)

NOW = datetime(2026, 10, 18, 7, tzinfo=UTC)
ZONE = "Europe/Moscow"


def history(user, assistant=TIME_REQUIRED):
    return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


@pytest.mark.parametrize(
    "text,moment",
    [
        ("Напомни завтра в 18:30 приготовить пирог", "2026-10-19T18:30:00+03:00"),
        (
            "Ты сможешь напомнить мне завтра в18 часов приготовить пирог?",
            "2026-10-19T18:00:00+03:00",
        ),
        ("Создай напоминание 25 октября в 09:00: отчёт", "2026-10-25T09:00:00+03:00"),
        ("Через два часа напомни сделать перерыв", "2026-10-18T12:00:00+03:00"),
        ("Напомни через 30 минут сделать перерыв", "2026-10-18T10:30:00+03:00"),
        ("Напомни: отчёт 2026-10-19T18:30:12.123456+03:00", "2026-10-19T18:30:12.123456+03:00"),
        ("Напомни: отчёт 2026-10-19T15:30:00Z", "2026-10-19T18:30:00+03:00"),
    ],
)
def test_explicit_user_date_time_and_duration_are_grounded(text, moment):
    context = reminder_context(text, [])
    assert context is not None and context.has_time and context.has_date
    assert context.matches(datetime.fromisoformat(moment), NOW, ZONE)
    assert not context.matches(datetime.fromisoformat(moment).replace(minute=59), NOW, ZONE)


def test_assistant_offer_and_textual_yes_never_supply_missing_time():
    request = "ты сможешь напомнить мне завтра приготовить пирог>"
    context = reminder_context("да", history(request, "Могу напомнить завтра в 18:00. Подойдёт?"))
    assert context is not None and context.has_date and not context.has_time
    assert context.prompt_text == request + "\nда"
    assert "18:00" not in context.prompt_text
    assert not context.matches(datetime.fromisoformat("2026-10-19T18:00:00+03:00"), NOW, ZONE)


def test_user_clock_followup_keeps_only_the_user_date():
    request = "Напомни завтра приготовить пирог"
    context = reminder_context("в 18:30", history(request))
    assert context is not None and context.has_time and context.has_date
    assert context.matches(datetime.fromisoformat("2026-10-19T18:30:00+03:00"), NOW, ZONE)
    assert not context.matches(datetime.fromisoformat("2026-10-19T18:00:00+03:00"), NOW, ZONE)


@pytest.mark.parametrize(
    "text",
    [
        "Напомни, что такое рекурсия",
        "Напомни мне синтаксис for в Python",
        "Напомни основы Python",
        "Как реализовать напоминания на Python?",
        "Объясни фразу «Напомни завтра в 18:30 испечь пирог»",
        'Что делает код `print("Напомни завтра в 18:30")`?',
    ],
)
def test_theory_and_quoted_examples_are_not_action_requests(text):
    assert reminder_context(text, []) is None


def test_new_question_and_new_request_do_not_inherit_old_parameters():
    old = history("Напомни завтра в 18:30 приготовить пирог")
    assert reminder_context("Что такое Python?", old) is None
    context = reminder_context("Напомни купить молоко", old)
    assert context is not None and not context.has_date and not context.has_time
    unrelated = [*old, *history("Что такое Python?", "Это язык программирования. Продолжить?")]
    assert reminder_context("да", unrelated) is None
    assert reminder_context("18:30", unrelated) is None


def test_short_reply_to_unrelated_assistant_question_does_not_reopen_reminder():
    assert (
        reminder_context("да", history("Напомни завтра испечь пирог", "Объяснить Python?")) is None
    )


@pytest.mark.parametrize("text", ["вечером", "в восемнадцать часов", "в шесть"])
def test_unsupported_clock_correction_does_not_keep_an_old_clock(text):
    context = reminder_context(text, history("Напомни завтра в 18:30 приготовить пирог"))
    assert context is not None and not context.has_time


def test_unsupported_weekday_correction_does_not_keep_an_old_date():
    context = reminder_context("в пятницу", history("Напомни завтра в 18:30 приготовить пирог"))
    assert context is not None and not context.has_date


@pytest.mark.parametrize("text", ["да", "в 18:30"])
def test_orphan_clarification_never_uses_assistant_parameters(text):
    context = reminder_context(text, [{"role": "assistant", "content": TIME_REQUIRED}])
    assert context is not None and context.is_orphaned
    assert not context.has_date and not context.has_time
    assert context.prompt_text == text


def test_orphan_detection_does_not_interrupt_programming_theory():
    assert (
        reminder_context(
            "да",
            [{"role": "assistant", "content": "Время в Python хранится в datetime. Продолжить?"}],
        )
        is None
    )


def test_only_eight_previous_pairs_are_examined():
    old = history("Напомни завтра приготовить пирог")
    for _ in range(8):
        old.extend(history("да"))
    context = reminder_context("в 18:30", old)
    assert context is not None and context.is_orphaned


@pytest.mark.parametrize("text", ["да", "давай", "ок", "Да!", "ОК."])
def test_simple_textual_confirmation(text):
    assert is_confirmation_reply(text)
    assert not is_confirmation_reply(text + " расскажи про Python")


@pytest.mark.parametrize(
    "text",
    [
        "Какая погода в Москве и напомни мне завтра в18:30 приготовить пирог",
        "Напомни завтра в18:30 приготовить пирог и покажи расписание на завтра",
        "Show the weather and remind me tomorrow at 18:30 to bake a pie",
    ],
)
def test_separate_read_and_reminder_requests_are_not_executed_partially(text):
    assert requests_reminder_and_read(text)
    assert not requests_reminder_and_read("Напомни завтра в18:30 посмотреть погоду")
    assert not requests_reminder_and_read("Напомни завтра в18:30 посмотреть погоду и расписание")


def test_compound_or_ambiguous_time_requires_clarification():
    for request in (
        "Напомни через два часа 30 минут испечь пирог",
        "Напомни через два часа и полчаса испечь пирог",
        "Напомни завтра через два часа испечь пирог",
        "Напомни завтра в18:30 или в19:00 испечь пирог",
        "Напомни завтра в18:30 и в19:00 испечь пирог",
    ):
        context = reminder_context(request, [])
        assert context is not None and not context.has_time


@pytest.mark.parametrize(
    "text,moment",
    [
        (
            "Remind me tomorrow at 11:00 to submit the report",
            "2026-10-19T11:00:00+03:00",
        ),
        ("Please remind me today at 3 PM to submit the report", "2026-10-18T15:00:00+03:00"),
        (
            "Could you please remind me the day after tomorrow at 11:30 a.m. to submit it?",
            "2026-10-20T11:30:00+03:00",
        ),
        ("Set a reminder tomorrow at 12 AM to submit the report", "2026-10-19T00:00:00+03:00"),
        ("Create a reminder tomorrow at 12 PM to submit the report", "2026-10-19T12:00:00+03:00"),
        ("I need a reminder tomorrow at 11:00 to submit the report", "2026-10-19T11:00:00+03:00"),
        ("In two hours remind me to submit the report", "2026-10-18T12:00:00+03:00"),
        ("Remind me in an hour to submit the report", "2026-10-18T11:00:00+03:00"),
        ("Remind me in half an hour to submit the report", "2026-10-18T10:30:00+03:00"),
        ("Remind me in 30 minutes to submit the report", "2026-10-18T10:30:00+03:00"),
        ("Remind me in 20 seconds to submit the report", "2026-10-18T10:00:20+03:00"),
        ("Remind me in two days to submit the report", "2026-10-20T10:00:00+03:00"),
    ],
)
def test_english_requests_use_only_explicit_user_date_clock_or_duration(text, moment):
    context = reminder_context(text, [])
    assert context is not None and context.has_date and context.has_time
    assert context.matches(datetime.fromisoformat(moment), NOW, ZONE)
    assert not context.matches(datetime.fromisoformat(moment).replace(minute=59), NOW, ZONE)


@pytest.mark.parametrize("clock", ["3 PM", "at 3 p.m.", "15:00", "at 15:00"])
def test_english_clock_followup_keeps_date_from_original_user_request(clock):
    context = reminder_context(
        clock, history("Remind me tomorrow to submit the report", "What time?")
    )
    assert context is not None and context.has_date and context.has_time
    assert context.matches(datetime.fromisoformat("2026-10-19T15:00:00+03:00"), NOW, ZONE)


def test_english_yes_does_not_use_time_invented_by_assistant():
    context = reminder_context(
        "yes",
        history(
            "Remind me tomorrow to submit the report", "I can remind you at 18:00. Is that OK?"
        ),
    )
    assert context is not None and context.has_date and not context.has_time
    assert "18:00" not in context.prompt_text


@pytest.mark.parametrize(
    "text",
    [
        "Remind me what recursion is",
        "Please remind me how Python decorators work",
        "Remind me the syntax of for in Python",
        'Explain the phrase "Remind me tomorrow at 11:00 to submit the report"',
    ],
)
def test_english_theory_and_quoted_examples_are_not_write_requests(text):
    assert reminder_context(text, []) is None


@pytest.mark.parametrize(
    "correction,has_date",
    [
        ("in the evening", True),
        ("at eleven", True),
        ("at 13 PM", True),
        ("at 0:30 AM", True),
        ("at 11:30 or 12:30", True),
        ("next Friday", False),
    ],
)
def test_english_unsupported_or_invalid_corrections_do_not_keep_old_parameter(correction, has_date):
    context = reminder_context(
        correction, history("Remind me tomorrow at 18:30 to submit the report")
    )
    assert context is not None and context.has_date == has_date
    assert not (context.has_date and context.has_time)


@pytest.mark.parametrize(
    "text",
    [
        "Remind me in two hours and 30 minutes to submit the report",
        "Remind me in two hours and half an hour to submit the report",
        "Remind me tomorrow in two hours to submit the report",
        "Remind me next Friday in two hours to submit the report",
    ],
)
def test_english_compound_or_conflicting_temporal_evidence_requires_clarification(text):
    context = reminder_context(text, [])
    assert context is not None and not context.has_time


def test_english_orphan_cannot_supply_original_request_or_assistant_time():
    context = reminder_context("at 11 AM", [{"role": "assistant", "content": "What time?"}])
    assert context is not None and context.is_orphaned
    assert not context.has_time and not context.has_date
