"""Напоминание требует явного времени пользователя и отдельной карточки."""

import json
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.agent import CHOOSE_ONE, AgentService
from app.llm import RoutingCompletion
from app.mcp.contracts import tool_schemas
from app.prompts import PROMPTS
from app.repository import UserSettings

NOW = datetime(2026, 10, 19, 7, tzinfo=UTC)
REQUEST = "Напомни мне завтра в 18:30 приготовить пирог"
FORGED_SUCCESS = "Напоминание установлено. Я напомню завтра в 18:30 приготовить пирог."
REMINDER_ARGUMENTS = {
    "text": "Приготовить пирог",
    "remind_at": "2026-10-20T18:30:00+03:00",
}


def structured(action="respond", **fields):
    return RoutingCompletion(
        text=json.dumps({"action": action, "reason": "Выбор действия.", **fields}),
        tool_calls=[],
        truncated=False,
    )


def reminder_decision(mode="structured", **overrides):
    arguments = {**REMINDER_ARGUMENTS, **overrides}
    if mode == "structured":
        return structured("call_tool", name="add_reminder", arguments=arguments)
    return RoutingCompletion(
        text=None,
        tool_calls=[
            {
                "id": "call_reminder",
                "type": "function",
                "function": {"name": "add_reminder", "arguments": json.dumps(arguments)},
            }
        ],
        truncated=False,
    )


@pytest.fixture
def reminder_agent(bot_settings):
    llm = SimpleNamespace(
        decide=AsyncMock(return_value=structured(text=FORGED_SUCCESS)),
        complete=AsyncMock(),
    )
    mcp = SimpleNamespace(
        available=True,
        tools={
            name: {"name": name, "description": name, **tool_schemas(name)}
            for name in ("get_weather", "get_schedule", "add_reminder")
        },
        resources={"schedule://current-week": {}},
        call_tool=AsyncMock(),
        read_resource=AsyncMock(),
    )
    repository = SimpleNamespace(
        timezone=AsyncMock(return_value="Europe/Moscow"),
        has_processed=AsyncMock(return_value=False),
        record=AsyncMock(),
    )
    agent = AgentService(llm, mcp, repository, bot_settings, now=lambda: NOW)
    return SimpleNamespace(agent=agent, llm=llm, mcp=mcp, repository=repository)


async def ask(setup, text=REQUEST, *, history=None):
    return await setup.agent.respond(
        UserSettings(1, "study", 0.3, 0),
        10,
        text,
        prompt=PROMPTS["study"],
        history=history or [],
    )


def assert_no_execution(setup, *, llm_calls):
    assert setup.llm.decide.await_count == llm_calls
    setup.llm.complete.assert_not_awaited()
    setup.mcp.call_tool.assert_not_awaited()
    setup.mcp.read_resource.assert_not_awaited()
    audit = setup.repository.record.call_args.kwargs
    assert (audit["llm_calls"], audit["mcp_calls"]) == (llm_calls, 0)


def assert_safe_refusal(reply):
    assert reply.reminder is None
    assert reply.text != FORGED_SUCCESS
    assert any(
        phrase in reply.text.casefold()
        for phrase in ("не сохран", "не установ", "уточни", "повтори", "подтвержден")
    )


async def test_missing_time_is_clarified_locally_without_model_or_mcp(reminder_agent):
    reply = await ask(reminder_agent, "ты сможешь напомнить мне завтра приготовить пирог")

    assert reply.reminder is None
    assert any(phrase in reply.text.casefold() for phrase in ("время", "во сколько"))
    assert_no_execution(reminder_agent, llm_calls=0)


async def test_yes_does_not_adopt_time_invented_by_legacy_assistant(reminder_agent):
    history = [
        {"role": "user", "content": "ты сможешь напомнить мне завтра приготовить пирог"},
        {
            "role": "assistant",
            "content": "Хорошо, напомню завтра в 18:00 приготовить пирог. Подтверждаешь?",
        },
    ]
    reply = await ask(reminder_agent, "да", history=history)

    assert reply.reminder is None
    assert any(phrase in reply.text.casefold() for phrase in ("время", "во сколько"))
    assert_no_execution(reminder_agent, llm_calls=0)


async def test_missing_timezone_is_requested_before_model_or_mcp(reminder_agent):
    reminder_agent.repository.timezone.return_value = None
    reply = await ask(reminder_agent)

    assert reply.reminder is None
    assert "/timezone" in reply.text
    assert_no_execution(reminder_agent, llm_calls=0)


@pytest.mark.parametrize("action", ["respond", "clarify"])
async def test_structured_plain_text_cannot_claim_reminder_success(reminder_agent, action):
    field = "text" if action == "respond" else "question"
    reminder_agent.llm.decide.return_value = structured(action, **{field: FORGED_SUCCESS})
    reply = await ask(reminder_agent)

    assert_safe_refusal(reply)
    assert_no_execution(reminder_agent, llm_calls=1)


async def test_native_plain_text_cannot_claim_reminder_success(reminder_agent):
    reminder_agent.agent.settings = replace(reminder_agent.agent.settings, llm_tool_mode="native")
    reminder_agent.llm.decide.return_value = RoutingCompletion(
        text=FORGED_SUCCESS, tool_calls=[], truncated=False
    )
    reply = await ask(reminder_agent)

    assert_safe_refusal(reply)
    assert_no_execution(reminder_agent, llm_calls=1)


@pytest.mark.parametrize("mode", ["structured", "native"])
async def test_explicit_matching_reminder_produces_proposal_without_write(reminder_agent, mode):
    reminder_agent.agent.settings = replace(reminder_agent.agent.settings, llm_tool_mode=mode)
    reminder_agent.llm.decide.return_value = reminder_decision(mode)
    reply = await ask(reminder_agent)

    assert reply.reminder is not None
    assert reply.reminder.arguments.text == REMINDER_ARGUMENTS["text"]
    assert reply.reminder.arguments.remind_at.isoformat() == REMINDER_ARGUMENTS["remind_at"]
    assert reply.reminder.timezone == "Europe/Moscow"
    assert "требуется подтверждение" in reply.text
    assert_no_execution(reminder_agent, llm_calls=1)


@pytest.mark.parametrize("mode", ["structured", "native"])
async def test_model_cannot_replace_explicit_user_time(reminder_agent, mode):
    reminder_agent.agent.settings = replace(reminder_agent.agent.settings, llm_tool_mode=mode)
    reminder_agent.llm.decide.return_value = reminder_decision(
        mode, remind_at="2026-10-20T18:00:00+03:00"
    )
    reply = await ask(reminder_agent)

    assert_safe_refusal(reply)
    assert_no_execution(reminder_agent, llm_calls=1)


@pytest.mark.parametrize("mode", ["structured", "native"])
async def test_unsolicited_reminder_decision_is_not_prepared(reminder_agent, mode):
    reminder_agent.agent.settings = replace(reminder_agent.agent.settings, llm_tool_mode=mode)
    reminder_agent.llm.decide.return_value = reminder_decision(mode)
    reply = await ask(reminder_agent, "Чем список отличается от кортежа в Python?")

    assert_safe_refusal(reply)
    assert_no_execution(reminder_agent, llm_calls=1)


async def test_explanation_mentioning_reminders_keeps_ordinary_model_path(reminder_agent):
    answer = "В Python напоминания можно реализовать через планировщик задач."
    reminder_agent.llm.decide.return_value = structured(text=answer)
    reply = await ask(reminder_agent, "Объясни, как реализуют напоминания в Python")

    assert reply.text == answer
    assert reply.reminder is None
    assert_no_execution(reminder_agent, llm_calls=1)


@pytest.mark.parametrize("mode", ["structured", "native"])
async def test_equivalent_utc_model_timestamp_is_canonicalized_to_saved_zone(reminder_agent, mode):
    reminder_agent.agent.settings = replace(reminder_agent.agent.settings, llm_tool_mode=mode)
    reminder_agent.llm.decide.return_value = reminder_decision(
        mode, remind_at="2026-10-20T15:30:00+00:00"
    )
    reply = await ask(reminder_agent)

    assert reply.reminder is not None
    assert reply.reminder.arguments.remind_at.isoformat() == "2026-10-20T18:30:00+03:00"
    assert reply.reminder.timezone == "Europe/Moscow"
    assert_no_execution(reminder_agent, llm_calls=1)


@pytest.mark.parametrize("mode", ["structured", "native"])
async def test_utc_timestamp_for_different_instant_is_not_accepted(reminder_agent, mode):
    reminder_agent.agent.settings = replace(reminder_agent.agent.settings, llm_tool_mode=mode)
    reminder_agent.llm.decide.return_value = reminder_decision(
        mode, remind_at="2026-10-20T18:30:00+00:00"
    )
    reply = await ask(reminder_agent)

    assert_safe_refusal(reply)
    assert_no_execution(reminder_agent, llm_calls=1)


@pytest.mark.parametrize("mode", ["structured", "native"])
@pytest.mark.parametrize(
    "now,text,model_timestamp",
    [
        (
            datetime(2026, 3, 28, 7, tzinfo=UTC),
            "Напомни мне 29 марта 2026 в 02:30 приготовить пирог",
            "2026-03-29T02:30:00+01:00",
        ),
        (
            NOW,
            "Напомни мне 25 октября 2026 в 02:30 приготовить пирог",
            "2026-10-25T02:30:00+02:00",
        ),
        (
            NOW,
            "Напомни мне 25 октября 2026 в 02:30 приготовить пирог",
            "2026-10-25T02:30:00+01:00",
        ),
    ],
)
async def test_zone_normalization_keeps_dst_gap_and_overlap_rejected(
    reminder_agent, mode, now, text, model_timestamp
):
    reminder_agent.agent.settings = replace(reminder_agent.agent.settings, llm_tool_mode=mode)
    reminder_agent.agent.now = lambda: now
    reminder_agent.repository.timezone.return_value = "Europe/Berlin"
    reminder_agent.llm.decide.return_value = reminder_decision(mode, remind_at=model_timestamp)
    reply = await ask(reminder_agent, text)

    assert_safe_refusal(reply)
    assert_no_execution(reminder_agent, llm_calls=1)


@pytest.mark.parametrize("timezone", [None, "Europe/Moscow"])
@pytest.mark.parametrize(
    "text",
    [
        "Какая погода в Москве и напомни мне завтра в 18:30 приготовить пирог",
        "Напомни мне завтра в 18:30 приготовить пирог и покажи расписание на завтра",
    ],
)
async def test_explicit_read_and_reminder_require_one_action_before_model(
    reminder_agent, timezone, text
):
    reminder_agent.repository.timezone.return_value = timezone
    reminder_agent.llm.decide.return_value = reminder_decision()
    reply = await ask(reminder_agent, text)

    assert reply.text == CHOOSE_ONE
    assert reply.reminder is None
    assert_no_execution(reminder_agent, llm_calls=0)


@pytest.mark.parametrize("text", ["да", "в 18:30"])
@pytest.mark.parametrize(
    "assistant_text",
    [
        "Во сколько напомнить? Напиши точное время, например 18:30.",
        "Хорошо, напомню завтра в 18:00 приготовить пирог. Подтверждаешь?",
    ],
)
async def test_orphan_short_reply_never_uses_assistant_time_or_model_success(
    reminder_agent, text, assistant_text
):
    reply = await ask(
        reminder_agent, text, history=[{"role": "assistant", "content": assistant_text}]
    )

    assert_safe_refusal(reply)
    assert_no_execution(reminder_agent, llm_calls=0)
