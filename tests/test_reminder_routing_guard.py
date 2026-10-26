"""One model chooses intent; code validates sources and keeps writes separate."""

import json
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.agent import CHOOSE_ONE, AgentService
from app.llm import ModelError, RoutingCompletion
from app.mcp.contracts import tool_schemas
from app.prompts import PROMPTS
from app.repository import UserSettings

NOW = datetime(2026, 10, 19, 7, tzinfo=UTC)
REQUEST = "Не забудь мне напомнить в 11:00 завтра приготовить пирог"
REMINDER_ARGUMENTS = {
    "text": "Приготовить пирог",
    "remind_at": "2026-10-20T11:00:00+03:00",
}


def quote(text, index=0):
    return {"source": "user", "message_index": index, "quote": text}


def evidence_for(text):
    return {name: quote(text) for name in ("request", "text", "date", "time")}


def structured(action="respond", **fields):
    return RoutingCompletion(
        json.dumps({"action": action, "reason": "Выбор действия.", **fields}), []
    )


def reminder_decision(mode="structured", *, question=REQUEST, evidence=None, **overrides):
    proposal = {
        "arguments": {**REMINDER_ARGUMENTS, **overrides},
        "evidence": evidence if evidence is not None else evidence_for(question),
    }
    if mode == "structured":
        return structured("call_tool", name="add_reminder", **proposal)
    return RoutingCompletion(
        None,
        [
            {
                "id": "prepare-card",
                "type": "function",
                "function": {"name": "prepare_reminder", "arguments": json.dumps(proposal)},
            }
        ],
    )


@pytest.fixture
def reminder_agent(bot_settings):
    llm = SimpleNamespace(
        decide=AsyncMock(return_value=structured(text="Ответ без инструмента.")),
        complete=AsyncMock(side_effect=AssertionError("No second model for a proposal")),
    )
    mcp = SimpleNamespace(
        available=True,
        tools={
            name: {"name": name, "description": name, **tool_schemas(name)}
            for name in ("get_weather", "get_schedule", "add_reminder")
        },
        resources={"schedule://current-week": {}},
        call_tool=AsyncMock(side_effect=AssertionError("Preparation cannot write")),
        read_resource=AsyncMock(side_effect=AssertionError("No partial execution")),
    )
    repository = SimpleNamespace(
        timezone=AsyncMock(return_value="Europe/Moscow"),
        has_processed=AsyncMock(return_value=False),
        start_request=AsyncMock(),
        record=AsyncMock(),
    )
    return SimpleNamespace(
        agent=AgentService(llm, mcp, repository, bot_settings, now=lambda: NOW),
        llm=llm,
        mcp=mcp,
        repository=repository,
    )


async def ask(setup, text=REQUEST, *, history=None, context=None):
    return await setup.agent.respond(
        UserSettings(1, "study", 0.3, 0),
        10,
        text,
        prompt=PROMPTS["study"],
        history=history or [],
        confirmation_context=context,
    )


def assert_preparation_only(setup):
    setup.llm.decide.assert_awaited_once()
    setup.llm.complete.assert_not_awaited()
    setup.mcp.call_tool.assert_not_awaited()
    setup.mcp.read_resource.assert_not_awaited()
    audit = setup.repository.record.call_args.kwargs
    assert (audit["llm_calls"], audit["mcp_calls"]) == (1, 0)


@pytest.mark.parametrize("mode", ["structured", "native"])
@pytest.mark.parametrize(
    "question",
    [
        REQUEST,
        "В 11:00 завтра надо приготовить пирог — можешь об этом напомнить?",
        "Чтобы я не забыл: завтра к одиннадцати напомни приготовить пирог",
        "Make sure I don't forget to bake a pie tomorrow at 11 AM",
    ],
)
async def test_unfamiliar_word_order_reaches_one_model_and_prepares_without_mcp(
    reminder_agent, mode, question
):
    reminder_agent.agent.settings = replace(reminder_agent.agent.settings, llm_tool_mode=mode)
    reminder_agent.llm.decide.return_value = reminder_decision(mode, question=question)
    reply = await ask(reminder_agent, question)
    assert reply.reminder is not None
    assert reply.reminder.arguments.text == REMINDER_ARGUMENTS["text"]
    assert reply.reminder.arguments.remind_at.isoformat() == REMINDER_ARGUMENTS["remind_at"]
    assert "требуется подтверждение" in reply.text
    assert reminder_agent.llm.decide.call_args.args[0][-1] == {"role": "user", "content": question}
    assert_preparation_only(reminder_agent)
    if mode == "native":
        names = {
            item["function"]["name"] for item in reminder_agent.llm.decide.call_args.kwargs["tools"]
        }
        assert {"prepare_reminder", "get_weather", "get_schedule"} <= names
        assert "add_reminder" not in names


@pytest.mark.parametrize(
    "question",
    [
        "Не забудь завтра напомнить про пирог",
        "Remind me tomorrow to bake a pie",
    ],
)
async def test_missing_time_is_a_model_clarification_without_mcp(reminder_agent, question):
    reminder_agent.llm.decide.return_value = structured(
        "clarify_reminder", question="Во сколько напомнить?"
    )
    reply = await ask(reminder_agent, question)
    assert reply.reminder is None
    assert "Во сколько" in reply.text and "ещё не сохранено" in reply.text
    assert_preparation_only(reminder_agent)


async def test_missing_timezone_is_not_a_gate_before_intent_selection(reminder_agent):
    reminder_agent.repository.timezone.return_value = None
    reminder_agent.llm.decide.return_value = reminder_decision()
    reply = await ask(reminder_agent)
    assert reply.reminder is None and "/timezone" in reply.text
    assert_preparation_only(reminder_agent)


@pytest.mark.parametrize(
    "question",
    [
        "Не напоминай мне завтра про пирог. Объясни рецепт.",
        "Разбери фразу «не забудь напомнить завтра в 11:00 приготовить пирог».",
        "Объясни, как реализуют напоминания в Python",
    ],
)
async def test_negative_quoted_and_theoretical_text_can_choose_ordinary_response(
    reminder_agent, question
):
    reminder_agent.llm.decide.return_value = structured(text="Объяснение без создания действия.")
    reply = await ask(reminder_agent, question)
    assert reply.text == "Объяснение без создания действия." and reply.reminder is None
    assert_preparation_only(reminder_agent)


async def test_several_requested_actions_are_clarified_by_the_model(reminder_agent):
    reminder_agent.llm.decide.return_value = structured("clarify", question=CHOOSE_ONE)
    reply = await ask(
        reminder_agent,
        "Покажи погоду в Москве и не забудь напомнить завтра в 11:00 приготовить пирог",
    )
    assert reply.text == CHOOSE_ONE and reply.reminder is None
    assert_preparation_only(reminder_agent)


async def test_followup_sources_reference_only_indexed_user_messages(reminder_agent):
    history = [
        {"role": "user", "content": "Завтра нужно приготовить пирог, напомни мне"},
        {"role": "assistant", "content": "Во сколько? Например, в 18:00?"},
    ]
    evidence = {
        "request": quote("напомни мне", 1),
        "text": quote("приготовить пирог", 1),
        "date": quote("Завтра", 1),
        "time": quote("11:00"),
    }
    reminder_agent.llm.decide.return_value = reminder_decision(evidence=evidence)
    reply = await ask(reminder_agent, "В 11:00", history=history)
    assert reply.reminder is not None
    messages = reminder_agent.llm.decide.call_args.args[0]
    assert json.loads(messages[1]["content"]) == {
        "message_index": 1,
        "text": history[0]["content"],
    }
    assert messages[2]["content"] == history[1]["content"]
    assert_preparation_only(reminder_agent)


@pytest.mark.parametrize("bad_source", [quote("18:00", 1), quote("11:00", 19)])
async def test_assistant_only_or_nonexistent_evidence_cannot_prepare(reminder_agent, bad_source):
    evidence = evidence_for(REQUEST)
    evidence["time"] = bad_source
    reminder_agent.llm.decide.return_value = reminder_decision(evidence=evidence)
    reply = await ask(
        reminder_agent,
        history=[{"role": "assistant", "content": "Напомню в 18:00"}],
    )
    assert reply.reminder is None
    assert reminder_agent.repository.record.call_args.kwargs["error_code"] == "INVALID_ARGUMENTS"
    assert_preparation_only(reminder_agent)


async def test_equivalent_utc_timestamp_is_canonicalized_without_language_reparsing(reminder_agent):
    reminder_agent.llm.decide.return_value = reminder_decision(
        remind_at="2026-10-20T08:00:00+00:00"
    )
    reply = await ask(reminder_agent)
    assert reply.reminder.arguments.remind_at.isoformat() == "2026-10-20T11:00:00+03:00"
    assert_preparation_only(reminder_agent)


@pytest.mark.parametrize(
    "now,question,moment",
    [
        (
            datetime(2026, 3, 28, 7, tzinfo=UTC),
            "Напомни 29 марта 2026 в 02:30 приготовить пирог",
            "2026-03-29T02:30:00+01:00",
        ),
        (NOW, "Напомни 25 октября 2026 в 02:30 приготовить пирог", "2026-10-25T02:30:00+02:00"),
        (NOW, "Напомни 25 октября 2026 в 02:30 приготовить пирог", "2026-10-25T02:30:00+01:00"),
    ],
)
async def test_model_selected_time_still_rejects_dst_gap_and_overlap(
    reminder_agent, now, question, moment
):
    reminder_agent.agent.now = lambda: now
    reminder_agent.repository.timezone.return_value = "Europe/Berlin"
    reminder_agent.llm.decide.return_value = reminder_decision(question=question, remind_at=moment)
    reply = await ask(reminder_agent, question)
    assert reply.reminder is None
    assert_preparation_only(reminder_agent)


async def test_native_model_cannot_expose_the_mcp_write_function(reminder_agent):
    reminder_agent.agent.settings = replace(reminder_agent.agent.settings, llm_tool_mode="native")
    result = reminder_decision("native")
    result.tool_calls[0]["function"]["name"] = "add_reminder"
    reminder_agent.llm.decide.return_value = result
    with pytest.raises(ModelError, match="некорректное действие"):
        await ask(reminder_agent)
    assert_preparation_only(reminder_agent)


@pytest.mark.parametrize(
    "name,parameters",
    [
        ("clarify_reminder", {"question": "Во сколько напомнить?", "reason": "Нужны часы."}),
        ("use_confirmation", {"reason": "Нужна кнопка владельца."}),
    ],
)
async def test_native_app_actions_do_not_become_mcp_calls(reminder_agent, name, parameters):
    reminder_agent.agent.settings = replace(reminder_agent.agent.settings, llm_tool_mode="native")
    reminder_agent.llm.decide.return_value = RoutingCompletion(
        None,
        [
            {
                "id": "local-step",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(parameters)},
            }
        ],
    )
    reply = await ask(reminder_agent, "Подготовь, если можешь")
    assert reply.reminder is None
    assert "не сохранено" in reply.text
    assert_preparation_only(reminder_agent)
