"""Incomplete routing may revoke a card safely, but cannot execute its decision."""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from app.agent import AgentService, RoutingContext
from app.confirmation_models import PreparedAction, format_card
from app.dialog import TRUNCATED_NOTICE
from app.llm import ModelError, RoutingCompletion
from app.mcp.contracts import ReminderArguments, tool_schemas
from app.repository import UserSettings

NOW = datetime(2026, 10, 26, 9, tzinfo=UTC)
QUESTION = "Нет, лучше в 21:00"
OWNER = UserSettings(1, "study", 0.2, 0)


def wire_decision(action, mode):
    fields = {"reason": "Нужно уточнение или подтверждение."}
    if action == "clarify_reminder":
        fields["question"] = "Во сколько напомнить?"
    elif action == "add_reminder":
        fields.update(
            arguments={"text": "Сдать отчёт", "remind_at": "2026-10-27T21:00:00+03:00"},
            evidence={
                "request": {"source": "user", "message_index": 0, "quote": QUESTION},
                "text": {"source": "pending_card"},
                "date": {"source": "pending_card"},
                "time": {"source": "user", "message_index": 0, "quote": "в 21:00"},
            },
        )
    if mode == "structured":
        value = (
            {"action": "call_tool", "name": "add_reminder", **fields}
            if action == "add_reminder"
            else {"action": action, **fields}
        )
        return RoutingCompletion(json.dumps(value, ensure_ascii=False), [], truncated=True)
    name = "prepare_reminder" if action == "add_reminder" else action
    if action == "add_reminder":
        fields.pop("reason")
    return RoutingCompletion(
        None,
        [
            {
                "id": "synthetic-incomplete-call",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(fields)},
            }
        ],
        truncated=True,
    )


def setup_agent(settings, result, *, pending=True):
    arguments = ReminderArguments(text="Сдать отчёт", remind_at="2026-10-27T19:00:00+03:00")
    card = PreparedAction(
        UUID("11111111-1111-4111-8111-111111111111"),
        format_card(arguments, "Europe/Moscow", NOW + timedelta(minutes=5)),
        arguments,
        "Europe/Moscow",
        NOW + timedelta(minutes=5),
        "pending",
    )
    repository = SimpleNamespace(
        has_processed=AsyncMock(return_value=False),
        start_request=AsyncMock(),
        timezone=AsyncMock(return_value="Europe/Moscow"),
        record=AsyncMock(),
        current_card=AsyncMock(return_value=card),
    )
    service = SimpleNamespace(
        repository=repository,
        begin_edit=AsyncMock(return_value=(card, replace(OWNER, revision=1), 2)),
        prepare=AsyncMock(side_effect=AssertionError("No new card from incomplete routing")),
    )
    context = RoutingContext(service, card if pending else None, 1)
    llm = SimpleNamespace(decide=AsyncMock(return_value=result), complete=AsyncMock())
    mcp = SimpleNamespace(
        available=True,
        tools={
            name: {"name": name, "description": name, **tool_schemas(name)}
            for name in ("get_weather", "get_schedule", "add_reminder")
        },
        resources={},
        call_tool=AsyncMock(),
        read_resource=AsyncMock(),
    )
    agent = AgentService(llm, mcp, repository, settings, now=lambda: NOW)
    return agent, context, repository, llm, mcp


@pytest.mark.parametrize("mode", ["structured", "native"])
@pytest.mark.parametrize(
    "action", ["clarify_reminder", "use_confirmation", "cancel_reminder", "add_reminder"]
)
async def test_truncated_semantic_control_is_rejected_before_dispatch(bot_settings, mode, action):
    settings = replace(bot_settings, llm_tool_mode=mode)
    agent, context, repository, llm, mcp = setup_agent(settings, wire_decision(action, mode))

    with pytest.raises(ModelError, match="Решение модели оборвалось"):
        await agent.respond(
            OWNER,
            20,
            QUESTION,
            prompt="Выбери действие по заданной схеме.",
            history=[],
            confirmation_context=context,
        )

    # The error path may revoke the old card, but must not treat model cancel as success.
    context.service.begin_edit.assert_awaited_once_with(OWNER, 20, expected_epoch=1)
    repository.current_card.assert_not_awaited()
    context.service.prepare.assert_not_awaited()
    llm.decide.assert_awaited_once()
    llm.complete.assert_not_awaited()
    mcp.call_tool.assert_not_awaited()
    mcp.read_resource.assert_not_awaited()
    audit = repository.record.call_args.kwargs
    assert audit["status"] == "FAILED"
    assert audit["error_code"] == "INVALID_RESULT"
    assert (audit["llm_calls"], audit["mcp_calls"]) == (1, 0)


@pytest.mark.parametrize("action", ["respond", "clarify"])
async def test_plain_partial_decision_keeps_flag_and_dialog_warning(dialog_setup, action):
    dialog, repository, _ = dialog_setup
    body = "Частичный обычный ответ." if action == "respond" else "Какую тему уточнить?"
    payload = {"action": action, "reason": "Обычный ответ."}
    payload["text" if action == "respond" else "question"] = body
    result = RoutingCompletion(json.dumps(payload), [], truncated=True)
    agent, _, _, llm, mcp = setup_agent(dialog.settings, result, pending=False)
    dialog.agent = agent

    answer = await dialog.respond(1, 20, "Объясни списки Python")

    assert answer == TRUNCATED_NOTICE + "\n\n" + body
    assert await repository.has_incomplete_answer(1)
    llm.decide.assert_awaited_once()
    llm.complete.assert_not_awaited()
    mcp.call_tool.assert_not_awaited()
