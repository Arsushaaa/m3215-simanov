import json
from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator, ValidationError
from pydantic import ValidationError as ModelValidationError

from app.decision_schema import model_schema
from app.mcp.contracts import AgentDecision, ReminderPreparation, WeatherArguments


def test_absent_card_cannot_be_offered_as_a_model_source():
    schema = model_schema(AgentDecision.model_json_schema())
    source = {"source": "user", "message_index": 0, "quote": "Напомни завтра в 11:00"}
    decision = {
        "action": "call_tool",
        "name": "add_reminder",
        "reason": "Подготовка",
        "arguments": {"text": "Пирог", "remind_at": "2030-01-02T11:00:00+03:00"},
        "evidence": {key: source for key in ("request", "text", "date", "time")},
    }
    Draft202012Validator(schema).validate(decision)
    decision["evidence"]["date"] = {"source": "pending_card"}
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(decision)


@pytest.mark.parametrize("model", [AgentDecision, ReminderPreparation])
def test_source_restriction_does_not_mutate_canonical_schema(model):
    canonical = model.model_json_schema()
    copied = model_schema(canonical)
    assert "PendingCardReference" in canonical["$defs"]
    assert "PendingCardReference" not in copied["$defs"]
    assert "oneOf" in canonical["$defs"]["ReminderEvidence"]["properties"]["time"]
    active = model_schema(canonical, pending=True)
    assert "PendingCardReference" in active["$defs"]


def test_provider_hint_does_not_weaken_weather_runtime_contract():
    canonical = WeatherArguments.model_json_schema()
    hint = model_schema(canonical)
    assert "pattern" in canonical["properties"]["city"]
    assert "pattern" not in hint["properties"]["city"]
    assert "русском" in hint["properties"]["city"]["description"]
    with pytest.raises(ModelValidationError):
        WeatherArguments(city="Kazan, Russia")
    assert WeatherArguments(city="Казань, Россия").city == "Казань, Россия"


def test_provider_routing_union_is_flat_and_keeps_all_tool_branches_selectable():
    canonical = AgentDecision.model_json_schema()
    original = deepcopy(canonical)
    provider = model_schema(canonical, pending=True)

    assert canonical == original
    assert "discriminator" in canonical
    assert any("oneOf" in branch for branch in canonical["oneOf"])
    assert all(set(branch) == {"$ref"} for branch in provider["oneOf"])
    references = {branch["$ref"] for branch in provider["oneOf"]}
    for name in ("WeatherDecision", "ScheduleDecision", "ReminderDecision"):
        assert f"#/$defs/{name}" in references
        assert list(provider["$defs"][name]["properties"])[:2] == ["action", "name"]

    nodes = [provider]
    while nodes:
        node = nodes.pop()
        if isinstance(node, dict):
            assert "discriminator" not in node
            nodes.extend(node.values())
        elif isinstance(node, list):
            nodes.extend(node)


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("get_weather", {"city": "Казань, Россия"}),
        ("get_schedule", {"date": "2030-01-02"}),
        ("add_reminder", {"text": "Пирог", "remind_at": "2030-01-02T11:00:00+03:00"}),
    ],
)
def test_each_tool_decision_is_valid_in_provider_and_canonical_schema(name, arguments):
    decision = {
        "action": "call_tool",
        "name": name,
        "arguments": arguments,
        "reason": "Выбрано по просьбе пользователя",
    }
    if name == "add_reminder":
        source = {"source": "user", "message_index": 0, "quote": "Напомни завтра в 11:00"}
        decision["evidence"] = {key: source for key in ("request", "text", "date", "time")}
    canonical = AgentDecision.model_json_schema()
    Draft202012Validator(canonical).validate(decision)
    Draft202012Validator(model_schema(canonical)).validate(decision)
    assert AgentDecision.model_validate(decision).root.name == name


@pytest.mark.parametrize("pending", [False, True])
def test_inline_native_reminder_schema_requires_objects_without_losing_validation(pending):
    canonical = ReminderPreparation.model_json_schema()
    original = deepcopy(canonical)
    provider = model_schema(canonical, pending=pending, inline_refs=True)
    assert canonical == original
    assert "$defs" in canonical
    nodes = [provider]
    while nodes:
        node = nodes.pop()
        if isinstance(node, dict):
            assert "$ref" not in node and "$defs" not in node
            nodes.extend(node.values())
        elif isinstance(node, list):
            nodes.extend(node)
    assert provider["properties"]["arguments"]["type"] == "object"
    assert provider["properties"]["evidence"]["type"] == "object"

    source = {"source": "user", "message_index": 0, "quote": "Напомни завтра в 11:00"}
    proposal = {
        "arguments": {"text": "Пирог", "remind_at": "2030-01-02T11:00:00+03:00"},
        "evidence": {key: dict(source) for key in ("request", "text", "date", "time")},
    }
    if pending:
        proposal["evidence"]["text"] = {"source": "pending_card"}
    Draft202012Validator(provider).validate(proposal)
    Draft202012Validator(canonical).validate(proposal)
    ReminderPreparation.model_validate(proposal)
    for name in ("arguments", "evidence"):
        serialized = {**proposal, name: json.dumps(proposal[name], ensure_ascii=False)}
        with pytest.raises(ValidationError):
            Draft202012Validator(provider).validate(serialized)
        with pytest.raises(ModelValidationError):
            ReminderPreparation.model_validate(serialized)
