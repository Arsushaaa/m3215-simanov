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
