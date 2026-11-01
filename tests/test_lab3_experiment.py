"""Experiment integrity tests. All responses are fake and saved only in tmp_path."""

import json
from unittest.mock import AsyncMock

import pytest

from app.config import Settings
from app.llm import ModelError, RoutingCompletion
from scripts import lab3_experiment as experiment


class FakeClient(experiment.MeasuredClient):
    instances = []
    fail_at = None

    async def start(self):
        self.payloads = []
        self.instances.append(self)

    async def close(self):
        pass

    async def _request(self, payload):
        self.attempts += 1
        self.payloads.append(payload)
        self.observation = {"payload_sha256": experiment.digest(experiment.encoded(payload))}
        if self.attempts == self.fail_at:
            raise ModelError("429")
        return RoutingCompletion(
            None,
            [
                {
                    "id": "route1",
                    "type": "function",
                    "function": {
                        "name": "select_route",
                        "arguments": json.dumps(
                            {"target": "clarify", "reason": "ambiguous", "confidence": 0.8}
                        ),
                    },
                }
            ],
        )


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    path = tmp_path / "protocol.json"
    protocol = experiment.freeze(path)
    FakeClient.instances = []
    FakeClient.fail_at = None
    monkeypatch.setattr(
        experiment.Settings,
        "load",
        lambda *args, **kwargs: Settings(
            bot_token="", postgres_password="", openrouter_api_key="fake"
        ),
    )
    return path, tmp_path / "results.json", protocol


def test_protocol_rejects_source_drift_and_never_overwrites_preregistration(frozen):
    path, _, protocol = frozen
    experiment.validate(protocol)
    assert len(json.loads((experiment.ROOT / experiment.DATASET).read_text())["cases"]) == 15
    bad = {**protocol, "source_sha256": {}}
    with pytest.raises(ValueError, match="source changed"):
        experiment.validate(bad)
    with pytest.raises(ValueError, match="already exists"):
        experiment.freeze(path)


@pytest.mark.parametrize("quota", [{"remaining": 1}, {}])
async def test_quota_gate_does_not_spend_partial_budget_or_relabel_as_success(frozen, quota):
    path, output, protocol = frozen
    result = await experiment.run(
        path,
        output,
        client_type=FakeClient,
        quota_reader=AsyncMock(return_value=quota),
        require_commit=False,
    )
    assert FakeClient.instances == []
    assert result["blocked_reason"] == "INSUFFICIENT_FREE_QUOTA"
    assert result["summary"]["denominator"] == 15 and result["summary"]["status"] == "incomplete"
    assert result["cases"] == []


async def test_every_attempt_and_failure_stays_in_fifteen_case_denominator(frozen):
    path, output, _ = frozen
    FakeClient.fail_at = 2
    result = await experiment.run(
        path,
        output,
        client_type=FakeClient,
        quota_reader=AsyncMock(return_value={"remaining": 50}),
        require_commit=False,
    )
    assert result["summary"]["denominator"] == 15 and result["summary"]["correct"] == 3
    assert result["summary"]["status"] == "not_passed"
    assert len(result["cases"]) == 15 and result["cases"][1]["error"] == "ModelError"
    assert sum(c["http_attempts"] for c in result["cases"]) == 15
    for payload in FakeClient.instances[0].payloads:
        assert payload["provider"]["max_price"] == {"prompt": 0, "completion": 0}
        raw = json.dumps(payload["messages"], ensure_ascii=False)
        assert "expected_target" not in raw and "actor_id" not in raw
        assert payload["temperature"] == 0.2 and payload["reasoning"] == {"enabled": False}
        assert payload["model"] == experiment.MODEL
    # Resuming the same checkpoint cannot repeat any started case.
    await experiment.run(
        path,
        output,
        client_type=FakeClient,
        quota_reader=AsyncMock(return_value={"remaining": 50}),
        require_commit=False,
    )
    assert sum(c.attempts for c in FakeClient.instances) == 15


async def test_interrupted_started_case_is_not_retried_or_dropped(frozen):
    path, output, _ = frozen
    experiment.save(
        output,
        {
            "kind": "lab3_real_router_experiment",
            "protocol_sha256": experiment.digest(path.read_bytes()),
            "cases": [{"id": "schedule_1", "state": "started"}],
        },
    )
    result = await experiment.run(
        path,
        output,
        client_type=FakeClient,
        quota_reader=AsyncMock(return_value={"remaining": 50}),
        require_commit=False,
    )
    assert result["cases"][0]["error"] == "INTERRUPTED_OUTCOME_UNKNOWN"
    assert result["cases"][0]["correct"] is False
    assert FakeClient.instances[0].attempts == 14
    assert result["summary"]["denominator"] == 15
