"""Holdout driver integrity; all LLM responses are substituted."""

import json
from unittest.mock import AsyncMock

import pytest

from app.config import Settings
from app.llm import ModelError, RoutingCompletion
from scripts import lab3_router_holdout as holdout
from scripts.lab3_experiment import encoded


async def test_four_call_limit_precedes_http():
    client = holdout.ProbeClient(Settings(bot_token="", postgres_password=""))
    client.attempts = 4
    with pytest.raises(ModelError, match="budget exhausted"):
        await client._request({})
    assert client.attempts == 4


def test_observer_never_persists_raw_response_or_prompts():
    client = holdout.ProbeClient(Settings(bot_token="", postgres_password=""))
    client.observation = {}
    secret = "Synthetic private prompt text"
    client._observe_response(
        {
            "usage": {"cost": 0, "prompt_tokens": 10, "private": secret},
            "choices": [{"message": {"content": secret}}],
        }
    )
    assert client.observation == {"usage": {"cost": 0, "prompt_tokens": 10}}


async def test_holdout_has_four_calls_no_expected_targets_and_no_checkpoint_replay(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(holdout, "FOLDER", tmp_path)
    monkeypatch.setattr(holdout, "assert_committed", lambda path: None)
    monkeypatch.setattr(
        holdout.Settings,
        "load",
        lambda *a, **k: Settings(
            bot_token="", postgres_password="", openrouter_api_key="synthetic-primary"
        ),
    )
    monkeypatch.setattr(
        holdout, "dotenv_values", lambda *a, **k: {"OPENROUTER_API_KEY_3": "synthetic-extra"}
    )
    monkeypatch.setattr(holdout, "read_quota", AsyncMock(return_value={"remaining": 4}))
    payloads = []
    routes = {text: target for _, text, target in holdout.CASES}

    class FakeClient(holdout.ProbeClient):
        async def start(self):
            assert self.settings.openrouter_api_key == "synthetic-extra"

        async def close(self):
            pass

        async def _request(self, payload):
            self.attempts += 1
            payloads.append(payload)
            self.observation = {"payload_sha256": holdout.digest(encoded(payload))}
            text = json.loads(payload["messages"][-1]["content"])["text"]
            target = routes[text]
            return RoutingCompletion(
                None,
                [
                    {
                        "id": "r",
                        "type": "function",
                        "function": {
                            "name": "select_route",
                            "arguments": json.dumps(
                                {
                                    "target": target,
                                    "reason": "ambiguous"
                                    if target == "clarify"
                                    else "general_study",
                                    "confidence": 0.8,
                                }
                            ),
                        },
                    }
                ],
            )

    monkeypatch.setattr(holdout, "ProbeClient", FakeClient)
    holdout.freeze()
    result = await holdout.run()
    assert result["summary"] == {"correct": 4, "denominator": 4, "attempts": 4, "complete": True}
    assert len(payloads) == 4
    for payload in payloads:
        model_input = json.dumps(payload["messages"])
        assert "expected" not in model_input and "actor_id" not in model_input
        assert payload["provider"]["max_price"] == {"prompt": 0, "completion": 0}
    await holdout.run()
    assert len(payloads) == 4
