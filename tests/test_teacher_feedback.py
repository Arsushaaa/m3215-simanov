"""Measurement integrity for teacher feedback; all network clients are substituted."""

import json
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from test_lab2_experiment import Answers, catalog, evidence

from app.llm import RoutingCompletion
from scripts import teacher_feedback_probe as probe
from scripts.lab2_experiment import load_dataset, run_trials


async def test_lr2_native_experiment_keeps_all_fixed_denominators(bot_settings, tmp_path):
    class NativeAnswers(Answers):
        async def decide(self, messages, temperature, **kwargs):
            self.calls.append((messages, kwargs))
            text = json.loads(messages[-1]["content"])["text"]
            case = next(c for c in load_dataset()["cases"] if c["query"] == text)
            action = case["expected_action"]
            if action == "respond":
                return RoutingCompletion("Учебный ответ.", [])
            arguments = case["expected_arguments"]
            name = action
            if action == "add_reminder":
                name = "prepare_reminder"
                arguments = {"arguments": arguments, "evidence": evidence(text)}
            elif action == "clarify":
                name = "clarify_reminder"
                arguments = {"question": "Во сколько напомнить?", "reason": "Не задано время."}
            return RoutingCompletion(
                None,
                [
                    {
                        "id": "c",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(arguments, ensure_ascii=False),
                        },
                    }
                ],
            )

    client = NativeAnswers()
    result = await run_trials(
        replace(bot_settings, llm_tool_mode="native"), client, catalog(), tmp_path / "native.json"
    )
    assert result["metrics"]["routing_accuracy"]["correct"] == 12
    assert result["metrics"]["argument_accuracy"]["correct"] == 9
    assert result["metrics"]["llm_calls"] == len(client.calls) == 12
    assert result["metrics"]["mcp_application_calls"] == 0


async def test_probe_checks_sources_quota_and_never_replays_started_cases(
    tmp_path, monkeypatch, bot_settings
):
    folder = tmp_path / "baseline"
    probe.freeze(folder, True)
    monkeypatch.setattr(probe, "assert_committed", lambda p: None)
    monkeypatch.setattr(probe.Settings, "load", lambda *a, **k: bot_settings)
    monkeypatch.setattr(
        probe, "dotenv_values", lambda *a, **k: {"OPENROUTER_API_KEY_3": "synthetic-extra"}
    )
    quota = AsyncMock(return_value={"remaining": 1})
    monkeypatch.setattr(probe, "read_quota", quota)
    start = AsyncMock()
    monkeypatch.setattr(probe.Client, "start", start)
    with pytest.raises(ValueError, match="Insufficient"):
        await probe.run(folder)
    start.assert_not_awaited()
    output = folder / "results.json"
    probe.save(
        output,
        {
            "protocol_sha256": probe.digest((folder / "protocol.json").read_bytes()),
            "cases": [{"id": "list_tuple", "state": "started"}],
        },
    )
    with pytest.raises(ValueError, match="never replayed"):
        await probe.run(folder)
    assert quota.await_count == 1
