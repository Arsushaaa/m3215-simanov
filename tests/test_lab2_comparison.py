"""Preregistration, fair selection and extra trials need no real model or HTTP."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from app.llm import RoutingCompletion
from app.mcp.contracts import tool_schemas
from app.mcp.server import TOOLS
from scripts import lab2_comparison as comparison
from scripts.lab2_experiment import CatalogOnlyMCP


def catalog():
    return CatalogOnlyMCP(
        {
            name: {"name": name, "description": entry[2], **tool_schemas(name)}
            for name, entry in TOOLS.items()
        },
        {"schedule://current-week": {"uri": "schedule://current-week"}},
    )


def test_comparing_inputs_accepts_same_messages_and_skips_local_decisions():
    row = {"id": "weather_1", "calls": [{"input_sha256": "fixed-input"}]}
    reports = {
        "first": {"attempts": [deepcopy(row), {"id": "local", "calls": []}]},
        "second": {"attempts": [deepcopy(row)]},
    }
    assert comparison.compare_inputs(reports) == {"weather_1": "fixed-input"}


def test_comparing_inputs_rejects_changed_messages_or_schema_between_cells():
    reports = {
        "first": {"attempts": [{"id": "case", "calls": [{"input_sha256": "original"}]}]},
        "second": {"attempts": [{"id": "case", "calls": [{"input_sha256": "changed"}]}]},
    }
    with pytest.raises(ValueError, match="Messages/schema changed"):
        comparison.compare_inputs(reports)


@pytest.mark.parametrize(
    "cells,results,winner",
    [
        (
            comparison.CELLS[:2],
            [("not_passed", 12, 9), ("passed", 10, 8)],
            "dots-t03",
        ),
        (comparison.CELLS[:2], [("passed", 12, 8), ("passed", 10, 9)], "dots-t03"),
        (comparison.CELLS[:2], [("passed", 10, 9), ("passed", 12, 9)], "dots-t03"),
        (
            (comparison.CELLS[1], comparison.CELLS[3]),
            [("passed", 12, 9), ("passed", 12, 9)],
            "xing-t00",
        ),
        (
            (comparison.CELLS[3], comparison.CELLS[2]),
            [("passed", 12, 9), ("passed", 12, 9)],
            "dots-t00",
        ),
    ],
)
def test_selection_obeys_preregistered_ranking_without_hiding_failed_cells(cells, results, winner):
    reports = {
        cell["id"]: {
            "metrics": {
                "acceptance": acceptance,
                "routing_accuracy": {"correct": actions, "total": 12},
                "argument_accuracy": {"correct": arguments, "total": 9},
            }
        }
        for cell, (acceptance, actions, arguments) in zip(cells, results, strict=True)
    }
    assert comparison.select_cell(reports, cells)["id"] == winner
    assert comparison.select_cell(reports, tuple(reversed(cells)))["id"] == winner


@pytest.fixture
def public_evidence(monkeypatch, bot_settings):
    required = ["temperature", "max_tokens", "reasoning", "response_format", "structured_outputs"]
    evidence = {
        "models": [
            {
                "id": model,
                "pricing": {"prompt": "0", "completion": "0"},
                "supported_parameters": required[:],
            }
            for model in comparison.MODELS
        ],
        "quota": {"remaining": 50, "limit": 50, "used": 0},
        "calls": [],
    }

    def public_json(url, *, key=None):
        evidence["calls"].append((url, key))
        if url.endswith("/models"):
            return deepcopy(evidence["models"])
        if url.endswith("/endpoints"):
            return {
                "endpoints": [
                    {
                        "provider_name": "Offline fixture",
                        "pricing": {"prompt": "0", "completion": "0"},
                        "supported_parameters": required[:],
                    }
                ]
            }
        assert url == bot_settings.openrouter_base_url + "/key"
        assert key == bot_settings.openrouter_api_key
        return {
            "label": bot_settings.openrouter_api_key,
            "hash": bot_settings.openrouter_api_key,
            "free_model_daily_requests": deepcopy(evidence["quota"]),
        }

    monkeypatch.setattr(comparison, "public_json", public_json)
    monkeypatch.setattr(comparison, "sources", lambda: {"app/prompts.py": "a" * 64})
    return evidence


def test_prepare_free_models_schema_and_fixed_budget_without_leaking_key(
    bot_settings, tmp_path, public_evidence, capsys
):
    document = comparison.prepare(bot_settings, tmp_path)
    saved = json.loads((tmp_path / "manifest.json").read_text())
    provider = json.loads((tmp_path / "provider-evidence.json").read_text())
    assert saved["source_sha256"] == document["source_sha256"] == {"app/prompts.py": "a" * 64}
    assert {cell["model"] for cell in saved["cells"]} == set(comparison.MODELS)
    assert len(saved["cells"]) == 4
    assert {cell["temperature"] for cell in saved["cells"]} == {0, 0.3}
    assert saved["budget"] == {"grid": 44, "extra": 4, "live_read": 2, "maximum": 50}
    assert saved["fixed"]["max_tokens"] == 2048
    assert saved["thresholds"] == {"actions": [10, 12], "arguments": [8, 9]}
    assert len(saved["extra_cases"]) == 4
    assert len(provider["models"]) == 2
    assert provider["free_model_daily_requests"]["remaining"] == 50
    captured = capsys.readouterr()
    published = (
        "".join(path.read_text() for path in tmp_path.iterdir()) + captured.out + captured.err
    )
    for secret in (
        bot_settings.openrouter_api_key,
        bot_settings.bot_token,
        bot_settings.postgres_password,
    ):
        assert secret not in published
    assert all(key is None for url, key in public_evidence["calls"] if not url.endswith("/key"))


@pytest.mark.parametrize("price_key", ["prompt", "completion"])
def test_prepare_rejects_paid_model_before_creating_manifest(
    bot_settings, tmp_path, public_evidence, price_key
):
    public_evidence["models"][1]["pricing"][price_key] = "0.001"
    with pytest.raises(ValueError, match="not free"):
        comparison.prepare(bot_settings, tmp_path)
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize("parameter", ["response_format", "structured_outputs"])
def test_prepare_requires_identical_structured_output_capabilities(
    bot_settings, tmp_path, public_evidence, parameter
):
    public_evidence["models"][0]["supported_parameters"].remove(parameter)
    with pytest.raises(ValueError, match="fixed request parameters"):
        comparison.prepare(bot_settings, tmp_path)
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize("remaining", [49, 0])
def test_prepare_rejects_insufficient_quota_for_full_campaign(
    bot_settings, tmp_path, public_evidence, remaining
):
    public_evidence["quota"]["remaining"] = remaining
    with pytest.raises(ValueError, match="50 remaining"):
        comparison.prepare(bot_settings, tmp_path)
    assert not (tmp_path / "manifest.json").exists()


def test_prepare_never_overwrites_preregistered_manifest(bot_settings, tmp_path, monkeypatch):
    original = '{"registered": "immutable"}\n'
    path = tmp_path / "manifest.json"
    path.write_text(original)

    def unexpected(*_args, **_kwargs):
        pytest.fail("Existing preregistration must be checked before any external request.")

    monkeypatch.setattr(comparison, "public_json", unexpected)
    with pytest.raises(ValueError, match="never overwrite"):
        comparison.prepare(bot_settings, tmp_path)
    assert path.read_text() == original


class ExtraAnswers:
    def __init__(self, wrong_case=None):
        self.calls = []
        self.wrong_case = wrong_case

    async def decide(self, messages, temperature, **kwargs):
        self.calls.append((deepcopy(messages), temperature, deepcopy(kwargs)))
        case = next(case for case in comparison.EXTRA if case["query"] == messages[-1]["content"])
        if case["id"] == self.wrong_case:
            return RoutingCompletion("WRONG RAW OUTPUT", [])
        decision = {"action": "respond", "reason": "Fixture result.", "text": "Answer."}
        if case["expected_arguments"]:
            decision = {
                "action": "call_tool",
                "name": case["expected_action"],
                "arguments": case["expected_arguments"],
                "reason": "Fixture result.",
            }
        return RoutingCompletion(json.dumps(decision, ensure_ascii=False), [])


async def test_extra_trials_score_four_fresh_cases_and_resume_without_more_api_calls(
    bot_settings, tmp_path
):
    client, output = ExtraAnswers(), tmp_path / "extra.json"
    settings = replace(bot_settings, openrouter_model=comparison.MODELS[0])
    report = await comparison.extra_trials(settings, client, catalog(), output, temperature=0)
    assert report["status"] == "completed"
    assert report["actions"] == 4
    assert report["arguments"] == 3
    assert report["denominators"] == {"actions": 4, "arguments": 3}
    assert len(client.calls) == 4
    assert [row["id"] for row in report["cases"]] == [case["id"] for case in comparison.EXTRA]
    assert all(len(row["calls"]) == 1 for row in report["cases"])
    for messages, temperature, kwargs in client.calls:
        assert [message["role"] for message in messages] == ["system", "user"]
        assert temperature == 0
        assert "decision_schema" in kwargs
        serialized = json.dumps({"messages": messages, **kwargs}, ensure_ascii=False)
        assert "expected_action" not in serialized
        assert "expected_arguments" not in serialized
    reopened = await comparison.extra_trials(settings, client, catalog(), output, temperature=0)
    assert reopened == report
    assert len(client.calls) == 4


async def test_extra_trials_keep_wrong_raw_output_and_failure_in_fixed_denominators(
    bot_settings, tmp_path
):
    client, output = ExtraAnswers(wrong_case="new_weather"), tmp_path / "wrong-extra.json"
    report = await comparison.extra_trials(bot_settings, client, catalog(), output, temperature=0.3)
    failed = report["cases"][0]
    assert failed["status"] == "failed"
    assert failed["calls"][0]["raw_content"] == "WRONG RAW OUTPUT"
    assert failed["score"]["routing_correct"] is False
    assert failed["score"]["arguments_correct"] is False
    assert report["actions"] == 3
    assert report["arguments"] == 2
    assert report["denominators"] == {"actions": 4, "arguments": 3}
    await comparison.extra_trials(bot_settings, client, catalog(), output, temperature=0.3)
    assert len(client.calls) == 4
