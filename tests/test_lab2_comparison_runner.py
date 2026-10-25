"""Parameter sweeps keep production inputs and the original scoring contract fixed."""

import asyncio
import json
from copy import deepcopy
from dataclasses import replace

import pytest

from app.llm import RoutingCompletion
from app.mcp.contracts import tool_schemas
from app.mcp.server import TOOLS
from scripts.lab2_experiment import (
    MODEL,
    USER,
    CatalogOnlyMCP,
    MemoryRepository,
    RecordingClient,
    digest,
    load_dataset,
    run_trials,
)

SECOND_MODEL = "example/second-model:free"
MODELS = (MODEL, SECOND_MODEL)


def catalog():
    return CatalogOnlyMCP(
        {
            name: {"name": name, "description": entry[2], **tool_schemas(name)}
            for name, entry in TOOLS.items()
        },
        {"schedule://current-week": {"uri": "schedule://current-week"}},
    )


class CapturingClient:
    """No expected action or argument is needed to observe model input."""

    def __init__(self):
        self.calls = []

    async def decide(self, messages, temperature, **kwargs):
        self.calls.append((deepcopy(messages), temperature, deepcopy(kwargs)))
        return RoutingCompletion(
            json.dumps({"action": "respond", "reason": "Fixture response.", "text": "Answer."}),
            [],
        )


async def test_two_free_models_and_temperatures_receive_identical_inputs(bot_settings, tmp_path):
    inputs, input_digests, request_digests = [], [], []
    dataset = load_dataset()
    for index, (model, temperature) in enumerate(
        (model, temperature) for model in MODELS for temperature in (0, 0.3)
    ):
        settings = replace(bot_settings, openrouter_model=model)
        client = CapturingClient()
        report = await run_trials(
            settings,
            client,
            catalog(),
            tmp_path / f"configuration-{index}.json",
            temperature=temperature,
            models=MODELS,
        )
        assert report["protocol"]["model"] == model
        assert report["protocol"]["temperature"] == temperature
        assert report["metrics"]["routing_accuracy"]["total"] == 12
        assert report["metrics"]["argument_accuracy"]["total"] == 9
        assert report["metrics"]["llm_calls"] == len(client.calls) == 11
        assert len(report["attempts"]) == 12
        assert report["attempts"][-1]["routing_source"] == "local_validation"
        assert all(call[1] == temperature for call in client.calls)
        payloads = [{"messages": messages, **kwargs} for messages, _, kwargs in client.calls]
        serialized = json.dumps(payloads, ensure_ascii=False, sort_keys=True).encode()
        assert b"expected_action" not in serialized
        assert b"expected_arguments" not in serialized
        assert all("expected" not in payload for payload in payloads)
        assert [call[0][-1]["content"] for call in client.calls] == [
            case["query"] for case in dataset["cases"][:-1]
        ]
        recorded_calls = [call for row in report["attempts"] for call in row["calls"]]
        assert [call["input_sha256"] for call in recorded_calls] == [
            digest(payload) for payload in payloads
        ]
        inputs.append(serialized)
        input_digests.append([call["input_sha256"] for call in recorded_calls])
        request_digests.append([call["request_sha256"] for call in recorded_calls])
    assert all(value == inputs[0] for value in inputs)
    assert all(value == input_digests[0] for value in input_digests)
    assert request_digests[0] != request_digests[1]  # Temperature is separately observable.
    assert request_digests[0] == request_digests[2]  # Model is outside the prompt payload.


async def test_memory_storage_returns_the_selected_trial_user_and_has_noop_start():
    user = replace(USER, temperature=0)
    repository = MemoryRepository("Europe/Moscow", user=user)
    assert await repository.user(USER.user_id) is user
    assert await repository.start_request(USER.user_id, 1) is None
    assert repository.audit is None and repository.exchange is None
    assert await MemoryRepository("Europe/Moscow").user(USER.user_id) is USER


@pytest.mark.parametrize("temperature", [None, True, "0.3", -0.1, 1.1, float("nan"), float("inf")])
async def test_invalid_temperature_is_rejected_before_any_model_call(
    bot_settings, tmp_path, temperature
):
    client, output = CapturingClient(), tmp_path / "invalid.json"
    with pytest.raises(ValueError, match="Temperature"):
        await run_trials(bot_settings, client, catalog(), output, temperature=temperature)
    assert not client.calls and not output.exists()


@pytest.mark.parametrize(
    "model,models",
    [
        ("example/paid", ("example/paid",)),
        (MODEL, (MODEL, "example/paid")),
        (SECOND_MODEL, (MODEL,)),
        (MODEL, ()),
        (MODEL, (MODEL, ":free")),
        (MODEL, (MODEL, "example/space model:free")),
        (MODEL, (MODEL, None)),
    ],
)
async def test_only_explicitly_allowed_free_models_are_accepted_before_calls(
    bot_settings, tmp_path, model, models
):
    client, output = CapturingClient(), tmp_path / "invalid.json"
    with pytest.raises(ValueError, match="selected free model"):
        await run_trials(
            replace(bot_settings, openrouter_model=model), client, catalog(), output, models=models
        )
    assert not client.calls and not output.exists()


async def test_default_still_allows_only_original_model_at_original_temperature(
    bot_settings, tmp_path
):
    client = CapturingClient()
    with pytest.raises(ValueError, match="selected free model"):
        await run_trials(
            replace(bot_settings, openrouter_model=SECOND_MODEL),
            client,
            catalog(),
            tmp_path / "other-model.json",
        )
    assert not client.calls
    report = await run_trials(
        bot_settings, client, catalog(), tmp_path / "original.json", case_ids=["respond_1"]
    )
    assert report["protocol"]["model"] == MODEL
    assert report["protocol"]["temperature"] == 0.3
    assert client.calls[0][1] == 0.3


async def test_reopening_comparison_checkpoint_uses_no_additional_quota(bot_settings, tmp_path):
    settings = replace(bot_settings, openrouter_model=SECOND_MODEL)
    client, output = CapturingClient(), tmp_path / "checkpoint.json"
    original = await run_trials(settings, client, catalog(), output, models=MODELS, temperature=0)
    assert len(client.calls) == 11
    reopened = await run_trials(settings, client, catalog(), output, models=MODELS, temperature=0)
    assert len(client.calls) == 11
    assert reopened["attempts"] == original["attempts"]
    with pytest.raises(ValueError, match="different protocol"):
        await run_trials(settings, client, catalog(), output, models=MODELS, temperature=0.3)
    assert len(client.calls) == 11


@pytest.mark.parametrize("temperature", [0, 1])
async def test_temperature_closed_interval_endpoints_are_allowed(
    bot_settings, tmp_path, temperature
):
    client = CapturingClient()
    report = await run_trials(
        bot_settings,
        client,
        catalog(),
        tmp_path / "endpoint.json",
        case_ids=["respond_1"],
        temperature=temperature,
    )
    assert report["protocol"]["temperature"] == client.calls[0][1] == temperature


async def test_twelve_single_case_invocations_keep_full_protocol_and_identical_inputs(
    bot_settings, tmp_path
):
    complete_client = CapturingClient()
    complete = await run_trials(
        bot_settings, complete_client, catalog(), tmp_path / "complete.json"
    )
    client, output = CapturingClient(), tmp_path / "interleaved.json"
    for index in range(12):
        report = await run_trials(bot_settings, client, catalog(), output, max_new_cases=1)
        assert report["protocol"] == complete["protocol"]
        assert len(report["attempts"]) == index + 1
        assert report["metrics"]["completed"] == (index == 11)
        assert report["metrics"]["scope"] == "full_12_cases"
        assert report["metrics"]["routing_accuracy"]["total"] == 12
        assert report["metrics"]["argument_accuracy"]["total"] == 9
    assert len(client.calls) == 11
    assert client.calls == complete_client.calls
    assert [row["id"] for row in report["attempts"]] == [row["id"] for row in complete["attempts"]]
    assert report["metrics"] == complete["metrics"]
    assert [call["input_sha256"] for row in report["attempts"] for call in row["calls"]] == [
        call["input_sha256"] for row in complete["attempts"] for call in row["calls"]
    ]
    # Completed reports remain idempotent even with an invocation limit.
    await run_trials(bot_settings, client, catalog(), output, max_new_cases=1)
    assert len(client.calls) == 11


@pytest.mark.parametrize("limit", [0, -1, True, 1.0, "1"])
async def test_invalid_batch_limit_is_rejected_before_any_api_call(bot_settings, tmp_path, limit):
    client, output = CapturingClient(), tmp_path / "invalid-limit.json"
    with pytest.raises(ValueError, match="new-case limit"):
        await run_trials(bot_settings, client, catalog(), output, max_new_cases=limit)
    assert not client.calls and not output.exists()


async def test_batch_resume_retains_interrupted_attempt_and_runs_only_next_case(
    bot_settings, tmp_path
):
    class Interrupted(CapturingClient):
        async def decide(self, messages, temperature, **kwargs):
            await super().decide(messages, temperature, **kwargs)
            raise asyncio.CancelledError

    output, interrupted = tmp_path / "interrupted-batch.json", Interrupted()
    with pytest.raises(asyncio.CancelledError):
        await run_trials(bot_settings, interrupted, catalog(), output, max_new_cases=1)
    assert len(interrupted.calls) == 1
    resumed = CapturingClient()
    report = await run_trials(bot_settings, resumed, catalog(), output, max_new_cases=1)
    assert len(resumed.calls) == 1
    assert len(report["attempts"]) == 2
    assert report["attempts"][0]["error"] == {"type": "InterruptedAttempt"}
    assert report["metrics"]["llm_calls"] == 2
    assert not report["metrics"]["completed"]
    assert resumed.calls[0][0][-1]["content"] != interrupted.calls[0][0][-1]["content"]


def test_recording_client_whitelists_bounded_provider_name(bot_settings):
    client = RecordingClient(bot_settings)
    client._observe_response({"provider": "Free provider", "internal": "hidden"})
    assert client.observation == {"provider": "Free provider"}
    for invalid in ("x" * 257, None, 42, {"name": "Untrusted object"}):
        client.observation = {}
        client._observe_response({"provider": invalid})
        assert not client.observation
