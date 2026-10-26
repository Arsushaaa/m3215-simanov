import asyncio
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.llm import ModelError, RoutingCompletion
from app.mcp.contracts import tool_schemas
from app.mcp.server import TOOLS
from scripts.lab2_experiment import (
    DATASET,
    CatalogOnlyMCP,
    RecordingClient,
    arguments_equal,
    checkpoint,
    load_dataset,
    main,
    metrics,
    run_trials,
    score,
    selected_cases,
)


def catalog():
    return CatalogOnlyMCP(
        {
            name: {"name": name, "description": entry[2], **tool_schemas(name)}
            for name, entry in TOOLS.items()
        },
        {"schedule://current-week": {"uri": "schedule://current-week"}},
    )


def evidence(query):
    source = {"source": "user", "message_index": 0, "quote": query}
    return {field: dict(source) for field in ("request", "text", "date", "time")}


class Answers:
    def __init__(self, overrides=None):
        self.calls = []
        self.overrides = overrides or {}

    async def decide(self, messages, temperature, **kwargs):
        self.calls.append((deepcopy(messages), temperature, deepcopy(kwargs)))
        case = next(c for c in load_dataset()["cases"] if c["query"] == messages[-1]["content"])
        if case["id"] in self.overrides:
            result = self.overrides[case["id"]]
            if isinstance(result, BaseException):
                raise result
            if isinstance(result, str):
                return RoutingCompletion(result, [])
            return RoutingCompletion(json.dumps(result, ensure_ascii=False), [])
        action = case["expected_action"]
        decision = {"action": action, "reason": "Краткое объяснение."}
        if action in {"get_weather", "get_schedule", "add_reminder"}:
            decision.update(action="call_tool", name=action, arguments=case["expected_arguments"])
            if action == "add_reminder":
                decision["evidence"] = evidence(case["query"])
        elif action == "respond":
            decision["text"] = "Ответ на вопрос."
        else:
            decision["action"] = "clarify_reminder"
            decision["question"] = "Во сколько напомнить?"
        return RoutingCompletion(json.dumps(decision, ensure_ascii=False), [])


async def test_all_fixed_cases_follow_real_dialog_route_and_never_execute_tools(
    bot_settings, tmp_path
):
    client, output = Answers(), tmp_path / "result.json"
    result = await run_trials(bot_settings, client, catalog(), output)
    assert result["metrics"]["acceptance"] == "passed"
    assert result["metrics"]["routing_accuracy"]["correct"] == 12
    assert result["metrics"]["argument_accuracy"]["correct"] == 9
    assert result["metrics"]["llm_calls"] == len(client.calls) == 12
    assert result["metrics"]["mcp_application_calls"] == 0
    assert result["metrics"]["database_writes"] == 0
    assert [row["id"] for row in result["attempts"]] == [c["id"] for c in load_dataset()["cases"]]
    assert [row["status"] for row in result["attempts"][:6]] == ["captured_before_mcp"] * 6
    for row in result["attempts"][6:9]:
        assert row["actual"]["action"] == "add_reminder"
        assert row["actual"]["timezone"] == "Europe/Moscow"
        assert "требуется подтверждение" in row["displayed_answer"]
        assert row["audit"]["mcp_calls"] == 0
    assert len(result["attempts"][-1]["calls"]) == 1
    assert result["attempts"][-1]["routing_source"] == "llm"
    assert result["attempts"][-1]["audit"]["target"] == "add_reminder"
    for messages, temperature, kwargs in client.calls:
        assert [m["role"] for m in messages] == ["system", "user"]
        assert temperature == 0.3
        assert "2026-10-12T09:00:00+03:00" in messages[0]["content"]
        assert '"reminder_timezone":"Europe/Moscow"' in messages[0]["content"]
        assert (
            f"Бюджет текущего ответа — {bot_settings.llm_max_output_tokens}"
            in messages[0]["content"]
        )
        assert "expected_action" not in json.dumps(messages)
        assert "expected_arguments" not in json.dumps(messages)
        assert "decision_schema" in kwargs and "tools" not in kwargs
    serialized = output.read_text()
    assert all(
        secret not in serialized
        for secret in (
            bot_settings.bot_token,
            bot_settings.postgres_password,
            bot_settings.openrouter_api_key,
        )
    )
    # Re-running a completed report makes no new model requests.
    await run_trials(bot_settings, client, catalog(), output)
    assert len(client.calls) == 12


async def test_wrong_or_missing_needed_actions_keep_nine_argument_denominator(
    bot_settings, tmp_path
):
    client = Answers(
        {
            "weather_1": {"action": "respond", "text": "Уточните город.", "reason": "Неясно."},
            "schedule_1": ModelError("Лимит запросов к модели исчерпан. Попробуй позже."),
        }
    )
    result = await run_trials(bot_settings, client, catalog(), tmp_path / "failure.json")
    assert result["metrics"]["routing_accuracy"]["correct"] == 10
    assert result["metrics"]["routing_accuracy"]["total"] == 12
    assert result["metrics"]["argument_accuracy"]["correct"] == 7
    assert result["metrics"]["argument_accuracy"]["total"] == 9
    assert result["metrics"]["acceptance"] == "not_passed"
    assert len(result["attempts"]) == 12
    assert len(client.calls) == 12
    failed = next(row for row in result["attempts"] if row["id"] == "schedule_1")
    assert failed["error"]["type"] == "ModelError"
    assert failed["calls"][0]["error"]["type"] == "ModelError"


async def test_unverified_reminder_source_is_refused_without_changing_recorded_model_output(
    bot_settings, tmp_path
):
    decision = {
        "action": "call_tool",
        "name": "add_reminder",
        "arguments": {"text": "отправить отчёт", "remind_at": "2026-10-13T15:00:00+03:00"},
        "evidence": evidence("Unverified source which the user did not send."),
        "reason": "Напоминание.",
    }
    result = await run_trials(
        bot_settings,
        Answers({"reminder_1": decision}),
        catalog(),
        tmp_path / "reminder.json",
        case_ids=["reminder_1"],
    )
    row = result["attempts"][0]
    assert row["actual"]["action"] == "clarify"
    assert not row["score"]["routing_correct"]
    assert not row["score"]["arguments_correct"]
    assert json.loads(row["calls"][0]["raw_content"]) == decision
    assert row["audit"]["error_code"] == "INVALID_ARGUMENTS"


async def test_wrong_derived_time_remains_a_scoring_failure_even_with_valid_source_quotes(
    bot_settings, tmp_path
):
    case = next(case for case in load_dataset()["cases"] if case["id"] == "reminder_1")
    decision = {
        "action": "call_tool",
        "name": "add_reminder",
        "arguments": {"text": "отправить отчёт", "remind_at": "2026-10-13T15:00:00+03:00"},
        "evidence": evidence(case["query"]),
        "reason": "Напоминание.",
    }
    result = await run_trials(
        bot_settings,
        Answers({case["id"]: decision}),
        catalog(),
        tmp_path / "wrong-derived-time.json",
        case_ids=[case["id"]],
    )
    row = result["attempts"][0]
    # Exact quotes prove provenance, not correct semantic interpretation. The
    # proposed time is preserved for owner review, never corrected to the oracle.
    assert row["actual"]["action"] == "add_reminder"
    assert row["actual"]["arguments"] == decision["arguments"]
    assert row["score"]["routing_correct"]
    assert not row["score"]["arguments_correct"]
    assert row["audit"]["mcp_calls"] == 0


async def test_failure_raw_output_is_preserved_without_implicit_retry(bot_settings, tmp_path):
    client, output = Answers({"weather_1": "not valid JSON"}), tmp_path / "invalid.json"
    result = await run_trials(bot_settings, client, catalog(), output, case_ids=["weather_1"])
    assert result["attempts"][0]["status"] == "failed"
    assert result["attempts"][0]["calls"][0]["raw_content"] == "not valid JSON"
    await run_trials(bot_settings, client, catalog(), output, case_ids=["weather_1"])
    assert len(client.calls) == 1
    assert result["metrics"]["acceptance"] == "not_applicable_subset"


async def test_checkpoint_protocol_and_case_selection_are_fail_closed(bot_settings, tmp_path):
    client, output = Answers(), tmp_path / "protocol.json"
    await run_trials(bot_settings, client, catalog(), output, case_ids=["weather_1"])
    for settings, cases in (
        (replace(bot_settings, llm_max_output_tokens=4096), ["weather_1"]),
        (bot_settings, ["weather_2"]),
    ):
        with pytest.raises(ValueError, match="different protocol"):
            await run_trials(settings, client, catalog(), output, case_ids=cases)
    for changed in (
        replace(bot_settings, llm_tool_mode="native"),
        replace(bot_settings, openrouter_model="other/model:free"),
        replace(bot_settings, schedule_timezone="UTC"),
    ):
        with pytest.raises(ValueError):
            await run_trials(changed, client, catalog(), tmp_path / "unused.json")
    assert len(client.calls) == 1
    for cases in (["missing"], ["weather_1", "weather_1"]):
        with pytest.raises(ValueError, match="Case IDs"):
            selected_cases(load_dataset(), cases)
    tampered = tmp_path / "dataset.json"
    tampered.write_bytes(DATASET.read_bytes() + b" ")
    with pytest.raises(ValueError, match="dataset has changed"):
        load_dataset(tampered)


async def test_legacy_checkpoint_version_is_preserved_and_never_resumed(bot_settings, tmp_path):
    client, output = Answers(), tmp_path / "legacy.json"
    report = await run_trials(bot_settings, client, catalog(), output, case_ids=["respond_1"])
    assert report["protocol"]["version"] == 2
    assert "app/reminder_intent.py" not in report["protocol"]["source_sha256"]
    assert {"app/agent_prompts.py", "app/reminder_grounding.py", "app/reminder_messages.py"} <= set(
        report["protocol"]["source_sha256"]
    )
    report["protocol"]["version"] = 1
    checkpoint(output, report)
    original = output.read_bytes()

    with pytest.raises(ValueError, match="different protocol"):
        await run_trials(bot_settings, client, catalog(), output, case_ids=["respond_1"])

    assert output.read_bytes() == original
    assert len(client.calls) == 1


async def test_budget_is_checkpointed_before_dispatch_and_interrupted_attempt_is_not_replayed(
    bot_settings, tmp_path
):
    output = tmp_path / "interrupt.json"

    class Interrupted:
        calls = 0

        async def decide(self, *_args, **_kwargs):
            self.calls += 1
            saved = json.loads(output.read_text())
            assert saved["attempts"][0]["status"] == "in_progress"
            assert len(saved["attempts"][0]["calls"]) == 1
            raise asyncio.CancelledError

    client = Interrupted()
    with pytest.raises(asyncio.CancelledError):
        await run_trials(bot_settings, client, catalog(), output, case_ids=["weather_1"])
    report = await run_trials(bot_settings, client, catalog(), output, case_ids=["weather_1"])
    assert client.calls == 1
    assert report["attempts"][0]["error"] == {"type": "InterruptedAttempt"}
    assert report["metrics"]["routing_accuracy"]["correct"] == 0
    assert report["metrics"]["llm_calls"] == 1


async def test_zero_budget_still_covers_all_twelve_cases_without_model(bot_settings, tmp_path):
    client = Answers()
    result = await run_trials(
        bot_settings, client, catalog(), tmp_path / "budget.json", max_api_calls=0
    )
    assert not client.calls
    assert len(result["attempts"]) == 12
    assert result["metrics"]["routing_accuracy"]["correct"] == 0
    assert result["metrics"]["argument_accuracy"]["correct"] == 0
    assert result["metrics"]["acceptance"] == "not_passed"


def test_fixed_denominators_include_omissions_and_subsets_never_pass_acceptance():
    dataset = load_dataset()
    row = {"id": "weather_1", "status": "ok", "score": {"routing_correct": True}}
    result = metrics(dataset, [row], dataset["cases"])
    assert result["routing_accuracy"] == {"correct": 1, "total": 12, "ratio": 1 / 12}
    assert result["argument_accuracy"]["total"] == 9
    assert result["acceptance"] == "not_passed"
    subset = metrics(dataset, [row], dataset["cases"][:1])
    assert subset["acceptance"] == "not_applicable_subset"


@pytest.mark.parametrize(
    "actual, expected, equal",
    [
        ({"city": "  САНКТ-ПЕТЕРБУРГ. "}, {"city": "Санкт-Петербург"}, True),
        ({"city": "СПб"}, {"city": "Санкт-Петербург"}, False),
        ({"text": "Отправить  отчет."}, {"text": "отправить отчёт"}, True),
        ({"text": "Отослать отчёт"}, {"text": "отправить отчёт"}, False),
        ({"remind_at": "2026-10-13T15:30:00Z"}, {"remind_at": "2026-10-13T18:30:00+03:00"}, True),
        ({"remind_at": "2026-10-13T18:30:00"}, {"remind_at": "2026-10-13T18:30:00+03:00"}, False),
        ({"remind_at": "bad"}, {"remind_at": "2026-10-13T18:30:00+03:00"}, False),
        ({"text": 7}, {"text": "7"}, False),
        ({"text": "A", "extra": "B"}, {"text": "A"}, False),
        (None, {"text": "A"}, False),
    ],
)
def test_scoring_normalization_has_explicit_narrow_semantics(actual, expected, equal):
    assert arguments_equal("add_reminder", actual, expected) is equal


def test_correct_argument_with_wrong_action_is_not_credited():
    case = load_dataset()["cases"][0]
    assert score(case, {"action": "respond", "arguments": case["expected_arguments"]}) == {
        "routing_correct": False,
        "arguments_required": True,
        "arguments_correct": False,
    }


def test_observer_whitelists_nested_metadata_and_malformed_content(bot_settings):
    client = RecordingClient(bot_settings)
    client._observe_response(
        {
            "model": "public/model",
            "id": "gen-public",
            "headers": {"Authorization": "secret"},
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "cost": 0.0,
                "reasoning": "secret",
                "debug": {"headers": "secret"},
                "total_tokens": "secret",
            },
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": {"reasoning": "secret"},
                        "reasoning": "secret",
                        "tool_calls": [
                            {
                                "id": "tool-public",
                                "type": "function",
                                "reasoning": "secret",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":"Казань"}',
                                    "reasoning": "secret",
                                },
                            }
                        ],
                    },
                }
            ],
        }
    )
    assert "secret" not in json.dumps(client.observation)
    assert client.observation["invalid_content_type"] == "dict"
    assert client.observation["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "cost": 0.0}
    assert client.observation["raw_tool_calls"][0]["function"] == {
        "name": "get_weather",
        "arguments": '{"city":"Казань"}',
    }


async def test_production_http_records_visible_decision_usage_and_safe_failure(
    bot_settings, monkeypatch, tmp_path
):
    bodies, requests = [], []
    secret = "private reasoning must never enter experiment artifacts"
    decision = {"action": "respond", "text": "Краткий ответ.", "reason": "Обычный вопрос."}
    bodies.append(
        {
            "model": "nvidia/nemotron-3-super-120b-a12b:free",
            "id": "gen-example",
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            "choices": [
                {
                    "message": {"content": json.dumps(decision), "reasoning": secret},
                    "finish_reason": "stop",
                }
            ],
        }
    )

    class Response:
        status = 200

        def __init__(self, data):
            self.body = json.dumps(data).encode()
            self.content_length = len(self.body)
            self.content = self

        async def iter_chunked(self, _size):
            yield self.body

    class Request:
        async def __aenter__(self):
            return Response(bodies.pop(0))

        async def __aexit__(self, *_args):
            pass

    class Session:
        def __init__(self, **_kwargs):
            pass

        def post(self, _url, **kwargs):
            requests.append(kwargs)
            return Request()

        async def close(self):
            pass

    monkeypatch.setattr("app.llm.aiohttp.ClientSession", Session)
    client = RecordingClient(bot_settings)
    await client.start()
    try:
        path = tmp_path / "http.json"
        result = await run_trials(bot_settings, client, catalog(), path, case_ids=["respond_1"])
    finally:
        await client.close()
    call = result["attempts"][0]["calls"][0]
    assert json.loads(call["raw_content"]) == decision
    assert call["model"] == bot_settings.openrouter_model
    assert call["id"] == "gen-example"
    assert call["usage"] == {"prompt_tokens": 100, "completion_tokens": 20}
    assert call["finish_reason"] == "stop"
    assert secret not in path.read_text()
    assert len(requests) == 1
    payload = requests[0]["json"]
    assert payload["provider"] == {
        "max_price": {"prompt": 0, "completion": 0},
        "require_parameters": True,
    }
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert payload["temperature"] == 0.3
    assert payload["reasoning"] == {"enabled": False}
    assert "tools" not in payload


@pytest.mark.parametrize(
    "model", ["nvidia/nemotron-3-super-120b-a12b:free", "dots-studio/dots-3-note-preview:free"]
)
async def test_main_uses_explicit_free_model_fixture_schedule_and_secret_free_discovery(
    bot_settings, monkeypatch, tmp_path, model
):
    observed = {}

    class Discovery:
        available = True
        tools = catalog().tools
        resources = catalog().resources

        def __init__(self, settings):
            observed["discovery_settings"] = settings

        async def start(self):
            pass

        async def close(self):
            pass

    class Client(Answers):
        def __init__(self, settings):
            super().__init__()
            observed["client_settings"] = settings

        async def start(self):
            pass

        async def close(self):
            pass

    monkeypatch.setattr(
        "scripts.lab2_experiment.Settings.load",
        lambda *_a, **_kw: replace(
            bot_settings,
            schedule_file="private/secret.json",
            schedule_timezone="UTC",
            openrouter_model=model,
        ),
    )
    monkeypatch.setattr("scripts.lab2_experiment.MCPClient", Discovery)
    monkeypatch.setattr("scripts.lab2_experiment.RecordingClient", Client)
    monkeypatch.setattr("scripts.lab2_experiment.configure_logging", lambda _settings: None)
    status = await main(
        SimpleNamespace(
            env_file=Path(".env"), output=tmp_path / "main.json", case=None, max_api_calls=12
        )
    )
    assert status == 0
    for settings in observed.values():
        assert settings.schedule_file.endswith("/data/schedule.json")
        assert settings.schedule_timezone == "Europe/Moscow"
        assert settings.openrouter_model == model
    settings = observed["discovery_settings"]
    assert settings.postgres_password == settings.bot_token == settings.openrouter_api_key == ""


async def test_corrupted_checkpoint_does_not_silently_duplicate_attempts(bot_settings, tmp_path):
    path = tmp_path / "duplicate.json"
    client = Answers()
    report = await run_trials(bot_settings, client, catalog(), path, case_ids=["respond_1"])
    report["attempts"].append(report["attempts"][0])
    checkpoint(path, report)
    with pytest.raises(ValueError, match="duplicate or unexpected"):
        await run_trials(bot_settings, client, catalog(), path, case_ids=["respond_1"])
    assert len(client.calls) == 1
