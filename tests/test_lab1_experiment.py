import json
from dataclasses import replace

import pytest

from app.llm import Completion, ModelError
from scripts.lab1_experiment import (
    COMPARISON_QUESTION,
    QUESTION,
    SYNTHETIC_USER,
    RecordingClient,
    RecordingResponse,
    protocol,
    run_trials,
    validate_isolation,
)


@pytest.fixture
def trial_time():
    class Clock:
        current = 0.0

        def __init__(self):
            self.waits = []

        def clock(self):
            return self.current

        async def sleep(self, seconds):
            self.waits.append(seconds)
            self.current += seconds

        @property
        def injection(self):
            return {"clock": self.clock, "sleeper": self.sleep}

    return Clock()


class FakeExperimentClient:
    def __init__(self, *, fail=False):
        self.last = None
        self.system_override = None
        self.calls = []
        self.fail = fail

    async def complete(self, messages, temperature):
        assert [m["role"] for m in messages] == ["system", "user"]
        self.calls.append((messages, temperature, self.system_override))
        self.last = {"messages": messages, "temperature": temperature}
        if self.fail:
            raise ModelError("Лимит исчерпан.")
        self.last.update(raw_answer="Корректный ответ.", truncated=False)
        return Completion("Корректный ответ.")


async def test_all_trials_reset_through_handlers_and_resume_without_more_calls(
    bot_settings, dialog_setup, tmp_path, trial_time
):
    _, repository, _ = dialog_setup
    client = FakeExperimentClient()
    output = tmp_path / "result.json"
    assert await run_trials(bot_settings, repository, client, output, **trial_time.injection)
    document = json.loads(output.read_text())
    assert len(client.calls) == len(document["attempts"]) == 14
    assert [call[1] for call in client.calls[:12]] == [0.0] * 3 + [0.3] * 3 + [0.7] * 3 + [1.0] * 3
    assert all(call[0][-1]["content"] == QUESTION for call in client.calls[:12])
    assert all(call[0][-1]["content"] == COMPARISON_QUESTION for call in client.calls[12:])
    assert client.calls[-2][2] != client.calls[-1][2]
    for attempt in document["attempts"]:
        assert attempt["reset"]["history_messages_after"] == 0
        assert attempt["reset"]["command"] == "/reset"
        assert attempt["reset"]["mode_after"] == "study"
        assert attempt["status"] == "ok"
        assert attempt["quality_review"] is None
        assert attempt["length_chars"] == len("Корректный ответ.")
    assert await repository.history(SYNTHETIC_USER, 20)
    assert len(trial_time.waits) == 13
    assert sum(trial_time.waits) >= 97.5
    waits = list(trial_time.waits)
    assert await run_trials(bot_settings, repository, client, output, **trial_time.injection)
    assert trial_time.waits == waits
    assert len(client.calls) == 14


async def test_failure_is_checkpointed_and_never_silently_retried(
    bot_settings, dialog_setup, tmp_path, trial_time
):
    _, repository, _ = dialog_setup
    client, output = FakeExperimentClient(fail=True), tmp_path / "result.json"
    assert not await run_trials(bot_settings, repository, client, output, **trial_time.injection)
    document = json.loads(output.read_text())
    assert document["attempts"][0]["status"] == "failed"
    assert document["attempts"][0]["telegram_simulated_messages"] == ["Лимит исчерпан."]
    assert not await run_trials(bot_settings, repository, client, output, **trial_time.injection)
    assert len(client.calls) == 1
    client.fail = False
    assert not await run_trials(
        bot_settings,
        repository,
        client,
        output,
        retry_failed=True,
        max_api_calls=2,
        **trial_time.injection,
    )
    document = json.loads(output.read_text())
    assert len(client.calls) == 2
    assert document["attempts"][0]["status"] == "failed"
    assert document["attempts"][1]["status"] == "ok"
    assert document["attempts"][1]["attempt"] == 2


async def test_truncated_observation_is_retained_and_batch_continues_without_reroll(
    bot_settings, dialog_setup, tmp_path, trial_time
):
    class TruncatedFirstClient(FakeExperimentClient):
        async def complete(self, messages, temperature):
            result = await super().complete(messages, temperature)
            if len(self.calls) == 1:
                self.last["truncated"] = True
                return Completion(result.text, truncated=True)
            return result

    _, repository, _ = dialog_setup
    client, output = TruncatedFirstClient(), tmp_path / "result.json"
    assert await run_trials(bot_settings, repository, client, output, **trial_time.injection)
    document = json.loads(output.read_text())
    assert len(client.calls) == len(document["attempts"]) == 14
    assert document["attempts"][0]["status"] == "truncated"
    assert document["attempts"][0]["call"]["raw_answer"] == "Корректный ответ."
    assert all(attempt["status"] == "ok" for attempt in document["attempts"][1:])
    assert await run_trials(bot_settings, repository, client, output, **trial_time.injection)
    assert len(client.calls) == 14


async def test_protocol_mismatch_rejected_before_api(
    bot_settings, dialog_setup, tmp_path, trial_time
):
    _, repository, _ = dialog_setup
    client, output = FakeExperimentClient(), tmp_path / "result.json"
    assert not await run_trials(
        bot_settings, repository, client, output, max_api_calls=0, **trial_time.injection
    )
    changed = replace(bot_settings, llm_max_output_tokens=4096)
    with pytest.raises(ValueError, match="different protocol"):
        await run_trials(changed, repository, client, output, **trial_time.injection)
    assert not client.calls
    assert "fake-openrouter" not in output.read_text()
    assert "fake-db" not in output.read_text()
    assert bot_settings.bot_token not in output.read_text()


async def test_unpaced_historical_checkpoint_is_preserved_and_not_replayed(
    bot_settings, dialog_setup, tmp_path, trial_time
):
    _, repository, _ = dialog_setup
    client, output = FakeExperimentClient(), tmp_path / "old-result.json"
    old_protocol = protocol(bot_settings)
    old_protocol["version"] = 1
    old_protocol.pop("pacing")
    original = json.dumps({"protocol": old_protocol, "attempts": []}, ensure_ascii=False)
    output.write_text(original)
    with pytest.raises(ValueError, match="different protocol"):
        await run_trials(bot_settings, repository, client, output, **trial_time.injection)
    assert output.read_text() == original
    assert client.calls == [] and trial_time.waits == []


def test_production_database_is_rejected(bot_settings):
    with pytest.raises(ValueError, match="isolated"):
        validate_isolation(bot_settings)
    validate_isolation(replace(bot_settings, postgres_db="lab1_eval_20261017"))


async def test_recording_response_preserves_usage_and_actual_model():
    class Response:
        status = 200

        async def json(self):
            return {
                "model": "actual/model",
                "id": "gen-test",
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                "choices": [{"finish_reason": "stop", "native_finish_reason": "stop"}],
            }

    metadata = {}
    response = RecordingResponse(Response(), metadata)
    data = await response.json()
    assert response.status == 200
    assert metadata["actual_model"] == "actual/model"
    assert metadata["usage"]["completion_tokens"] == 20
    assert metadata["response_id"] == data["id"]
    assert metadata["finish_reason"] == "stop"


async def test_recording_client_uses_production_parser_and_parameters(bot_settings, monkeypatch):
    calls = []

    class Response:
        status = 200

        def __init__(self):
            self.body = json.dumps(
                {
                    "choices": [{"message": {"content": "Ответ"}, "finish_reason": "length"}],
                    "model": "provider/model",
                    "usage": {"completion_tokens": 5},
                }
            ).encode()
            self.content_length = len(self.body)
            self.content = self

        async def iter_chunked(self, _size):
            yield self.body

    class Request:
        async def __aenter__(self):
            return Response()

        async def __aexit__(self, *args):
            return None

    class Session:
        def __init__(self, **kwargs):
            pass

        def post(self, *args, **kwargs):
            calls.append(kwargs)
            return Request()

        async def close(self):
            pass

    monkeypatch.setattr("app.llm.aiohttp.ClientSession", Session)
    client = RecordingClient(bot_settings)
    await client.start()
    try:
        client.system_override = "RCTF instruction"
        answer = await client.complete(
            [{"role": "system", "content": "original"}, {"role": "user", "content": "question"}],
            0.7,
        )
    finally:
        await client.close()
    assert answer == Completion("Ответ", truncated=True)
    assert calls[0]["json"]["temperature"] == 0.7
    assert calls[0]["json"]["provider"] == {"max_price": {"prompt": 0, "completion": 0}}
    assert calls[0]["json"]["messages"][0]["content"].startswith("RCTF instruction")
    assert client.last["raw_answer"] == "Ответ"
    assert client.last["actual_model"] == "provider/model"
    assert client.last["usage"] == {"completion_tokens": 5}
    assert client.last["duration_seconds"] >= 0
