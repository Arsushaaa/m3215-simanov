"""Controlled LR1 experiment; real OpenRouter/isolated PostgreSQL, simulated Telegram.

Run with ``python -m scripts.lab1_experiment --env-file /tmp/lab1-eval.env
--output docs/evaluation/lab1-results.json``. No API retries are implicit. A failed attempt
is checkpointed and stops the run; ``--retry-failed`` explicitly permits another
attempt, still within ``--max-api-calls`` across all recorded attempts.
"""

import argparse
import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

from aiogram import Bot, Dispatcher
from aiogram.methods import SendMessage
from aiogram.types import Chat, Message, Update, User

from app.config import Settings
from app.db import create_pool
from app.dialog import DialogService
from app.handlers.assistant import create_router
from app.llm import OpenRouterClient
from app.migration import migrate
from app.prompts import PROMPTS
from app.repository import DialogRepository
from app.request_limits import REQUESTS_PER_MINUTE, WINDOW_SECONDS

SYNTHETIC_USER = 899000001
TEMPERATURES = (0.0, 0.3, 0.7, 1.0)
QUESTION = (
    "Объясни первокурснику, чем список (list) отличается от кортежа (tuple) в Python. "
    "Ответь на русском в пределах 120–180 слов, ровно в трёх разделах: "
    "«Различия», «Пример», «Выбор». В первом назови изменяемость и общие свойства, "
    "во втором дай небольшой корректный пример Python, в третьем предложи по одному "
    "случаю применения. Отметь, может ли кортеж содержать изменяемый объект."
)
COMPARISON_QUESTION = "Расскажи о копировании списков в Python."
WEAK_PROMPT = "Ты помощник. Помоги с Python."
IMPROVED_PROMPT = PROMPTS["study"] + (
    "\nКонтекст: первокурсник уже знает списки, но путает присваивание и копирование. "
    "Задача: объясни присваивание, поверхностную и глубокую копию; покажи, "
    "почему вложенный список может измениться в двух переменных. "
    "Формат: по-русски, 120–180 слов, три раздела «Различия», «Пример», «Выбор»; "
    "один выполняемый пример Python и практическая рекомендация."
)


def now() -> str:
    return datetime.now(UTC).isoformat()


def protocol(settings: Settings) -> dict:
    """Public fixed parameters only; credentials and connection details are omitted."""
    return {
        "version": 2,
        "question": QUESTION,
        "study_prompt": PROMPTS["study"],
        "mode": "study",
        "model": settings.openrouter_model,
        "max_tokens": settings.llm_max_output_tokens,
        "timeout_seconds": settings.llm_timeout_seconds,
        "history_max_messages": settings.history_max_messages,
        "history_max_tokens": settings.history_max_tokens,
        "reasoning": {"enabled": False},
        "provider": {"max_price": {"prompt": 0, "completion": 0}},
        "temperatures": list(TEMPERATURES),
        "repetitions": 3,
        "telegram": "synthetic private updates through actual aiogram handlers; no network",
        "database": "isolated PostgreSQL database named lab1_eval*",
        "pacing": {
            "minimum_seconds_after_previous_completion": WINDOW_SECONDS / REQUESTS_PER_MINUTE,
            "method": "wait before the next generation; ordinary application limits remain enabled",
        },
        "rubric_before_execution": {
            "format": "Russian; exactly Различия/Пример/Выбор; 120–180 words; Python block",
            "completeness": "mutability; shared sequence properties; two use cases; nested mutable",
            "clarity": "suitable for a first-year student; clear explanation and example",
            "facts": "correct Python example; no claim that tuple contents are always immutable",
            "analysis": "human assessment after execution; no automated quality verdict",
        },
        "comparison": {
            "question": COMPARISON_QUESTION,
            "temperature": 0.3,
            "weak_prompt": WEAK_PROMPT,
            "improved_prompt": IMPROVED_PROMPT,
            "method": "replace only system message at client boundary, preserve user and budget",
            "rubric_before_execution": [
                "Distinguish aliasing, shallow copy and deepcopy accurately",
                "Explain nested mutable object sharing with a correct example",
                "Meet improved requested format and length",
                "Give a useful recommendation understandable to a beginner",
            ],
        },
    }


def trial_specs() -> list[dict]:
    trials = [
        {
            "id": f"temperature-{temperature:.1f}-{repeat}",
            "temperature": temperature,
            "repetition": repeat,
            "question": QUESTION,
            "system_override": None,
        }
        for temperature in TEMPERATURES
        for repeat in range(1, 4)
    ]
    trials.extend(
        {
            "id": "prompt-" + name,
            "temperature": 0.3,
            "question": COMPARISON_QUESTION,
            "system_override": prompt,
        }
        for name, prompt in (("weak", WEAK_PROMPT), ("improved", IMPROVED_PROMPT))
    )
    return trials


def checkpoint(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_checkpoint(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _record_metadata(data: dict, metadata: dict) -> None:
    metadata.update(
        actual_model=data.get("model"),
        response_id=data.get("id"),
        usage=data.get("usage"),
    )
    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        choice = choices[0]
        metadata["finish_reason"] = choice.get("finish_reason")
        metadata["native_finish_reason"] = choice.get("native_finish_reason")


class RecordingResponse:
    def __init__(self, response, metadata: dict):
        self.response, self.metadata = response, metadata

    @property
    def status(self):
        return self.response.status

    def __getattr__(self, name):
        return getattr(self.response, name)

    async def json(self):
        data = await self.response.json()
        _record_metadata(data, self.metadata)
        return data


class RecordingRequest:
    def __init__(self, request, metadata: dict):
        self.request, self.metadata = request, metadata

    async def __aenter__(self):
        response = await self.request.__aenter__()
        self.metadata["http_status"] = response.status
        return RecordingResponse(response, self.metadata)

    async def __aexit__(self, *args):
        return await self.request.__aexit__(*args)


class RecordingSession:
    def __init__(self, session, client):
        self.session, self.client = session, client

    def post(self, *args, **kwargs):
        return RecordingRequest(self.session.post(*args, **kwargs), self.client.last)

    async def close(self):
        await self.session.close()


class RecordingClient(OpenRouterClient):
    """Use the production HTTP implementation, merely observing public metadata."""

    def __init__(self, settings):
        super().__init__(settings)
        self.last = None
        self.system_override = None

    async def start(self):
        await super().start()
        self.session = RecordingSession(self.session, self)

    def _observe_response(self, data):
        if self.last is not None:
            _record_metadata(data, self.last)

    async def complete(self, messages, temperature, **kwargs):
        if self.system_override:
            messages = [
                {
                    "role": "system",
                    "content": self.system_override
                    + f"\nБюджет текущего ответа — {self.settings.llm_max_output_tokens} токенов.",
                },
                *messages[1:],
            ]
        self.last = {
            "started_at": now(),
            "messages": messages,
            "temperature": temperature,
            "requested_model": self.settings.openrouter_model,
        }
        started = time.monotonic()
        try:
            result = await super().complete(messages, temperature, **kwargs)
            self.last.update(raw_answer=result.text, truncated=result.truncated)
            return result
        finally:
            self.last.update(
                finished_at=now(), duration_seconds=round(time.monotonic() - started, 3)
            )


def synthetic_update(text: str, message_id: int) -> Update:
    return Update(
        update_id=message_id,
        message=Message(
            message_id=message_id,
            date=datetime.now(UTC),
            chat=Chat(id=SYNTHETIC_USER, type="private"),
            from_user=User(id=SYNTHETIC_USER, is_bot=False, first_name="LR1 experiment"),
            text=text,
        ),
    )


async def run_trials(
    settings,
    repository,
    client,
    output: Path,
    *,
    retry_failed=False,
    max_api_calls=14,
    clock=time.monotonic,
    sleeper=asyncio.sleep,
) -> bool:
    expected = protocol(settings)
    document = await asyncio.to_thread(load_checkpoint, output)
    if document is not None:
        if document["protocol"] != expected:
            raise ValueError("Existing result uses a different protocol; use a new output file.")
    else:
        document = {"created_at": now(), "protocol": expected, "attempts": []}
        checkpoint(output, document)  # Declare conditions and rubric before the first API call.
    bot, dispatcher = Bot(settings.bot_token), Dispatcher()
    sent = []

    async def fake_telegram(_bot, method, **kwargs):
        if isinstance(method, SendMessage):
            sent.append(method.text)
            return Message(
                message_id=len(sent),
                date=datetime.now(UTC),
                chat=Chat(id=SYNTHETIC_USER, type="private"),
                text=method.text,
            )
        return True

    bot.session = AsyncMock(side_effect=fake_telegram)
    dispatcher.include_router(create_router(request_limits_clock=clock))
    service = DialogService(repository, client, settings)
    message_id = len(document["attempts"]) * 10 + 1
    last_completed = None

    async def feed(text):
        nonlocal message_id
        message_id += 1
        await dispatcher.feed_update(bot, synthetic_update(text, message_id), dialog=service)

    try:
        await feed("/study")
        for spec in trial_specs():
            previous = [a for a in document["attempts"] if a["trial_id"] == spec["id"]]
            if any(a["status"] in {"ok", "truncated"} for a in previous):
                continue
            if previous and not retry_failed:
                return False
            if len(document["attempts"]) >= max_api_calls:
                return False
            if last_completed is not None:
                # Pace a legitimate experiment rather than disable the public
                # handler's rate guard or retry a rejected generation.
                interval = WINDOW_SECONDS / REQUESTS_PER_MINUTE
                while (remaining := last_completed + interval - clock()) > 0:
                    await sleeper(remaining)
            await feed(f"/settings {spec['temperature']}")
            await feed("/reset")
            history = await repository.history(SYNTHETIC_USER, settings.history_max_messages)
            user = await repository.user(SYNTHETIC_USER)
            if history or user.mode != "study" or user.temperature != spec["temperature"]:
                raise RuntimeError(
                    "Reset/mode/temperature precondition failed; no API request sent."
                )
            attempt = {
                "trial_id": spec["id"],
                "attempt": len(previous) + 1,
                "status": "running",
                "temperature": spec["temperature"],
                "question": spec["question"],
                "reset": {
                    "command": "/reset",
                    "handler": "app.handlers.assistant.create_router",
                    "history_messages_after": len(history),
                    "mode_after": user.mode,
                    "temperature_after": user.temperature,
                    "checked_at": now(),
                },
                "quality_review": None,
            }
            document["attempts"].append(attempt)
            checkpoint(output, document)
            client.system_override, client.last = spec["system_override"], None
            sent.clear()
            try:
                await feed(spec["question"])
                call = client.last
                attempt["call"] = call
                attempt["telegram_simulated_messages"] = list(sent)
                if call and "raw_answer" in call:
                    attempt["length_chars"] = len(call["raw_answer"])
                    attempt["status"] = "truncated" if call["truncated"] else "ok"
                else:
                    attempt["status"] = "failed"
            except Exception as exc:
                # Never include exception bodies: remote responses may contain sensitive material.
                attempt.update(status="failed", error_type=type(exc).__name__, call=client.last)
            finally:
                last_completed = clock()
                attempt["finished_at"] = now()
                checkpoint(output, document)
            if attempt["status"] not in {"ok", "truncated"}:
                return False
        return True
    finally:
        await dispatcher.emit_shutdown()
        await bot.session.close()


def validate_isolation(settings: Settings):
    if not settings.postgres_db.startswith("lab1_eval"):
        raise ValueError("Use an isolated PostgreSQL database whose name starts with lab1_eval.")


async def main(args):
    settings = Settings.load(args.env_file, environ={}, require_llm=True)
    validate_isolation(settings)
    pool = await create_pool(settings)
    client = RecordingClient(settings)
    try:
        await migrate(pool)
        await client.start()
        complete = await run_trials(
            settings,
            DialogRepository(pool),
            client,
            args.output,
            retry_failed=args.retry_failed,
            max_api_calls=args.max_api_calls,
        )
    finally:
        await client.close()
        await pool.close()
    print(
        "Experiment complete: 12 temperature trials + 2 prompt comparisons."
        if complete
        else "Stopped; inspect checkpoint. No automatic retry was performed."
    )
    return 0 if complete else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        required=True,
        type=Path,
        help="Dedicated env with isolated lab1_eval* DB; never use production .env",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--max-api-calls",
        type=int,
        default=14,
        help="Total attempt ceiling including prior failures (default: 14)",
    )
    raise SystemExit(asyncio.run(main(parser.parse_args())))
