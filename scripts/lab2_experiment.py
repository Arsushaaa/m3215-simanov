"""Fixed LR2 routing experiment with real LLM; no tool execution or database writes.

Run ``python -m scripts.lab2_experiment --env-file .env --output
docs/evaluation/lab2-results.json``. Real stdio discovery supplies the catalog, then
the production DialogService/AgentService run with isolated in-memory storage.
Read decisions stop at the tool boundary; reminders pass normal proposal/time
validation. There is no final generation, confirmation, weather HTTP or SQL.
Use a new output and ``--case ID`` to repeat an explicitly selected failed case;
the full original report is preserved. No attempt is automatically retried.
"""

import argparse
import asyncio
import hashlib
import json
import math
import time
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from app.agent import AgentService
from app.config import ConfigError, Settings
from app.dialog import DialogService
from app.llm import ModelError, OpenRouterClient
from app.logging_setup import configure_logging
from app.mcp.client import MCPClient
from app.prompts import PROMPTS
from app.repository import UserSettings

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "docs/evaluation/lab2-routing.json"
DATASET_SHA256 = "0a9c9f06f85e094ae42aaa3b684ea368b8f051055f5476ed676203e2a1afdc98"
MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
TEMPERATURE = 0.3
USER = UserSettings(1, "study", TEMPERATURE, 0)
SCORING = (
    "Exact action; argument keys exact. Text: casefold, ё→е, collapsed whitespace, "
    "terminal .!? stripped. Datetime: equal aware UTC instant. Missing/wrong actions "
    "remain failures in the fixed denominators 12 and 9. No semantic aliases."
)


def timestamp():
    return datetime.now(UTC).isoformat()


def digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_dataset(path=DATASET):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATASET_SHA256:
        raise ValueError("The fixed 12-case dataset has changed.")
    return json.loads(raw)


def selected_cases(dataset, identifiers=None):
    identifiers = identifiers or []
    known = {case["id"] for case in dataset["cases"]}
    if len(set(identifiers)) != len(identifiers) or set(identifiers) - known:
        raise ValueError("Case IDs must be distinct members of the fixed dataset.")
    return [case for case in dataset["cases"] if not identifiers or case["id"] in identifiers]


def checkpoint(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _normalized(text):
    return " ".join(text.casefold().replace("ё", "е").split()).rstrip(".!?")


def arguments_equal(action, actual, expected):
    if not isinstance(actual, dict) or actual.keys() != expected.keys():
        return False
    for key, value in expected.items():
        candidate = actual[key]
        if not isinstance(candidate, str):
            return False
        if key == "remind_at":
            try:
                left, right = datetime.fromisoformat(candidate), datetime.fromisoformat(value)
                if left.utcoffset() is None or right.utcoffset() is None or left != right:
                    return False
            except ValueError:
                return False
        elif action == "get_schedule":
            if candidate != value:
                return False
        elif _normalized(candidate) != _normalized(value):
            return False
    return True


def score(case, actual):
    routing = actual.get("action") == case["expected_action"]
    requires_arguments = bool(case["expected_arguments"])
    return {
        "routing_correct": routing,
        "arguments_required": requires_arguments,
        "arguments_correct": (
            routing
            and arguments_equal(
                case["expected_action"], actual.get("arguments"), case["expected_arguments"]
            )
            if requires_arguments
            else None
        ),
    }


def metrics(dataset, attempts, selected):
    by_id = {row["id"]: row for row in attempts}
    routing = sum(
        by_id.get(case["id"], {}).get("score", {}).get("routing_correct") is True
        for case in dataset["cases"]
    )
    arguments = sum(
        by_id.get(case["id"], {}).get("score", {}).get("arguments_correct") is True
        for case in dataset["cases"]
        if case["expected_arguments"]
    )
    full_run = len(selected) == 12
    completed = len(attempts) == len(selected) and all(
        row["status"] != "in_progress" for row in attempts
    )
    return {
        "scope": "full_12_cases" if full_run else "selected_cases_only",
        "completed": completed,
        "routing_accuracy": {"correct": routing, "total": 12, "ratio": routing / 12},
        "argument_accuracy": {"correct": arguments, "total": 9, "ratio": arguments / 9},
        "acceptance": (
            "passed"
            if full_run and completed and routing >= 10 and arguments >= 8
            else "not_passed"
            if full_run
            else "not_applicable_subset"
        ),
        "llm_calls": sum(len(row.get("calls", [])) for row in attempts),
        "mcp_application_calls": 0,
        "database_writes": 0,
    }


class ReadBoundary(Exception):
    """A validated production decision, deliberately stopped before any tool IO."""

    def __init__(self, action, arguments):
        self.action, self.arguments = action, arguments


class CatalogOnlyMCP:
    def __init__(self, tools, resources):
        self.available = True
        self.tools, self.resources = deepcopy(tools), deepcopy(resources)

    async def call_tool(self, name, arguments):
        if name == "add_reminder":
            raise RuntimeError("Reminder writes are forbidden in the routing experiment.")
        raise ReadBoundary(name, arguments)

    async def read_resource(self, uri):
        raise ReadBoundary("read_resource", {"uri": uri})


class MemoryRepository:
    """Only the narrow storage methods used by one fresh production dialog."""

    def __init__(self, timezone, user=USER):
        self.zone, self.audit, self.exchange = timezone, None, None
        self._user = user

    async def user(self, _user_id):
        return self._user

    async def answer(self, _user_id, _message_id):
        return None

    async def history(self, _user_id, _maximum):
        return []

    async def has_incomplete_answer(self, _user_id):
        return False

    async def has_processed(self, _user_id, _message_id):
        return False

    async def start_request(self, _user_id, _message_id):
        pass

    async def timezone(self, _user_id):
        return self.zone

    async def record(self, _user_id, _message_id, **audit):
        self.audit = audit

    async def save_exchange(self, _user, _message_id, question, answer, **_kwargs):
        self.exchange = (question, answer)
        return True


class ObservedAgent(AgentService):
    async def respond(self, *args, **kwargs):
        self.reply = await super().respond(*args, **kwargs)
        return self.reply


class RecordingClient(OpenRouterClient):
    """Observe whitelisted public metadata at the production HTTP parsing boundary."""

    def __init__(self, settings):
        super().__init__(settings)
        self.observation = {}

    async def _request(self, payload):
        self.observation = {}
        return await super()._request(payload)

    def _observe_response(self, data):
        for key in ("model", "id", "provider"):
            value = data.get(key)
            if isinstance(value, str) and len(value) <= 256:
                self.observation[key] = value
        usage = data.get("usage")
        if isinstance(usage, dict):
            self.observation["usage"] = {
                key: value
                for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost")
                if type(value := usage.get(key)) in (int, float)
                and math.isfinite(value)
                and value >= 0
            }
        choices = data.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            choice = choices[0]
            finish_reason = choice.get("finish_reason")
            if isinstance(finish_reason, str) and len(finish_reason) <= 100:
                self.observation["finish_reason"] = finish_reason
            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) or content is None:
                    self.observation["raw_content"] = content
                else:
                    self.observation["invalid_content_type"] = type(content).__name__
                calls = message.get("tool_calls")
                if isinstance(calls, list):
                    public_calls = []
                    for call in calls:
                        if not isinstance(call, dict):
                            continue
                        public = {
                            key: call[key]
                            for key in ("id", "type")
                            if isinstance(call.get(key), str)
                        }
                        function = call.get("function")
                        if isinstance(function, dict):
                            public["function"] = {
                                key: function[key]
                                for key in ("name", "arguments")
                                if isinstance(function.get(key), str)
                            }
                        public_calls.append(public)
                    self.observation["raw_tool_calls"] = public_calls


class TrialLLM:
    """One routing call per case; preserve raw result even when local validation fails."""

    def __init__(self, client, record, *, permit, persist):
        self.client, self.record, self.permit, self.persist = client, record, permit, persist

    async def decide(self, messages, temperature, **kwargs):
        if self.record["calls"]:
            raise RuntimeError("A second LLM call is forbidden in a routing-only trial.")
        if not self.permit:
            raise ModelError("Лимит вызовов эксперимента достигнут.")
        call = {
            "started_at": timestamp(),
            "request_sha256": digest({"messages": messages, "temperature": temperature, **kwargs}),
            "input_sha256": digest({"messages": messages, **kwargs}),
            "temperature": temperature,
            "message_roles": [item["role"] for item in messages],
            "context_estimate_tokens": (
                len(
                    json.dumps(
                        {"messages": messages, "schemas": kwargs.get("decision_schema")},
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    ).encode()
                )
                + 2
            )
            // 3,
        }
        self.record["calls"].append(call)
        # Save the consumed slot before dispatch: an interrupted request is not free.
        self.persist()
        started = time.monotonic()
        if hasattr(self.client, "observation"):
            self.client.observation = {}
        try:
            result = await self.client.decide(messages, temperature, **kwargs)
            call.update(
                raw_content=result.text,
                raw_tool_calls=deepcopy(result.tool_calls),
                truncated=result.truncated,
            )
            return result
        except ModelError as error:
            call["error"] = {"type": "ModelError", "message": error.user_message}
            raise
        except Exception as error:
            call["error"] = {"type": type(error).__name__}
            raise
        finally:
            call.update(deepcopy(getattr(self.client, "observation", {})))
            call["duration_ms"] = round((time.monotonic() - started) * 1000)
            call["finished_at"] = timestamp()
            self.persist()

    async def complete(self, *_args, **_kwargs):
        raise RuntimeError("Final generation is forbidden in the routing-only experiment.")


def protocol(settings, dataset, cases, catalog, *, temperature=TEMPERATURE, models=(MODEL,)):
    if (
        not models
        or any(
            not isinstance(model, str)
            or not model.endswith(":free")
            or not model.removesuffix(":free")
            or any(character.isspace() for character in model)
            for model in models
        )
        or settings.openrouter_model not in models
        or settings.llm_tool_mode != "structured"
    ):
        raise ValueError("Use the selected free model and structured routing, without fallback.")
    if (
        type(temperature) not in (int, float)
        or not math.isfinite(temperature)
        or not 0 <= temperature <= 1
    ):
        raise ValueError("Temperature must be a finite number between zero and one.")
    if settings.schedule_timezone != dataset["evaluation_context"]["timezone"]:
        raise ValueError("Schedule timezone must match the fixed evaluation timezone.")
    return {
        "version": 1,
        "dataset_sha256": DATASET_SHA256,
        "evaluation_context": dataset["evaluation_context"],
        "case_ids": [case["id"] for case in cases],
        "model": settings.openrouter_model,
        "tool_mode": settings.llm_tool_mode,
        "mode": "study",
        "temperature": temperature,
        "max_output_tokens": settings.llm_max_output_tokens,
        "llm_timeout_seconds": settings.llm_timeout_seconds,
        "agent_timeout_seconds": settings.agent_timeout_seconds,
        "history_max_messages": settings.history_max_messages,
        "history_max_tokens": settings.history_max_tokens,
        "history": "fresh empty per case; trusted saved timezone from evaluation_context",
        "reasoning": {"enabled": False},
        "provider": {"max_price": {"prompt": 0, "completion": 0}},
        "catalog": {"tools": catalog.tools, "resources": catalog.resources},
        "catalog_sha256": digest({"tools": catalog.tools, "resources": catalog.resources}),
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "scripts/lab2_experiment.py",
                "app/dialog.py",
                "app/agent.py",
                "app/llm.py",
                "app/prompts.py",
                "app/reminder_intent.py",
                "app/mcp/contracts.py",
            )
        },
        "study_prompt_sha256": digest(PROMPTS["study"]),
        "scoring": SCORING,
        "boundaries": {
            "llm": "real OpenRouter client when invoked through CLI",
            "mcp": "real stdio discovery; stop before every application call",
            "storage": "isolated memory only; no PostgreSQL connection",
            "reminders": "validated proposal only; no confirmation or capability",
            "telegram": "no API calls",
            "expected_answers_in_model_context": False,
            "automatic_retries": False,
        },
    }


async def run_trials(
    settings,
    client,
    catalog,
    output,
    *,
    case_ids=None,
    max_api_calls=12,
    temperature=TEMPERATURE,
    models=(MODEL,),
    max_new_cases=None,
):
    if max_api_calls < 0 or max_api_calls > 12:
        raise ValueError("The API budget must be between zero and twelve calls.")
    if max_new_cases is not None and (type(max_new_cases) is not int or max_new_cases <= 0):
        raise ValueError("The new-case limit must be a positive integer or None.")
    dataset = load_dataset()
    cases = selected_cases(dataset, case_ids)
    expected_protocol = protocol(
        settings, dataset, cases, catalog, temperature=temperature, models=models
    )
    if output.exists():
        document = json.loads(output.read_text(encoding="utf-8"))
        if document.get("protocol") != expected_protocol:
            raise ValueError("Existing result uses a different protocol; choose a new output.")
    else:
        document = {"created_at": timestamp(), "protocol": expected_protocol, "attempts": []}
    rows = document["attempts"]
    known = {case["id"] for case in cases}
    identifiers = [row["id"] for row in rows]
    if len(identifiers) != len(set(identifiers)) or set(identifiers) - known:
        raise ValueError("Checkpoint contains duplicate or unexpected case IDs.")
    # An interrupted API request may already have consumed quota: retain it, never replay it.
    for row in rows:
        if row["status"] == "in_progress":
            row.update(status="failed", error={"type": "InterruptedAttempt"})
    fixed_now = datetime.fromisoformat(dataset["evaluation_context"]["now"])
    zone = dataset["evaluation_context"]["timezone"]
    user = replace(USER, temperature=temperature)
    new_cases = 0
    for case in cases:
        if case["id"] in identifiers:
            continue
        if max_new_cases is not None and new_cases >= max_new_cases:
            break
        consumed = sum(len(row.get("calls", [])) for row in rows)
        row = {
            "id": case["id"],
            "query": case["query"],
            "expected_action": case["expected_action"],
            "expected_arguments": case["expected_arguments"],
            "started_at": timestamp(),
            "status": "in_progress",
            "calls": [],
            "actual": {"action": None, "arguments": None},
        }
        rows.append(row)
        new_cases += 1
        checkpoint(output, document)
        repository = MemoryRepository(zone, user=user)
        measured = TrialLLM(
            client,
            row,
            permit=consumed < max_api_calls,
            persist=lambda: checkpoint(output, document),
        )
        agent = ObservedAgent(measured, catalog, repository, settings, now=lambda: fixed_now)
        dialog = DialogService(repository, measured, settings, agent=agent)
        try:
            # Only the user's synthetic query reaches the production entry point;
            # expected_action/expected_arguments are used below, after generation.
            answer = await dialog.respond(user.user_id, 1, case["query"])
            proposal = getattr(getattr(agent, "reply", None), "reminder", None)
            if proposal is not None:
                actual = {
                    "action": "add_reminder",
                    "arguments": proposal.arguments.model_dump(mode="json"),
                    "timezone": proposal.timezone,
                }
            else:
                action = (repository.audit or {}).get("decision", "respond")
                actual = {"action": action, "arguments": {}}
            row.update(status="ok", actual=actual, displayed_answer=answer)
        except ReadBoundary as selected:
            row.update(
                status="captured_before_mcp",
                actual={"action": selected.action, "arguments": selected.arguments},
            )
        except ModelError as error:
            row.update(status="failed", error={"type": "ModelError", "message": error.user_message})
        except Exception as error:
            row.update(status="failed", error={"type": type(error).__name__})
        row["audit"] = repository.audit
        row["routing_source"] = "llm" if row["calls"] else "local_validation"
        row["score"] = score(case, row["actual"])
        row["finished_at"] = timestamp()
        document["metrics"] = metrics(dataset, rows, cases)
        document["updated_at"] = timestamp()
        checkpoint(output, document)
        print(
            f"{case['id']}: {row['status']}; action={row['actual']['action']}; "
            f"LLM={len(row['calls'])}",
            flush=True,
        )
    document["metrics"] = metrics(dataset, rows, cases)
    document["updated_at"] = timestamp()
    checkpoint(output, document)
    return document


async def main(args):
    settings = Settings.load(args.env_file, require_llm=True)
    # A personal local schedule or machine timezone must not affect this experiment.
    settings = replace(
        settings,
        schedule_file=str(ROOT / "data/schedule.json"),
        schedule_timezone="Europe/Moscow",
    )
    configure_logging(settings)
    catalog_source = MCPClient(
        replace(settings, postgres_password="", bot_token="", openrouter_api_key="")
    )
    client = RecordingClient(settings)
    try:
        # No tool/resource operation is performed. The server connects to PostgreSQL
        # lazily only on add_reminder, which is unreachable here.
        await catalog_source.start()
        if not catalog_source.available:
            raise ValueError("MCP discovery is unavailable; no LLM quota was used.")
        catalog = CatalogOnlyMCP(catalog_source.tools, catalog_source.resources)
        await catalog_source.close()
        await client.start()
        report = await run_trials(
            settings,
            client,
            catalog,
            args.output,
            case_ids=args.case,
            max_api_calls=args.max_api_calls,
        )
    finally:
        await client.close()
        await catalog_source.close()
    summary = report["metrics"]
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if summary["scope"] == "selected_cases_only":
        return 0 if all(row["status"] != "failed" for row in report["attempts"]) else 1
    return 0 if summary["acceptance"] == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case", action="append", help="Repeat selected case(s) into a new report."
    )
    parser.add_argument("--max-api-calls", type=int, default=12)
    try:
        raise SystemExit(asyncio.run(main(parser.parse_args())))
    except (ConfigError, ValueError) as error:
        print(f"Experiment refused: {error}", flush=True)
        raise SystemExit(2) from None
