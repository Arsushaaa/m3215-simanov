"""Freeze once, commit, then run all 15 real Router decisions without tuning cases.

No Telegram, user data, MCP write or paid fallback. Checkpoints never replay a
started attempt. Insufficient quota stops before any generation.
"""

import argparse
import asyncio
import fcntl
import hashlib
import json
import math
import subprocess
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from dotenv import dotenv_values

from app.config import Settings
from app.llm import OpenRouterClient
from app.routing.contracts import AgentRequest
from app.routing.router import Router
from scripts.semantic_routing_smoke import quota_status

ROOT = Path(__file__).resolve().parents[1]
DATASET = "docs/evaluation/lab3-routing.json"
DEFAULT_PROTOCOL = "docs/evaluation/lab3-experiment-v9-protocol.json"
DEFAULT_OUTPUT = "docs/evaluation/lab3-experiment-v9-results.json"
MODEL = "nvidia/nemotron-3.5-super-vl:free"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def source_hashes(root=ROOT):
    paths = [
        *sorted((root / "app").rglob("*.py")),
        *sorted((root / "app/migrations").glob("*.sql")),
        root / "scripts/lab3_experiment.py",
        root / DATASET,
        root / "data/schedule.json",
        *sorted((root / "data/materials").glob("*")),
        root / "requirements.txt",
        root / "constraints.txt",
    ]
    return {p.relative_to(root).as_posix(): digest(p.read_bytes()) for p in paths if p.is_file()}


def freeze(path, root=ROOT, *, key_env="OPENROUTER_API_KEY"):
    if Path(path).exists():
        raise ValueError("Protocol already exists; never overwrite a preregistration")
    dataset = json.loads((root / DATASET).read_text())
    assert len(dataset["cases"]) == 15
    protocol = {
        "kind": "lab3_router_preregistration",
        "credential_env": key_env,
        "created_at": datetime.now(UTC).isoformat(),
        "model": MODEL,
        "temperature": 0.2,
        "tool_mode": "native",
        "history_max_tokens": 6000,
        "llm_max_output_tokens": 2048,
        "llm_timeout_seconds": 60,
        "api_base": "https://openrouter.ai/api/v1",
        "reasoning_enabled": False,
        "max_price": {"prompt": 0, "completion": 0},
        "max_http_attempts": 15,
        "minimum_correct": 12,
        "denominator": 15,
        "dataset": DATASET,
        "dataset_sha256": digest((root / DATASET).read_bytes()),
        "source_sha256": source_hashes(root),
        "scope": (
            "One production Router decision per frozen query. "
            "No specialist/final LLM, MCP or Telegram calls."
        ),
        "evaluation": "Expected target is compared only after route(); never included in messages.",
        "failure_policy": (
            "HTTP errors/timeouts/invalid DTO/started interrupted attempts "
            "count as failures; no retries or replacement model."
        ),
        "correction_policy": (
            "Only after complete baseline: explain one change, "
            "preregister a separate protocol before any new calls, "
            "retain baseline and all 15 denominators."
        ),
        "quota_policy": (
            "Require remaining quota for every not-yet-started case before calls; "
            "leave blocked checkpoint otherwise."
        ),
        "model_catalog": "https://openrouter.ai/nvidia/nemotron-3.5-super-vl:free",
    }
    save(path, protocol)
    return protocol


def validate(protocol, root=ROOT):
    if protocol["source_sha256"] != source_hashes(root):
        raise ValueError("Frozen source changed")
    if (
        protocol["model"] != MODEL
        or protocol["denominator"] != 15
        or protocol["max_http_attempts"] != 15
    ):
        raise ValueError("Invalid fixed experiment")
    if digest((root / protocol["dataset"]).read_bytes()) != protocol["dataset_sha256"]:
        raise ValueError("Dataset changed")


class MeasuredClient(OpenRouterClient):
    def __init__(self, settings):
        super().__init__(settings)
        self.attempts = 0
        self.observation = {}

    async def _request(self, payload):
        self.attempts += 1
        if self.attempts > 15:
            raise ValueError("Experiment API limit")
        self.observation = {"payload_sha256": digest(encoded(payload))}
        return await super()._request(payload)

    def _observe_response(self, data):
        usage = data.get("usage", {})
        usage = usage if isinstance(usage, dict) else {}
        safe = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost"):
            v = usage.get(key)
            if type(v) in (int, float) and math.isfinite(v) and v >= 0:
                safe[key] = v
        self.observation["usage"] = safe
        # Only route output, no system/user input dump or identifying provider metadata.
        try:
            message = data["choices"][0]["message"]
            route = {"content": message.get("content"), "tool_calls": message.get("tool_calls")}
            if len(encoded(route)) <= 8192:
                self.observation["response"] = route
        except (TypeError, KeyError, IndexError):
            pass


def summarize(result, protocol):
    cases = result["cases"]
    correct = sum(c.get("correct") is True for c in cases)
    complete = len(cases) == 15 and all(c["state"] == "finished" for c in cases)
    return {
        "correct": correct,
        "denominator": 15,
        "accuracy": correct / 15,
        "started_attempts": len(cases),
        "http_attempts": sum(c.get("http_attempts", 0) for c in cases),
        "status": "passed"
        if complete and correct >= protocol["minimum_correct"]
        else "not_passed"
        if complete
        else "incomplete",
        "api_cost_reported": sum(
            c.get("observation", {}).get("usage", {}).get("cost", 0) for c in cases
        )
        if cases and all("cost" in c.get("observation", {}).get("usage", {}) for c in cases)
        else None,
    }


async def _run(
    protocol_path,
    output_path,
    *,
    env_file=".env",
    key_env="OPENROUTER_API_KEY",
    client_type=MeasuredClient,
    quota_reader=quota_status,
    require_commit=True,
):
    protocol_path, output_path = Path(protocol_path), Path(output_path)
    protocol = json.loads(protocol_path.read_text())
    validate(protocol)
    if require_commit:
        rel = protocol_path.resolve().relative_to(ROOT).as_posix()
        stored = (
            await asyncio.to_thread(
                subprocess.run,
                ["git", "show", f"HEAD:{rel}"],
                cwd=ROOT,
                capture_output=True,
                check=True,
            )
        ).stdout
        if stored != protocol_path.read_bytes():
            raise ValueError("Commit protocol before real API")
    if key_env != protocol.get("credential_env", "OPENROUTER_API_KEY"):
        raise ValueError("Credential slot differs from preregistration")
    settings = Settings.load(env_file, require_llm=True)
    if key_env != "OPENROUTER_API_KEY":
        key = dotenv_values(env_file, interpolate=False).get(key_env)
        if not key:
            raise ValueError("Registered credential variable is missing")
        settings = replace(settings, openrouter_api_key=key)
    if settings.openrouter_base_url != protocol["api_base"]:
        raise ValueError("Unexpected OpenRouter endpoint")
    settings = replace(
        settings,
        openrouter_model=protocol["model"],
        llm_tool_mode=protocol["tool_mode"],
        history_max_tokens=protocol["history_max_tokens"],
        llm_max_output_tokens=protocol["llm_max_output_tokens"],
        llm_timeout_seconds=protocol["llm_timeout_seconds"],
        schedule_file="data/schedule.json",
    )
    phash = digest(protocol_path.read_bytes())
    result = (
        json.loads(output_path.read_text())
        if output_path.exists()
        else {
            "kind": "lab3_real_router_experiment",
            "protocol_sha256": phash,
            "cases": [],
            "scope": protocol["scope"],
            "created_at": datetime.now(UTC).isoformat(),
        }
    )
    if result["protocol_sha256"] != phash:
        raise ValueError("Checkpoint belongs to another protocol")
    dataset = json.loads((ROOT / DATASET).read_text())
    ids = [c["id"] for c in dataset["cases"]]
    started = [c["id"] for c in result["cases"]]
    if started != ids[: len(started)]:
        raise ValueError("Invalid checkpoint ordering")
    # A crash after recording start must never silently cause a second attempt.
    for entry in result["cases"]:
        if entry["state"] == "started":
            entry.update(state="finished", correct=False, error="INTERRUPTED_OUTCOME_UNKNOWN")
    remaining = 15 - len(result["cases"])
    quota = await quota_reader(settings)
    result["quota_before"] = quota
    if remaining and quota.get("remaining", 0) < remaining:
        result["blocked_reason"] = "INSUFFICIENT_FREE_QUOTA"
        result["summary"] = summarize(result, protocol)
        save(output_path, result)
        return result
    result.pop("blocked_reason", None)
    client = client_type(settings)
    await client.start()
    try:
        for case in dataset["cases"][len(result["cases"]) :]:
            entry = {
                "id": case["id"],
                "query": case["query"],
                "state": "started",
                "started_at": datetime.now(UTC).isoformat(),
            }
            result["cases"].append(entry)
            save(output_path, result)
            client.observation = {}
            before = client.attempts
            start = time.monotonic()
            context = dataset["evaluation_context"]
            request = AgentRequest(
                request_id=UUID(int=1),
                trace_id=UUID(int=2),
                actor_id=101,
                text=case["query"],
                intent="user_query",
                occurred_at=datetime.fromisoformat(context["now"]),
                timezone=context["timezone"],
            )
            try:
                decision = await Router(client, settings).route(
                    request, [], protocol["temperature"]
                )
                entry["actual_target"] = decision.target if decision is not None else None
                entry["confidence"] = decision.confidence if decision is not None else None
                entry["error"] = None if decision is not None else "INVALID_ROUTE"
            except Exception as error:
                entry.update(actual_target=None, confidence=None, error=type(error).__name__)
            entry.update(
                state="finished",
                expected_target=case["expected_target"],
                correct=entry["actual_target"] == case["expected_target"],
                duration_ms=round((time.monotonic() - start) * 1000),
                http_attempts=client.attempts - before,
                observation=client.observation,
            )
            result["summary"] = summarize(result, protocol)
            save(output_path, result)
            print(f"{entry['id']}: {entry['actual_target'] or entry['error']}", flush=True)
    finally:
        await client.close()
    result["quota_after"] = await quota_reader(settings)
    result["summary"] = summarize(result, protocol)
    save(output_path, result)
    return result


async def run(protocol_path, output_path, **kwargs):
    lock_path = ROOT / ".tools/lab3-experiment.lock"
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another experiment is running") from None
        try:
            return await _run(protocol_path, output_path, **kwargs)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument(
        "--key-env",
        choices=("OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2", "OPENROUTER_API_KEY_3"),
        default="OPENROUTER_API_KEY",
        help="Explicit registered credential variable; no automatic key fallback",
    )
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.protocol, key_env=args.key_env)
        print("Protocol frozen. Commit before run.")
        return 0
    result = asyncio.run(
        run(args.protocol, args.output, env_file=args.env_file, key_env=args.key_env)
    )
    print(json.dumps(result["summary"], ensure_ascii=False))
    if result.get("blocked_reason"):
        print(result["blocked_reason"])
    return 0 if result["summary"]["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
