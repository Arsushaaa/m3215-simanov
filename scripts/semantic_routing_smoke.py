"""Preregister six semantic-routing checks; no Telegram, SQL or weather calls.

Prepare and commit the exact manifest before --run. Each case permits one real
LLM decision. Interrupted/failed attempts remain failures and are never retried.
Only synthetic visible outputs and whitelisted usage metadata are persisted.
"""

import argparse
import asyncio
import fcntl
import hashlib
import json
import subprocess
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from uuid import UUID

import aiohttp

from app.agent import parse_decision
from app.config import Settings
from app.confirmation_models import ConfirmationError, PreparedAction, format_card
from app.dialog import DialogService
from app.llm import ModelError
from app.mcp.client import MCPClient
from app.mcp.contracts import ReminderArguments
from app.repository import UserSettings
from scripts.lab2_experiment import (
    ROOT,
    CatalogOnlyMCP,
    MemoryRepository,
    ObservedAgent,
    ReadBoundary,
    RecordingClient,
    TrialLLM,
    arguments_equal,
    checkpoint,
    digest,
    timestamp,
)

NOW = datetime.fromisoformat("2026-10-26T10:00:00+03:00")
ZONE = "Europe/Moscow"
TEMPERATURE = 0.2
CASE_LIMIT = 6
_PIE = {"text": "Приготовить пирог", "remind_at": "2026-10-27T11:00:00+03:00"}
CASES = (
    {
        "id": "natural_reminder",
        "query": "не забудь мне напомнить в 11:00 завтра приготовить пирог",
        "expected_action": "add_reminder",
        "expected_arguments": dict(_PIE),
    },
    {
        "id": "negated_reminder",
        "query": "Не напоминай мне про пирог. Просто объясни, как его приготовить.",
        "expected_action": "respond",
        "expected_arguments": {},
    },
    {
        "id": "english_weather",
        "query": "What's the weather in Kazan, Russia?",
        "expected_action": "get_weather",
        "expected_arguments": {"city": "Казань, Россия"},
    },
    {
        "id": "unspecified_clock",
        "query": "Не забудь завтра вечером напомнить про пирог.",
        "expected_action": "clarify_reminder",
        "expected_arguments": {},
    },
    {
        "id": "reordered_reminder",
        "query": "Завтра к одиннадцати хочу приготовить пирог — напомни об этом.",
        "expected_action": "add_reminder",
        "expected_arguments": dict(_PIE),
    },
    {
        "id": "relative_card_edit",
        "query": "Нет, лучше на два часа позже",
        "pending_card": {"text": "Приготовить пирог", "remind_at": "2026-10-27T19:00:00+03:00"},
        "expected_action": "add_reminder",
        "expected_arguments": {
            "text": "Приготовить пирог",
            "remind_at": "2026-10-27T21:00:00+03:00",
        },
    },
)
PLAN = {
    "version": 1,
    "now": NOW.isoformat(),
    "timezone": ZONE,
    "temperature": TEMPERATURE,
    "tool_mode": "structured",
    "cases": CASES,
    "limits": {
        "llm_per_case": 1,
        "maximum_llm_calls": CASE_LIMIT,
        "executed_mcp_calls": 0,
        "sql_writes": 0,
        "telegram_calls": 0,
    },
    "acceptance": "6/6 actions and all four exact/normalized argument sets; all six cases pass",
    "normalization": (
        "casefold, ё→е, collapsed whitespace, terminal .!?; aware datetime equal instant"
    ),
    "automatic_retries": False,
}


def fixed_settings(settings):
    model = settings.openrouter_model
    if (
        not isinstance(model, str)
        or not model.endswith(":free")
        or not model.removesuffix(":free")
        or any(c.isspace() for c in model)
    ):
        raise ValueError("An explicitly configured free model is required; no fallback is used.")
    return replace(
        settings,
        llm_tool_mode="structured",
        schedule_timezone=ZONE,
        schedule_file=str(ROOT / "data/schedule.json"),
        llm_max_output_tokens=2048,
        llm_timeout_seconds=60,
        agent_timeout_seconds=135,
        history_max_tokens=6000,
        history_max_messages=20,
    )


def source_snapshot():
    paths = sorted(p for p in (ROOT / "app").rglob("*") if p.suffix in {".py", ".sql"})
    paths += [
        ROOT / "scripts/semantic_routing_smoke.py",
        ROOT / "scripts/lab2_experiment.py",
        ROOT / "data/schedule.json",
        ROOT / "requirements.txt",
        ROOT / "constraints.txt",
    ]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def protocol(settings, catalog):
    settings = fixed_settings(settings)
    return {
        "version": 1,
        "plan_sha256": digest(PLAN),
        "model": settings.openrouter_model,
        "temperature": TEMPERATURE,
        "tool_mode": "structured",
        "max_output_tokens": 2048,
        "llm_timeout_seconds": 60,
        "agent_timeout_seconds": 135,
        "history_max_tokens": 6000,
        "history_max_messages": 20,
        "provider_max_price": {"prompt": 0, "completion": 0},
        "reasoning": {"enabled": False},
        "api_base": settings.openrouter_base_url,
        "source_sha256": source_snapshot(),
        "catalog_sha256": digest({"tools": catalog.tools, "resources": catalog.resources}),
    }


async def quota_status(settings):
    """Read only bounded quota metadata. Never persist the response/key label."""
    timeout = aiohttp.ClientTimeout(total=20, sock_connect=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(
            settings.openrouter_base_url + "/key",
            headers={"Authorization": "Bearer " + settings.openrouter_api_key},
            allow_redirects=False,
        ) as response:
            if response.status != 200:
                raise ValueError("OpenRouter quota metadata is unavailable.")
            raw = bytearray()
            async for chunk in response.content.iter_chunked(4096):
                raw.extend(chunk)
                if len(raw) > 64 * 1024:
                    raise ValueError("Quota metadata exceeds the allowed size.")
            data = json.loads(raw)
    values = data.get("data", {}).get("free_model_daily_requests", {})
    return {
        k: v for k in ("limit", "used", "remaining") if type(v := values.get(k)) is int and v >= 0
    }


async def discover(settings):
    source = MCPClient(replace(settings, bot_token="", openrouter_api_key="", postgres_password=""))
    try:
        await source.start()
        if not source.available:
            raise ValueError("MCP discovery unavailable; no model call was made.")
        return CatalogOnlyMCP(source.tools, source.resources)
    finally:
        await source.close()


async def prepare(settings, folder, *, catalog=None, quota=None):
    settings, folder = fixed_settings(settings), Path(folder)
    target = folder / "manifest.json"
    if target.exists():
        raise ValueError("Never overwrite a preregistration; use a new folder.")
    quota = await quota_status(settings) if quota is None else quota
    if type(quota.get("remaining")) is not int or quota["remaining"] < CASE_LIMIT:
        raise ValueError("This preregistration requires six remaining free requests.")
    catalog = await discover(settings) if catalog is None else catalog
    manifest = {
        "registered_at": timestamp(),
        "protocol": protocol(settings, catalog),
        "plan": deepcopy(PLAN),
        "quota": {k: v for k, v in quota.items() if k in {"limit", "used", "remaining"}},
    }
    folder.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    return manifest


def assert_committed(path):
    try:
        relative = path.resolve().relative_to(ROOT)
    except ValueError:
        raise ValueError("The preregistration must be inside the repository.") from None
    result = subprocess.run(
        ["git", "show", f"HEAD:{relative}"], cwd=ROOT, capture_output=True, check=False
    )
    if result.returncode or result.stdout != path.read_bytes():
        raise ValueError("Commit the exact preregistration before --run.")


@contextmanager
def exclusive_run(folder):
    with (folder / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another process owns this smoke run.") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


class MemoryConfirmations:
    """Trusted synthetic pending state; no capability, confirmation or database."""

    def __init__(self, repository, case):
        self.repository, self.epoch, self.revoked = repository, 1, False
        self.pending = None
        if "pending_card" in case:
            arguments = ReminderArguments.model_validate(case["pending_card"])
            self.pending = PreparedAction(
                UUID("00000000-0000-4000-8000-000000000006"),
                format_card(arguments, ZONE, NOW + timedelta(minutes=5)),
                arguments,
                ZONE,
                NOW + timedelta(minutes=5),
                "pending",
            )

    async def begin_semantic_request(self, _user, _message_id):
        return self.pending, self.epoch

    async def begin_edit(self, user, _message_id, *, expected_epoch=None):
        if expected_epoch != self.epoch:
            raise ConfirmationError("ACTION_CANCELLED", "Подтверждение отозвано.")
        source, self.pending = self.pending, None
        if source is None:
            return None
        self.revoked, self.epoch = True, self.epoch + 1
        self.repository._user = replace(user, revision=user.revision + 1)
        return source, self.repository._user, self.epoch

    async def prepare(self, user, message_id, question, proposal, *, expected_epoch=None):
        if expected_epoch != self.epoch or user.revision != self.repository._user.revision:
            raise ConfirmationError("ACTION_CANCELLED", "Подтверждение отозвано.")
        if self.pending is not None:
            raise AssertionError("The former pending card was not revoked.")
        card = PreparedAction(
            UUID("00000000-0000-4000-8000-000000000007"),
            format_card(proposal.arguments, ZONE, NOW + timedelta(minutes=5)),
            proposal.arguments,
            ZONE,
            NOW + timedelta(minutes=5),
            "pending",
        )
        await self.repository.save_exchange(user, message_id, question, card.card_text)
        self.pending = card
        return card


def score_case(case, row):
    actual = row["actual"]
    action = actual["action"] == case["expected_action"]
    required = bool(case["expected_arguments"])
    arguments = (
        (
            action
            and arguments_equal(
                case["expected_action"], actual["arguments"], case["expected_arguments"]
            )
        )
        if required
        else None
    )
    one_call = len(row["calls"]) == 1
    revoked = "pending_card" not in case or row.get("former_card_revoked") is True
    return {
        "action_correct": action,
        "arguments_required": required,
        "arguments_correct": arguments,
        "one_llm_call": one_call,
        "pending_replacement_safe": revoked,
        "passed": action
        and (arguments is not False)
        and one_call
        and revoked
        and row["status"] in {"ok", "captured_before_mcp"},
    }


def summarize(document):
    rows = document["attempts"]
    scores = [row["score"] for row in rows if "score" in row]
    costs = [call.get("usage", {}).get("cost") for row in rows for call in row["calls"]]
    document["metrics"] = {
        "cases": {"correct": sum(s["passed"] for s in scores), "total": 6},
        "actions": {"correct": sum(s["action_correct"] for s in scores), "total": 6},
        "arguments": {"correct": sum(s["arguments_correct"] is True for s in scores), "total": 4},
        "llm_calls": sum(len(row["calls"]) for row in rows),
        "executed_mcp_calls": 0,
        "database_writes": 0,
        "telegram_calls": 0,
        "api_cost": sum(costs) if costs and all(cost is not None for cost in costs) else None,
        "acceptance": "passed"
        if len(scores) == 6 and all(s["passed"] for s in scores)
        else "not_passed",
    }


async def run(settings, folder, *, catalog=None, client=None):
    settings, folder = fixed_settings(settings), Path(folder)
    manifest_path, output = folder / "manifest.json", folder / "results.json"
    manifest = json.loads(manifest_path.read_text())
    if digest(manifest.get("plan")) != digest(PLAN):
        raise ValueError("Frozen smoke plan changed; preserve this run and register a new folder.")
    # Reject changed sources before discovery, client startup or any model dispatch.
    saved = manifest.get("protocol", {})
    if saved.get("source_sha256") != source_snapshot():
        raise ValueError("Frozen smoke sources changed after preregistration.")
    assert_committed(manifest_path)
    catalog = await discover(settings) if catalog is None else catalog
    if saved != protocol(settings, catalog):
        raise ValueError("Frozen smoke settings/catalog changed after preregistration.")
    with exclusive_run(folder):
        document = (
            json.loads(output.read_text())
            if output.exists()
            else {"created_at": timestamp(), "manifest_sha256": digest(manifest), "attempts": []}
        )
        if document.get("manifest_sha256") != digest(manifest):
            raise ValueError("Existing smoke results belong to another preregistration.")
        ids = [row["id"] for row in document["attempts"]]
        if len(ids) != len(set(ids)) or not set(ids) <= {case["id"] for case in CASES}:
            raise ValueError("Duplicate or unknown smoke attempt IDs.")
        for row in document["attempts"]:
            if row["status"] == "in_progress":
                row.update(status="failed", error={"type": "InterruptedAttempt"})
                row["score"] = score_case(next(c for c in CASES if c["id"] == row["id"]), row)
        checkpoint(output, document)
        pending = [case for case in CASES if case["id"] not in ids]
        if not pending:
            summarize(document)
            checkpoint(output, document)
            return document
        client = client if client is not None else RecordingClient(settings)
        await client.start()
        try:
            for case in pending:
                row = {
                    "id": case["id"],
                    "query": case["query"],
                    "started_at": timestamp(),
                    "status": "in_progress",
                    "calls": [],
                    "actual": {"action": None, "arguments": {}},
                }
                document["attempts"].append(row)
                checkpoint(output, document)
                repo = MemoryRepository(ZONE, UserSettings(1, "study", TEMPERATURE, 0))
                confirmations = MemoryConfirmations(repo, case)
                measured = TrialLLM(
                    client,
                    row,
                    permit=sum(len(r["calls"]) for r in document["attempts"]) < CASE_LIMIT,
                    persist=lambda: checkpoint(output, document),
                )
                agent = ObservedAgent(measured, catalog, repo, settings, now=lambda: NOW)
                dialog = DialogService(
                    repo, measured, settings, agent=agent, confirmations=confirmations
                )
                try:
                    row["displayed_answer"] = await dialog.respond(1, 1, case["query"])
                    proposal = getattr(getattr(agent, "reply", None), "reminder", None)
                    raw_action = parse_decision(row["calls"][0]["raw_content"]).action
                    action = (
                        "clarify_reminder"
                        if raw_action == "clarify_reminder"
                        else (repo.audit or {}).get("decision")
                    )
                    row.update(
                        status="ok",
                        actual={
                            "action": "add_reminder" if proposal else action,
                            "arguments": proposal.arguments.model_dump(mode="json")
                            if proposal
                            else {},
                        },
                    )
                except ReadBoundary as boundary:
                    row.update(
                        status="captured_before_mcp",
                        actual={"action": boundary.action, "arguments": boundary.arguments},
                    )
                except BaseException as error:
                    row.update(status="failed", error={"type": type(error).__name__})
                    if isinstance(error, ModelError):
                        row["error"]["message"] = error.user_message
                    if not isinstance(error, Exception):
                        raise
                finally:
                    row.update(
                        audit=repo.audit,
                        former_card_revoked=confirmations.revoked,
                        finished_at=timestamp(),
                    )
                    row["score"] = score_case(case, row)
                    summarize(document)
                    checkpoint(output, document)
        finally:
            await client.close()
        document["finished_at"] = timestamp()
        checkpoint(output, document)
        return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--folder", type=Path, required=True)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--prepare", action="store_true")
    operation.add_argument("--run", action="store_true")
    args = parser.parse_args()
    settings = Settings.load(args.env_file, require_llm=True)
    if args.prepare:
        asyncio.run(prepare(settings, args.folder))
        print("Six-case preregistration saved. Commit manifest.json before --run.")
    else:
        result = asyncio.run(run(settings, args.folder))
        print(json.dumps(result["metrics"], ensure_ascii=False))


if __name__ == "__main__":
    main()
