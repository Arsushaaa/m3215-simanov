"""Four new live LLM attempts verify the repaired weather clarification flow.

This is a separate preregistered regression, not a rescore/replay of either
previous ten-flow report. Uses its unchanged weather case and isolated runtime.
The first focused run failed after one real model call because Open-Meteo was
unreachable. The second used four calls and exposed changed place labels and a
lost country qualifier. Both failures remain part of the frozen session ledger.
"""

import argparse
import asyncio
import hashlib
import json
import subprocess
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from aiogram import Bot, Dispatcher

from app.agent import AgentService
from app.agent_repository import AgentRepository, ConfirmationRepository
from app.config import Settings
from app.confirmations import ConfirmationService
from app.db import create_pool
from app.dialog import DialogService
from app.handlers.assistant import create_router
from app.llm import ModelError
from app.mcp.client import MCPClient
from app.migration import migrate
from app.repository import DialogRepository
from scripts import lab2_adversarial_live as base
from scripts.lab2_experiment import checkpoint, digest, timestamp

ROOT = base.ROOT
FOLDERS = (
    (ROOT / "docs/evaluation/lab2-adversarial-live-20261028", 15),
    (ROOT / "docs/evaluation/lab2-adversarial-live-20261028-v2", 17),
    (ROOT / "docs/evaluation/lab2-weather-verify-20261028", 1),
    (ROOT / "docs/evaluation/lab2-weather-verify-20261028-v2", 4),
)
CASE = deepcopy(next(case for case in base.CASES if case["id"] == "weather_read"))
BASELINE_CALLS = 37
MAXIMUM = 4
PLAN = {
    "version": 3,
    "case": CASE,
    "queries": {name: base.QUERIES[name] for name in ("weather_read", "weather_refinement")},
    "maximum_llm_calls": MAXIMUM,
    "maximum_mcp_calls": 2,
    "maximum_session_llm_calls": 50,
    "previous_started_llm_calls": BASELINE_CALLS,
    "automatic_retries": False,
    "acceptance": (
        "The unchanged v2 weather flow passes clarification, argument and final-fact checks."
    ),
    "boundaries": deepcopy(base.PLAN["boundaries"]),
}


def baseline():
    files, total = {}, 0
    for folder, expected in FOLDERS:
        result_path = folder / "results.json"
        result = json.loads(result_path.read_text())
        started = sum(len(row["llm_calls"]) for row in result["attempts"])
        if (
            result.get("status") != "completed"
            or started != expected
            or result["metrics"]["llm_calls"] != started
            or any(
                row.get("status") == "in_progress"
                or any(call.get("status") not in {"ok", "failed"} for call in row["llm_calls"])
                for row in result["attempts"]
            )
        ):
            raise ValueError("Previous attempt ledger is inconsistent or unfinished.")
        total += started
        for path in (folder / "manifest.json", result_path):
            files[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if total != BASELINE_CALLS:
        raise ValueError("Previous attempts changed; freeze a new protocol before continuing.")
    return {"started_llm_calls": total, "files_sha256": files}


def sources():
    return {
        **base.source_snapshot(),
        str(Path(__file__).relative_to(ROOT)): hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }


async def prepare(settings, folder, *, quota=None, catalog=None):
    settings, folder = base.fixed_settings(settings), Path(folder)
    target = folder / "manifest.json"
    if target.exists():
        raise ValueError("Never overwrite a verification manifest.")
    quota = await base.quota_status(settings) if quota is None else quota
    if type(quota.get("remaining")) is not int or quota["remaining"] < MAXIMUM:
        raise ValueError("Four remaining requests are required.")
    previous = baseline()
    if previous["started_llm_calls"] + MAXIMUM > PLAN["maximum_session_llm_calls"]:
        raise ValueError("The focused verification exceeds the authorized session budget.")
    owned = catalog is None
    source = (
        MCPClient(replace(settings, openrouter_api_key="", postgres_password=""))
        if owned
        else catalog
    )
    try:
        if owned:
            await source.start()
        base.require(source.available, "MCP discovery unavailable")
        document = {
            "registered_at": timestamp(),
            "plan": deepcopy(PLAN),
            "protocol": {
                "now": datetime.now(UTC).replace(microsecond=0).isoformat(),
                "model": settings.openrouter_model,
                "tool_mode": settings.llm_tool_mode,
                "temperature": base.TEMPERATURE,
                "max_output_tokens": settings.llm_max_output_tokens,
                "api_base": settings.openrouter_base_url,
                "history_max_tokens": settings.history_max_tokens,
                "history_max_messages": settings.history_max_messages,
                "llm_timeout_seconds": settings.llm_timeout_seconds,
                "agent_timeout_seconds": settings.agent_timeout_seconds,
                "provider_max_price": {"prompt": 0, "completion": 0},
                "source_sha256": sources(),
                "session_baseline": previous,
                "catalog_sha256": digest({"tools": source.tools, "resources": source.resources}),
            },
            "quota": {key: quota.get(key) for key in ("limit", "used", "remaining")},
        }
        folder.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as output:
            json.dump(document, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
        return document
    finally:
        if owned:
            await source.close()


def validate(settings, folder):
    path = folder / "manifest.json"
    manifest = json.loads(path.read_text())
    protocol = manifest["protocol"]
    if (
        manifest["plan"] != PLAN
        or protocol["source_sha256"] != sources()
        or protocol["session_baseline"] != baseline()
        or protocol["model"] != settings.openrouter_model
        or protocol["tool_mode"] != settings.llm_tool_mode
        or protocol["api_base"] != settings.openrouter_base_url
    ):
        raise ValueError("Frozen verification plan, source, model or ledger changed.")
    base.assert_committed(path)
    for name, expected in {
        **protocol["source_sha256"],
        **protocol["session_baseline"]["files_sha256"],
    }.items():
        raw = subprocess.run(
            ["git", "show", f"HEAD:{name}"], cwd=ROOT, capture_output=True, check=False
        )
        if raw.returncode or hashlib.sha256(raw.stdout).hexdigest() != expected:
            raise ValueError("Commit exact sources and prior reports before verification.")
    return manifest


class WeatherBudget(base.BudgetLLM):
    async def _call(self, phase, messages, temperature, kwargs):
        used = sum(len(row["llm_calls"]) for row in self.document["attempts"])
        if used >= MAXIMUM or BASELINE_CALLS + used >= 50:
            raise ModelError("Weather verification/session budget rejected before HTTP.")
        return await super()._call(phase, messages, temperature, kwargs)


def summarize(document):
    rows = document["attempts"]
    calls = [call for row in rows for call in row["llm_calls"]]
    passed = len(rows) == 1 and rows[0]["status"] == "passed" and document["status"] == "completed"
    costs = [call.get("usage", {}).get("cost") for call in calls]
    document["metrics"] = {
        "cases": {"passed": int(passed), "total": 1},
        "acceptance": "passed" if passed else "not_passed",
        "llm_calls": len(calls),
        "session_started_llm_calls": BASELINE_CALLS + len(calls),
        "mcp_calls": sum(len(row["mcp_calls"]) for row in rows),
        "api_cost": sum(costs) if costs and all(value is not None for value in costs) else None,
        "real_telegram_http_calls": 0,
    }


async def run_isolated(settings, folder, manifest, document):
    def persist():
        checkpoint(folder / "results.json", document)

    now = datetime.fromisoformat(manifest["protocol"]["now"])
    pool = await create_pool(settings)
    source, client = MCPClient(settings), base.AuditClient(settings)
    dispatcher, bot, sink = Dispatcher(), Bot(base.FAKE_TOKEN), base.TelegramSink(now)
    bot.session = sink.session
    dispatcher.include_router(create_router())
    try:
        await migrate(pool)
        await source.start()
        base.require(source.available, "MCP discovery failed")
        base.require(
            digest({"tools": source.tools, "resources": source.resources})
            == manifest["protocol"]["catalog_sha256"],
            "MCP catalog changed",
        )
        await client.start()
        llm = WeatherBudget(client, document, persist)
        mcp = base.AuditMCP(source, llm, persist)
        confirmations = ConfirmationService(ConfirmationRepository(pool), mcp)
        agent = AgentService(llm, mcp, AgentRepository(pool), settings, now=lambda: now)
        dialog = DialogService(
            DialogRepository(pool), llm, settings, agent=agent, confirmations=confirmations
        )
        runtime = base.Runtime(pool, dispatcher, bot, sink, dialog, llm, now, persist)
        row = {
            "id": CASE["id"],
            "status": "in_progress",
            "maximum_llm_calls": MAXIMUM,
            "started_at": timestamp(),
            "llm_calls": [],
            "mcp_calls": [],
            "steps": [],
        }
        document["attempts"].append(row)
        llm.row = row
        persist()
        try:
            await base.execute_case(runtime, CASE, owner=21001)
            row["status"] = "passed"
        except BaseException as error:
            row.update(status="failed", error_type=type(error).__name__)
            if isinstance(error, base.CaseFailure):
                row["failure"] = str(error)
            if not isinstance(error, Exception):
                raise
        finally:
            row["finished_at"] = timestamp()
            persist()
    finally:
        await base.close_resources(
            [
                ("dispatcher", dispatcher.emit_shutdown, None),
                ("model", client.close, None),
                ("mcp", source.close, None),
                ("postgresql", pool.close, pool.terminate),
            ]
        )


async def run(settings, folder):
    settings, folder = base.fixed_settings(settings), Path(folder)
    manifest = validate(settings, folder)
    output = folder / "results.json"
    with base.exclusive_run(folder):
        if output.exists():
            result = json.loads(output.read_text())
            if result.get("manifest_sha256") != digest(manifest):
                raise ValueError("Existing verification belongs to another manifest.")
            return result
        result = {
            "created_at": timestamp(),
            "manifest_sha256": digest(manifest),
            "status": "in_progress",
            "attempts": [],
        }
        checkpoint(output, result)
        try:
            with base.isolated_database(settings) as isolated:
                await run_isolated(isolated, folder, manifest, result)
            result["status"] = "completed"
        except BaseException as error:
            result.update(status="failed", error_type=type(error).__name__)
            if not isinstance(error, Exception):
                raise
        finally:
            summarize(result)
            result["finished_at"] = timestamp()
            checkpoint(output, result)
        return result


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
        print("Weather-only verification preregistered; commit manifest and sources before --run.")
    else:
        print(json.dumps(asyncio.run(run(settings, args.folder))["metrics"], ensure_ascii=False))


if __name__ == "__main__":
    main()
