"""New-session regression for place specificity after the teacher's feedback.

Focused network-recovery check: eight weather calls, no new semantic calls.
All41startedattempts remain counted; this task has a maximum49attempts.
Uses real model/MCP/isolated PostgreSQL; Telegram transport is substituted.
Prepare/commit fresh manifests before API. Never overwrite or replay attempts.
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
from app.mcp.contracts import WeatherResult
from app.migration import migrate
from app.repository import DialogRepository
from scripts import lab2_adversarial_live as base
from scripts.lab2_experiment import checkpoint, digest, timestamp

ROOT = base.ROOT
SMOKE_FOLDER = ROOT / "docs/evaluation/semantic-routing-smoke-20261101-v4"
MAXIMUM, COHORT_MAXIMUM, PRIOR_CALLS, SESSION_MAXIMUM = 8, 8, 35, 49
PRIOR_SMOKE = ROOT / "docs/evaluation/semantic-routing-smoke-20261101"
PRIOR_WEATHER = ROOT / "docs/evaluation/weather-specificity-20261101"
CASES = (
    {
        "id": "rostov_followup",
        "maximum_llm_calls": 6,
        "steps": (
            {"query": "Какая сейчас погода в Ростове?", "city": "Ростов", "status": "ambiguous"},
            {"query": "В России", "city": "Ростов, Россия", "status": "ambiguous"},
            {
                "query": "В Ростове-на-Дону",
                "city": "Ростов-на-Дону",
                "allowed_cities": ["Ростов-на-Дону", "Ростов-на-Дону, Россия"],
                "status": "ok",
            },
        ),
    },
    {
        "id": "english_rostov",
        "maximum_llm_calls": 2,
        "steps": (
            {"query": "What's the weather in Rostov?", "city": "Ростов", "status": "ambiguous"},
        ),
    },
    {
        "id": "explicit_full_name",
        "maximum_llm_calls": 2,
        "steps": ({"query": "Погода в Ростове-на-Дону", "city": "Ростов-на-Дону", "status": "ok"},),
    },
)
CASES = CASES[:2]  # Before preregistration: only the two ambiguity/clarification flows.
PLAN = {
    "version": 4,
    "session": "teacher-feedback-20261101",
    "cases": CASES,
    "maximum_llm_calls": MAXIMUM,
    "related_unchanged_smoke_maximum_llm_calls": 6,
    "maximum_session_llm_calls": SESSION_MAXIMUM,
    "previous_started_llm_calls": PRIOR_CALLS,
    "maximum_mcp_calls_per_message": 1,
    "maximum_llm_calls_per_message": 2,
    "automatic_retries": False,
    "acceptance": (
        "Both flows and four messages pass city arguments and actual MCP status; "
        "failed/unstarted flows stay in denominator two."
    ),
    "boundaries": deepcopy(base.PLAN["boundaries"]),
}


def related_smoke():
    path = SMOKE_FOLDER / "manifest.json"
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def previous_runs():
    folders = (
        (PRIOR_SMOKE, "calls"),
        (PRIOR_WEATHER, "llm_calls"),
        (ROOT / "docs/evaluation/semantic-routing-smoke-20261101-v3", "calls"),
        (ROOT / "docs/evaluation/weather-specificity-20261101-v2", "llm_calls"),
        (ROOT / "docs/evaluation/weather-specificity-20261101-v3", "llm_calls"),
    )
    count, files = 0, {}
    for folder, key in folders:
        result = json.loads((folder / "results.json").read_text())
        if any(row["status"] == "in_progress" for row in result["attempts"]):
            raise ValueError("Previous round remains unresolved.")
        count += sum(len(row[key]) for row in result["attempts"])
        for path in (folder / "manifest.json", folder / "results.json"):
            files[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if count != PRIOR_CALLS:
        raise ValueError("Previous rounds' consumed slots changed.")
    return {"started_llm_calls": count, "files_sha256": files}


def smoke_calls():
    path = SMOKE_FOLDER / "results.json"
    if not path.exists():
        return 0
    result = json.loads(path.read_text())
    count = sum(len(row["calls"]) for row in result["attempts"])
    if count > 6:
        raise ValueError("The related smoke exceeded its six-slot ledger.")
    return count


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
    if type(quota.get("remaining")) is not int or quota["remaining"] < COHORT_MAXIMUM:
        raise ValueError("Eight remaining requests are required for this focused check.")
    previous = related_smoke()
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
                "related_smoke": previous,
                "previous_round": previous_runs(),
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
        digest(manifest["plan"]) != digest(PLAN)
        or protocol["source_sha256"] != sources()
        or protocol["related_smoke"] != related_smoke()
        or protocol["previous_round"] != previous_runs()
        or protocol["model"] != settings.openrouter_model
        or protocol["tool_mode"] != settings.llm_tool_mode
        or protocol["api_base"] != settings.openrouter_base_url
    ):
        raise ValueError("Frozen verification plan, source, model or ledger changed.")
    base.assert_committed(path)
    for name, expected in {
        **protocol["source_sha256"],
        **protocol["previous_round"]["files_sha256"],
        protocol["related_smoke"]["path"]: protocol["related_smoke"]["sha256"],
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
        if used >= MAXIMUM or PRIOR_CALLS + smoke_calls() + used >= SESSION_MAXIMUM:
            raise ModelError("Weather verification/session budget rejected before HTTP.")
        return await super()._call(phase, messages, temperature, kwargs)


def summarize(document):
    rows = document["attempts"]
    calls = [call for row in rows for call in row["llm_calls"]]
    passed = (
        len(rows) == len(CASES)
        and all(row["status"] == "passed" for row in rows)
        and document["status"] == "completed"
    )
    costs = [call.get("usage", {}).get("cost") for call in calls]
    document["metrics"] = {
        "cases": {"passed": sum(row["status"] == "passed" for row in rows), "total": len(CASES)},
        "acceptance": "passed" if passed else "not_passed",
        "llm_calls": len(calls),
        "session_started_llm_calls": PRIOR_CALLS + smoke_calls() + len(calls),
        "mcp_calls": sum(len(row["mcp_calls"]) for row in rows),
        "api_cost": sum(costs) if costs and all(value is not None for value in costs) else None,
        "real_telegram_http_calls": 0,
    }


async def execute_case(runtime, case, owner):
    row = runtime.llm.row
    await runtime.dialog.repository.user(owner)
    await runtime.dialog.repository.set_temperature(owner, base.TEMPERATURE)
    await runtime.dialog.confirmations.repository.set_timezone(owner, base.ZONE)
    for index, expected in enumerate(case["steps"]):
        replies = await runtime.step(owner, expected["query"], maximum_llm_calls=2)
        calls = row["llm_calls"][2 * index :]
        base.require(
            len(calls) == 2 and all(call["status"] == "ok" for call in calls),
            "Read did not finish two successful model phases",
        )
        base.require(len(row["mcp_calls"]) == index + 1, "Wrong application MCP count")
        call = row["mcp_calls"][-1]
        base.require(
            call["target"] == "get_weather"
            and any(
                base.arguments_equal("get_weather", call["arguments"], {"city": city})
                for city in expected.get("allowed_cities", [expected["city"]])
            ),
            "City was expanded or received unprovided country/region",
        )
        base.require(call["status"] == "ok", "MCP did not return a validated result")
        observation = WeatherResult.model_validate(call["result"]).root
        base.require(
            observation.status == expected["status"],
            "Expected ambiguity/selected place was not returned",
        )
        base.require(
            calls[-1]["disable_tools"] is True and calls[-1]["wire"].get("tool_choice") == "none",
            "Final allowed another tool",
        )
        displayed = "\n".join(reply.text for reply in replies)
        base.require(bool(displayed) and "Open-Meteo" in displayed, "No final/source displayed")
        if observation.status == "ambiguous":
            base.require(
                "уточни" in displayed.casefold(), "Ambiguous request did not ask for selection"
            )
            names = {place.name for place in observation.candidates}
            base.require(
                {"Ростов-на-Дону", "Ростов Великий"} <= names, "Russian homonyms disappeared"
            )
            for place in observation.candidates:
                base.require(
                    place.name in displayed and place.country in displayed,
                    "Candidate labels were changed/lost",
                )
        else:
            base.require(
                observation.place.name == "Ростов-на-Дону"
                and observation.place.country == "Россия",
                "Selected observation belongs to another city",
            )
            number = str(observation.temperature_c).removesuffix(".0")
            base.require(
                number in displayed or number.replace(".", ",") in displayed, "Temperature changed"
            )
        base.require(
            await runtime.current_card(owner) is None and await runtime.reminder_count(owner) == 0,
            "Weather request prepared or saved a reminder",
        )


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
        for index, case in enumerate(CASES):
            row = {
                "id": case["id"],
                "status": "in_progress",
                "maximum_llm_calls": case["maximum_llm_calls"],
                "started_at": timestamp(),
                "llm_calls": [],
                "mcp_calls": [],
                "steps": [],
            }
            document["attempts"].append(row)
            llm.row = row
            persist()
            try:
                await execute_case(runtime, case, owner=51001 + index)
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
        print(
            "Place-specificity regression preregistered; "
            "commit exact sources and both manifests before --run."
        )
    else:
        print(json.dumps(asyncio.run(run(settings, args.folder))["metrics"], ensure_ascii=False))


if __name__ == "__main__":
    main()
