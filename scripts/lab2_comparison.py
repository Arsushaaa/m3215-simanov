"""Preregistered real comparison: frozen inputs, two free models, temperature only.

Prepare/commit the manifest before --run. No production .env is rewritten.
The current protocol needs 54 attempts: all 12 cases now reach the LLM.
Archived 50-attempt manifests belong to the former router and are never reused.
"""

import argparse
import asyncio
import hashlib
import json
import subprocess
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

from app.config import Settings
from app.dialog import DialogService
from app.llm import ModelError
from app.logging_setup import configure_logging
from app.mcp.client import MCPClient
from scripts.lab2_experiment import (
    ROOT,
    USER,
    CatalogOnlyMCP,
    MemoryRepository,
    ObservedAgent,
    ReadBoundary,
    RecordingClient,
    TrialLLM,
    checkpoint,
    digest,
    load_dataset,
    run_trials,
    score,
    timestamp,
)

MODELS = ("xingchen-agi/xing4.0-29b-a4b:free", "dots-studio/dots-3-note-preview:free")
CELLS = (
    {"id": "xing-t03", "model": MODELS[0], "temperature": 0.3},
    {"id": "dots-t03", "model": MODELS[1], "temperature": 0.3},
    {"id": "dots-t00", "model": MODELS[1], "temperature": 0.0},
    {"id": "xing-t00", "model": MODELS[0], "temperature": 0.0},
)
PROTOCOL_VERSION = 2
EXTRA = (
    {
        "id": "new_weather",
        "query": "Какая текущая температура в Лиссабоне, Португалия?",
        "expected_action": "get_weather",
        "expected_arguments": {"city": "Лиссабон, Португалия"},
    },
    {
        "id": "new_schedule",
        "query": "Покажи расписание на послезавтра.",
        "expected_action": "get_schedule",
        "expected_arguments": {"date": "2026-10-14"},
    },
    {
        "id": "new_reminder",
        "query": "Remind me tomorrow at 11:00 to submit the report",
        "expected_action": "add_reminder",
        "expected_arguments": {
            "text": "submit the report",
            "remind_at": "2026-10-13T11:00:00+03:00",
        },
    },
    {
        "id": "new_respond",
        "query": "Для чего нужен оператор break в цикле Python?",
        "expected_action": "respond",
        "expected_arguments": {},
    },
)


def campaign_budget():
    grid = len(CELLS) * len(load_dataset()["cases"])
    extra, live_read = len(EXTRA), 2
    return {
        "grid": grid,
        "extra": extra,
        "live_read": live_read,
        "maximum": grid + extra + live_read,
    }


def sources():
    paths = sorted((ROOT / "app").rglob("*.py")) + [
        ROOT / "scripts" / name
        for name in ("lab2_experiment.py", "lab2_comparison.py", "lab2_comparison_live.py")
    ]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }


def public_json(url, *, key=None):
    headers = {"Authorization": "Bearer " + key} if key else {}
    with urlopen(Request(url, headers=headers), timeout=30) as response:
        return json.load(response)["data"]


def prepare(settings, folder):
    if (folder / "manifest.json").exists():
        raise ValueError("A preregistration already exists; never overwrite it.")
    available = public_json("https://openrouter.ai/api/v1/models")
    selected = []
    for model_id in MODELS:
        model = next(model for model in available if model["id"] == model_id)
        if any(model["pricing"].get(key) != "0" for key in ("prompt", "completion")):
            raise ValueError("A selected model is not free.")
        needed = {"temperature", "max_tokens", "reasoning", "response_format", "structured_outputs"}
        if not needed <= set(model["supported_parameters"]):
            raise ValueError("A selected model lacks the fixed request parameters.")
        endpoints = public_json("https://openrouter.ai/api/v1/models/" + model_id + "/endpoints")
        selected.append({"model": model, "endpoints": endpoints})
    account = public_json(settings.openrouter_base_url + "/key", key=settings.openrouter_api_key)
    quota = account.get("free_model_daily_requests", {})
    budget = campaign_budget()
    if quota.get("remaining", 0) < budget["maximum"]:
        raise ValueError(
            f"The semantic-routing campaign needs {budget['maximum']} remaining free requests; "
            "the archived 50-attempt protocol cannot be reused."
        )
    document = {
        "version": PROTOCOL_VERSION,
        "registered_at": timestamp(),
        "source_sha256": sources(),
        "dataset_sha256": hashlib.sha256(
            (ROOT / "docs/evaluation/lab2-routing.json").read_bytes()
        ).hexdigest(),
        "context": load_dataset()["evaluation_context"],
        "cells": CELLS,
        "order": "case order fixed; rotate cell order by case index; one new case per checkpoint",
        "extra_cases": EXTRA,
        "selection": (
            "passed first, arguments descending, actions descending, "
            "temperature ascending, model ID ascending"
        ),
        "fixed": {
            "max_tokens": 2048,
            "llm_timeout_seconds": 60,
            "agent_timeout_seconds": 135,
            "history_max_tokens": 6000,
            "history_max_messages": 20,
            "reasoning": {"enabled": False},
            "tool_mode": "structured",
            "top_p": "omitted",
            "seed": "omitted",
            "penalties": "omitted",
            "provider_max_price": {"prompt": 0, "completion": 0},
        },
        "thresholds": {"actions": [10, 12], "arguments": [8, 9]},
        "api_base": settings.openrouter_base_url,
        "extra_acceptance": "4/4 actions and 3/3 arguments; not used for selection or tuning",
        "budget": budget,
        "live_read": {
            "query": "Покажи расписание на 13 октября 2026 года.",
            "tool": "get_schedule",
            "arguments": {"date": "2026-10-13"},
        },
        "limitations": [
            "The original 12 cases were used during earlier prompt development.",
            "One sample per model/temperature is not a stability estimate.",
            "Extra cases are fixed before calls and never used for tuning.",
            "API errors, truncation and invalid JSON remain failures; no retries.",
            "A failed campaign will be reported as failed, never relabelled.",
            "Every case requires one semantic LLM decision, including clarification.",
            "This new 54-attempt protocol is distinct from the archived 50-attempt experiment.",
        ],
    }
    checkpoint(
        folder / "provider-evidence.json",
        {
            "checked_at": timestamp(),
            "models": selected,
            "free_model_daily_requests": quota,
            "sources": [
                "https://openrouter.ai/api/v1/models",
                "https://openrouter.ai/docs/api_reference/limits",
                "https://openrouter.ai/docs/api_reference/parameters",
            ],
        },
    )
    checkpoint(folder / "manifest.json", document)
    return document


class PacedClient(RecordingClient):
    last_dispatch = 0.0

    async def _request(self, payload):
        # Across models as well: <=15 starts/min, without retrying any request.
        await asyncio.sleep(max(0, 4.1 - (time.monotonic() - PacedClient.last_dispatch)))
        PacedClient.last_dispatch = time.monotonic()
        started = time.monotonic()
        try:
            return await super()._request(payload)
        finally:
            self.observation["http_duration_ms"] = round((time.monotonic() - started) * 1000)


def select_cell(reports, cells=CELLS):
    def key(cell):
        metrics = reports[cell["id"]]["metrics"]
        return (
            metrics["acceptance"] != "passed",
            -metrics["argument_accuracy"]["correct"],
            -metrics["routing_accuracy"]["correct"],
            cell["temperature"],
            cell["model"],
        )

    return min(cells, key=key)


def compare_inputs(reports):
    by_case = {}
    for report in reports.values():
        for row in report["attempts"]:
            if row["calls"]:
                value = row["calls"][0]["input_sha256"]
                if row["id"] in by_case and by_case[row["id"]] != value:
                    raise ValueError("Messages/schema changed between experimental cells.")
                by_case[row["id"]] = value
    return by_case


async def extra_trials(settings, client, catalog, path, *, temperature):
    context = load_dataset()["evaluation_context"]
    expected_protocol = {
        "version": PROTOCOL_VERSION,
        "source_sha256": sources(),
        "extra_cases_sha256": digest(EXTRA),
        "context": context,
        "model": settings.openrouter_model,
        "temperature": temperature,
        "catalog_sha256": digest({"tools": catalog.tools, "resources": catalog.resources}),
        "max_output_tokens": settings.llm_max_output_tokens,
        "llm_timeout_seconds": settings.llm_timeout_seconds,
        "agent_timeout_seconds": settings.agent_timeout_seconds,
        "history_max_tokens": settings.history_max_tokens,
        "history_max_messages": settings.history_max_messages,
        "tool_mode": settings.llm_tool_mode,
        "maximum_llm_calls": len(EXTRA),
        "automatic_retries": False,
    }
    if path.exists():
        report = json.loads(path.read_text())
        if report.get("protocol") != expected_protocol:
            raise ValueError(
                "Existing extra result uses a different protocol; choose a new output."
            )
        return report  # Interrupted attempts are not replayed.
    report = {
        "protocol": expected_protocol,
        "cases": [],
        "started_at": timestamp(),
        "status": "in_progress",
    }
    checkpoint(path, report)
    for case in EXTRA:
        row = {
            **case,
            "calls": [],
            "status": "in_progress",
            "actual": {"action": None, "arguments": None},
        }
        report["cases"].append(row)
        checkpoint(path, report)
        repo = MemoryRepository(context["timezone"], user=replace(USER, temperature=temperature))
        measured = TrialLLM(client, row, permit=True, persist=lambda: checkpoint(path, report))
        agent = ObservedAgent(
            measured, catalog, repo, settings, now=lambda: datetime.fromisoformat(context["now"])
        )
        dialog = DialogService(repo, measured, settings, agent=agent)
        try:
            row["displayed_answer"] = await dialog.respond(USER.user_id, 1, case["query"])
            proposal = getattr(getattr(agent, "reply", None), "reminder", None)
            row["actual"] = (
                {"action": "add_reminder", "arguments": proposal.arguments.model_dump(mode="json")}
                if proposal
                else {"action": (repo.audit or {}).get("decision", "respond"), "arguments": {}}
            )
            row["status"] = "ok"
        except ReadBoundary as boundary:
            row.update(
                status="captured_before_mcp",
                actual={"action": boundary.action, "arguments": boundary.arguments},
            )
        except ModelError as error:
            row.update(status="failed", error={"type": "ModelError", "message": error.user_message})
        except Exception as error:
            row.update(status="failed", error={"type": type(error).__name__})
        row["score"] = score(case, row["actual"])
        row["finished_at"] = timestamp()
        checkpoint(path, report)
    report.update(
        status="completed",
        actions=sum(row["score"]["routing_correct"] for row in report["cases"]),
        arguments=sum(row["score"]["arguments_correct"] is True for row in report["cases"]),
        denominators={"actions": 4, "arguments": 3},
    )
    checkpoint(path, report)
    return report


async def campaign(settings, folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("version") != PROTOCOL_VERSION or manifest.get("budget") != campaign_budget():
        raise ValueError(
            "Legacy comparison protocol/budget cannot run with the current semantic router; "
            "preserve the historical result and prepare a new preregistration."
        )
    registered = await asyncio.to_thread(
        subprocess.run,
        ["git", "show", "HEAD:" + str((folder / "manifest.json").resolve().relative_to(ROOT))],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if registered.returncode or registered.stdout != (folder / "manifest.json").read_bytes():
        raise ValueError("Commit the exact preregistration before real API calls.")
    if manifest["source_sha256"] != sources() or digest(manifest["extra_cases"]) != digest(EXTRA):
        raise ValueError("Frozen source/extra cases changed after preregistration.")
    if digest(manifest["cells"]) != digest(CELLS):
        raise ValueError("Frozen configurations changed.")
    if manifest["api_base"] != settings.openrouter_base_url:
        raise ValueError("The API endpoint changed after preregistration.")
    settings = replace(
        settings,
        schedule_file=str(ROOT / "data/schedule.json"),
        schedule_timezone="Europe/Moscow",
        llm_max_output_tokens=2048,
        llm_timeout_seconds=60,
        agent_timeout_seconds=135,
        history_max_tokens=6000,
        history_max_messages=20,
        llm_tool_mode="structured",
    )
    configure_logging(settings)
    channel = MCPClient(
        replace(settings, bot_token="", openrouter_api_key="", postgres_password="")
    )
    clients = {model: PacedClient(replace(settings, openrouter_model=model)) for model in MODELS}
    reports = {}
    try:
        await channel.start()
        if not channel.available:
            raise ValueError("MCP discovery unavailable; no LLM was called.")
        catalog = CatalogOnlyMCP(channel.tools, channel.resources)
        for client in clients.values():
            await client.start()
        for index in range(12):
            rotated = CELLS[index % 4 :] + CELLS[: index % 4]
            for cell in rotated:
                print(f"cell={cell['id']} case_index={index}", flush=True)
                reports[cell["id"]] = await run_trials(
                    replace(settings, openrouter_model=cell["model"]),
                    clients[cell["model"]],
                    catalog,
                    folder / (cell["id"] + ".json"),
                    temperature=cell["temperature"],
                    models=MODELS,
                    max_new_cases=1,
                )
        hashes = compare_inputs(reports)
        chosen = select_cell(reports)
        extra = await extra_trials(
            replace(settings, openrouter_model=chosen["model"]),
            clients[chosen["model"]],
            catalog,
            folder / "extra.json",
            temperature=chosen["temperature"],
        )
        from scripts.lab2_comparison_live import run_live

        live = await run_live(
            replace(settings, openrouter_model=chosen["model"]),
            channel,
            folder / "live-read.json",
            temperature=chosen["temperature"],
        )
        summary = {
            "manifest_sha256": digest(manifest),
            "finished_at": timestamp(),
            "selected": chosen,
            "metrics": {name: report["metrics"] for name, report in reports.items()},
            "equal_input_hashes": hashes,
            "extra": {
                key: extra.get(key) for key in ("status", "actions", "arguments", "denominators")
            },
            "live_status": live.get("metrics", {}).get("acceptance", "not_passed"),
            "routing_acceptance": reports[chosen["id"]]["metrics"]["acceptance"],
            "status": "passed"
            if (
                reports[chosen["id"]]["metrics"]["acceptance"] == "passed"
                and extra.get("status") == "completed"
                and extra.get("actions") == 4
                and extra.get("arguments") == 3
                and live.get("metrics", {}).get("acceptance") == "passed"
            )
            else "not_passed",
        }
        checkpoint(folder / "summary.json", summary)
        return summary
    finally:
        for client in clients.values():
            await client.close()
        await channel.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--folder", type=Path, default=Path("docs/evaluation/lab2-comparison"))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    settings = Settings.load(args.env_file, require_llm=True)
    if args.prepare:
        prepare(settings, args.folder)
        print("Preregistration saved. Commit it before --run.")
    else:
        result = asyncio.run(campaign(settings, args.folder))
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
