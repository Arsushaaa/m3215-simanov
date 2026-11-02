"""Four preregistered novel Router inputs; no specialists or execution tools."""

import argparse
import asyncio
import json
import math
import socket
from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

import aiohttp
from dotenv import dotenv_values

from app.config import Settings
from app.llm import ModelError
from app.routing.contracts import AgentRequest
from app.routing.router import Router
from scripts.lab3_experiment import ROOT, MeasuredClient, digest, save, source_hashes
from scripts.lab3_redteam_console import read_quota
from scripts.semantic_routing_smoke import assert_committed, exclusive_run

FOLDER = ROOT / "docs/evaluation/lab3-quality-holdout-20261102"
CASES = (
    ("vague_delegation", "Возьми всё в свои руки.", "clarify"),
    ("unknown_priorities", "Помоги разобраться, что сейчас важнее.", "clarify"),
    ("idiomatic_remind", "Напомни, зачем в Python нужен finally.", "legacy_study"),
    ("thanks", "Спасибо, мне стало понятнее!", "legacy_study"),
)


def snapshot():
    return {
        **source_hashes(),
        **{
            name: digest((ROOT / name).read_bytes())
            for name in ("scripts/lab3_router_holdout.py", "scripts/lab3_redteam_console.py")
        },
    }


class ProbeClient(MeasuredClient):
    async def start(self):
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(family=socket.AF_INET),
            timeout=aiohttp.ClientTimeout(total=60, sock_connect=10),
            headers={"Authorization": "Bearer " + self.settings.openrouter_api_key},
        )

    async def _request(self, payload):
        if self.attempts >= len(CASES):
            raise ModelError("Registered four-call budget exhausted before HTTP")
        return await super()._request(payload)

    def _observe_response(self, data):
        usage = data.get("usage")
        self.observation["usage"] = {
            k: v
            for k, v in (usage.items() if isinstance(usage, dict) else [])
            if k in {"prompt_tokens", "completion_tokens", "cost"}
            and type(v) in (int, float)
            and math.isfinite(v)
            and v >= 0
        }


def freeze():
    FOLDER.mkdir(exist_ok=True)
    path = FOLDER / "protocol.json"
    if path.exists():
        raise ValueError("Never overwrite an existing preregistration")
    save(
        path,
        {
            "kind": "lab3_router_novel_inputs_preregistration",
            "created_at": datetime.now(UTC).isoformat(),
            "source_sha256": snapshot(),
            "credential_env": "OPENROUTER_API_KEY_3",
            "cases": [dict(id=i, text=t, expected=e) for i, t, e in CASES],
            "max_http_attempts": 4,
            "model": "nvidia/nemotron-3.5-super-vl:free",
            "temperature": 0.2,
            "clock": "2026-10-12T09:00:00+03:00",
            "policy": (
                "One production Router per input; no specialists/MCP/Telegram. "
                "Expected targets never enter model context. Started attempts are not retried. "
                "No raw model/system prompts are persisted."
            ),
        },
    )


async def run():
    path, output = FOLDER / "protocol.json", FOLDER / "results.json"
    protocol = json.loads(path.read_text())
    if protocol["source_sha256"] != snapshot():
        raise ValueError("Source changed after preregistration")
    assert_committed(path)
    result = (
        json.loads(output.read_text())
        if output.exists()
        else {
            "kind": "lab3_router_novel_inputs",
            "protocol_sha256": digest(path.read_bytes()),
            "cases": [],
        }
    )
    if result["protocol_sha256"] != digest(path.read_bytes()):
        raise ValueError("Checkpoint belongs to another protocol")
    for row in result["cases"]:
        if row["state"] == "started":
            row.update(state="finished", actual=None, error="INTERRUPTED_OUTCOME_UNKNOWN")
    settings = Settings.load(".env", require_llm=True)
    key = dotenv_values(".env", interpolate=False).get(protocol["credential_env"])
    if not key or settings.openrouter_base_url != "https://openrouter.ai/api/v1":
        raise ValueError("Registered credential or endpoint unavailable")
    settings = replace(
        settings,
        openrouter_api_key=key,
        openrouter_model=protocol["model"],
        llm_tool_mode="native",
        history_max_tokens=6000,
        llm_max_output_tokens=2048,
    )
    result["quota_before"] = await read_quota(settings)
    if result["quota_before"].get("remaining", 0) < len(CASES) - len(result["cases"]):
        result["blocked"] = "INSUFFICIENT_FREE_QUOTA"
        save(output, result)
        return result
    client = ProbeClient(settings)
    await client.start()
    try:
        for case in protocol["cases"]:
            if any(r["id"] == case["id"] for r in result["cases"]):
                continue
            row = {"id": case["id"], "text": case["text"], "state": "started"}
            result["cases"].append(row)
            save(output, result)
            request = AgentRequest(
                request_id=uuid4(),
                trace_id=uuid4(),
                actor_id=101,
                text=case["text"],
                intent="user_query",
                occurred_at=datetime.fromisoformat(protocol["clock"]),
                timezone="Europe/Moscow",
            )
            before = client.attempts
            client.observation = {}
            try:
                decision = await Router(client, settings).route(
                    request, [], protocol["temperature"]
                )
                row["actual"] = decision.target if decision is not None else None
            except Exception as error:
                row.update(actual=None, error=type(error).__name__)
            row.update(
                state="finished",
                observation=client.observation,
                http_attempts=client.attempts - before,
            )
            save(output, result)
    finally:
        await client.close()
    expected = {c["id"]: c["expected"] for c in protocol["cases"]}
    result["summary"] = {
        "correct": sum(r.get("actual") == expected[r["id"]] for r in result["cases"]),
        "denominator": 4,
        "attempts": len(result["cases"]),
        "complete": len(result["cases"]) == 4,
    }
    save(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    else:
        with exclusive_run(FOLDER):
            result = asyncio.run(run())
        print(json.dumps(result.get("summary", {"blocked": result.get("blocked")})))


if __name__ == "__main__":
    main()
