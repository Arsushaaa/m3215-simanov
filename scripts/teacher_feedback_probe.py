"""Preregister teacher examples; real Router and optional full study responses."""

import argparse
import asyncio
import json
import socket
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import aiohttp
from dotenv import dotenv_values

from app.agents.knowledge import KnowledgeAgent
from app.config import Settings
from app.mcp.client import MCPClient
from app.prompts import PROMPTS
from app.routing.contracts import AgentRequest
from app.routing.router import Router
from app.routing.supervisor import Supervisor
from scripts.lab2_experiment import USER, MemoryRepository
from scripts.lab3_experiment import MeasuredClient, digest, save, source_hashes
from scripts.lab3_redteam_console import read_quota
from scripts.semantic_routing_smoke import assert_committed, exclusive_run

EXAMPLES = (
    ("list_tuple", "Чем список отличается от кортежа в Python?", ["legacy_study"]),
    ("revision", "Что повторить по идемпотентности?", ["knowledge", "legacy_study"]),
)
CONTROLS = (
    ("concrete_help", "Помоги разобраться с замыканиями в Python.", ["legacy_study"]),
    ("quoted_weather", 'Объясни строку Python: weather = "погода в Казани"', ["legacy_study"]),
    ("idiom", "Напомни, почему повтор POST иногда создаёт две записи.", ["legacy_study"]),
    ("materials", "Что конспект говорит про transactional outbox?", ["knowledge"]),
    ("vague", "Помоги подготовиться.", ["clarify"]),
    ("multiple", "Какая погода в Казани и какие пары завтра?", ["clarify"]),
)


def snapshot():
    return {**source_hashes(), __file__: digest(Path(__file__).read_bytes())}


class Client(MeasuredClient):
    async def start(self):
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(family=socket.AF_INET),
            timeout=aiohttp.ClientTimeout(total=60, sock_connect=10),
            headers={"Authorization": "Bearer " + self.settings.openrouter_api_key},
        )

    async def _request(self, payload):
        if self.attempts >= self.limit:
            raise ValueError("Preregistered HTTP budget exhausted")
        return await super()._request(payload)

    def _observe_response(self, data):
        usage = data.get("usage", {})
        self.observation["usage"] = {
            k: usage[k]
            for k in ("prompt_tokens", "completion_tokens", "cost")
            if type(usage.get(k)) in (int, float)
        }


def freeze(folder, baseline):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "protocol.json"
    if path.exists():
        raise ValueError("Protocol is immutable")
    cases = EXAMPLES if baseline else EXAMPLES + CONTROLS
    save(
        path,
        {
            "created_at": datetime.now(UTC).isoformat(),
            "source_sha256": snapshot(),
            "baseline": baseline,
            "credential_env": "OPENROUTER_API_KEY_3",
            "cases": [dict(id=i, text=t, allowed_targets=a) for i, t, a in cases],
            "model": "nvidia/nemotron-3.5-super-vl:free",
            "temperature": 0.2,
            "max_http_attempts": 2 if baseline else 14,
            "clock": "2026-11-04T10:00:00+03:00",
            "scope": "Router for all cases; fixed phase also full Supervisor for teacher examples. "
            "Fresh memory storage, real MCP read only, no Telegram or writes. "
            "Expected targets not sent to model. Started cases never replayed. "
            "No raw prompts saved.",
        },
    )


async def run(folder):
    path, output = folder / "protocol.json", folder / "results.json"
    p = json.loads(path.read_text())
    assert_committed(path)
    if p["source_sha256"] != snapshot():
        raise ValueError("Sources changed after registration")
    d = (
        json.loads(output.read_text())
        if output.exists()
        else {"protocol_sha256": digest(path.read_bytes()), "cases": []}
    )
    if d["protocol_sha256"] != digest(path.read_bytes()):
        raise ValueError("Checkpoint mismatch")
    if d["cases"]:
        raise ValueError("Started experiment is never replayed")
    s = Settings.load(".env", require_llm=True)
    key = dotenv_values(".env", interpolate=False)[p["credential_env"]]
    s = replace(
        s,
        openrouter_api_key=key,
        openrouter_model=p["model"],
        llm_tool_mode="native",
        history_max_tokens=6000,
        llm_max_output_tokens=2048,
    )
    d["quota_before"] = await read_quota(s)
    if d["quota_before"].get("remaining", 0) < p["max_http_attempts"]:
        raise ValueError("Insufficient free quota")
    client = Client(s)
    client.limit = p["max_http_attempts"]
    mcp = MCPClient(s)
    await client.start()
    try:
        for c in p["cases"]:
            row = dict(id=c["id"], state="started")
            d["cases"].append(row)
            save(output, d)
            req = AgentRequest(
                request_id=uuid4(),
                trace_id=uuid4(),
                actor_id=1,
                text=c["text"],
                intent="user_query",
                occurred_at=datetime.fromisoformat(p["clock"]),
                timezone="Europe/Moscow",
            )
            try:
                decision = await Router(client, s).route(req, [], p["temperature"])
                row["actual"] = decision.target if decision else None
            except Exception as e:
                row.update(actual=None, error=type(e).__name__)
            row.update(
                state="finished",
                correct=row["actual"] in c["allowed_targets"],
                observation=client.observation,
            )
            save(output, d)
        if not p["baseline"]:
            await mcp.start()
            knowledge = await KnowledgeAgent.initialize(mcp)
            for i, text, _ in EXAMPLES:
                row = dict(id=i + "_full_study", state="started")
                d["cases"].append(row)
                save(output, d)
                repo = MemoryRepository("Europe/Moscow")
                supervisor = Supervisor(
                    client,
                    mcp,
                    repo,
                    s,
                    knowledge=knowledge,
                    now=lambda: datetime.fromisoformat(p["clock"]),
                )
                before = client.attempts
                try:
                    reply = await supervisor.respond(
                        USER, 1, text, prompt=PROMPTS["study"], history=[]
                    )
                    row.update(
                        answer=reply.text,
                        status=reply.agent_result.status,
                        used_tools=list(reply.agent_result.used_tools),
                    )
                except Exception as e:
                    row["error"] = type(e).__name__
                row.update(state="finished", http_attempts=client.attempts - before)
                save(output, d)
    finally:
        await mcp.close()
        await client.close()
        d["http_attempts"] = client.attempts
        save(output, d)
    d["quota_after"] = await read_quota(s)
    save(output, d)
    print(
        json.dumps(
            {
                "http_attempts": d["http_attempts"],
                "router_correct": sum(r.get("correct", False) for r in d["cases"]),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.folder, args.baseline)
    else:
        with exclusive_run(args.folder):
            asyncio.run(run(args.folder))
