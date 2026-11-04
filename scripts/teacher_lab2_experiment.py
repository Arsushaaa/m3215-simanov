"""Fresh fixed 12-case LR2 experiment in the deployed native configuration."""

import argparse
import asyncio
import json
import socket
from dataclasses import replace
from pathlib import Path

import aiohttp
from dotenv import dotenv_values

from app.config import Settings
from app.mcp.client import MCPClient
from scripts.lab2_experiment import (
    DATASET,
    CatalogOnlyMCP,
    RecordingClient,
    checkpoint,
    load_dataset,
    protocol,
    run_trials,
)
from scripts.lab3_redteam_console import read_quota
from scripts.semantic_routing_smoke import (
    ROOT,
    assert_committed,
    exclusive_run,
    fixed_settings,
    source_snapshot,
)


class IPv4Client(RecordingClient):
    async def start(self):
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(family=socket.AF_INET),
            timeout=aiohttp.ClientTimeout(total=60, sock_connect=10),
            headers={"Authorization": "Bearer " + self.settings.openrouter_api_key},
        )


def snapshot():
    from scripts.lab3_experiment import digest

    return {
        **source_snapshot(),
        "scripts/teacher_lab2_experiment.py": digest(Path(__file__).read_bytes()),
    }


async def execute(folder, freeze=False):
    s = Settings.load(".env", require_llm=True)
    if s.openrouter_base_url != "https://openrouter.ai/api/v1":
        raise ValueError("Only registered OpenRouter endpoint is allowed")
    s = fixed_settings(
        replace(
            s, openrouter_api_key=dotenv_values(".env", interpolate=False)["OPENROUTER_API_KEY_2"]
        )
    )
    mcp = MCPClient(replace(s, bot_token="", openrouter_api_key="", postgres_password=""))
    try:
        await mcp.start()
        if not mcp.available:
            raise ValueError("Discovery unavailable; no LLM was called")
        catalog = CatalogOnlyMCP(mcp.tools, mcp.resources)
    finally:
        await mcp.close()
    dataset = load_dataset()
    manifest = {
        "experiment_protocol": protocol(
            s, dataset, dataset["cases"], catalog, temperature=0.3, models=(s.openrouter_model,)
        ),
        "credential_env": "OPENROUTER_API_KEY_2",
        "source_sha256": snapshot(),
        "dataset": str(DATASET.relative_to(ROOT)),
        "scope": "One real LLM per fixed input; native mode. Real stdio discovery. "
        "Stop before MCP calls; memory storage, no Telegram, SQL or confirmations. "
        "Failures/interrupted attempts never retried, no paid models or fallback.",
    }
    path, output = folder / "protocol.json", folder / "results.json"
    if freeze:
        folder.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise ValueError("Preregistration cannot be overwritten")
        checkpoint(path, manifest)
        return
    assert_committed(path)
    if json.loads(path.read_text()) != manifest:
        raise ValueError("Source/settings/catalog changed after registration")
    quota = await read_quota(s)
    previous = json.loads(output.read_text()) if output.exists() else {"attempts": []}
    if quota.get("remaining", 0) < 12 - len(previous["attempts"]):
        raise ValueError("Insufficient free quota for all remaining cases")
    client = IPv4Client(s)
    await client.start()
    try:
        report = await run_trials(s, client, catalog, output, models=(s.openrouter_model,))
    finally:
        await client.close()
    checkpoint(folder / "quota.json", {"before": quota, "after": await read_quota(s)})
    print(json.dumps(report["metrics"], ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    args.folder.mkdir(parents=True, exist_ok=True)
    with exclusive_run(args.folder):
        asyncio.run(execute(args.folder, args.action == "freeze"))
