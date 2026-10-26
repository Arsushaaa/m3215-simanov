"""One preregistered real read through the production dialog, without Telegram/SQL.

The caller owns an already started MCPClient. An existing output is never replayed,
including a checkpoint interrupted after consuming a request slot. Expectations are
used only for scoring after production processing, never in the model context.
"""

import hashlib
import json
import math
import time
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from app.agent import AgentService
from app.dialog import DialogService
from app.llm import ModelError
from app.mcp.contracts import ScheduleResult
from app.prompts import PROMPTS
from scripts.lab2_experiment import (
    ROOT,
    USER,
    MemoryRepository,
    RecordingClient,
    checkpoint,
    digest,
    timestamp,
)

QUERY = "Покажи расписание на 13 октября 2026 года."
NOW = datetime.fromisoformat("2026-10-12T09:00:00+03:00")
ZONE = "Europe/Moscow"
EXPECTED_DATE = "2026-10-13"


class LiveRecordingClient(RecordingClient):
    """Observe the actual production payload without changing the HTTP request."""

    def __init__(self, settings):
        super().__init__(settings)
        self.capture_payload = None

    async def _request(self, payload):
        if self.capture_payload is not None:
            self.capture_payload(payload)
        return await super()._request(payload)


class LiveLLM:
    def __init__(self, client, document, persist):
        self.client, self.document, self.persist = client, document, persist
        client.capture_payload = self._payload

    def _payload(self, payload):
        call = self.document["raw_data"][-1]
        call["wire"] = {
            "request_sha256": digest(payload),
            "model": payload.get("model"),
            "temperature": payload.get("temperature"),
            "max_tokens": payload.get("max_tokens"),
            "tools_present": "tools" in payload,
            "tools": deepcopy(payload.get("tools")),
            "tool_choice": payload.get("tool_choice"),
            "reasoning": deepcopy(payload.get("reasoning")),
            "provider": deepcopy(payload.get("provider")),
        }
        self.persist()

    async def _call(self, phase, messages, temperature, kwargs):
        calls = self.document["raw_data"]
        if len(calls) >= 2 or phase != ("decide" if not calls else "complete"):
            raise ModelError("Live experiment LLM call budget/order rejected.")
        if phase == "complete" and (
            kwargs.get("disable_tools") is not True or kwargs.get("tools") not in (None, [])
        ):
            raise ModelError("Live experiment final tools must be disabled.")
        call = {
            "phase": phase,
            "status": "in_progress",
            "started_at": timestamp(),
            "input_sha256": digest({"messages": messages, **kwargs}),
            "request_sha256": digest({"messages": messages, "temperature": temperature, **kwargs}),
            "temperature": temperature,
            "message_roles": [message["role"] for message in messages],
            "disable_tools": kwargs.get("disable_tools", False),
        }
        calls.append(call)
        self.client.observation = {}
        self.persist()  # A consumed slot is durable before HTTP dispatch.
        started = time.monotonic()
        try:
            result = await getattr(self.client, phase)(messages, temperature, **kwargs)
            call.update(
                status="ok",
                raw_content=result.text,
                raw_tool_calls=deepcopy(getattr(result, "tool_calls", [])),
                truncated=result.truncated,
            )
            return result
        except BaseException as error:
            call.update(status="failed", error={"type": type(error).__name__})
            raise
        finally:
            call.update(deepcopy(self.client.observation))
            call["duration_ms"] = round((time.monotonic() - started) * 1000)
            call["finished_at"] = timestamp()
            self.persist()

    async def decide(self, messages, temperature, **kwargs):
        return await self._call("decide", messages, temperature, kwargs)

    async def complete(self, messages, temperature, **kwargs):
        return await self._call("complete", messages, temperature, kwargs)


class LiveMCP:
    def __init__(self, source, document, persist):
        self.source, self.document, self.persist = source, document, persist
        self.available = source.available
        self.tools, self.resources = deepcopy(source.tools), deepcopy(source.resources)

    async def _call(self, operation, name, arguments):
        if self.document["mcp_calls"]:
            raise ModelError("Live experiment MCP call budget exceeded.")
        if operation == "tools/call" and name == "add_reminder":
            raise ModelError("Writes are forbidden in the live read experiment.")
        call = {
            "operation": operation,
            "name": name,
            "arguments": deepcopy(arguments),
            "status": "in_progress",
            "started_at": timestamp(),
        }
        self.document["mcp_calls"].append(call)
        self.persist()
        try:
            result = (
                await self.source.call_tool(name, arguments)
                if operation == "tools/call"
                else await self.source.read_resource(name)
            )
            call.update(status="ok", result=deepcopy(result))
            return result
        except BaseException as error:
            call.update(status="failed", error={"type": type(error).__name__})
            raise
        finally:
            call["finished_at"] = timestamp()
            self.persist()

    async def call_tool(self, name, arguments, **kwargs):
        if kwargs:
            raise ModelError("Write authorization is forbidden in the live read experiment.")
        return await self._call("tools/call", name, arguments)

    async def read_resource(self, uri):
        return await self._call("resources/read", uri, {})


def _protocol(settings, source, temperature):
    if (
        not settings.openrouter_model.endswith(":free")
        or settings.llm_tool_mode != "structured"
        or settings.schedule_timezone != ZONE
        or type(temperature) not in (int, float)
        or not math.isfinite(temperature)
        or not 0 <= temperature <= 1
    ):
        raise ValueError("Use a free model, structured routing, fixed timezone and temperature.")
    return {
        "version": 2,
        "query": QUERY,
        "now": NOW.isoformat(),
        "timezone": ZONE,
        "model": settings.openrouter_model,
        "temperature": temperature,
        "max_output_tokens": settings.llm_max_output_tokens,
        "tool_mode": settings.llm_tool_mode,
        "llm_timeout_seconds": settings.llm_timeout_seconds,
        "agent_timeout_seconds": settings.agent_timeout_seconds,
        "history_max_tokens": settings.history_max_tokens,
        "history_max_messages": settings.history_max_messages,
        "study_prompt_sha256": digest(PROMPTS["study"]),
        "catalog_sha256": digest({"tools": source.tools, "resources": source.resources}),
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "scripts/lab2_comparison_live.py",
                "app/dialog.py",
                "app/agent.py",
                "app/llm.py",
                "app/prompts.py",
                "app/agent_prompts.py",
                "app/reminder_grounding.py",
                "app/reminder_messages.py",
                "app/mcp/contracts.py",
            )
        },
        "expected": {"tool": "get_schedule", "arguments": {"date": EXPECTED_DATE}},
        "limits": {"llm_calls": 2, "mcp_application_calls": 1},
        "storage": "isolated memory; no PostgreSQL or Telegram",
        "expectations_in_model_context": False,
        "automatic_retries": False,
    }


def _score(document, repository):
    calls, mcp = document["raw_data"], document["mcp_calls"]
    result = mcp[0].get("result") if len(mcp) == 1 else None
    result_valid = False
    try:
        parsed = ScheduleResult.model_validate(result).root
        result_valid = (
            parsed.status == "ok"
            and parsed.date.isoformat() == EXPECTED_DATE
            and parsed.source == "local_schedule"
            and parsed.timezone == ZONE
        )
    except (ValueError, AttributeError, TypeError):
        pass
    wire = calls[1].get("wire", {}) if len(calls) == 2 else {}
    criteria = {
        "exact_read_action": len(mcp) == 1
        and mcp[0]["operation"] == "tools/call"
        and mcp[0]["name"] == "get_schedule"
        and mcp[0]["arguments"] == {"date": EXPECTED_DATE}
        and mcp[0]["status"] == "ok",
        "result_matches_date_and_source": result_valid,
        "two_successful_llm_calls": len(calls) == 2
        and [call["phase"] for call in calls] == ["decide", "complete"]
        and all(call["status"] == "ok" and not call.get("truncated") for call in calls),
        "final_http_tools_disabled": wire.get("tools_present") is True
        and wire.get("tools") == []
        and calls[1]["disable_tools"] is True,
        "answer_saved": bool(document.get("displayed_answer", "").strip())
        and repository.exchange == (QUERY, document.get("displayed_answer")),
    }
    costs = [call.get("usage", {}).get("cost") for call in calls]
    document["metrics"] = {
        "criteria": criteria,
        "acceptance": "passed" if all(criteria.values()) else "not_passed",
        "llm_calls": len(calls),
        "mcp_application_calls": len(mcp),
        "api_cost": sum(costs) if costs and all(cost is not None for cost in costs) else None,
        "database_writes": 0,
        "telegram_calls": 0,
    }


def _create_output(output, document):
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents concurrent invocations from consuming the same trial twice.
    with output.open("x", encoding="utf-8") as file:
        json.dump(document, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")


async def run_live(settings, catalog_source, output, *, temperature):
    """Execute once, or read the same checkpoint without replaying any operation."""
    output = Path(output)
    expected_protocol = _protocol(settings, catalog_source, temperature)
    if output.exists():
        document = json.loads(output.read_text(encoding="utf-8"))
        if document.get("protocol") != expected_protocol:
            raise ValueError("Existing live result uses a different protocol; choose a new output.")
        return document
    if not catalog_source.available:
        raise ValueError("MCP discovery is unavailable; no LLM request was made.")
    document = {
        "created_at": timestamp(),
        "protocol": expected_protocol,
        "status": "in_progress",
        "raw_data": [],
        "mcp_calls": [],
    }
    _create_output(output, document)

    def persist():
        checkpoint(output, document)

    repository = MemoryRepository(ZONE, user=replace(USER, temperature=temperature))
    client = LiveRecordingClient(settings)
    llm, mcp = LiveLLM(client, document, persist), LiveMCP(catalog_source, document, persist)
    agent = AgentService(llm, mcp, repository, settings, now=lambda: NOW)
    dialog = DialogService(repository, llm, settings, agent=agent)
    try:
        await client.start()
        document["displayed_answer"] = await dialog.respond(USER.user_id, 1, QUERY)
        document["status"] = "completed"
    except BaseException as error:
        document.update(status="failed", error={"type": type(error).__name__})
        if not isinstance(error, Exception):
            raise
    finally:
        document["audit"] = repository.audit
        _score(document, repository)
        document["finished_at"] = timestamp()
        persist()
        await client.close()
    return document
