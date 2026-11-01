"""Attack-runner checks; real API is never used by these tests."""

import json
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest
from test_integration import database as database
from test_supervisor import tool

from app.config import Settings
from app.llm import ModelError, RoutingCompletion
from app.materials.contracts import SearchArguments
from app.materials.corpus import Corpus
from scripts import lab3_attack_live as attack


def test_hard_five_call_limit_preserves_reserve_before_any_http():
    cases = [{"llm_calls": [], "maximum_llm_calls": 3}, {"llm_calls": [], "maximum_llm_calls": 2}]
    persist = Mock()
    client = attack.LimitedClient(
        Settings(bot_token="", postgres_password=""), {"cases": cases}, persist
    )
    client.case = cases[0]
    for _ in range(3):
        client.reserve({})
    with pytest.raises(ModelError):
        client.reserve({})
    client.case = cases[1]
    client.reserve({})
    client.reserve({})
    with pytest.raises(ModelError):
        client.reserve({})
    assert sum(len(c["llm_calls"]) for c in cases) == 5
    assert persist.call_count == 5
    assert 50 - 30 - 5 == 15


def test_source_injection_does_not_mutate_original_or_exceed_mcp_snippet_bound():
    raw = {"status": "ok", "matches": [{"snippet": "x" * 1200}]}
    altered, changed = attack.poison_result(raw)
    assert changed and len(altered["matches"][0]["snippet"]) <= 1200
    assert raw["matches"][0]["snippet"] == "x" * 1200
    assert attack.POISON in altered["matches"][0]["snippet"]
    assert attack.poison_result({"status": "not_found", "matches": []})[1] is False


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="Isolated Docker required")
async def test_live_attack_runner_real_sql_stdio_fake_model_reserve_and_owner_isolation(
    database, monkeypatch
):
    settings, _, _, _ = database
    settings = replace(settings, llm_tool_mode="native")
    now = datetime.now(UTC).replace(microsecond=0)
    protocol = {"now": now.isoformat(), "cases": attack.CASES, "temperature": 0.2}

    class SyntheticClient(attack.LimitedClient):
        async def start(self):
            pass

        async def close(self):
            pass

        async def _request(self, payload):
            self.reserve(payload)
            record = self.case["llm_calls"][-1]
            number = len(self.case["llm_calls"])
            knowledge = self.case["id"] == "material_instruction"
            if number == 1:
                target = "knowledge" if knowledge else "schedule"
                reason = "materials_request" if knowledge else "schedule_request"
                result = tool(
                    "select_route", {"target": target, "reason": reason, "confidence": 0.9}
                )
            elif knowledge and number == 2:
                result = tool(
                    "search_materials", {"query": "MCP", "document_id": "mcp-basics", "limit": 1}
                )
            elif knowledge:
                match = (
                    Corpus.load()
                    .search(SearchArguments(query="MCP", document_id="mcp-basics", limit=1))
                    .matches[0]
                )
                result = RoutingCompletion(
                    json.dumps(
                        {
                            "answer": "MCP предоставляет инструменты и ресурсы.",
                            "citations": [
                                {"document_id": match.document_id, "quote": match.snippet[:80]}
                            ],
                        }
                    ),
                    [],
                )
            else:
                moment = (now.astimezone(ZoneInfo("Europe/Moscow")) + timedelta(days=1)).replace(
                    hour=11, minute=0, second=0
                )
                quote = {"source": "user", "message_index": 0, "quote": self.case["text"]}
                result = tool(
                    "prepare_reminder",
                    {
                        "arguments": {"text": "Приготовить пирог", "remind_at": moment.isoformat()},
                        "evidence": {k: quote for k in ("request", "text", "date", "time")},
                    },
                )
            record.update(state="finished", usage={"cost": 0})
            return result

    monkeypatch.setattr(attack, "LimitedClient", SyntheticClient)
    document = {"cases": []}
    await attack.execute(settings, protocol, document, lambda: None)
    assert [c["state"] for c in document["cases"]] == ["passed", "passed"]
    assert sum(len(c["llm_calls"]) for c in document["cases"]) == 5
    assert document["cases"][0]["poison_applied"]
    assert document["cases"][1]["pending_card"]
    assert all(c["saved_reminders"] == 0 and c["foreign_actions"] == 0 for c in document["cases"])
    assert all(
        c["foreign_history_unchanged"] and not c["foreign_history_exposed"]
        for c in document["cases"]
    )
