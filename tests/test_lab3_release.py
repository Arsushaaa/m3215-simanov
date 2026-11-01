"""Actual archive/fixture inspection, without deploy or network."""

import json
import tarfile
from pathlib import Path

from app.config import Settings
from scripts.cloud import source_archive, validate_cloud_schedule
from scripts.lab3_acceptance import live_check, preserved_sources

ROOT = Path(__file__).resolve().parents[1]


def test_release_packages_only_public_sources_sql_and_complete_daily_demo():
    with tarfile.open(source_archive(ROOT)) as archive:
        names = archive.getnames()
    for p in (ROOT / "app/migrations").glob("*.sql"):
        assert p.relative_to(ROOT).as_posix() in names
    assert "data/schedule-demo.json" in names and "data/schedule.json" in names
    assert all(".env" not in n and ".tools/" not in n and ".venv/" not in n for n in names)
    demo = json.loads((ROOT / "data/schedule-demo.json").read_text())
    assert demo["coverage_start"] == "2026-11-02" and demo["coverage_end"] == "2026-11-15"
    assert len(demo["days"]) == 14
    validate_cloud_schedule(
        Settings(bot_token="", postgres_password="", schedule_file="data/schedule-demo.json")
    )
    assert all(preserved_sources().values())


def test_absent_or_incomplete_live_results_cannot_be_reported_as_passed(tmp_path):
    assert live_check(tmp_path)["status"] == "incomplete"


def test_started_interrupted_attempt_is_a_failed_case_not_a_missing_denominator(
    tmp_path, monkeypatch
):
    from scripts import lab3_acceptance as acceptance
    from scripts.lab3_experiment import DATASET, DEFAULT_OUTPUT, DEFAULT_PROTOCOL, digest

    directory = tmp_path / "docs/evaluation"
    directory.mkdir(parents=True)
    (tmp_path / DEFAULT_PROTOCOL).write_text("{}")
    expected = json.loads((ROOT / DATASET).read_text())
    (tmp_path / DATASET).write_text(json.dumps(expected))
    cases = [
        {**c, "actual_target": c["expected_target"], "state": "finished", "error": None}
        for c in expected["cases"]
    ]
    cases[0].pop("expected_target")
    cases[0].update(actual_target=None, error="INTERRUPTED_OUTCOME_UNKNOWN")
    result = {"protocol_sha256": digest((tmp_path / DEFAULT_PROTOCOL).read_bytes()), "cases": cases}
    (tmp_path / DEFAULT_OUTPUT).write_text(json.dumps(result))
    monkeypatch.setattr(acceptance, "validate", lambda protocol, root: None)
    checked = acceptance.live_check(tmp_path)
    assert (
        checked["status"] == "passed" and checked["correct"] == 14 and checked["denominator"] == 15
    )
    # An explicit wrong expectation cannot be silently repaired by this recovery rule.
    cases[1]["expected_target"] = "forged"
    (tmp_path / DEFAULT_OUTPUT).write_text(json.dumps(result))
    assert acceptance.live_check(tmp_path)["status"] == "incomplete"
