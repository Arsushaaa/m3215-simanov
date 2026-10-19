"""Воспроизводимые проверки этапа 6: без реальных Telegram, LLM и погоды.

Запуск из корня: source scripts/env.sh; python -m scripts.lab2_reliability
Docker нужен только для изолированных PostgreSQL-контейнеров тестов.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/evaluation/lab2-stage6-cases.json"
LAB1_COMMIT = "2d10cf0b9b750b5dc9ce67b849d1c774b1413ff9"
STAGE5_COMMIT = "52ab91fea85018ffc9d607ca0e6f0e9636dc20fd"


def junit_outcomes(path: Path) -> dict[str, str]:
    results = {}
    for case in ET.parse(path).getroot().iter("testcase"):
        node = case.attrib["classname"].replace(".", "/") + ".py::" + case.attrib["name"]
        status = (
            "failed"
            if case.find("failure") is not None or case.find("error") is not None
            else "skipped"
            if case.find("skipped") is not None
            else "passed"
        )
        # Повторный node не должен скрыть неуспешный исход первым/последним PASS.
        if results.get(node) in {"failed", "skipped"}:
            continue
        results[node] = status
    return results


def evaluate_cases(cases: list[dict], outcomes: dict[str, str]) -> list[dict]:
    result = []
    for case in cases:
        checks = []
        for required in case["tests"]:
            matches = {
                node: status
                for node, status in outcomes.items()
                if node == required or node.startswith(required + "[")
            }
            checks.append(
                {
                    "test": required,
                    "matched": len(matches),
                    "status": (
                        "passed"
                        if matches and all(s == "passed" for s in matches.values())
                        else "missing"
                        if not matches
                        else "not_passed"
                    ),
                }
            )
        result.append(
            {
                **case,
                "status": "passed"
                if checks and all(c["status"] == "passed" for c in checks)
                else "not_passed",
                "checks": checks,
            }
        )
    return result


def _run(name, arguments, *, env):
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [sys.executable, "-m", *arguments],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1800,
            check=False,
        )
        exit_code = completed.returncode
    except (OSError, subprocess.TimeoutExpired):
        exit_code = -1
    # Текст сторонних программ/traceback не переносится в отчёт или журнал.
    print(f"{name}: {'PASS' if exit_code == 0 else 'FAIL'}", flush=True)
    return {
        "name": name,
        "command": ["python", "-m", *arguments],
        "exit_code": exit_code,
        "duration_ms": round((time.monotonic() - started) * 1000),
    }


def source_checks():
    baseline_result = subprocess.run(
        ["git", "show", f"{STAGE5_COMMIT}:docs/evaluation/lab2-stage5-checks.json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    baseline = json.loads(baseline_result.stdout)
    hashes = baseline["source_checks"]["migration_sha256"]
    migrations = {
        name: (ROOT / "app/migrations" / name).is_file()
        and hashlib.sha256((ROOT / "app/migrations" / name).read_bytes()).hexdigest() == digest
        for name, digest in hashes.items()
    }
    tag = subprocess.run(
        ["git", "rev-parse", "lab1^{commit}"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    dataset = ROOT / "docs/evaluation/lab2-routing.json"
    original = subprocess.run(
        ["git", "show", f"{LAB1_COMMIT}:docs/evaluation/lab2-routing.json"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    return {
        "python_312": sys.version_info[:2] == (3, 12),
        "migrations_001_to_006_unchanged": bool(migrations) and all(migrations.values()),
        "lab1_commit_unchanged": tag.returncode == 0 and tag.stdout.strip() == LAB1_COMMIT,
        "evaluation_dataset_unchanged": original.returncode == 0
        and original.stdout == dataset.read_bytes(),
    }


def test_environment():
    env = {**os.environ, "RUN_INTEGRATION": "1"}
    # Воспроизводимый полный прогон не наследует -k/-m/--deselect из оболочки.
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "docs/evaluation/lab2-stage6-checks.json"
    )
    args = parser.parse_args()
    cases = json.loads(MANIFEST.read_text(encoding="utf-8"))["cases"]
    env = test_environment()
    with tempfile.TemporaryDirectory(prefix="lab2-reliability-") as directory:
        junit = Path(directory) / "tests.xml"
        checks = [
            _run("pip check", ["pip", "check"], env=env),
            _run("Ruff", ["ruff", "check", "."], env=env),
            _run("format", ["ruff", "format", "--check", "."], env=env),
            _run("pytest + PostgreSQL", ["pytest", "-q", f"--junitxml={junit}"], env=env),
        ]
        outcomes = junit_outcomes(junit) if junit.exists() else {}
    matrix = evaluate_cases(cases, outcomes)
    sources = source_checks()
    success = (
        all(check["exit_code"] == 0 for check in checks)
        and bool(matrix)
        and all(case["status"] == "passed" for case in matrix)
        and all(sources.values())
    )
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "stage": 6,
        "status": "passed" if success else "not_passed",
        "checks": checks,
        "suite": {
            status: sum(value == status for value in outcomes.values())
            for status in ("passed", "failed", "skipped")
        },
        "source_checks": sources,
        "cases": matrix,
        "boundaries": {
            "real_postgresql": any(
                node.startswith("tests/test_integration.py::") and status == "passed"
                for node, status in outcomes.items()
            ),
            "real_stdio": outcomes.get(
                "tests/test_mcp_client.py::test_real_stdio_start_and_close_from_different_tasks"
            )
            == "passed",
            "telegram_llm_and_weather": "substituted in tests",
            "real_model_experiment": "stage 7; not performed here",
            "lab2_tag": "not created by this script",
        },
    }
    # В отчёте pytest путь временного файла заменён воспроизводимым обозначением.
    report["checks"][-1]["command"][-1] = "--junitxml=<temporary>/tests.xml"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Stage 6: {report['status']}; {report['suite']['passed']} tests passed", flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
