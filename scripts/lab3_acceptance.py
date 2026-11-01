"""Full automated acceptance plus a separately verified real 15-case experiment."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from scripts.lab2_reliability import (
    _run,
    evaluate_cases,
    junit_outcomes,
    source_checks,
    test_environment,
)
from scripts.lab3_experiment import (
    DATASET,
    DEFAULT_OUTPUT,
    DEFAULT_PROTOCOL,
    ROOT,
    digest,
    validate,
)


def live_check(root=ROOT):
    path = root / DEFAULT_OUTPUT
    try:
        protocol = json.loads((root / DEFAULT_PROTOCOL).read_text())
        validate(protocol, root)
        result = json.loads(path.read_text())
        expected = json.loads((root / DATASET).read_text())["cases"]
        cases = result["cases"]
        complete = (
            len(cases) == 15
            and result["protocol_sha256"] == digest((root / DEFAULT_PROTOCOL).read_bytes())
            and all(
                c.get("state") == "finished"
                and c.get("id") == e["id"]
                and c.get("query") == e["query"]
                and (
                    c.get("expected_target") == e["expected_target"]
                    or (
                        c.get("error") == "INTERRUPTED_OUTCOME_UNKNOWN"
                        and c.get("expected_target") is None
                        and c.get("actual_target") is None
                    )
                )
                for c, e in zip(cases, expected, strict=True)
            )
        )
        correct = (
            sum(
                c.get("actual_target") == e["expected_target"]
                for c, e in zip(cases, expected, strict=True)
            )
            if complete
            else 0
        )
        return {
            "id": "routing_experiment",
            "status": "passed"
            if complete and correct >= 12
            else "not_passed"
            if complete
            else "incomplete",
            "correct": correct,
            "denominator": 15,
            "report": str(path.relative_to(root)),
            "sha256": digest(path.read_bytes()),
        }
    except (OSError, ValueError, KeyError, TypeError):
        return {
            "id": "routing_experiment",
            "status": "incomplete",
            "correct": 0,
            "denominator": 15,
            "report": DEFAULT_OUTPUT,
        }


def preserved_sources(root=ROOT):
    baseline = json.loads((root / "docs/evaluation/lab3-stage6-checks.json").read_text())
    migrations = {
        p: digest((root / p).read_bytes()) == h
        for p, h in baseline["runtime_source_sha256"].items()
        if p.startswith("app/migrations/")
    }
    tags = baseline["source_checks"]["historical_tags"]
    tagchecks = {
        t: subprocess.run(
            ["git", "rev-parse", t], cwd=root, capture_output=True, text=True, check=False
        ).stdout.strip()
        == h
        for t, h in tags.items()
    }
    original = subprocess.run(
        ["git", "show", f"19c469a:{DATASET}"], cwd=root, capture_output=True, check=True
    ).stdout
    return {
        "migrations_001_to_010": all(migrations.values()),
        "historical_tags": all(tagchecks.values()),
        "fifteen_cases": original == (root / DATASET).read_bytes(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "docs/evaluation/lab3-final-checks.json"
    )
    args = parser.parse_args()
    env = test_environment()
    with tempfile.TemporaryDirectory(prefix="lab3-acceptance-") as folder:
        junit = Path(folder) / "tests.xml"
        checks = [
            _run("pip check", ["pip", "check"], env=env),
            _run("Ruff", ["ruff", "check", "."], env=env),
            _run("format", ["ruff", "format", "--check", "."], env=env),
            _run(
                "pytest + PostgreSQL",
                ["pytest", "-q", "-o", "faulthandler_timeout=60", f"--junitxml={junit}"],
                env=env,
            ),
        ]
        outcomes = junit_outcomes(junit) if junit.exists() else {}
    manifest = json.loads((ROOT / "docs/evaluation/lab3-acceptance-matrix.json").read_text())
    cases = evaluate_cases([c for c in manifest["cases"] if c["kind"] == "tests"], outcomes)
    cases.append(live_check())
    inherited = evaluate_cases(
        json.loads((ROOT / "docs/evaluation/lab2-stage6-cases.json").read_text())["cases"], outcomes
    )
    sources = {**source_checks(), **preserved_sources()}
    automated = (
        all(c["exit_code"] == 0 for c in checks)
        and bool(outcomes)
        and all(c["status"] == "passed" for c in cases if c["id"] != "routing_experiment")
        and all(c["status"] == "passed" for c in inherited)
        and all(sources.values())
    )
    ready = automated and all(c["status"] == "passed" for c in cases)
    status = (
        "passed"
        if ready
        else "incomplete"
        if automated and cases[-1]["status"] == "incomplete"
        else "not_passed"
    )
    report = {
        "kind": "lab3_final_acceptance",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": status,
        "automated_status": "passed" if automated else "not_passed",
        "release_ready": ready,
        "checks": checks,
        "suite": {
            s: sum(v == s for v in outcomes.values()) for s in ("passed", "failed", "skipped")
        },
        "criteria_passed": sum(c["status"] == "passed" for c in cases),
        "criteria_total": len(cases),
        "cases": cases,
        "inherited_cases": inherited,
        "source_checks": sources,
        "runtime_source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "app").rglob("*"))
            if p.suffix in (".py", ".sql")
        },
        "boundaries": {
            "real_llm": "Only the separately registered experiment, never pytest.",
            "telegram_http": False,
            "postgresql": "Real isolated databases and restart tests",
            "mcp": "Real stdio/SDK plus fake HTTP where controlled failures are necessary",
            "tag": "Readiness gate; annotated lab3 is created only after release_ready=true.",
        },
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Lab3: {status}; {report['suite']}; criteria {report['criteria_passed']}/{len(cases)}")
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
