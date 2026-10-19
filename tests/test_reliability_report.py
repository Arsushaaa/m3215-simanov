"""Проверка отчёта не засчитывает пропущенный тест или частично зелёную параметризацию."""

import pytest

from scripts.lab2_reliability import evaluate_cases, junit_outcomes, source_checks
from scripts.lab2_reliability import test_environment as reliability_environment


@pytest.mark.parametrize(
    "outcomes,expected",
    [
        ({}, "not_passed"),
        ({"tests/test_tool.py::test_case[ok]": "passed"}, "passed"),
        (
            {
                "tests/test_tool.py::test_case[ok]": "passed",
                "tests/test_tool.py::test_case[failure]": "failed",
            },
            "not_passed",
        ),
        ({"tests/test_tool.py::test_case[skip]": "skipped"}, "not_passed"),
        ({"tests/test_tool.py::test_case_extra": "passed"}, "not_passed"),
    ],
)
def test_case_requires_exact_name_and_every_collected_parameter_to_pass(outcomes, expected):
    cases = [{"id": "test", "tests": ["tests/test_tool.py::test_case"]}]
    assert evaluate_cases(cases, outcomes)[0]["status"] == expected


def test_junit_preserves_failure_even_if_duplicate_later_passes(tmp_path):
    path = tmp_path / "junit.xml"
    path.write_text(
        "<testsuites><testsuite>"
        '<testcase classname="tests.test_tool" name="test_case"><failure/></testcase>'
        '<testcase classname="tests.test_tool" name="test_case"/>'
        "</testsuite></testsuites>"
    )
    assert junit_outcomes(path) == {"tests/test_tool.py::test_case": "failed"}


def test_baseline_migrations_tag_and_model_dataset_are_preserved():
    assert all(source_checks().values())


def test_external_pytest_selection_cannot_make_a_partial_run_look_complete(monkeypatch):
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k only_one_test")
    monkeypatch.setenv("PYTEST_PLUGINS", "external_plugin")
    env = reliability_environment()
    assert "PYTEST_ADDOPTS" not in env and "PYTEST_PLUGINS" not in env
    assert env["RUN_INTEGRATION"] == "1"
