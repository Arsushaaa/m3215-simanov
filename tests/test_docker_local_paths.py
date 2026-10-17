"""Запуск через абсолютный симлинк использует физические пути RootlessKit."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def linked_project(tmp_path):
    project = tmp_path / "Каталог с пробелами"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "scripts"
    for name in ("env.sh", "docker-local.sh"):
        shutil.copyfile(source / name, scripts / name)
    activate = project / ".venv/bin/activate"
    activate.parent.mkdir(parents=True)
    activate.write_text(":\n")
    binaries = project / ".tools/docker/bin"
    binaries.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(project, target_is_directory=True)
    return project, alias, binaries


def executable(path, script):
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + script)
    path.chmod(0o755)


def test_env_from_symlink_configures_physical_docker_paths(linked_project):
    project, alias, binaries = linked_project
    executable(binaries / "docker", "exit 0\n")
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1/scripts/env.sh"; '
            'printf "%s\\n" "$DOCKER_HOST" "$DOCKER_CONFIG" "$PIP_CACHE_DIR"',
            "env-test",
            str(alias),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.stdout.splitlines() == [
        f"unix://{project}/.tools/docker-run/docker.sock",
        f"{project}/.tools/docker-config",
        f"{project}/.tools/pip-cache",
    ]


@pytest.mark.parametrize("action", ["run", "start"])
def test_rootless_launch_from_symlink_receives_only_physical_paths(linked_project, action):
    project, alias, binaries = linked_project
    trace = project / "launch.txt"
    executable(binaries / "docker", '[[ "$1" == info && -f "$LABBI_TEST_LAUNCH" ]]\n')
    executable(
        binaries / "dockerd-rootless.sh",
        'printf "%s\\n" "$DOCKER_HOST" "$XDG_RUNTIME_DIR" "$DOCKER_CONFIG" "$@" '
        '>"$LABBI_TEST_LAUNCH"\n',
    )
    subprocess.run(
        ["bash", str(alias / "scripts/docker-local.sh"), action],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
        env={**os.environ, "LABBI_TEST_LAUNCH": str(trace)},
    )
    actual = trace.read_text().splitlines()
    runtime = project / ".tools/docker-run"
    assert actual == [
        f"unix://{runtime}/docker.sock",
        str(runtime),
        str(project / ".tools/docker-config"),
        "--host",
        f"unix://{runtime}/docker.sock",
        "--data-root",
        str(project / ".tools/docker-data"),
        "--exec-root",
        str(project / ".tools/docker-exec"),
        "--pidfile",
        str(runtime / "dockerd.pid"),
    ]
