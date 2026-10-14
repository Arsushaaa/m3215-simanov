#!/usr/bin/env bash
# Выполните source scripts/env.sh из терминала для выбора окружения проекта.
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "$task_root/.venv/bin/activate" ]]; then
    echo 'Виртуальное окружение .venv ещё не подготовлено.' >&2
    return 1 2>/dev/null || exit 1
fi
source "$task_root/.venv/bin/activate"
export PIP_CACHE_DIR="$task_root/.tools/pip-cache"
if [[ -x "$task_root/.tools/docker/bin/docker" ]]; then
    export PATH="$task_root/.tools/docker/bin:$task_root/.tools/bin:$PATH"
    export DOCKER_CONFIG="$task_root/.tools/docker-config"
    export DOCKER_HOST="unix://$task_root/.tools/docker-run/docker.sock"
fi
unset task_root
