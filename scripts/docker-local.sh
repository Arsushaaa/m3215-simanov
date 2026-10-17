#!/usr/bin/env bash
# Управление изолированным Docker проекта без sudo и системных сервисов.
set -euo pipefail
# RootlessKit открывает namespace через /proc/PID/root: абсолютный симлинк
# в пути может вывести его в mount namespace родителя вместо дочернего.
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
task_runtime="$task_root/.tools/docker-run"
task_pidfile="$task_runtime/supervisor.pid"
task_log="$task_runtime/daemon.log"
export PATH="$task_root/.tools/docker/bin:$PATH"
export DOCKER_CONFIG="$task_root/.tools/docker-config"
export DOCKER_HOST="unix://$task_runtime/docker.sock"

if [[ ! -x "$task_root/.tools/docker/bin/dockerd-rootless.sh" ]]; then
    echo 'Локальные компоненты Docker не установлены.' >&2
    exit 1
fi

case "${1:-status}" in
    run)
        if docker info >/dev/null 2>&1; then
            echo 'Локальный Docker уже работает.'
            exit 0
        fi
        mkdir -p "$task_runtime" "$task_root/.tools/docker-data" "$task_root/.tools/docker-exec"
        chmod 700 "$task_runtime"
        printf '%s\n' "$$" >"$task_pidfile"
        export XDG_RUNTIME_DIR="$task_runtime"
        exec dockerd-rootless.sh \
            --host "$DOCKER_HOST" \
            --data-root "$task_root/.tools/docker-data" \
            --exec-root "$task_root/.tools/docker-exec" \
            --pidfile "$task_runtime/dockerd.pid"
        ;;
    start)
        if docker info >/dev/null 2>&1; then
            echo 'Локальный Docker уже работает.'
            exit 0
        fi
        mkdir -p "$task_runtime" "$task_root/.tools/docker-data" "$task_root/.tools/docker-exec"
        chmod 700 "$task_runtime"
        nohup env XDG_RUNTIME_DIR="$task_runtime" \
            dockerd-rootless.sh \
            --host "$DOCKER_HOST" \
            --data-root "$task_root/.tools/docker-data" \
            --exec-root "$task_root/.tools/docker-exec" \
            --pidfile "$task_runtime/dockerd.pid" \
            >"$task_log" 2>&1 </dev/null &
        task_pid=$!
        printf '%s\n' "$task_pid" >"$task_pidfile"
        for ((task_attempt = 0; task_attempt < 30; task_attempt++)); do
            if docker info >/dev/null 2>&1; then
                echo 'Локальный Docker готов.'
                exit 0
            fi
            if ! kill -0 "$task_pid" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        echo "Docker не запустился. Подробности: $task_log" >&2
        exit 1
        ;;
    stop)
        if [[ -f "$task_pidfile" ]]; then
            read -r task_pid <"$task_pidfile"
            if [[ "$task_pid" =~ ^[0-9]+$ ]] && kill -0 "$task_pid" 2>/dev/null; then
                if [[ -r "/proc/$task_pid/cmdline" ]] && \
                    tr '\0' '\n' <"/proc/$task_pid/cmdline" | \
                    grep -Fq -- "$task_root/.tools/docker/bin/dockerd-rootless.sh"; then
                    kill "$task_pid"
                else
                    echo 'PID больше не относится к Docker проекта; остановка отменена.' >&2
                    exit 1
                fi
            fi
            rm -f "$task_pidfile"
        fi
        echo 'Локальный Docker остановлен; данные сохранены.'
        ;;
    status)
        task_version="$(docker info --format '{{.ServerVersion}}')"
        printf 'Docker %s: готов\n' "$task_version"
        ;;
    *)
        echo 'Использование: bash scripts/docker-local.sh start|run|stop|status' >&2
        exit 2
        ;;
esac
