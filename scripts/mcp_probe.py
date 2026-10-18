"""Независимый stdio MCP-клиент: discovery, расписание и отказ записи без права.

Погода, Telegram и модель не вызываются. Настройки дочернего процесса ограничены
серверным whitelist; личные поля расписания и секреты не выводятся в отчёт.
"""

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from mcp import Client, StdioServerParameters

from app.config import ConfigError, MCPSettings
from app.mcp.contracts import ReminderResult, ScheduleResult, WeekResult

ROOT = Path(__file__).resolve().parents[1]


class ProbeError(ValueError):
    """Безопасная ошибка проверки без содержимого протокола и окружения."""


async def probe(settings: MCPSettings, *, day: str = "2026-10-12") -> dict:
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp.server"],
        cwd=ROOT,
        env=settings.environment(),
    )
    async with Client(parameters, read_timeout_seconds=settings.mcp_timeout_seconds + 5) as client:
        tools = (await client.list_tools()).tools
        resources = (await client.list_resources()).resources
        if {tool.name for tool in tools} != {"get_weather", "get_schedule", "add_reminder"}:
            raise ProbeError("MCP не обнаружил три ожидаемых инструмента.")
        if any(not tool.input_schema or not tool.output_schema for tool in tools):
            raise ProbeError("Инструмент не опубликовал входную или выходную схему.")
        if {resource.uri for resource in resources} != {"schedule://current-week"}:
            raise ProbeError("MCP не обнаружил ресурс текущей недели.")

        schedule_result = await client.call_tool("get_schedule", {"date": day})
        schedule = ScheduleResult.model_validate(schedule_result.structured_content).root
        if schedule_result.is_error or schedule.status == "error":
            raise ProbeError("Чтение учебного расписания завершилось ошибкой.")
        resource_result = await client.read_resource("schedule://current-week")
        if len(resource_result.contents) != 1:
            raise ProbeError("Недельный ресурс имеет неожиданную структуру.")
        content = resource_result.contents[0]
        if not hasattr(content, "text") or content.mime_type != "application/json":
            raise ProbeError("Недельный ресурс не содержит JSON-текст.")
        week = WeekResult.model_validate_json(content.text).root
        if getattr(week, "status", None) == "error":
            raise ProbeError("Чтение недельного ресурса завершилось ошибкой.")

        # Права записи не выпускаются: сервер обязан отказать до обращения к БД.
        denied_result = await client.call_tool(
            "add_reminder",
            {
                "text": "Проверка независимого MCP-клиента",
                "remind_at": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
            },
        )
        denied = ReminderResult.model_validate(denied_result.structured_content).root
        if (
            not denied_result.is_error
            or denied.status != "error"
            or denied.error.code != "UNAUTHORIZED"
        ):
            raise ProbeError("Независимый клиент не получил ожидаемый отказ записи.")

        return {
            "transport": "stdio",
            "tools": [
                {
                    "name": tool.name,
                    "input_fields": sorted(tool.input_schema.get("properties", {})),
                    "output_schema": bool(tool.output_schema),
                    "read_only": tool.annotations.read_only_hint if tool.annotations else None,
                }
                for tool in tools
            ],
            "resources": [resource.uri for resource in resources],
            "schedule": {
                "date": schedule.date.isoformat(),
                "status": schedule.status,
                "lessons": len(schedule.lessons) if schedule.status == "ok" else None,
            },
            "week": {
                "start": week.week_start.isoformat(),
                "end": week.week_end.isoformat(),
                "timezone": week.timezone,
                "days": len(week.days),
            },
            "write_without_context": {
                "is_error": denied_result.is_error,
                "code": denied.error.code,
            },
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Независимая проверка MCP через stdio")
    parser.add_argument("--env-file", type=Path, help="Необязательные серверные настройки")
    parser.add_argument("--date", default="2026-10-12", help="Дата учебного расписания YYYY-MM-DD")
    args = parser.parse_args()
    try:
        result = asyncio.run(probe(MCPSettings.load(args.env_file), day=args.date))
    except (ConfigError, ProbeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"MCP-проверка не завершена: {type(exc).__name__}.", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
