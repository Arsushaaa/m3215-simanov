"""Собственный MCP stdio-сервер. Telegram и LLM подключаются на следующих этапах."""

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import aiohttp
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError
from mcp.types import (
    CallToolResult,
    ListResourcesResult,
    ListToolsResult,
    ReadResourceResult,
    Resource,
    TextContent,
    TextResourceContents,
    Tool,
    ToolAnnotations,
)
from pydantic import ValidationError

from app.agent_repository import ReminderRepository, valid_capability_context
from app.config import ConfigError, MCPSettings
from app.db import create_pool
from app.mcp.contracts import (
    ErrorDetails,
    ErrorResult,
    ReminderArguments,
    ReminderResult,
    ScheduleArguments,
    ScheduleResult,
    WeatherArguments,
    WeatherResult,
    WeekResult,
    tool_schemas,
)
from app.mcp.schedule import ScheduleService, ScheduleSourceError

logger = logging.getLogger(__name__)
WEEK_URI = "schedule://current-week"
CAPABILITY_KEY = "labbi/action-capability"
MAX_RESULT_BYTES = 32 * 1024
TOOLS = {
    "get_weather": (
        WeatherArguments,
        WeatherResult,
        "Текущая погода в городе. При одноимённых городах нужно уточнение страны/региона. "
        "Это наблюдение, а не прогноз на весь день.",
    ),
    "get_schedule": (
        ScheduleArguments,
        ScheduleResult,
        "Учебные занятия на календарную дату YYYY-MM-DD в зоне расписания. "
        "Вне покрытия данных нет; пустой список внутри покрытия означает свободный день.",
    ),
    "add_reminder": (
        ReminderArguments,
        ReminderResult,
        "Сохранить напоминание только по уже подтверждённому действию владельца. "
        "До вызова приложение обязано показать параметры и получить подтверждение. "
        "Сам инструмент не рассылает уведомления.",
    ),
}


def _error(code, message, *, retryable=False):
    return ErrorResult(
        status="error", error=ErrorDetails(code=code, message=message, retryable=retryable)
    ).model_dump(mode="json")


def _json(value):
    text = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(text.encode("utf-8")) > MAX_RESULT_BYTES:
        raise ValueError("MCP payload exceeds limit")
    return text


def _tool_result(value):
    try:
        text = _json(value)
        result = CallToolResult(
            content=[TextContent(type="text", text=text)],
            structuredContent=value,
            isError=value.get("status") == "error",
        )
        _json(result.model_dump(mode="json", by_alias=True, exclude_none=True))
        return result
    except (ValueError, TypeError, RecursionError):
        value = _error("INVALID_RESULT", "Результат инструмента превышает допустимый размер.")
        text = _json(value)
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structuredContent=value,
        isError=value.get("status") == "error",
    )


class _Runtime:
    def __init__(self, settings, weather, schedule, reminders):
        self.settings = settings
        self.weather, self.schedule, self.reminders = weather, schedule, reminders
        self.pool = None
        self.session = None
        self.pool_lock = asyncio.Lock()

    async def get_weather(self):
        if self.weather is None:
            from app.mcp.weather import WeatherService

            if self.session is None:
                self.session = aiohttp.ClientSession(trust_env=False)
            self.weather = WeatherService(
                self.session, timeout_seconds=self.settings.mcp_timeout_seconds
            )
        return self.weather

    async def get_schedule(self):
        if self.schedule is None:
            self.schedule = await asyncio.to_thread(
                ScheduleService.from_file,
                self.settings.schedule_file,
                self.settings.schedule_timezone,
            )
        return self.schedule

    async def get_reminders(self):
        async with self.pool_lock:
            if self.reminders is None:
                self.pool = await create_pool(self.settings)
                self.reminders = ReminderRepository(self.pool)
        return self.reminders

    async def close(self):
        if self.session is not None:
            await self.session.close()
        if self.pool is not None:
            try:
                async with asyncio.timeout(5):
                    await self.pool.close()
            except TimeoutError:
                self.pool.terminate()


def create_server(settings: MCPSettings, *, weather=None, schedule=None, reminders=None, now=None):
    """Транспорт и прикладные границы можно проверить независимым SDK-клиентом."""
    clock = now or (lambda: datetime.now(UTC))

    @asynccontextmanager
    async def lifespan(_server):
        runtime = _Runtime(settings, weather, schedule, reminders)
        try:
            yield runtime
        finally:
            await runtime.close()

    async def list_tools(_ctx, _params):
        return ListToolsResult(
            tools=[
                Tool(
                    name=name,
                    description=definition[2],
                    **tool_schemas(name),
                    annotations=ToolAnnotations(
                        readOnlyHint=name != "add_reminder",
                        destructiveHint=False,
                        idempotentHint=True,
                        openWorldHint=name == "get_weather",
                    ),
                )
                for name, definition in TOOLS.items()
            ]
        )

    async def call_tool(ctx, params):
        started, request_id = time.monotonic(), uuid.uuid4().hex
        definition = TOOLS.get(params.name)
        if definition is None:
            return _tool_result(_error("UNKNOWN_TOOL", "Такой инструмент не разрешён."))
        try:
            _json(params.arguments)
            arguments = definition[0].model_validate(params.arguments)
        except (ValidationError, ValueError, TypeError, RecursionError):
            return _tool_result(_error("INVALID_ARGUMENTS", "Аргументы не соответствуют схеме."))

        capability = (params.meta or {}).get(CAPABILITY_KEY)
        if params.name == "add_reminder" and not valid_capability_context(capability):
            return _tool_result(
                _error("UNAUTHORIZED", "Нет действующего права на запись напоминания.")
            )
        try:
            async with asyncio.timeout(settings.mcp_timeout_seconds):
                runtime = ctx.lifespan_context
                if params.name == "get_weather":
                    result = await (await runtime.get_weather()).get_weather(arguments.city)
                elif params.name == "get_schedule":
                    result = (await runtime.get_schedule()).get_schedule(arguments.date)
                else:
                    result = await (await runtime.get_reminders()).add_reminder(
                        arguments, capability
                    )
                raw = result.model_dump(mode="json") if hasattr(result, "model_dump") else result
                raw = definition[1].model_validate(raw).model_dump(mode="json")
        except TimeoutError:
            code = "OUTCOME_UNKNOWN" if params.name == "add_reminder" else "TIMEOUT"
            message = (
                "Результат записи неизвестен. Проверь состояние того же подтверждения."
                if params.name == "add_reminder"
                else "Инструмент не успел ответить. Попробуй позже."
            )
            raw = _error(code, message, retryable=True)
        except ScheduleSourceError:
            raw = _error("SOURCE_UNAVAILABLE", "Источник расписания недоступен или некорректен.")
        except (ValidationError, ValueError, TypeError, RecursionError):
            if params.name == "add_reminder":
                raw = _error(
                    "OUTCOME_UNKNOWN",
                    "Результат записи требует проверки того же подтверждения.",
                    retryable=True,
                )
            else:
                raw = _error("INVALID_RESULT", "Инструмент вернул некорректный результат.")
        except Exception:
            code = "OUTCOME_UNKNOWN" if params.name == "add_reminder" else "SOURCE_UNAVAILABLE"
            raw = _error(
                code,
                "Сервис инструмента сейчас недоступен. Проверь результат позже.",
                retryable=True,
            )
        result = _tool_result(raw)
        # Neither arguments, user identifiers, capabilities nor exception bodies are logged.
        logger.info(
            "request=%s tool=%s status=%s duration_ms=%d",
            request_id,
            params.name,
            "error" if result.is_error else "ok",
            round((time.monotonic() - started) * 1000),
        )
        return result

    async def list_resources(_ctx, _params):
        return ListResourcesResult(
            resources=[
                Resource(
                    uri=WEEK_URI,
                    name="current-week",
                    description="Учебное расписание на текущую неделю.",
                    mimeType="application/json",
                )
            ]
        )

    async def read_resource(ctx, params):
        if str(params.uri) != WEEK_URI:
            raise MCPError(-32002, "Ресурс не найден.")
        try:
            async with asyncio.timeout(settings.mcp_timeout_seconds):
                service = await ctx.lifespan_context.get_schedule()
                raw = WeekResult.model_validate(
                    service.current_week(now=clock()).model_dump(mode="json")
                ).model_dump(mode="json")
                text = _json(raw)
        except TimeoutError:
            text = _json(
                _error("TIMEOUT", "Не удалось получить расписание вовремя.", retryable=True)
            )
        except ScheduleSourceError:
            text = _json(
                _error("SOURCE_UNAVAILABLE", "Источник расписания недоступен или некорректен.")
            )
        except (ValidationError, ValueError, TypeError, RecursionError):
            text = _json(_error("INVALID_RESULT", "Ресурс вернул некорректный результат."))
        except Exception:
            text = _json(_error("SOURCE_UNAVAILABLE", "Ресурс сейчас недоступен.", retryable=True))
        result = ReadResourceResult(
            contents=[TextResourceContents(uri=WEEK_URI, mimeType="application/json", text=text)]
        )
        try:
            _json(result.model_dump(mode="json", by_alias=True, exclude_none=True))
        except (ValueError, TypeError, RecursionError):
            result.contents[0].text = _json(
                _error("INVALID_RESULT", "Результат ресурса превышает допустимый размер.")
            )
        return result

    server = Server(
        "labbi-tools",
        version="0.1.0",
        lifespan=lifespan,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
        on_list_resources=list_resources,
        on_read_resource=read_resource,
        get_tool_input_schema=lambda name: (
            tool_schemas(name)["inputSchema"] if name in TOOLS else None
        ),
    )
    # Avoid third-party tracing of tool metadata and arguments in this local server.
    server.middleware.clear()
    return server


def _configure_logging():
    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(lambda record: record.name in {__name__, "app.mcp.weather"})
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)


async def run(settings: MCPSettings):
    server = create_server(settings)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file", default=None, help="Отдельный env сервера; по умолчанию только env"
    )
    args = parser.parse_args()
    _configure_logging()
    try:
        asyncio.run(run(MCPSettings.load(args.env_file)))
    except KeyboardInterrupt:
        return 0
    except ConfigError as error:
        logger.error("%s", error)
        return 1
    except Exception as error:
        logger.error("MCP-сервер остановлен: %s.", type(error).__name__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
