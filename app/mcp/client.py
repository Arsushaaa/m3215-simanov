"""Постоянный MCP stdio-канал с одним владельцем жизненного цикла SDK.

Одна прикладная операция вызывает один session-запрос: высокоуровневые драйверы
SDK с повторами HEADER_MISMATCH/InputRequired и кешированием не используются.
"""

import asyncio
import json
import logging
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as SchemaError
from mcp import Client, StdioServerParameters, types
from mcp.shared.exceptions import MCPError
from pydantic import ValidationError

from app.agent_repository import valid_capability_context
from app.config import MCPSettings, Settings
from app.mcp.contracts import (
    ReminderArguments,
    ReminderResult,
    ScheduleArguments,
    ScheduleResult,
    WeatherArguments,
    WeatherResult,
    WeekResult,
    tool_schemas,
)

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
WEEK_URI = "schedule://current-week"
MAX_RESULT_BYTES = 32 * 1024
MODELS = {
    "get_weather": (WeatherArguments, WeatherResult),
    "get_schedule": (ScheduleArguments, ScheduleResult),
    "add_reminder": (ReminderArguments, ReminderResult),
}
MESSAGES = {
    "MCP_UNAVAILABLE": "Инструменты сейчас недоступны. Попробуй позже.",
    "UNKNOWN_TOOL": "Такой инструмент не разрешён.",
    "INVALID_ARGUMENTS": "Аргументы инструмента не соответствуют допустимому формату.",
    "INVALID_RESULT": "Инструмент вернул некорректный результат.",
    "TIMEOUT": "Инструмент не успел ответить. Попробуй позже.",
    "UNAUTHORIZED": "Нет действующего права на запись напоминания.",
    "CONFIRMATION_REQUIRED": "Создание напоминания требует подтверждения.",
    "ACTION_CANCELLED": "Это подтверждение отменено.",
    "ACTION_EXPIRED": "Срок подтверждения истёк. Подготовь напоминание заново.",
    "ACTION_CONFLICT": "Параметры действия изменились. Нужно новое подтверждение.",
    "TIMEZONE_REQUIRED": "Сначала задай часовой пояс через /timezone.",
    "AMBIGUOUS_TIME": "Это локальное время неоднозначно или не существует. Уточни время.",
    "AMBIGUOUS_PLACE": (
        "Сервис нашёл несколько одноимённых населённых пунктов с одинаковыми страной и регионом. "
        "Уточни название и район; выбрать точное место по этим данным нельзя."
    ),
    "UPSTREAM_ERROR": "Внешний сервис сейчас недоступен. Попробуй позже.",
    "SOURCE_UNAVAILABLE": "Источник данных сейчас недоступен. Попробуй позже.",
    "CALL_LIMIT": "Для одного сообщения разрешено только одно действие.",
    "OUTCOME_UNKNOWN": "Результат записи неизвестен. Проверь состояние того же подтверждения.",
}


class MCPClientError(Exception):
    def __init__(self, code: str, user_message: str | None = None):
        self.code = code
        self.user_message = user_message or MESSAGES.get(code, MESSAGES["MCP_UNAVAILABLE"])
        super().__init__(self.user_message)


def _bounded(value) -> None:
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(raw.encode("utf-8")) > MAX_RESULT_BYTES:
        raise ValueError("MCP result exceeds limit")


def _local_references(value) -> None:
    if isinstance(value, dict):
        for name, item in value.items():
            if name in {"$ref", "$dynamicRef", "$recursiveRef"} and (
                not isinstance(item, str) or not item.startswith("#")
            ):
                raise ValueError("External schema reference")
            _local_references(item)
    elif isinstance(value, list):
        for item in value:
            _local_references(item)


def _unique_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON key")
        value[key] = item
    return value


@dataclass(eq=False)
class _Operation:
    method: str
    target: str
    arguments: dict | None = field(repr=False)
    future: asyncio.Future = field(repr=False)
    task: asyncio.Task | None = field(default=None, repr=False)
    action_context: dict | None = field(default=None, repr=False)

    @property
    def writing(self) -> bool:
        return self.method == "tools/call" and self.target == "add_reminder"


class MCPClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.available = False
        self._tools: dict = {}
        self._resources: dict = {}
        self._validators: dict = {}
        self._queue: asyncio.Queue[_Operation | None] = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        self._ready: asyncio.Future | None = None
        self._closing = False

    @property
    def tools(self) -> dict:
        return deepcopy(self._tools)

    @property
    def resources(self) -> dict:
        return deepcopy(self._resources)

    def _parameters(self) -> StdioServerParameters:
        server = MCPSettings(
            postgres_password=self.settings.postgres_password,
            postgres_host=self.settings.postgres_host,
            postgres_port=self.settings.postgres_port,
            postgres_db=self.settings.postgres_db,
            postgres_user=self.settings.postgres_user,
            schedule_file=self.settings.schedule_file,
            schedule_timezone=self.settings.schedule_timezone,
            mcp_timeout_seconds=self.settings.mcp_timeout_seconds,
        )
        return StdioServerParameters(
            command=sys.executable,
            args=["-m", "app.mcp.server"],
            cwd=ROOT,
            env=server.environment(),
        )

    async def start(self) -> None:
        if self._closing:
            return
        if self._worker is None:
            self._ready = asyncio.get_running_loop().create_future()
            self._worker = asyncio.create_task(self._run(), name="labbi-mcp-client")
        try:
            async with asyncio.timeout(self.settings.mcp_timeout_seconds):
                await asyncio.shield(self._ready)
        except TimeoutError:
            self._invalidate()
            self._closing = True
            self._worker.cancel()
            logger.warning("MCP startup status=TIMEOUT")

    async def close(self) -> None:
        self._closing = True
        self._invalidate()
        if self._worker is None:
            return
        if self._ready is not None and not self._ready.done():
            # Повторный native cancel мог бы прервать shielded cleanup SDK.
            if not self._worker.cancelling():
                self._worker.cancel()
        else:
            self._queue.put_nowait(None)
        # Только worker выходит из SDK-context: task-владелец AnyIO scope не меняется.
        try:
            await asyncio.shield(self._worker)
        except asyncio.CancelledError:
            if asyncio.current_task().cancelling():
                raise
        finally:
            if self._ready is not None and not self._ready.done() and self._worker.done():
                self._ready.set_result(None)

    def _invalidate(self) -> None:
        self.available = False
        self._tools.clear()
        self._resources.clear()

    async def _discover(self, sdk) -> None:
        tools = await sdk.session.list_tools()
        resources = await sdk.session.list_resources()
        _bounded(tools.model_dump(mode="json", by_alias=True, exclude_none=True))
        _bounded(resources.model_dump(mode="json", by_alias=True, exclude_none=True))
        if tools.next_cursor or resources.next_cursor:
            raise ValueError("Unexpected paginated catalog")
        if len(tools.tools) != 3 or {tool.name for tool in tools.tools} != set(MODELS):
            raise ValueError("Unexpected tools")
        if len(resources.resources) != 1 or resources.resources[0].uri != WEEK_URI:
            raise ValueError("Unexpected resources")
        catalog, validators = {}, {}
        for tool in tools.tools:
            expected = tool_schemas(tool.name)
            for schema in (tool.input_schema, tool.output_schema):
                _local_references(schema)
                Draft202012Validator.check_schema(schema)
            if (
                tool.input_schema != expected["inputSchema"]
                or tool.output_schema != expected["outputSchema"]
            ):
                raise ValueError("Unexpected schema")
            if (
                not tool.description
                or len(tool.description) > 500
                or tool.annotations is None
                or tool.annotations.read_only_hint is not (tool.name != "add_reminder")
            ):
                raise ValueError("Invalid tool metadata")
            catalog[tool.name] = {
                "name": tool.name,
                "description": tool.description,
                **expected,
                "annotations": tool.annotations.model_dump(
                    mode="json", by_alias=True, exclude_none=True
                ),
            }
            validators[tool.name] = (
                Draft202012Validator(tool.input_schema),
                Draft202012Validator(tool.output_schema),
            )
        resource = resources.resources[0]
        if resource.mime_type != "application/json":
            raise ValueError("Unexpected resource MIME type")
        self._tools = catalog
        self._validators = validators
        self._resources = {
            WEEK_URI: {
                "uri": WEEK_URI,
                "name": resource.name,
                "description": resource.description or "Расписание текущей недели.",
                "mimeType": resource.mime_type,
            }
        }

    async def _run(self) -> None:
        active: set[asyncio.Task] = set()
        try:
            # Нет hooks для аргументов/metadata, SDK-cache и драйверы повторов отключены.
            # SDK 2.3 outbound spans содержат только method/request ID; exporter не настраивается.
            async with Client(
                self._parameters(),
                cache=None,
                read_timeout_seconds=self.settings.mcp_timeout_seconds,
            ) as sdk:
                await self._discover(sdk)
                if self._closing:
                    return
                self.available = True
                if not self._ready.done():
                    self._ready.set_result(None)
                try:
                    while (operation := await self._queue.get()) is not None:
                        if operation.future.cancelled():
                            continue
                        task = asyncio.create_task(self._execute(sdk, operation))
                        operation.task = task
                        active.add(task)
                        task.add_done_callback(active.discard)
                finally:
                    for task in active:
                        task.cancel()
                    if active:
                        await asyncio.gather(*active, return_exceptions=True)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("MCP connection status=FAILED error_type=%s", type(exc).__name__)
        finally:
            self._invalidate()
            if self._ready is not None and not self._ready.done():
                self._ready.set_result(None)
            while not self._queue.empty():
                operation = self._queue.get_nowait()
                if operation is not None and not operation.future.done():
                    operation.future.set_exception(MCPClientError("MCP_UNAVAILABLE"))

    async def _execute(self, sdk, operation: _Operation) -> None:
        try:
            if operation.method == "tools/call":
                metadata = (
                    {"meta": {"labbi/action-capability": operation.action_context}}
                    if operation.action_context is not None
                    else {}
                )
                result = await sdk.session.call_tool(
                    operation.target,
                    operation.arguments,
                    allow_input_required=True,
                    **metadata,
                )
                value = self._tool_value(operation.target, result, operation.arguments)
            else:
                result = await sdk.session.read_resource(
                    operation.target, allow_input_required=True
                )
                value = self._resource_value(result)
        except asyncio.CancelledError:
            if not operation.future.done():
                operation.future.set_exception(
                    MCPClientError("OUTCOME_UNKNOWN" if operation.writing else "MCP_UNAVAILABLE")
                )
            raise
        except MCPClientError as exc:
            if not operation.future.done():
                operation.future.set_exception(exc)
        except MCPError as exc:
            codes = {
                types.REQUEST_TIMEOUT: "TIMEOUT",
                types.INVALID_PARAMS: "INVALID_ARGUMENTS",
                types.METHOD_NOT_FOUND: "UNKNOWN_TOOL",
                -32002: "SOURCE_UNAVAILABLE",
            }
            code = codes.get(exc.code, "MCP_UNAVAILABLE")
            if operation.writing and code in {"TIMEOUT", "MCP_UNAVAILABLE"}:
                code = "OUTCOME_UNKNOWN"
            if code == "MCP_UNAVAILABLE" or (
                code == "OUTCOME_UNKNOWN" and exc.code != types.REQUEST_TIMEOUT
            ):
                self._invalidate()
                self._queue.put_nowait(None)
            if not operation.future.done():
                operation.future.set_exception(MCPClientError(code))
        except (ValidationError, SchemaError, ValueError, TypeError, RuntimeError, RecursionError):
            if not operation.future.done():
                operation.future.set_exception(
                    MCPClientError("OUTCOME_UNKNOWN" if operation.writing else "INVALID_RESULT")
                )
        except Exception:
            self._invalidate()
            self._queue.put_nowait(None)
            if not operation.future.done():
                operation.future.set_exception(
                    MCPClientError("OUTCOME_UNKNOWN" if operation.writing else "MCP_UNAVAILABLE")
                )
        else:
            if not operation.future.done():
                operation.future.set_result(value)

    def _tool_value(self, name: str, result, arguments: dict) -> dict:
        if not isinstance(result, types.CallToolResult):
            raise ValueError("Unsupported MCP result")
        _bounded(result.model_dump(mode="json", by_alias=True, exclude_none=True))
        value = MODELS[name][1].model_validate(result.structured_content).model_dump(mode="json")
        self._validators[name][1].validate(value)
        if result.is_error is not (value["status"] == "error"):
            raise ValueError("Mismatched error flag")
        # A valid schema does not prove that a response belongs to this request.
        # Reject a different day before it can become a fact in the final answer;
        # a mismatched write receipt has an unknown outcome and is reconciled in DB.
        if value["status"] != "error":
            if name == "get_schedule" and value["date"] != arguments["date"]:
                raise ValueError("Mismatched schedule date")
            if name == "add_reminder" and datetime.fromisoformat(
                value["remind_at"]
            ) != datetime.fromisoformat(arguments["remind_at"]):
                raise ValueError("Mismatched reminder moment")
        return self._domain_result(value)

    def _resource_value(self, result) -> dict:
        if not isinstance(result, types.ReadResourceResult):
            raise ValueError("Unsupported resource result")
        _bounded(result.model_dump(mode="json", by_alias=True, exclude_none=True))
        if len(result.contents) != 1 or not isinstance(
            result.contents[0], types.TextResourceContents
        ):
            raise ValueError("Unexpected resource contents")
        content = result.contents[0]
        if content.uri != WEEK_URI or content.mime_type != "application/json":
            raise ValueError("Unexpected resource identity")
        raw = json.loads(content.text, object_pairs_hook=_unique_keys)
        value = WeekResult.model_validate(raw).model_dump(mode="json")
        return self._domain_result(value)

    @staticmethod
    def _domain_result(value: dict) -> dict:
        if value.get("status") == "error":
            raise MCPClientError(value["error"]["code"])
        return value

    async def _request(
        self,
        method: str,
        target: str,
        arguments: dict | None = None,
        *,
        action_context: dict | None = None,
    ) -> dict:
        if not self.available or self._closing:
            raise MCPClientError("MCP_UNAVAILABLE")
        future = asyncio.get_running_loop().create_future()
        operation = _Operation(method, target, arguments, future, action_context=action_context)
        self._queue.put_nowait(operation)
        try:
            async with asyncio.timeout(self.settings.mcp_timeout_seconds):
                return await future
        except TimeoutError:
            raise MCPClientError("OUTCOME_UNKNOWN" if operation.writing else "TIMEOUT") from None
        finally:
            if future.cancelled() and operation.task is not None:
                operation.task.cancel()

    async def call_tool(self, name: str, args: dict, *, action_context: dict | None = None) -> dict:
        if name not in MODELS:
            raise MCPClientError("UNKNOWN_TOOL")
        if name == "add_reminder":
            if not valid_capability_context(action_context):
                raise MCPClientError("UNAUTHORIZED")
        elif action_context is not None:
            raise MCPClientError("INVALID_ARGUMENTS")
        if not self.available:
            raise MCPClientError("MCP_UNAVAILABLE")
        try:
            arguments = MODELS[name][0].model_validate(args).model_dump(mode="json")
            _bounded(arguments)
            self._validators[name][0].validate(arguments)
        except (ValidationError, SchemaError, ValueError, TypeError, RecursionError):
            raise MCPClientError("INVALID_ARGUMENTS") from None
        return await self._request(
            "tools/call", name, arguments, action_context=deepcopy(action_context)
        )

    async def read_resource(self, uri: str) -> dict:
        if uri != WEEK_URI:
            raise MCPClientError("UNKNOWN_TOOL")
        return await self._request("resources/read", uri)
