"""Асинхронный OpenRouter-клиент без платных моделей и скрытых повторов."""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field

import aiohttp

from app.config import Settings

logger = logging.getLogger(__name__)
MAX_RESPONSE_BYTES = 256 * 1024
MAX_TOOL_ARGUMENT_BYTES = 32 * 1024
MAX_TOOL_CALLS = 16
REQUEST_REJECTED_MESSAGE = "Провайдер модели отклонил этот запрос. Ответ не получен."


@dataclass(frozen=True)
class Completion:
    text: str = field(repr=False)
    truncated: bool = False


@dataclass(frozen=True)
class RoutingCompletion:
    text: str | None = field(repr=False)
    tool_calls: list[dict] = field(repr=False)
    truncated: bool = False


class ModelError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.user_message = message


class OpenRouterClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.settings.llm_timeout_seconds, sock_connect=10),
            headers={"Authorization": "Bearer " + self.settings.openrouter_api_key},
        )

    async def close(self) -> None:
        if self.session is not None:
            await self.session.close()

    def _payload(self, messages: list[dict], temperature: float) -> dict:
        return {
            "model": self.settings.openrouter_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.settings.llm_max_output_tokens,
            "reasoning": {"enabled": False},
            "provider": {"max_price": {"prompt": 0, "completion": 0}},
        }

    async def complete(
        self,
        messages: list[dict],
        temperature: float,
        *,
        disable_tools: bool = False,
        tools: list[dict] | None = None,
    ) -> Completion:
        if tools is not None and not disable_tools:
            raise ValueError("Определения инструментов здесь нужны только для запрещённого финала.")
        payload = self._payload(messages, temperature)
        if disable_tools:
            # Structured-финал не требует неподдерживаемого некоторыми endpoint tool_choice.
            payload["tools"] = tools if tools is not None else []
            if tools is not None:
                # Native-история сохраняет исходные схемы, но новые calls запрещены.
                payload["tool_choice"] = "none"
                payload["provider"]["require_parameters"] = True
        result = await self._request(payload)
        if result.tool_calls:
            raise ModelError("Модель предложила новое действие вместо итогового ответа.")
        return Completion(result.text, truncated=result.truncated)

    async def decide(
        self,
        messages: list[dict],
        temperature: float,
        *,
        tools: list[dict] | None = None,
        decision_schema: dict | None = None,
    ) -> RoutingCompletion:
        if tools is not None and decision_schema is not None:
            raise ValueError("Нужно выбрать один режим: native tools или structured decision.")
        payload = self._payload(messages, temperature)
        if tools is not None:
            payload.update(tools=tools, tool_choice="auto", parallel_tool_calls=False)
            payload["provider"]["require_parameters"] = True
        elif decision_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "labbi_decision",
                    "strict": True,
                    "schema": decision_schema,
                },
            }
            payload["provider"]["require_parameters"] = True
        return await self._request(payload)

    def _observe_response(self, data: dict) -> None:
        """Extension point for the isolated experiment; production records no response body."""

    @staticmethod
    async def _read_json(response) -> dict:
        if response.content_length is not None and response.content_length > MAX_RESPONSE_BYTES:
            raise ModelError("Ответ модели превышает допустимый размер.")
        raw = bytearray()
        async for chunk in response.content.iter_chunked(16384):
            raw.extend(chunk)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise ModelError("Ответ модели превышает допустимый размер.")
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Ожидался объект ответа.")
        return data

    @staticmethod
    def _calls(message: dict) -> list[dict]:
        calls = message.get("tool_calls")
        if calls is None:
            return []
        if not isinstance(calls, list) or len(calls) > MAX_TOOL_CALLS:
            raise ValueError("Неверный список инструментов.")
        canonical = []
        for call in calls:
            if not isinstance(call, dict) or call.get("type") != "function":
                raise ValueError("Неверный инструмент.")
            identifier = call.get("id")
            function = call.get("function")
            if (
                not isinstance(identifier, str)
                or not identifier.strip()
                or not 1 <= len(identifier.encode("utf-8")) <= 256
            ):
                raise ValueError("Неверный идентификатор вызова.")
            if not isinstance(function, dict):
                raise ValueError("Неверный формат функции.")
            name, arguments = function.get("name"), function.get("arguments")
            if (
                not isinstance(name, str)
                or not name.strip()
                or not 1 <= len(name.encode("utf-8")) <= 128
            ):
                raise ValueError("Неверное имя функции.")
            if (
                not isinstance(arguments, str)
                or len(arguments.encode("utf-8")) > MAX_TOOL_ARGUMENT_BYTES
            ):
                raise ValueError("Неверные аргументы функции.")
            canonical.append(
                {
                    "id": identifier,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            )
        return canonical

    async def _request(self, payload: dict) -> RoutingCompletion:
        if self.session is None:
            raise RuntimeError("LLM-клиент не запущен.")
        request_id = uuid.uuid4().hex[:12]
        started = time.monotonic()
        outcome = "failed"
        http_status = None
        logger.info("LLM начало request=%s model=%s", request_id, self.settings.openrouter_model)
        try:
            async with self.session.post(
                self.settings.openrouter_base_url + "/chat/completions",
                json=payload,
                allow_redirects=False,
            ) as response:
                http_status = response.status
                if response.status == 400:
                    outcome = "rejected"
                    raise ModelError(REQUEST_REJECTED_MESSAGE)
                if response.status == 429:
                    raise ModelError("Лимит запросов к модели исчерпан. Попробуй позже.")
                if response.status in {401, 402, 403}:
                    raise ModelError("Модель недоступна: нужно проверить ключ и доступ OpenRouter.")
                if response.status != 200:
                    raise ModelError("Сервис модели сейчас недоступен. Попробуй позже.")
                try:
                    data = await self._read_json(response)
                    self._observe_response(data)
                    choice = data["choices"][0]
                    message = choice["message"]
                    if not isinstance(message, dict):
                        raise ValueError("Неверный формат сообщения.")
                    text = message.get("content")
                    if text is not None and not isinstance(text, str):
                        raise ValueError("Неверный формат текста.")
                    if isinstance(text, str):
                        text.encode("utf-8")
                    calls = self._calls(message)
                    finish_reason = choice.get("finish_reason")
                    if finish_reason not in {None, "stop", "length", "tool_calls"}:
                        raise ValueError("Неуспешная причина завершения.")
                except (ValueError, KeyError, IndexError, TypeError, RecursionError):
                    raise ModelError(
                        "Модель вернула некорректный ответ. Попробуй ещё раз."
                    ) from None
                if not calls and (not isinstance(text, str) or not text.strip()):
                    raise ModelError("Модель вернула пустой ответ. Попробуй ещё раз.")
                truncated = finish_reason == "length"
                outcome = "truncated" if truncated else "ok"
        except TimeoutError:
            raise ModelError("Модель не успела ответить. Попробуй ещё раз позже.") from None
        except aiohttp.ClientError:
            raise ModelError("Не удалось связаться с моделью. Попробуй позже.") from None
        finally:
            logger.info(
                "LLM результат request=%s status=%s http_status=%s duration=%.2f",
                request_id,
                outcome,
                http_status,
                time.monotonic() - started,
            )
        return RoutingCompletion(text, calls, truncated=truncated)
