"""Асинхронный OpenRouter-клиент без платных моделей и скрытых повторов."""

import logging
import time
import uuid
from dataclasses import dataclass, field

import aiohttp

from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Completion:
    text: str = field(repr=False)
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

    async def complete(self, messages: list[dict[str, str]], temperature: float) -> Completion:
        if self.session is None:
            raise RuntimeError("LLM-клиент не запущен.")
        request_id = uuid.uuid4().hex[:12]
        started = time.monotonic()
        outcome = "failed"
        logger.info("LLM начало request=%s model=%s", request_id, self.settings.openrouter_model)
        try:
            async with self.session.post(
                self.settings.openrouter_base_url + "/chat/completions",
                json={
                    "model": self.settings.openrouter_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": self.settings.llm_max_output_tokens,
                    "reasoning": {"enabled": False},
                    "provider": {"max_price": {"prompt": 0, "completion": 0}},
                },
                allow_redirects=False,
            ) as response:
                if response.status == 429:
                    raise ModelError("Лимит запросов к модели исчерпан. Попробуй позже.")
                if response.status in {401, 402, 403}:
                    raise ModelError("Модель недоступна: нужно проверить ключ и доступ OpenRouter.")
                if response.status != 200:
                    raise ModelError("Сервис модели сейчас недоступен. Попробуй позже.")
                try:
                    data = await response.json()
                    choice = data["choices"][0]
                    text = choice["message"]["content"]
                except (ValueError, KeyError, IndexError, TypeError):
                    raise ModelError(
                        "Модель вернула некорректный ответ. Попробуй ещё раз."
                    ) from None
                if not isinstance(text, str) or not text.strip():
                    raise ModelError("Модель вернула пустой ответ. Попробуй ещё раз.")
                truncated = choice.get("finish_reason") == "length"
                outcome = "truncated" if truncated else "ok"
        except TimeoutError:
            raise ModelError("Модель не успела ответить. Попробуй ещё раз позже.") from None
        except aiohttp.ClientError:
            raise ModelError("Не удалось связаться с моделью. Попробуй позже.") from None
        finally:
            logger.info(
                "LLM результат request=%s status=%s duration=%.2f",
                request_id,
                outcome,
                time.monotonic() - started,
            )
        return Completion(text, truncated=truncated)
