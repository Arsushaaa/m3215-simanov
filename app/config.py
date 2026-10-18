"""Единственное место чтения и проверки настроек приложения."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from aiogram.utils.token import TokenValidationError, validate_token
from dotenv import dotenv_values


class ConfigError(ValueError):
    """Ошибка настройки без секретных значений в сообщении."""


@dataclass(frozen=True)
class Settings:
    bot_token: str = field(repr=False)
    postgres_password: str = field(repr=False)
    telegram_proxy_url: str = field(default="", repr=False)
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    postgres_db: str = "bot"
    postgres_user: str = "bot"
    log_level: str = "INFO"
    health_port: int = 8080
    openrouter_api_key: str = field(default="", repr=False)
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    llm_timeout_seconds: int = 60
    llm_max_output_tokens: int = 2048
    history_max_messages: int = 20
    history_max_tokens: int = 6000
    schedule_file: str = field(default="data/schedule.json", repr=False)
    schedule_timezone: str = "Europe/Moscow"
    mcp_timeout_seconds: int = 15
    agent_timeout_seconds: int = 135
    llm_tool_mode: str = "native"

    @classmethod
    def load(
        cls,
        env_file: Path | str = ".env",
        *,
        environ: Mapping[str, str] | None = None,
        require_llm: bool = False,
    ) -> "Settings":
        values = {
            **dotenv_values(env_file, interpolate=False),
            **(os.environ if environ is None else environ),
        }

        def value(key: str, default: str = "") -> str:
            return values.get(key) or default

        def port(key: str, default: str) -> int:
            try:
                result = int(value(key, default))
                if not 1 <= result <= 65535:
                    raise ValueError
                return result
            except ValueError:
                raise ConfigError(f"{key}: нужен номер порта от 1 до 65535.") from None

        def number(key: str, default: str, minimum: int, maximum: int) -> int:
            try:
                result = int(value(key, default))
                if not minimum <= result <= maximum:
                    raise ValueError
                return result
            except ValueError:
                raise ConfigError(f"{key}: нужно целое число от {minimum} до {maximum}.") from None

        token = value("BOT_TOKEN")
        try:
            validate_token(token)
        except TokenValidationError:
            raise ConfigError("BOT_TOKEN: укажите токен, полученный у BotFather.") from None
        password = value("POSTGRES_PASSWORD")
        if not password:
            raise ConfigError("POSTGRES_PASSWORD: пароль базы данных не задан.")
        proxy = value("TELEGRAM_PROXY_URL")
        if proxy:
            try:
                parsed = urlsplit(proxy)
                if (
                    parsed.scheme not in {"http", "socks5"}
                    or not parsed.hostname
                    or not parsed.port
                    or parsed.path not in {"", "/"}
                    or parsed.query
                    or parsed.fragment
                ):
                    raise ValueError
            except ValueError:
                raise ConfigError(
                    "TELEGRAM_PROXY_URL: нужен http://host:port или socks5://host:port; "
                    "при необходимости добавьте user:password@."
                ) from None
        level = value("LOG_LEVEL", "INFO").upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigError("LOG_LEVEL: используйте DEBUG, INFO, WARNING, ERROR или CRITICAL.")
        api_key = value("OPENROUTER_API_KEY").strip()
        if require_llm and not api_key:
            raise ConfigError("OPENROUTER_API_KEY: добавьте ключ OpenRouter в env-файл.")
        base_url = value("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        if base_url != "https://openrouter.ai/api/v1":
            raise ConfigError("OPENROUTER_BASE_URL: используйте https://openrouter.ai/api/v1.")
        model = value("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
        if not model.endswith(":free") or "/" not in model or any(c.isspace() for c in model):
            raise ConfigError("OPENROUTER_MODEL: укажите идентификатор бесплатной модели с :free.")
        message_limit = number("HISTORY_MAX_MESSAGES", "20", 4, 200)
        if message_limit % 2:
            raise ConfigError("HISTORY_MAX_MESSAGES: нужно чётное число для целых пар диалога.")
        schedule_file = values.get("SCHEDULE_FILE", "data/schedule.json")
        try:
            if (
                not isinstance(schedule_file, str)
                or not schedule_file.strip()
                or any(ord(character) < 32 or ord(character) == 127 for character in schedule_file)
            ):
                raise ValueError
            # На Windows буква диска не является схемой URI.
            windows_drive = (
                len(schedule_file) >= 3
                and schedule_file[0].isascii()
                and schedule_file[0].isalpha()
                and schedule_file[1] == ":"
                and schedule_file[2] in {"/", "\\"}
            )
            if urlsplit(schedule_file).scheme and not windows_drive:
                raise ValueError
        except ValueError:
            raise ConfigError(
                "SCHEDULE_FILE: нужен непустой локальный путь без URI и управляющих символов."
            ) from None
        schedule_timezone = values.get("SCHEDULE_TIMEZONE", "Europe/Moscow")
        try:
            if (
                not isinstance(schedule_timezone, str)
                or not schedule_timezone
                or schedule_timezone in {"localtime", "posixrules"}
                or schedule_timezone.startswith(("posix/", "right/"))
            ):
                raise ValueError
            ZoneInfo(schedule_timezone)
        except (ValueError, ZoneInfoNotFoundError):
            raise ConfigError(
                "SCHEDULE_TIMEZONE: укажите существующую IANA-зону, например Europe/Moscow."
            ) from None
        llm_tool_mode = values.get("LLM_TOOL_MODE", "native")
        if llm_tool_mode not in {"native", "structured"}:
            raise ConfigError("LLM_TOOL_MODE: используйте native или structured.")
        return cls(
            bot_token=token,
            postgres_password=password,
            telegram_proxy_url=proxy,
            postgres_host=value("POSTGRES_HOST", "127.0.0.1"),
            postgres_port=port("POSTGRES_PORT", "5432"),
            postgres_db=value("POSTGRES_DB", "bot"),
            postgres_user=value("POSTGRES_USER", "bot"),
            log_level=level,
            health_port=port("HEALTH_PORT", "8080"),
            openrouter_api_key=api_key,
            openrouter_base_url=base_url,
            openrouter_model=model,
            llm_timeout_seconds=number("LLM_TIMEOUT_SECONDS", "60", 1, 300),
            llm_max_output_tokens=number("LLM_MAX_OUTPUT_TOKENS", "2048", 64, 8192),
            history_max_messages=message_limit,
            history_max_tokens=number("HISTORY_MAX_TOKENS", "6000", 1024, 100000),
            schedule_file=schedule_file,
            schedule_timezone=schedule_timezone,
            mcp_timeout_seconds=number("MCP_TIMEOUT_SECONDS", "15", 1, 60),
            agent_timeout_seconds=number("AGENT_TIMEOUT_SECONDS", "135", 1, 300),
            llm_tool_mode=llm_tool_mode,
        )
