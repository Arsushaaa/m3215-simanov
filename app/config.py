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


def _schedule_values(values: Mapping[str, str | None]) -> tuple[str, str]:
    path = values.get("SCHEDULE_FILE", "data/schedule.json")
    try:
        if (
            not isinstance(path, str)
            or not path.strip()
            or any(ord(character) < 32 or ord(character) == 127 for character in path)
        ):
            raise ValueError
        windows_drive = (
            len(path) >= 3
            and path[0].isascii()
            and path[0].isalpha()
            and path[1] == ":"
            and path[2] in {"/", "\\"}
        )
        if urlsplit(path).scheme and not windows_drive:
            raise ValueError
    except ValueError:
        raise ConfigError(
            "SCHEDULE_FILE: нужен непустой локальный путь без URI и управляющих символов."
        ) from None
    zone = values.get("SCHEDULE_TIMEZONE", "Europe/Moscow")
    try:
        if (
            not isinstance(zone, str)
            or not zone
            or zone in {"localtime", "posixrules"}
            or zone.startswith(("posix/", "right/"))
        ):
            raise ValueError
        ZoneInfo(zone)
    except (ValueError, ZoneInfoNotFoundError):
        raise ConfigError(
            "SCHEDULE_TIMEZONE: укажите существующую IANA-зону, например Europe/Moscow."
        ) from None
    return path, zone


@dataclass(frozen=True)
class MCPSettings:
    """Минимальные настройки сервера без Telegram и OpenRouter-секретов.

    По умолчанию читается только окружение процесса, не корневой .env бота.
    Отдельный env-файл допускается для независимого локального клиента.
    """

    postgres_password: str = field(default="", repr=False)
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    postgres_db: str = "bot"
    postgres_user: str = "bot"
    schedule_file: str = field(default="data/schedule.json", repr=False)
    schedule_timezone: str = "Europe/Moscow"
    mcp_timeout_seconds: int = 15

    @classmethod
    def load(
        cls,
        env_file: Path | str | None = None,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> "MCPSettings":
        values = {
            **(dotenv_values(env_file, interpolate=False) if env_file is not None else {}),
            **(os.environ if environ is None else environ),
        }
        path, zone = _schedule_values(values)
        try:
            port = int(values.get("POSTGRES_PORT") or "5432")
            timeout = int(values.get("MCP_TIMEOUT_SECONDS") or "15")
            if not 1 <= port <= 65535 or not 1 <= timeout <= 60:
                raise ValueError
        except ValueError:
            raise ConfigError(
                "POSTGRES_PORT/MCP_TIMEOUT_SECONDS: нужны порт 1–65535 и тайм-аут 1–60 секунд."
            ) from None
        return cls(
            postgres_password=values.get("POSTGRES_PASSWORD") or "",
            postgres_host=values.get("POSTGRES_HOST") or "127.0.0.1",
            postgres_port=port,
            postgres_db=values.get("POSTGRES_DB") or "bot",
            postgres_user=values.get("POSTGRES_USER") or "bot",
            schedule_file=path,
            schedule_timezone=zone,
            mcp_timeout_seconds=timeout,
        )

    def environment(self) -> dict[str, str]:
        """Явный allowlist для дочернего процесса; произвольный env не копируется."""
        return {
            "POSTGRES_HOST": self.postgres_host,
            "POSTGRES_PORT": str(self.postgres_port),
            "POSTGRES_DB": self.postgres_db,
            "POSTGRES_USER": self.postgres_user,
            "POSTGRES_PASSWORD": self.postgres_password,
            "SCHEDULE_FILE": self.schedule_file,
            "SCHEDULE_TIMEZONE": self.schedule_timezone,
            "MCP_TIMEOUT_SECONDS": str(self.mcp_timeout_seconds),
        }


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
    llm_tool_mode: str = "structured"

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
        schedule_file, schedule_timezone = _schedule_values(values)
        llm_tool_mode = values.get("LLM_TOOL_MODE", "structured")
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
