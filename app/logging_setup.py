import logging
import re
from urllib.parse import unquote, urlsplit

from app.config import Settings


class PrivacyFilter(logging.Filter):
    """Library diagnostics may contain Telegram methods, updates and HTTP bodies."""

    _LIBRARIES = (
        "aiogram",
        "aiohttp",
        "httpx",
        "httpcore",
        "httpx2",
        "httpcore2",
        "mcp",
        "asyncio",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        # logging.exception вне except передаёт (None, None, None).
        # Сам фильтр не должен падать и отдавать payload стандартному handleError.
        exception_class = record.exc_info[0] if record.exc_info else None
        error_type = exception_class.__name__ if isinstance(exception_class, type) else None
        if record.name.split(".", 1)[0] in self._LIBRARIES:
            # Не разбираем и не редактируем произвольный payload: текст ошибки,
            # args, HTTP-заголовки и Update целиком остаются вне журнала.
            record.msg = "Сторонний компонент: событие уровня %s"
            record.args = (record.levelname,)
            if error_type:
                record.msg += "; error_type=%s"
                record.args += (error_type,)
        elif error_type:
            # app логирует собственные фиксированные сообщения и классы ошибок.
            # Текст исключения и стек могут содержать запрос, capability или SQL.
            message = record.getMessage()
            record.msg, record.args = "%s; error_type=%s", (message, error_type)
        record.exc_info = record.exc_text = record.stack_info = None
        return True


class SecretFilter(logging.Filter):
    def __init__(self, secrets: list[str]):
        super().__init__()
        self.secrets = sorted({s for s in secrets if s}, key=len, reverse=True)

    def filter(self, record: logging.LogRecord) -> bool:
        text = record.getMessage()
        if record.exc_info:
            text += "\n" + logging.Formatter().formatException(record.exc_info)
        if record.stack_info:
            text += "\n" + record.stack_info
        for secret in self.secrets:
            text = text.replace(secret, "[скрыто]")
        text = re.sub(r"(https?://|socks5://)[^\s/@]+:[^\s/@]+@", r"\1[скрыто]@", text)
        record.msg, record.args = text, ()
        record.exc_info = record.exc_text = record.stack_info = None
        return True


def configure_logging(settings: Settings) -> None:
    proxy = urlsplit(settings.telegram_proxy_url)
    handler = logging.StreamHandler()
    handler.addFilter(PrivacyFilter())
    handler.addFilter(
        SecretFilter(
            [
                settings.bot_token,
                settings.openrouter_api_key,
                settings.postgres_password,
                settings.telegram_proxy_url,
                proxy.password or "",
                unquote(proxy.password or ""),
            ]
        )
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=settings.log_level, handlers=[handler], force=True)
