import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config import ConfigError, Settings
from app.health import HealthState, health_result
from app.logging_setup import SecretFilter
from app.telegram import create_bot

TOKEN = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk"


@pytest.fixture
def settings(tmp_path):
    path = tmp_path / ".env"
    path.write_text(f"BOT_TOKEN={TOKEN}\nPOSTGRES_PASSWORD=secret-db\n", encoding="utf-8")
    return Settings.load(path, environ={})


def test_settings_environment_overrides_file(tmp_path):
    # Arrange
    path = tmp_path / ".env"
    path.write_text(
        f"BOT_TOKEN={TOKEN}\nPOSTGRES_PASSWORD='p$a#ss'\nPOSTGRES_PORT=5432\n", encoding="utf-8"
    )
    # Act
    config = Settings.load(path, environ={"POSTGRES_PORT": "55432"})
    # Assert
    assert config.postgres_port == 55432
    assert config.postgres_password == "p$a#ss"
    assert "p$a#ss" not in repr(config)
    assert TOKEN not in repr(config)


@pytest.mark.parametrize("value", ["abc", "0", "65536"])
def test_invalid_port_has_safe_error(tmp_path, value):
    # Arrange
    path = tmp_path / ".env"
    # Act / Assert
    with pytest.raises(ConfigError, match="POSTGRES_PORT"):
        Settings.load(
            path,
            environ={"BOT_TOKEN": TOKEN, "POSTGRES_PASSWORD": "secret", "POSTGRES_PORT": value},
        )


@pytest.mark.parametrize("proxy", ["http://user:pass@localhost:8080", "socks5://localhost:1080"])
async def test_proxy_is_attached_to_bot_session(settings, proxy):
    # Arrange
    from dataclasses import replace

    config = replace(settings, telegram_proxy_url=proxy)
    # Act
    bot = create_bot(config)
    # Assert
    assert bot.session._proxy == proxy
    await bot.session.close()


def test_bad_proxy_does_not_leak_credentials(tmp_path):
    # Arrange
    proxy = "ftp://user:very-secret@host:123"
    # Act / Assert
    with pytest.raises(ConfigError) as exc:
        Settings.load(
            tmp_path / ".env",
            environ={"BOT_TOKEN": TOKEN, "POSTGRES_PASSWORD": "db", "TELEGRAM_PROXY_URL": proxy},
        )
    assert "TELEGRAM_PROXY_URL" in str(exc.value)
    assert "very-secret" not in str(exc.value)


async def test_health_tracks_pool_failure_and_recovery():
    # Arrange
    pool = MagicMock()
    pool.fetchval = AsyncMock(return_value=1)
    task = asyncio.create_task(asyncio.Event().wait())
    state = HealthState(pool=pool, initialized=True, polling_task=task)
    try:
        # Act / Assert
        assert (await health_result(state))[0] == 200
        pool.fetchval.assert_awaited_with("SELECT 1", timeout=3)
        pool.fetchval.side_effect = ConnectionError("secret-password")
        status, body = await health_result(state)
        assert status == 503
        assert body == {"status": "not_ready", "database": "unavailable", "polling": "running"}
        pool.fetchval.side_effect = None
        assert (await health_result(state))[0] == 200
        state.initialized = False
        assert (await health_result(state))[0] == 503
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_health_without_polling_is_not_ready():
    # Arrange
    state = HealthState(pool=None)
    # Act
    status, body = await health_result(state)
    # Assert
    assert status == 503
    assert body["polling"] == "stopped"


def test_log_filter_redacts_secrets_and_exception():
    # Arrange
    secret_filter = SecretFilter([TOKEN, "db-secret", "proxy-secret"])
    try:
        raise ValueError("proxy-secret")
    except ValueError:
        import sys

        record = logging.LogRecord(
            "test",
            logging.ERROR,
            "",
            1,
            "Ошибка %s https://api.telegram.org/bot%s",
            ("db-secret", TOKEN),
            sys.exc_info(),
        )
    # Act
    secret_filter.filter(record)
    rendered = logging.Formatter().format(record)
    # Assert
    assert all(secret not in rendered for secret in (TOKEN, "db-secret", "proxy-secret"))
    assert "ValueError" in rendered


def test_llm_key_is_required_for_application_but_not_database_setup(tmp_path):
    path = tmp_path / ".env"
    values = {"BOT_TOKEN": TOKEN, "POSTGRES_PASSWORD": "fake-db"}
    Settings.load(path, environ=values)
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        Settings.load(path, environ=values, require_llm=True)
    values["OPENROUTER_API_KEY"] = "fake-secret-key"
    settings = Settings.load(path, environ=values, require_llm=True)
    assert "fake-secret-key" not in repr(settings)


@pytest.mark.parametrize(
    "field,value",
    [
        ("OPENROUTER_MODEL", "paid/model"),
        ("OPENROUTER_BASE_URL", "https://user:private-key@untrusted.example"),
        ("HISTORY_MAX_MESSAGES", "3"),
        ("HISTORY_MAX_TOKENS", "0"),
        ("LLM_TIMEOUT_SECONDS", "bad"),
    ],
)
def test_llm_configuration_errors_do_not_leak_credentials(tmp_path, field, value):
    with pytest.raises(ConfigError) as exc:
        Settings.load(
            tmp_path / ".env",
            environ={
                "BOT_TOKEN": TOKEN,
                "POSTGRES_PASSWORD": "fake-db",
                field: value,
            },
        )
    assert field in str(exc.value)
    assert "private-key" not in str(exc.value)


def test_logging_masks_openrouter_key(bot_settings, capsys):
    from app.logging_setup import configure_logging

    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    try:
        configure_logging(bot_settings)
        logging.getLogger("test").error("Authorization: Bearer %s", bot_settings.openrouter_api_key)
        output = capsys.readouterr().err
        assert bot_settings.openrouter_api_key not in output
        assert "[скрыто]" in output
    finally:
        root.handlers = handlers
        root.setLevel(level)
