"""Общий shutdown закрывает остальные ресурсы даже после ошибки одного адаптера."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app import __main__ as application
from app.config import Settings


def startup(monkeypatch, *, polling_error=None):
    pool = Mock(close=AsyncMock())
    bot = Mock(
        get_me=AsyncMock(return_value=Mock(username="private-bot-name")),
        get_webhook_info=AsyncMock(return_value=Mock(url="")),
        session=Mock(close=AsyncMock()),
    )
    llm = Mock(start=AsyncMock(), close=AsyncMock())
    mcp = Mock(start=AsyncMock(), close=AsyncMock(), available=False)
    timezone = Mock(close=AsyncMock())
    runner = Mock(cleanup=AsyncMock())
    dispatcher = Mock(start_polling=AsyncMock(side_effect=polling_error))
    monkeypatch.setattr(application, "create_pool", AsyncMock(return_value=pool))
    monkeypatch.setattr(application, "migrate", AsyncMock())
    monkeypatch.setattr(application, "create_bot", Mock(return_value=bot))
    monkeypatch.setattr(application, "OpenRouterClient", Mock(return_value=llm))
    monkeypatch.setattr(application, "MCPClient", Mock(return_value=mcp))
    monkeypatch.setattr(application.aiohttp, "ClientSession", Mock(return_value=timezone))
    monkeypatch.setattr(application, "Dispatcher", Mock(return_value=dispatcher))
    monkeypatch.setattr(application, "start_health_server", AsyncMock(return_value=runner))
    return SimpleNamespace(
        pool=pool,
        bot=bot,
        llm=llm,
        mcp=mcp,
        timezone=timezone,
        runner=runner,
        dispatcher=dispatcher,
    )


@pytest.mark.parametrize("resource", ["health", "telegram", "timezone", "mcp", "llm", "pool"])
async def test_failing_resource_does_not_skip_other_cleanup_or_leak_details(
    monkeypatch, caplog, resource
):
    resources = startup(monkeypatch, polling_error=RuntimeError("original failure"))
    closers = {
        "health": resources.runner.cleanup,
        "telegram": resources.bot.session.close,
        "timezone": resources.timezone.close,
        "mcp": resources.mcp.close,
        "llm": resources.llm.close,
        "pool": resources.pool.close,
    }
    closers[resource].side_effect = ValueError("private-close-error-full-message-capability")
    with pytest.raises(RuntimeError, match="original failure"):
        await application.run(Settings(bot_token="unused", postgres_password="unused"))
    for close in closers.values():
        close.assert_awaited_once()
    if resource == "pool":
        resources.pool.terminate.assert_called_once()
    assert "private-close-error" not in caplog.text
    assert "private-bot-name" not in caplog.text
    assert "ValueError" in caplog.text


async def test_cancel_during_cleanup_waits_for_remaining_closes_and_propagates(monkeypatch):
    resources = startup(monkeypatch)
    started, release = asyncio.Event(), asyncio.Event()

    async def slow_close():
        started.set()
        await release.wait()

    resources.runner.cleanup.side_effect = slow_close
    task = asyncio.create_task(
        application.run(Settings(bot_token="unused", postgres_password="unused"))
    )
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    resources.bot.session.close.assert_awaited_once()
    resources.timezone.close.assert_awaited_once()
    resources.mcp.close.assert_awaited_once()
    resources.llm.close.assert_awaited_once()
    resources.pool.close.assert_awaited_once()


async def test_pool_close_failure_terminates_and_does_not_mask_original_error(monkeypatch):
    resources = startup(monkeypatch, polling_error=RuntimeError("original failure"))
    resources.pool.close.side_effect = TimeoutError("private-dsn")
    with pytest.raises(RuntimeError, match="original failure"):
        await application.run(Settings(bot_token="unused", postgres_password="unused"))
    resources.pool.terminate.assert_called_once()


def test_main_reports_error_type_without_untrusted_exception_body(monkeypatch, caplog):
    monkeypatch.setattr(application.Settings, "load", Mock(return_value=Settings("unused", "x")))
    monkeypatch.setattr(application, "configure_logging", Mock())
    monkeypatch.setattr(
        application.argparse.ArgumentParser,
        "parse_args",
        Mock(return_value=SimpleNamespace(env_file="unused")),
    )

    async def fail(_settings):
        raise ValueError("private-message-id-user-id-system-prompt-capability")

    monkeypatch.setattr(application, "run", fail)
    assert application.main() == 1
    assert "ValueError" in caplog.text
    assert "private-message" not in caplog.text
