import argparse
import asyncio
import logging

import aiohttp
from aiogram import Dispatcher

from app.agent import AgentService
from app.agent_repository import AgentRepository, ConfirmationRepository
from app.config import ConfigError, Settings
from app.confirmations import ConfirmationService
from app.db import create_pool
from app.dialog import DialogService
from app.handlers.assistant import create_router
from app.health import HealthState, start_health_server
from app.llm import OpenRouterClient
from app.logging_setup import configure_logging
from app.mcp.client import MCPClient
from app.migration import migrate
from app.repository import DialogRepository
from app.telegram import create_bot
from app.timezone_lookup import TimezoneLookup
from app.timezone_ui import TimezoneUI

logger = logging.getLogger("app")


async def _close_resource(label, close, *, terminate=None):
    try:
        async with asyncio.timeout(10):
            await close()
    except (Exception, asyncio.CancelledError) as error:
        logger.error("Завершение %s: ошибка %s.", label, type(error).__name__)
        if terminate is not None:
            try:
                terminate()
            except Exception as fallback_error:
                logger.error("Завершение %s: ошибка %s.", label, type(fallback_error).__name__)


async def _shutdown(state, runner, bot, timezone_session, mcp, llm):
    if state.polling_task and not state.polling_task.done():
        state.polling_task.cancel()
        await asyncio.gather(state.polling_task, return_exceptions=True)
    if runner:
        await _close_resource("healthcheck", runner.cleanup)
    await _close_resource("Telegram", bot.session.close)
    if timezone_session:
        await _close_resource("поиска часового пояса", timezone_session.close)
    if mcp:
        await _close_resource("MCP", mcp.close)
    if llm:
        await _close_resource("LLM", llm.close)
    if state.pool:
        await _close_resource("PostgreSQL", state.pool.close, terminate=state.pool.terminate)


async def run(settings: Settings) -> None:
    state = HealthState()
    bot = create_bot(settings)
    runner = None
    llm = None
    mcp = None
    timezone_session = None
    try:
        state.pool = await create_pool(settings)
        logger.info("PostgreSQL подключён: SELECT 1 выполнен.")
        await migrate(state.pool)
        # Начальная проверка токена и маршрута через прокси ограничена по времени.
        async with asyncio.timeout(30):
            await bot.get_me()
            webhook = await bot.get_webhook_info()
        if webhook.url:
            raise ConfigError(
                "У бота установлен webhook. Удалите его перед запуском polling "
                "или используйте отдельного учебного бота."
            )
        logger.info("Telegram доступен. Запускается polling.")
        llm = OpenRouterClient(settings)
        await llm.start()
        mcp = MCPClient(settings)
        await mcp.start()
        agent = AgentService(llm, mcp, AgentRepository(state.pool), settings)
        confirmations = ConfirmationService(ConfirmationRepository(state.pool), mcp)
        timezone_session = aiohttp.ClientSession()
        timezone_ui = TimezoneUI(confirmations, TimezoneLookup(timezone_session))
        dialog = DialogService(
            DialogRepository(state.pool),
            llm,
            settings,
            agent=agent,
            confirmations=confirmations,
            timezone_ui=timezone_ui,
        )
        dispatcher = Dispatcher()
        dispatcher.include_router(create_router())
        runner = await start_health_server(state, settings.health_port)
        state.polling_task = asyncio.create_task(
            dispatcher.start_polling(
                bot,
                db=state.pool,
                dialog=dialog,
                allowed_updates=dispatcher.resolve_used_update_types(),
                close_bot_session=False,
            )
        )
        state.initialized = True
        await state.polling_task
    finally:
        state.initialized = False
        cleanup = asyncio.create_task(_shutdown(state, runner, bot, timezone_session, mcp, llm))
        interrupted = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                interrupted = True
        await cleanup
        if interrupted:
            raise asyncio.CancelledError


def main() -> int:
    parser = argparse.ArgumentParser(description="Лабби — учебный AI-помощник")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()
    try:
        settings = Settings.load(args.env_file, require_llm=True)
    except ConfigError as exc:
        print(str(exc))
        return 1
    configure_logging(settings)
    try:
        asyncio.run(run(settings))
    except KeyboardInterrupt:
        logger.info("Бот остановлен.")
    except Exception as error:
        logger.error(
            "Не удалось запустить бот. Проверьте БД, токен и прокси. Ошибка %s.",
            type(error).__name__,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
