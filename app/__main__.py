import argparse
import asyncio
import logging

from aiogram import Dispatcher

from app.agent import AgentService
from app.agent_repository import AgentRepository
from app.config import ConfigError, Settings
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

logger = logging.getLogger("app")


async def run(settings: Settings) -> None:
    state = HealthState()
    bot = create_bot(settings)
    runner = None
    llm = None
    mcp = None
    try:
        state.pool = await create_pool(settings)
        logger.info("PostgreSQL подключён: SELECT 1 выполнен.")
        await migrate(state.pool)
        # Начальная проверка токена и маршрута через прокси ограничена по времени.
        async with asyncio.timeout(30):
            me = await bot.get_me()
            webhook = await bot.get_webhook_info()
        if webhook.url:
            raise ConfigError(
                "У бота установлен webhook. Удалите его перед запуском polling "
                "или используйте отдельного учебного бота."
            )
        logger.info("Telegram доступен. Бот @%s запускает polling.", me.username)
        llm = OpenRouterClient(settings)
        await llm.start()
        mcp = MCPClient(settings)
        await mcp.start()
        agent = AgentService(llm, mcp, AgentRepository(state.pool), settings)
        dialog = DialogService(DialogRepository(state.pool), llm, settings, agent=agent)
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
        if state.polling_task and not state.polling_task.done():
            state.polling_task.cancel()
            await asyncio.gather(state.polling_task, return_exceptions=True)
        if runner:
            await runner.cleanup()
        await bot.session.close()
        if mcp:
            await mcp.close()
        if llm:
            await llm.close()
        if state.pool:
            try:
                async with asyncio.timeout(10):
                    await state.pool.close()
            except TimeoutError:
                state.pool.terminate()


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
    except Exception:
        logger.exception("Не удалось запустить бот. Проверьте БД, токен и прокси.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
