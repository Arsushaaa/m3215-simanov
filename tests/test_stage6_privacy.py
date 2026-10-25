"""Ошибки SDK и Telegram не выносят пользовательские данные в текстовый журнал."""

import asyncio
import json
import logging
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot, Dispatcher
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import SendMessage
from aiogram.types import Chat, Message, Update, User

from app.agent import AgentService
from app.handlers.assistant import create_router
from app.llm import RoutingCompletion
from app.logging_setup import configure_logging
from app.repository import UserSettings

OWNER = 918273645001
PRIVATE = (
    str(OWNER),
    "private_telegram_username",
    "private-full-question-about-a-pie",
    "private-system-instruction",
    "private-256-bit-capability-value",
)


@contextmanager
def private_logging(settings):
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    try:
        configure_logging(settings)
        yield
    finally:
        root.handlers = handlers
        root.setLevel(level)


@pytest.mark.parametrize(
    "source",
    [
        "aiogram.event",
        "aiogram.dispatcher",
        "aiohttp.client",
        "httpx",
        "httpcore.http11",
        "httpx2",
        "httpcore2.connection",
        "mcp.client",
        "asyncio",
    ],
)
def test_library_payloads_are_replaced_before_formatting_and_never_logged(
    bot_settings, capsys, source
):
    payload = " ".join(PRIVATE)
    with private_logging(bot_settings):
        try:
            raise RuntimeError(payload)
        except RuntimeError:
            logging.getLogger(source).exception("Update and HTTP payload: %s", payload)
    rendered = capsys.readouterr().err
    assert source in rendered
    assert "RuntimeError" in rendered
    assert "Traceback" not in rendered
    assert all(value not in rendered for value in PRIVATE)


def test_application_error_keeps_fixed_diagnostic_and_type_without_exception_payload(
    bot_settings, capsys
):
    with private_logging(bot_settings):
        try:
            raise ValueError(" ".join(PRIVATE))
        except ValueError:
            logging.getLogger("app").exception("Запуск не завершён: готовность 50%%")
        logging.getLogger("app.llm").info("LLM завершён status=OK llm_calls=1")
    rendered = capsys.readouterr().err
    assert "Запуск не завершён" in rendered
    assert "ValueError" in rendered
    assert "status=OK" in rendered
    assert "Traceback" not in rendered
    assert all(value not in rendered for value in PRIVATE)


@pytest.mark.parametrize("source", ["aiogram.event", "httpx2"])
def test_library_exception_outside_except_is_safe_and_filter_does_not_fail(
    bot_settings, capsys, source
):
    with private_logging(bot_settings):
        logging.getLogger(source).exception("Untrusted SDK payload: %s", " ".join(PRIVATE))
    rendered = capsys.readouterr().err
    assert source in rendered
    assert "Logging error" not in rendered
    assert "Traceback" not in rendered
    assert all(value not in rendered for value in PRIVATE)


async def test_agent_audit_failure_does_not_log_question_prompt_or_model_reason(
    bot_settings, capsys
):
    repository = SimpleNamespace(
        has_processed=AsyncMock(return_value=False),
        start_request=AsyncMock(),
        timezone=AsyncMock(return_value=None),
        record=AsyncMock(side_effect=RuntimeError(" ".join(PRIVATE))),
    )
    llm = SimpleNamespace(
        decide=AsyncMock(
            return_value=RoutingCompletion(
                text=json.dumps({"action": "respond", "text": PRIVATE[2], "reason": PRIVATE[3]}),
                tool_calls=[],
            )
        )
    )
    mcp = SimpleNamespace(available=False, tools={}, resources={})
    agent = AgentService(llm, mcp, repository, bot_settings)
    with private_logging(bot_settings):
        reply = await agent.respond(
            UserSettings(OWNER, "study", 0.3, 0),
            3,
            PRIVATE[2],
            prompt=PRIVATE[3],
            history=[],
        )
    assert reply.text == PRIVATE[2]
    rendered = capsys.readouterr().err
    assert "Аудит решения недоступен" in rendered
    assert "RuntimeError" in rendered
    assert all(value not in rendered for value in PRIVATE)
    assert PRIVATE[3] not in repository.record.call_args.kwargs["reason"]


async def test_actual_dispatcher_send_failure_does_not_log_update_identity_or_telegram_payload(
    bot_settings, capsys
):
    bot = Bot(bot_settings.bot_token)
    bot.session = AsyncMock()
    dispatcher = Dispatcher()
    dispatcher.include_router(create_router())
    lock = asyncio.Lock()
    dialog = SimpleNamespace(
        lock=lambda _owner: lock,
        timezone_search_active=lambda _owner: False,
        respond=AsyncMock(return_value=PRIVATE[2]),
        confirmation_card=AsyncMock(return_value=None),
    )
    update = Update(
        update_id=765432198,
        message=Message(
            message_id=3,
            date=datetime.now(UTC),
            chat=Chat(id=OWNER, type="private"),
            from_user=User(
                id=OWNER,
                is_bot=False,
                first_name="Private Name",
                username=PRIVATE[1],
            ),
            text=PRIVATE[2],
        ),
    )

    async def send(_bot, method, **_kwargs):
        if isinstance(method, SendMessage):
            raise TelegramBadRequest(method=method, message=" ".join(PRIVATE))

    bot.session.side_effect = send
    with private_logging(bot_settings):
        assert await dispatcher._process_update(bot, update, dialog=dialog)
    rendered = capsys.readouterr().err
    assert "TelegramBadRequest" in rendered
    assert "aiogram.event" in rendered
    assert "Traceback" not in rendered
    assert str(update.update_id) not in rendered
    assert str(bot.id) not in rendered
    assert all(value not in rendered for value in PRIVATE)
    dialog.respond.assert_awaited_once_with(OWNER, 3, PRIVATE[2])
