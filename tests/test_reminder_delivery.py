"""HTTP failures, privacy, bounded concurrency and lifecycle without Telegram HTTP/LLM."""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from aiogram.exceptions import (
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramNetworkError,
    TelegramRetryAfter,
    TelegramServerError,
    TelegramUnauthorizedError,
)
from aiogram.methods import SendMessage
from test_confirmation_handlers import setup_ui as setup_ui
from test_confirmation_handlers import text_update
from test_supervisor import MemoryTraces

from app.notifications.delivery_models import (
    DeliveryFailure,
    Lease,
    classify_failure,
    reminder_body,
    retry_delay,
    valid_payload,
)
from app.notifications.worker import ReminderWorker
from app.tracing.runtime import TraceManager

METHOD = SendMessage(chat_id=9191919191, text="private request text")


@pytest.mark.parametrize(
    "error,code,transient,uncertain",
    [
        (TimeoutError(), "NETWORK_UNKNOWN", True, True),
        (OSError("private address"), "NETWORK_UNKNOWN", True, True),
        (TelegramNetworkError(METHOD, "private token"), "NETWORK_UNKNOWN", True, True),
        (TelegramServerError(METHOD, "internal body"), "SERVER_ERROR", True, True),
        (TelegramForbiddenError(METHOD, "blocked"), "CHAT_UNAVAILABLE", False, False),
        (TelegramUnauthorizedError(METHOD, "secret"), "CHAT_UNAVAILABLE", False, False),
        (TelegramBadRequest(METHOD, "bad"), "REQUEST_REJECTED", False, False),
        (RuntimeError("private exception"), "UNEXPECTED_ERROR", True, True),
    ],
)
def test_http_failure_policy_is_bounded_and_does_not_use_provider_text(
    error, code, transient, uncertain
):
    actual = classify_failure(error)
    assert (actual.code, actual.transient, actual.uncertain) == (code, transient, uncertain)
    assert "private" not in repr(actual)


@pytest.mark.parametrize(
    "value,code",
    [
        (-1, "INVALID_RETRY_AFTER"),
        (float("nan"), "INVALID_RETRY_AFTER"),
        (float("inf"), "INVALID_RETRY_AFTER"),
        (True, "INVALID_RETRY_AFTER"),
        (90, "RATE_LIMIT"),
        (0, "RATE_LIMIT"),
    ],
)
def test_retry_after_is_finite_nonnegative_and_not_a_boolean(value, code):
    result = classify_failure(TelegramRetryAfter(METHOD, "private", value))
    assert result.code == code
    if code == "RATE_LIMIT":
        assert result.retry_after == value


@pytest.mark.parametrize(
    "attempt,after,expected", [(1, 0, 30), (2, 0, 120), (3, 0, 120), (1, 99, 99)]
)
def test_retry_policy_has_no_exponential_unbounded_loop(attempt, after, expected):
    assert retry_delay(attempt, after) == expected


@pytest.mark.parametrize(
    "body,valid",
    [
        ("", False),
        (" " * 3, False),
        ("x" * 4096, True),
        ("x" * 4097, False),
        ("🦉" * 2048, True),
        ("🦉" * 2049, False),
        ("\ud800", False),
        ("Hello", True),
    ],
)
def test_single_telegram_payload_is_bounded_in_utf16(body, valid):
    assert valid_payload(body) is valid


def test_reminder_payload_is_literal_and_shows_zone_and_delay():
    now = datetime(2026, 11, 1, 10, tzinfo=UTC)
    text = reminder_body(
        "<b>read</b> https://example.invalid", now, "Europe/Moscow", now + timedelta(seconds=2)
    )
    assert "с задержкой" in text and "13:00 (UTC+03:00)" in text
    assert "<b>read</b>" in text
    assert "задержкой" not in reminder_body("Read", now, "Europe/Moscow", now)


@pytest.fixture
def delivery_setup():
    lease = Lease(uuid4(), 1, uuid4(), 2, "literal <script> private body", 9191919191, 1)
    repo = SimpleNamespace(
        recover=AsyncMock(),
        claim_preparations=AsyncMock(return_value=[]),
        release_preparations=AsyncMock(),
        candidates=AsyncMock(return_value=[(lease.event_id, lease.owner)]),
        begin_send=AsyncMock(return_value=lease),
        preflight=AsyncMock(return_value=True),
        sent=AsyncMock(return_value=True),
        failed=AsyncMock(return_value=True),
    )
    bot = SimpleNamespace(send_message=AsyncMock(return_value=SimpleNamespace(message_id=123)))
    worker = ReminderWorker(repo, bot)
    return worker, repo, bot, lease


async def test_worker_sends_once_to_stored_destination_with_no_formatting_or_paid_broadcast(
    delivery_setup,
):
    worker, repo, bot, lease = delivery_setup
    await worker.tick()
    bot.send_message.assert_awaited_once()
    values = bot.send_message.call_args.kwargs
    assert values["chat_id"] == lease.destination and values["chat_id"] != lease.owner
    assert values["text"] == lease.body and values["parse_mode"] is None
    assert values["allow_paid_broadcast"] is False and values["request_timeout"] == 20
    assert values["link_preview_options"].is_disabled
    repo.sent.assert_awaited_once_with(lease, 123)
    repo.failed.assert_not_awaited()
    assert not hasattr(worker, "llm") and not hasattr(worker, "mcp")


async def test_delivery_failure_records_policy_once_and_does_not_retry_in_http_loop(
    delivery_setup, caplog
):
    worker, repo, bot, lease = delivery_setup
    bot.send_message.side_effect = TelegramRetryAfter(METHOD, "sensitive provider body", 90)
    await worker.tick()
    bot.send_message.assert_awaited_once()
    repo.failed.assert_awaited_once_with(lease, DeliveryFailure("RATE_LIMIT", True, retry_after=90))
    assert "sensitive provider body" not in caplog.text and "9191919191" not in caplog.text
    repo.sent.assert_not_awaited()


async def test_invalid_acknowledgement_is_unknown_not_a_success(delivery_setup):
    worker, repo, bot, lease = delivery_setup
    bot.send_message.return_value = SimpleNamespace(message_id=True)
    await worker.tick()
    repo.sent.assert_not_awaited()
    assert repo.failed.call_args.args[1].uncertain


@pytest.mark.parametrize("body", ["x" * 4097, "🦉" * 2049, ""])
async def test_invalid_payload_never_calls_telegram(delivery_setup, body):
    from dataclasses import replace

    worker, repo, bot, lease = delivery_setup
    repo.begin_send.return_value = replace(lease, body=body)
    await worker.tick()
    bot.send_message.assert_not_awaited()
    assert repo.failed.call_args.args[1].code == "INVALID_PAYLOAD"


async def test_revoked_preflight_or_stale_candidate_never_sends(delivery_setup):
    worker, repo, bot, _ = delivery_setup
    repo.preflight.return_value = False
    await worker.tick()
    bot.send_message.assert_not_awaited()
    repo.begin_send.return_value = None
    await worker.tick()
    bot.send_message.assert_not_awaited()


async def test_preflight_is_rechecked_after_trace_write_before_http(delivery_setup):
    worker, repo, bot, _ = delivery_setup
    repo.preflight.side_effect = [True, False]
    await worker.tick()
    bot.send_message.assert_not_awaited()


async def test_only_two_active_http_calls_and_waiting_tasks_do_not_hold_leases(delivery_setup):
    worker, repo, bot, lease = delivery_setup
    ids = [uuid4() for _ in range(10)]
    repo.candidates.return_value = [(i, 1) for i in ids]
    active = peak = 0
    entered = asyncio.Event()
    release = asyncio.Event()

    async def send(**kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if active == 2:
            entered.set()
        await release.wait()
        active -= 1
        return SimpleNamespace(message_id=1)

    bot.send_message.side_effect = send
    task = asyncio.create_task(worker.tick())
    await asyncio.wait_for(entered.wait(), 2)
    assert repo.begin_send.await_count == 2
    release.set()
    await task
    assert peak == 2 and bot.send_message.await_count == 10


async def test_graceful_close_waits_for_active_send_and_skips_queued_claims(delivery_setup):
    worker, repo, bot, lease = delivery_setup
    repo.candidates.return_value = [(lease.event_id, 1)] * 5
    started, release = asyncio.Event(), asyncio.Event()

    async def send(**kwargs):
        started.set()
        await release.wait()
        return SimpleNamespace(message_id=1)

    bot.send_message.side_effect = send
    worker.start()
    await asyncio.wait_for(started.wait(), 2)
    closing = asyncio.create_task(worker.close())
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    await asyncio.wait_for(closing, 2)
    assert bot.send_message.await_count <= 2 and worker.task.done()
    assert repo.begin_send.await_count <= 2


async def test_user_stop_does_not_cancel_an_active_committed_notification(delivery_setup, setup_ui):
    worker, repo, bot, _ = delivery_setup
    dispatcher, telegram, dialog = setup_ui
    started, release = asyncio.Event(), asyncio.Event()

    async def send(**kwargs):
        started.set()
        await release.wait()
        return SimpleNamespace(message_id=9)

    bot.send_message.side_effect = send
    task = asyncio.create_task(worker.tick())
    await started.wait()
    await dispatcher.feed_update(telegram, text_update("/stop"), dialog=dialog)
    assert not task.done()
    release.set()
    await task
    bot.send_message.assert_awaited_once()
    repo.sent.assert_awaited_once()


async def test_cancelled_inflight_http_records_unknown_outcome_for_recovery(delivery_setup):
    worker, repo, bot, lease = delivery_setup
    started = asyncio.Event()

    async def send(**kwargs):
        started.set()
        await asyncio.Event().wait()

    bot.send_message.side_effect = send
    task = asyncio.create_task(worker._deliver(lease.event_id, lease.owner))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    repo.failed.assert_awaited_once_with(lease, DeliveryFailure("PROCESS_INTERRUPTED", True, True))


async def test_receipt_commit_ack_loss_reconciles_same_lease_without_second_http(delivery_setup):
    worker, repo, bot, lease = delivery_setup
    repo.sent.side_effect = [ConnectionError("commit response lost"), True]
    await worker.tick()
    assert repo.sent.await_count == 2
    bot.send_message.assert_awaited_once()
    repo.failed.assert_not_awaited()


async def test_worker_trace_records_http_without_private_payload(delivery_setup):
    worker, repo, _, lease = delivery_setup
    memory = MemoryTraces()
    worker.traces = TraceManager(memory)
    await worker.tick()
    spans = list(memory.spans.values())
    assert any(s.operation == "send_message" and s.status == "ok" for s in spans)
    assert not any(s.operation.startswith("model_") for s in spans)
    assert all(
        "private body" not in s.model_dump_json()
        and str(lease.destination) not in s.model_dump_json()
        for s in spans
    )


async def test_finished_worker_makes_health_unready():
    from app.health import HealthState, health_result

    polling = asyncio.create_task(asyncio.Event().wait())
    stopped = asyncio.create_task(asyncio.sleep(0))
    await stopped
    state = HealthState(
        pool=Mock(fetchval=AsyncMock(return_value=1)),
        initialized=True,
        polling_task=polling,
        delivery_task=stopped,
    )
    try:
        status, body = await health_result(state)
        assert status == 503 and body["reminders"] == "stopped"
    finally:
        polling.cancel()
        await asyncio.gather(polling, return_exceptions=True)


@pytest.mark.parametrize("failure", [False, True])
async def test_shutdown_stops_delivery_before_bot_and_pool_even_if_worker_close_fails(failure):
    from app.__main__ import _shutdown
    from app.health import HealthState

    order = []

    async def delivery_close():
        order.append("delivery")
        if failure:
            raise RuntimeError("private exception")

    async def bot_close():
        order.append("telegram")

    async def pool_close():
        order.append("database")

    worker = SimpleNamespace(close=delivery_close)
    bot = SimpleNamespace(session=SimpleNamespace(close=bot_close))
    pool = Mock(close=pool_close)
    await _shutdown(HealthState(pool=pool), None, bot, None, None, None, worker)
    assert order == ["delivery", "telegram", "database"]


async def test_app_fails_fast_if_reminder_worker_stops_while_polling_is_alive(monkeypatch):
    from app import __main__ as application
    from app.config import Settings

    pool = Mock(close=AsyncMock())
    bot = Mock(
        get_me=AsyncMock(),
        get_webhook_info=AsyncMock(return_value=Mock(url="")),
        session=Mock(close=AsyncMock()),
    )
    llm = Mock(start=AsyncMock(), close=AsyncMock())
    mcp = Mock(start=AsyncMock(), close=AsyncMock(), available=False)
    polling_cancelled = asyncio.Event()

    async def poll(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            polling_cancelled.set()

    worker = Mock(
        start=Mock(side_effect=lambda: asyncio.create_task(asyncio.sleep(0))), close=AsyncMock()
    )
    monkeypatch.setattr(application, "create_pool", AsyncMock(return_value=pool))
    monkeypatch.setattr(application, "migrate", AsyncMock())
    monkeypatch.setattr(application, "create_bot", Mock(return_value=bot))
    monkeypatch.setattr(application, "OpenRouterClient", Mock(return_value=llm))
    monkeypatch.setattr(application, "MCPClient", Mock(return_value=mcp))
    monkeypatch.setattr(application, "ReminderWorker", Mock(return_value=worker))
    monkeypatch.setattr(application, "Dispatcher", Mock(return_value=Mock(start_polling=poll)))
    monkeypatch.setattr(
        application, "start_health_server", AsyncMock(return_value=Mock(cleanup=AsyncMock()))
    )
    with pytest.raises(RuntimeError, match="напоминаний"):
        await asyncio.wait_for(
            application.run(Settings(bot_token="unused", postgres_password="unused")), 2
        )
    assert polling_cancelled.is_set()
    worker.close.assert_awaited_once()
    bot.session.close.assert_awaited_once()
    pool.close.assert_awaited_once()
