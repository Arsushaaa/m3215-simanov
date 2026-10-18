"""Подтверждения не вызывают LLM и восстанавливают результат того же action."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from app.agent import ReminderProposal
from app.confirmation_models import (
    Authorization,
    ConfirmationError,
    Outcome,
    PreparedAction,
    SafeAgentEvent,
    format_card,
)
from app.confirmations import UNKNOWN_OUTCOME, ConfirmationService
from app.mcp.client import MCPClientError
from app.mcp.contracts import ReminderArguments
from app.repository import UserSettings

USER = UserSettings(99999999999, "study", 0.3, 7)
ACTION_ID = UUID("a1cf8b47-b6fb-4e95-9931-b21dd239b782")
REMINDER_ID = UUID("f4b780fd-4c99-42be-8cab-30ffb8502f5e")
NOW = datetime(2026, 10, 19, 6, tzinfo=UTC)
ARGS = ReminderArguments(text="Отправить отчёт", remind_at="2026-10-20T18:30:12.123456+03:00")
CONTEXT = {"action_id": str(ACTION_ID), "capability": "A" * 43}
PROPOSAL = ReminderProposal(ARGS, "Europe/Moscow")
COMMITTED = Outcome(
    status="committed",
    code=None,
    message="Напоминание сохранено.",
    reminder_id=REMINDER_ID,
    remind_at=ARGS.remind_at,
    timezone="Europe/Moscow",
)


@pytest.fixture
def setup_service():
    expires = NOW + timedelta(minutes=5)
    prepared = PreparedAction(
        action_id=ACTION_ID,
        card_text=format_card(ARGS, "Europe/Moscow", expires),
        arguments=ARGS,
        timezone="Europe/Moscow",
        expires_at=expires,
        state="pending",
    )
    repository = SimpleNamespace(
        prepare=AsyncMock(return_value=prepared),
        approve=AsyncMock(return_value=Authorization(ACTION_ID, ARGS, CONTEXT)),
        reconcile=AsyncMock(return_value=COMMITTED),
        cancel=AsyncMock(
            return_value=Outcome("cancelled", "ACTION_CANCELLED", "Подтверждение отменено.")
        ),
        invalidate=AsyncMock(return_value=True),
        timezone=AsyncMock(return_value="Europe/Moscow"),
        set_timezone=AsyncMock(return_value="Europe/Moscow"),
        now=AsyncMock(return_value=NOW),
        latest_event=AsyncMock(return_value=None),
    )
    mcp = SimpleNamespace(call_tool=AsyncMock(return_value={"status": "ok"}))
    return ConfirmationService(repository, mcp), repository, mcp


async def test_prepare_delegates_atomic_pair_and_card_without_mcp(setup_service):
    service, repository, mcp = setup_service
    prepared = await service.prepare(USER, 20, "Напомни отправить отчёт", PROPOSAL)
    assert prepared.action_id == ACTION_ID
    assert "2026-10-20T18:30:12.123456+03:00" in prepared.card_text
    assert "5 минут" in prepared.card_text
    assert "ещё не сохранено" in prepared.card_text
    repository.prepare.assert_awaited_once_with(
        USER, 20, "Напомни отправить отчёт", ARGS, timezone="Europe/Moscow"
    )
    mcp.call_tool.assert_not_awaited()


async def test_prepare_checks_proposal_before_repository(setup_service):
    service, repository, _ = setup_service
    with pytest.raises(ConfirmationError) as error:
        await service.prepare(USER, 20, "Вопрос", replace(PROPOSAL, timezone="localtime"))
    assert error.value.code == "INVALID_ARGUMENTS"
    repository.prepare.assert_not_awaited()


async def test_confirm_uses_one_trusted_metadata_call_and_database_receipt(setup_service):
    service, repository, mcp = setup_service
    result = await service.confirm(USER.user_id, ACTION_ID)
    assert "Напоминание сохранено" in result
    assert str(REMINDER_ID) in result
    assert "2026-10-20T18:30:12.123456+03:00" in result
    assert "Europe/Moscow" in result
    assert CONTEXT["capability"] not in result
    repository.approve.assert_awaited_once_with(USER.user_id, ACTION_ID)
    repository.reconcile.assert_awaited_once_with(USER.user_id, ACTION_ID)
    mcp.call_tool.assert_awaited_once_with(
        "add_reminder", ARGS.model_dump(mode="json"), action_context=CONTEXT
    )
    assert set(mcp.call_tool.call_args.args[1]) == {"text", "remind_at"}


async def test_committed_replay_returns_receipt_without_new_authorization_or_mcp(setup_service):
    service, repository, mcp = setup_service
    repository.approve.return_value = COMMITTED
    result = await service.confirm(USER.user_id, ACTION_ID)
    assert str(REMINDER_ID) in result
    mcp.call_tool.assert_not_awaited()
    repository.reconcile.assert_not_awaited()


@pytest.mark.parametrize(
    "status,code,message",
    [
        ("expired", "ACTION_EXPIRED", "Срок подтверждения истёк."),
        ("cancelled", "ACTION_CANCELLED", "Подтверждение отменено."),
        ("error", "UNAUTHORIZED", "Нет действующего права."),
        ("unknown", "OUTCOME_UNKNOWN", "Состояние неизвестно."),
    ],
)
async def test_rejected_or_unresolved_approval_never_calls_mcp(
    setup_service, status, code, message
):
    service, repository, mcp = setup_service
    repository.approve.return_value = Outcome(status, code, message)
    result = await service.confirm(USER.user_id, ACTION_ID)
    assert result == (UNKNOWN_OUTCOME if status == "unknown" else message)
    mcp.call_tool.assert_not_awaited()


async def test_foreign_action_is_rejected_before_metadata_or_receipt(setup_service):
    service, repository, mcp = setup_service
    repository.approve.side_effect = ConfirmationError("UNAUTHORIZED", "Нет действующего права.")
    result = await service.confirm(2, ACTION_ID)
    assert result == "Нет действующего права."
    assert str(REMINDER_ID) not in result
    repository.approve.assert_awaited_once_with(2, ACTION_ID)
    repository.reconcile.assert_not_awaited()
    mcp.call_tool.assert_not_awaited()


@pytest.mark.parametrize(
    "failure",
    [
        MCPClientError("OUTCOME_UNKNOWN"),
        TimeoutError("SECRET"),
        RuntimeError("SECRET SQL"),
    ],
)
async def test_lost_commit_ack_recovers_same_database_receipt_without_retry(setup_service, failure):
    service, repository, mcp = setup_service
    mcp.call_tool.side_effect = failure
    result = await service.confirm(USER.user_id, ACTION_ID)
    assert str(REMINDER_ID) in result
    assert "SECRET" not in result
    mcp.call_tool.assert_awaited_once()
    repository.approve.assert_awaited_once()
    repository.reconcile.assert_awaited_once_with(USER.user_id, ACTION_ID)
    repository.prepare.assert_not_awaited()


async def test_timeout_with_uncommitted_action_reports_unknown_not_success(setup_service):
    service, repository, mcp = setup_service
    mcp.call_tool.side_effect = MCPClientError("OUTCOME_UNKNOWN")
    repository.reconcile.return_value = Outcome("unknown", "OUTCOME_UNKNOWN", "Неизвестно.")
    assert await service.confirm(USER.user_id, ACTION_ID) == UNKNOWN_OUTCOME
    mcp.call_tool.assert_awaited_once()
    repository.prepare.assert_not_awaited()


async def test_database_recovery_failure_does_not_claim_rollback_or_reveal_details(setup_service):
    service, repository, mcp = setup_service
    repository.reconcile.side_effect = RuntimeError("SECRET-password SQL")
    mcp.call_tool.side_effect = MCPClientError("OUTCOME_UNKNOWN")
    result = await service.confirm(USER.user_id, ACTION_ID)
    assert result == UNKNOWN_OUTCOME
    assert "SECRET" not in result
    assert "отменено" not in result


async def test_cancelled_write_reconciles_same_action_then_propagates_cancellation(setup_service):
    service, repository, mcp = setup_service
    started = asyncio.Event()

    async def waiting(*_args, **_kwargs):
        started.set()
        await asyncio.Event().wait()

    mcp.call_tool.side_effect = waiting
    task = asyncio.create_task(service.confirm(USER.user_id, ACTION_ID))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    repository.reconcile.assert_awaited_once_with(USER.user_id, ACTION_ID)
    mcp.call_tool.assert_awaited_once()


async def test_cancelled_approval_recovers_same_action_without_calling_mcp(setup_service):
    service, repository, mcp = setup_service
    started = asyncio.Event()

    async def waiting(*_args):
        started.set()
        await asyncio.Event().wait()

    repository.approve.side_effect = waiting
    task = asyncio.create_task(service.confirm(USER.user_id, ACTION_ID))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    repository.reconcile.assert_awaited_once_with(USER.user_id, ACTION_ID)
    mcp.call_tool.assert_not_awaited()


async def test_cancel_committed_action_returns_existing_receipt_not_false_cancellation(
    setup_service,
):
    service, repository, mcp = setup_service
    repository.cancel.return_value = COMMITTED
    result = await service.cancel(USER.user_id, ACTION_ID)
    assert "Напоминание сохранено" in result
    assert "отменено" not in result
    mcp.call_tool.assert_not_awaited()


async def test_invalidate_delegates_user_scope_without_mcp(setup_service):
    service, repository, mcp = setup_service
    assert await service.invalidate(USER.user_id)
    repository.invalidate.assert_awaited_once_with(USER.user_id)
    mcp.call_tool.assert_not_awaited()


async def test_timezones_use_explicit_iana_and_database_clock(setup_service):
    service, repository, mcp = setup_service
    result = await service.timezone(USER.user_id, " Europe/Moscow ")
    repository.set_timezone.assert_awaited_once_with(USER.user_id, "Europe/Moscow")
    repository.now.assert_awaited_once()
    assert "2026-10-19T09:00:00+03:00" in result
    assert "При смене часового пояса" in result
    mcp.call_tool.assert_not_awaited()


@pytest.mark.parametrize("zone", ["localtime", "+03:00", "No/Zone", "", "posix/UTC"])
async def test_invalid_zone_does_not_update_or_invalidate_existing_preference(setup_service, zone):
    service, repository, _ = setup_service
    assert "IANA" in await service.timezone(USER.user_id, zone)
    repository.set_timezone.assert_not_awaited()
    repository.invalidate.assert_not_awaited()


async def test_timezone_show_and_missing_zone_do_not_require_llm(setup_service):
    service, repository, _ = setup_service
    result = await service.timezone(USER.user_id)
    assert "Часовой пояс: Europe/Moscow." in result
    assert "2026-10-19T09:00:00+03:00" in result
    repository.now.assert_awaited_once()
    repository.timezone.return_value = None
    assert "/timezone Europe/Moscow" in await service.timezone(USER.user_id)
    repository.set_timezone.assert_not_awaited()


async def test_why_shows_owner_event_safe_arguments_and_fixed_reason(setup_service):
    service, repository, mcp = setup_service
    repository.latest_event.return_value = SafeAgentEvent(
        created_at=NOW,
        decision="call_tool",
        target="get_weather",
        status="OK",
        reason="Нужны актуальные погодные данные.",
        argument_summary="Город: Казань",
        error_code=None,
        llm_calls=2,
        mcp_calls=1,
        duration_ms=10,
    )
    result = await service.why(USER.user_id)
    assert "2026-10-19T09:00:00+03:00" in result
    assert "get_weather" in result and "Город: Казань" in result
    assert "Проверка: пройдена" in result and "Выполнение: успешно" in result
    assert "Нужны актуальные погодные данные" in result
    assert str(USER.user_id) not in result and "capability" not in result
    repository.latest_event.assert_awaited_once_with(USER.user_id)
    mcp.call_tool.assert_not_awaited()


async def test_why_without_decisions_is_explicit(setup_service):
    service, _, _ = setup_service
    assert await service.why(USER.user_id) == "Пока нет запросов с решением агента."


async def test_malformed_committed_receipt_cannot_be_claimed_success(setup_service):
    service, repository, mcp = setup_service
    repository.approve.return_value = Outcome("committed", None, "Успех без данных")
    assert await service.confirm(USER.user_id, ACTION_ID) == UNKNOWN_OUTCOME
    mcp.call_tool.assert_not_awaited()
