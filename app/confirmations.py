"""Подтверждения и команды настроек: детерминированные ответы без вызова модели."""

import asyncio
from typing import TYPE_CHECKING
from uuid import UUID
from zoneinfo import ZoneInfo

from pydantic import TypeAdapter, ValidationError

from app.confirmation_models import Authorization, ConfirmationError, Outcome, PreparedAction
from app.mcp.client import MCPClient, MCPClientError
from app.mcp.contracts import ReminderArguments, TimezoneName
from app.reminder_intent import BUTTON_REQUIRED

if TYPE_CHECKING:
    from app.agent import ReminderProposal
    from app.agent_repository import ConfirmationRepository
    from app.repository import UserSettings

_TIMEZONE = TypeAdapter(TimezoneName)
UNKNOWN_OUTCOME = (
    "Результат записи пока неизвестен. Нажми ту же кнопку подтверждения ещё раз: "
    "сначала я проверю состояние этого действия. Новое напоминание создавать не нужно."
)


class ConfirmationService:
    def __init__(self, repository: "ConfirmationRepository", mcp: MCPClient):
        self.repository, self.mcp = repository, mcp

    async def button_reply(self, user_id: int, message_id: int) -> str | None:
        """A textual yes to an active card never grants permission to write."""
        if await self.repository.current_card(user_id) is None:
            return None
        if await self.repository.has_processed(user_id, message_id):
            raise ConfirmationError(
                "ACTION_CONFLICT", "Этот запрос уже обработан. Отправь новое сообщение."
            )
        await self.repository.record(
            user_id,
            message_id,
            decision="clarify",
            target="add_reminder",
            status="CLARIFY",
            reason="Для записи подготовленного напоминания требуется кнопка владельца.",
            error_code="CONFIRMATION_REQUIRED",
            llm_calls=0,
            mcp_calls=0,
        )
        return BUTTON_REQUIRED

    async def prepare(
        self,
        user: "UserSettings",
        message_id: int,
        question: str,
        proposal: "ReminderProposal",
    ) -> PreparedAction:
        try:
            arguments = ReminderArguments.model_validate(proposal.arguments.model_dump(mode="json"))
            timezone = _TIMEZONE.validate_python(proposal.timezone)
        except (ValidationError, AttributeError, ValueError, TypeError):
            raise ConfirmationError(
                "INVALID_ARGUMENTS", "Параметры напоминания некорректны."
            ) from None
        # Репозиторий повторно проверяет параметры и DB-clock под блокировкой пользователя.
        return await self.repository.prepare(
            user, message_id, question, arguments, timezone=timezone
        )

    @staticmethod
    def _outcome(outcome: Outcome) -> str:
        if outcome.status != "committed":
            return UNKNOWN_OUTCOME if outcome.status == "unknown" else outcome.message
        if outcome.reminder_id is None or outcome.remind_at is None or outcome.timezone is None:
            return UNKNOWN_OUTCOME
        try:
            timezone = _TIMEZONE.validate_python(outcome.timezone)
            if outcome.remind_at.tzinfo is None or outcome.remind_at.utcoffset() is None:
                return UNKNOWN_OUTCOME
            moment = outcome.remind_at.astimezone(ZoneInfo(timezone)).isoformat()
        except (ValidationError, ValueError, AttributeError):
            return UNKNOWN_OUTCOME
        return (
            f"Напоминание сохранено.\nID: {outcome.reminder_id}\n"
            f"Когда: {moment}\nЧасовой пояс: {timezone}\n"
            "Отправка уведомлений по времени пока не подключена."
        )

    async def _reconcile(self, user_id: int, action_id: UUID | str) -> Outcome | None:
        try:
            async with asyncio.timeout(3):
                return await self.repository.reconcile(user_id, action_id)
        except Exception:
            # Ни недоступная БД, ни её диагностика не доказывают успех или откат записи.
            return None

    async def confirm(self, user_id: int, action_id: UUID | str) -> str:
        try:
            return await self._confirm(user_id, action_id)
        except asyncio.CancelledError:
            # Включая отмену во время approve или чтения квитанции после MCP.
            await self._reconcile(user_id, action_id)
            raise

    async def _confirm(self, user_id: int, action_id: UUID | str) -> str:
        try:
            authorization = await self.repository.approve(user_id, action_id)
        except ConfirmationError as error:
            return error.user_message
        if isinstance(authorization, Outcome):
            # Повтор после commit возвращает квитанцию без нового права и MCP-вызова.
            return self._outcome(authorization)
        if not isinstance(authorization, Authorization):
            return UNKNOWN_OUTCOME
        try:
            await self.mcp.call_tool(
                "add_reminder",
                authorization.arguments.model_dump(mode="json"),
                action_context=authorization.context,
            )
        except (MCPClientError, TimeoutError):
            pass
        except Exception:
            # Не раскрываем исключение и не запускаем новую операцию/второй MCP-вызов.
            pass
        outcome = await self._reconcile(user_id, authorization.action_id)
        return self._outcome(outcome) if outcome is not None else UNKNOWN_OUTCOME

    async def cancel(self, user_id: int, action_id: UUID | str) -> str:
        try:
            return self._outcome(await self.repository.cancel(user_id, action_id))
        except ConfirmationError as error:
            return error.user_message

    async def invalidate(self, user_id: int) -> bool:
        return await self.repository.invalidate(user_id)

    async def timezone(self, user_id: int, argument: str | None = None) -> str:
        if argument is None:
            timezone = await self.repository.timezone(user_id)
            if timezone is None:
                return "Часовой пояс не задан. Используй /timezone Europe/Moscow."
        else:
            try:
                timezone = _TIMEZONE.validate_python(argument)
            except ValidationError:
                return "Укажи существующую IANA-зону, например /timezone Europe/Moscow."
            try:
                timezone = await self.repository.set_timezone(user_id, timezone)
            except ConfirmationError as error:
                return error.user_message
        now = await self.repository.now()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ConfirmationError("SOURCE_UNAVAILABLE", "Не удалось получить текущее время.")
        moment = now.astimezone(ZoneInfo(timezone)).isoformat(timespec="seconds")
        result = f"Часовой пояс: {timezone}.\nТекущее время: {moment}."
        if argument is not None:
            result += "\nПри смене часового пояса прежнее неподтверждённое действие отменяется."
        return result

    async def why(self, user_id: int) -> str:
        event = await self.repository.latest_event(user_id)
        if event is None:
            return "Пока нет запросов с решением агента."
        timezone = await self.repository.timezone(user_id) or "UTC"
        moment = event.created_at.astimezone(ZoneInfo(timezone)).isoformat(timespec="seconds")
        actions = {
            "respond": "ответ без инструмента",
            "clarify": "уточнение",
            "call_tool": "вызов инструмента",
            "read_resource": "чтение ресурса",
            "confirm": "подтверждение напоминания",
            "cancel": "отзыв подтверждения",
        }
        statuses = {
            "OK": "успешно",
            "CLARIFY": "нужно уточнение или подтверждение; действие не выполнено",
            "REJECTED": "отклонено",
            "FAILED": "ошибка",
            "CANCELLED": "отменено",
            "OUTCOME_UNKNOWN": "результат пока неизвестен",
        }
        validation_errors = {
            "INVALID_ARGUMENTS",
            "UNKNOWN_TOOL",
            "UNAUTHORIZED",
            "CONFIRMATION_REQUIRED",
            "ACTION_CANCELLED",
            "ACTION_EXPIRED",
            "ACTION_CONFLICT",
            "TIMEZONE_REQUIRED",
            "AMBIGUOUS_TIME",
        }
        validation = (
            "отклонено"
            if event.error_code in validation_errors
            else "пройдена"
            if event.status == "OK"
            else "не завершена или не требовалась"
        )
        target = f" {event.target}" if event.target else ""
        return (
            f"Последний запрос: {moment} ({timezone})\n"
            f"Действие: {actions.get(event.decision, 'обработка запроса')}{target}\n"
            f"Аргументы: {event.argument_summary or 'нет'}\n"
            f"Проверка: {validation}\n"
            f"Выполнение: {statuses.get(event.status, 'не завершено')}\n"
            f"Причина: {event.reason}"
        )
