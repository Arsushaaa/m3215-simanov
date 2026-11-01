"""Trusted orchestration: router -> one scoped specialist, never an agent loop."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

from app.agent import AgentReply, fit_context
from app.agents.specialists import LifestyleAgent, ScheduleAgent
from app.llm import ModelError
from app.routing.contracts import (
    AgentRequest,
    AgentResult,
    EmptyFacts,
    ScheduleFacts,
    WeatherFacts,
    WeekFacts,
)
from app.routing.policy import Budget, ScopedMCP
from app.routing.router import CLARIFY, Router
from app.tracing.runtime import TracedLLM, current_trace, span


class Supervisor:
    def __init__(self, llm, mcp, repository, settings, *, now=None):
        self.llm, self.mcp, self.repository, self.settings = llm, mcp, repository, settings
        self.now = now or (lambda: datetime.now(UTC))

    async def _audit(self, user, message_id, *, status="OK", llm_calls=0, error_code=None):
        await self.repository.record(
            user.user_id,
            message_id,
            decision="respond",
            target=None,
            status=status,
            reason="Завершён маршрут без прикладного MCP.",
            error_code=error_code,
            llm_calls=llm_calls,
            mcp_calls=0,
            duration_ms=0,
            argument_summary="",
        )

    async def respond(
        self,
        user,
        message_id,
        text,
        *,
        prompt,
        history,
        continuing=False,
        confirmation_context=None,
    ):
        context = confirmation_context
        specialist_started = False
        budget = Budget()
        trace = current_trace.get()
        request = AgentRequest(
            request_id=trace.request_id if trace else uuid4(),
            trace_id=trace.trace_id if trace else uuid4(),
            actor_id=user.user_id,
            text=text,
            intent="user_query",
            occurred_at=self.now(),
            timezone=await self.repository.timezone(user.user_id),
        )
        if context is None:
            if await self.repository.has_processed(user.user_id, message_id):
                raise ModelError("Этот запрос уже обработан. Отправь новое сообщение.")
            await self.repository.start_request(user.user_id, message_id)
        try:
            async with asyncio.timeout(165):
                router_llm = TracedLLM(self.llm, "router", self.settings.openrouter_model, budget)
                decision = await Router(router_llm, self.settings).route(
                    request,
                    history,
                    user.temperature,
                    pending=context is not None and context.pending is not None,
                )
                target = decision.target if decision is not None else "clarify"
                confidence = decision.confidence if decision is not None else None
                scoped = ScopedMCP(self.mcp, target, budget=budget)
                llm = TracedLLM(
                    self.llm,
                    target if target != "clarify" else "supervisor",
                    self.settings.openrouter_model,
                    budget,
                )
                async with span(
                    target if target != "clarify" else "supervisor",
                    "agent",
                    target=target,
                    confidence=confidence,
                ) as agent_metadata:
                    if target in {"schedule", "lifestyle"}:
                        specialist_started = True
                        agent_type = ScheduleAgent if target == "schedule" else LifestyleAgent
                        reply = await agent_type(
                            llm, scoped, self.repository, self.settings, now=self.now
                        ).respond(
                            user,
                            message_id,
                            text,
                            prompt=prompt,
                            history=history if target == "schedule" else history[-4:],
                            continuing=continuing,
                            confirmation_context=context,
                        )
                    elif target == "legacy_study":
                        messages = fit_context(
                            prompt + "\nNo tools or writes are available. "
                            "Answer the study question; never claim an external action completed.",
                            history,
                            text,
                            self.settings,
                            preserve_last=continuing,
                        )
                        result = await llm.complete(messages, user.temperature, disable_tools=True)
                        reply = AgentReply(result.text, result.truncated)
                    elif target == "knowledge":
                        agent_metadata.update(status="failed", error_code="SOURCE_UNAVAILABLE")
                        reply = AgentReply(
                            "Источник учебных материалов ещё не подключён. "
                            "Поиск по материалам будет доступен после следующего этапа ЛР3."
                        )
                    else:
                        # A failed/uncertain interpretation must not leave an old card live.
                        if context is not None and context.pending is not None:
                            await self._revoke(user, message_id, context)
                        reply = AgentReply(CLARIFY)
                if not specialist_started:
                    await self._audit(
                        user,
                        message_id,
                        llm_calls=budget.llm_calls - 1,
                        status="FAILED" if target == "knowledge" or decision is None else "OK",
                        error_code="SOURCE_UNAVAILABLE"
                        if target == "knowledge"
                        else ("INVALID_RESULT" if decision is None else None),
                    )
                facts, status, tools, resources = EmptyFacts(), "success", (), ()
                error = None
                if target == "knowledge":
                    status, error = "failed", "SOURCE_UNAVAILABLE"
                elif target == "clarify" or reply.reminder is not None:
                    status = "needs_clarification"
                if scoped.observation is not None:
                    kind, name, raw = scoped.observation
                    facts = (
                        {"get_schedule": ScheduleFacts, "get_weather": WeatherFacts}[name]
                        if kind == "tool"
                        else WeekFacts
                    )(data=raw)
                    status = {
                        "ok": "success",
                        "ambiguous": "needs_clarification",
                        "not_found": "not_found",
                        "outside_coverage": "not_found",
                        "error": "failed",
                    }[getattr(facts.data.root, "status", "ok")]
                    tools, resources = ((name,), ()) if kind == "tool" else ((), (name,))
                    if status == "failed":
                        error = facts.data.root.error.code
                outcome = AgentResult(
                    request_id=request.request_id,
                    trace_id=request.trace_id,
                    status=status,
                    message=reply.text,
                    confidence=confidence,
                    facts=facts,
                    used_tools=tools,
                    used_resources=resources,
                    error_code=error,
                )
                return replace(
                    reply,
                    agent_result=outcome,
                    revision=context.revision if context else reply.revision,
                    confirmation_epoch=context.epoch if context else reply.confirmation_epoch,
                )
        except BaseException as error:
            try:
                async with asyncio.timeout(5):
                    await self._revoke(user, message_id, context)
            except Exception:
                pass
            if not specialist_started:
                try:
                    async with asyncio.timeout(5):
                        await self._audit(
                            user,
                            message_id,
                            status="CANCELLED"
                            if isinstance(error, asyncio.CancelledError)
                            else "FAILED",
                            llm_calls=max(0, budget.llm_calls - 1),
                        )
                except Exception:
                    pass  # Request marker remains unfinished; /why never falls back to old audit.
            if isinstance(error, TimeoutError):
                raise ModelError("Не удалось завершить ответ вовремя. Попробуй позже.") from None
            raise

    async def _revoke(self, user, message_id, context):
        if context is None or context.pending is None or context.revoked:
            return
        updated = await context.service.begin_edit(user, message_id, expected_epoch=context.epoch)
        if updated is not None:
            _, revised, context.epoch = updated
            context.revision = revised.revision
        context.revoked = True
