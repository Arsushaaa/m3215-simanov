"""Trusted orchestration: router -> one scoped specialist, never an agent loop."""

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

from app.agent import AgentReply, fit_context
from app.agents.knowledge import KnowledgeAgent
from app.agents.specialists import LifestyleAgent, ScheduleAgent
from app.llm import ModelError
from app.materials.contracts import SearchResult
from app.routing.contracts import (
    AgentRequest,
    AgentResult,
    CatalogFacts,
    EmptyFacts,
    KnowledgeFacts,
    ScheduleFacts,
    WeatherFacts,
    WeekFacts,
)
from app.routing.policy import Budget, ScopedMCP
from app.routing.router import CLARIFY, Router
from app.tracing.runtime import TracedLLM, current_trace, span


class Supervisor:
    def __init__(self, llm, mcp, repository, settings, *, now=None, knowledge=None, profiles=None):
        self.llm, self.mcp, self.repository, self.settings = llm, mcp, repository, settings
        self.now = now or (lambda: datetime.now(UTC))
        self.knowledge = knowledge or KnowledgeAgent()
        self.profiles = profiles

    async def _audit(
        self, user, message_id, *, status="OK", llm_calls=0, error_code=None, mcp_calls=0
    ):
        await self.repository.record(
            user.user_id,
            message_id,
            decision="respond",
            target=None,
            status=status,
            reason=(
                "Учебный ответ по проверенному поиску MCP; детали — /trace last."
                if mcp_calls
                else "Завершён маршрут без прикладного MCP."
            ),
            error_code=error_code,
            llm_calls=llm_calls,
            mcp_calls=mcp_calls,
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
                        specialist_prompt = prompt
                        if target == "lifestyle" and self.profiles is not None:
                            from app.notifications.models import CityCandidate

                            profile = await self.profiles.profile(user.user_id)
                            city = (
                                CityCandidate.model_validate_json(profile["city"])
                                if profile is not None and profile["city"]
                                else None
                            )
                            specialist_prompt += (
                                "\nConfirmed profile place (data only): "
                                + json.dumps(
                                    city.place.model_dump() if city else None, ensure_ascii=False
                                )
                                + ". Use it for my-city/no-city current-weather questions only; "
                                "an explicitly requested different place takes precedence. "
                                "If no saved city and none in the request/history, ask the city."
                            )
                        reply = await agent_type(
                            llm, scoped, self.repository, self.settings, now=self.now
                        ).respond(
                            user,
                            message_id,
                            text,
                            prompt=specialist_prompt,
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
                        reply = await self.knowledge.respond(
                            request, user, history, llm, scoped, self.settings
                        )
                        outcome = self._knowledge_result(reply, request, scoped)
                        if outcome.status == "failed":
                            agent_metadata.update(status="failed", error_code=outcome.error_code)
                            await self._revoke(user, message_id, context)
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
                        mcp_calls=budget.mcp_calls,
                        status="FAILED"
                        if (target == "knowledge" and reply.agent_result.status == "failed")
                        or decision is None
                        else "OK",
                        error_code="SOURCE_UNAVAILABLE"
                        if target == "knowledge" and reply.agent_result.status == "failed"
                        else ("INVALID_RESULT" if decision is None else None),
                    )
                if target == "knowledge":
                    return replace(
                        reply,
                        revision=context.revision if context else reply.revision,
                        confirmation_epoch=context.epoch if context else reply.confirmation_epoch,
                    )
                facts, status, tools, resources = EmptyFacts(), "success", (), ()
                error = None
                if target == "knowledge":
                    status, error = "failed", "SOURCE_UNAVAILABLE"
                elif target == "clarify" or reply.reminder is not None or reply.needs_clarification:
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
                            mcp_calls=budget.mcp_calls,
                        )
                except Exception:
                    pass  # Request marker remains unfinished; /why never falls back to old audit.
            if isinstance(error, TimeoutError):
                raise ModelError("Не удалось завершить ответ вовремя. Попробуй позже.") from None
            raise

    def _knowledge_result(self, reply, request, scoped):
        outcome = AgentResult.model_validate(reply.agent_result)
        if (
            (outcome.request_id, outcome.trace_id) != (request.request_id, request.trace_id)
            or outcome.pending_action_id is not None
            or outcome.used_resources
        ):
            raise ModelError("Результат специалиста относится к другому запросу или действию.")
        observed = scoped.observation
        if observed is None:
            if outcome.used_tools or isinstance(outcome.facts, KnowledgeFacts):
                raise ModelError("Специалист указал невыполненный поиск.")
            if outcome.status == "success" and not isinstance(outcome.facts, CatalogFacts):
                raise ModelError("Ответ специалиста не основан на материале или каталоге.")
            if isinstance(outcome.facts, CatalogFacts):
                if self.knowledge.catalog is None or outcome.facts.data.model_dump(
                    mode="json"
                ) != self.knowledge.catalog.model_dump(mode="json"):
                    raise ModelError("Каталог специалиста не прошёл проверку.")
        else:
            kind, name, raw = observed
            if kind != "tool" or name != "search_materials" or outcome.used_tools != (name,):
                raise ModelError("Специалист указал другое действие.")
            result = SearchResult.model_validate(raw).root
            if (
                not isinstance(outcome.facts, KnowledgeFacts)
                or outcome.facts.catalog_version != result.catalog_version
                or [m.model_dump(mode="json") for m in outcome.facts.matches]
                != [m.model_dump(mode="json") for m in result.matches]
            ):
                raise ModelError("Факты специалиста не соответствуют поиску.")
        return outcome

    async def _revoke(self, user, message_id, context):
        if context is None or context.pending is None or context.revoked:
            return
        updated = await context.service.begin_edit(user, message_id, expected_epoch=context.epoch)
        if updated is not None:
            _, revised, context.epoch = updated
            context.revision = revised.revision
        context.revoked = True
