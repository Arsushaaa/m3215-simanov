"""Context-local spans follow a request through model, policy and Telegram delivery."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from uuid import uuid4

from app.routing.contracts import TraceSpan

logger = logging.getLogger(__name__)
current_trace = ContextVar("labbi_trace", default=None)
parent_span = ContextVar("labbi_parent_span", default=None)


class TraceUnavailable(Exception):
    """Cannot record a request safely; do not call a paid/limited external service."""


class Trace:
    def __init__(self, repository):
        self.repository = repository
        self.trace_id, self.request_id = uuid4(), uuid4()
        self.status = "ok"
        self.count = 0

    async def persist(self, method, *args):
        try:
            async with asyncio.timeout(5):
                return await getattr(self.repository, method)(*args)
        except Exception:
            raise TraceUnavailable from None

    @asynccontextmanager
    async def span(self, component, operation, **fields):
        self.count += 1
        if self.count > 64:
            raise RuntimeError("Trace span budget exhausted")
        start, clock = datetime.now(UTC), time.monotonic()
        value = TraceSpan(
            trace_id=self.trace_id,
            request_id=self.request_id,
            span_id=uuid4(),
            parent_span_id=parent_span.get(),
            component=component,
            operation=operation,
            status="running",
            started_at=start,
            finished_at=None,
            duration_ms=None,
            **fields,
        )
        await self.persist("span", value)
        token = parent_span.set(value.span_id)
        finish_fields = {}
        status = "ok"
        try:
            yield finish_fields
        except asyncio.CancelledError:
            status = self.status = "cancelled"
            raise
        except BaseException as error:
            status = self.status = "failed"
            if getattr(error, "trace_code", None) == "POLICY_DENIED":
                finish_fields["error_code"] = "POLICY_DENIED"
            raise
        finally:
            parent_span.reset(token)
            finished = max(start, datetime.now(UTC))
            completed = TraceSpan.model_validate(
                {
                    **value.model_dump(),
                    **finish_fields,
                    "status": (
                        finish_fields.get("status", self.status)
                        if value.parent_span_id is None
                        else finish_fields.get("status", status)
                    )
                    if status == "ok"
                    else status,
                    "finished_at": finished,
                    "duration_ms": round((time.monotonic() - clock) * 1000),
                }
            )
            if completed.status != "ok":
                self.status = completed.status
            try:
                await self.persist("span", completed)
            except Exception as error:
                self.status = "failed"
                logger.error("Трасса: ошибка записи %s", type(error).__name__)


@asynccontextmanager
async def span(component, operation, **fields):
    trace = current_trace.get()
    if trace is None:
        yield {}
    else:
        async with trace.span(component, operation, **fields) as result:
            yield result


def failed():
    trace = current_trace.get()
    if trace is not None:
        trace.status = "failed"


class TraceManager:
    def __init__(self, repository):
        self.repository = repository

    @asynccontextmanager
    async def request(self, owner, *, component="supervisor", operation="receive"):
        trace = Trace(self.repository)
        await trace.persist("start", owner, trace.trace_id, trace.request_id, datetime.now(UTC))
        token = current_trace.set(trace)
        parent_token = parent_span.set(None)
        try:
            async with trace.span(component, operation):
                yield trace
        finally:
            parent_span.reset(parent_token)
            current_trace.reset(token)
            try:
                await trace.persist("finish", trace.trace_id, trace.status, datetime.now(UTC))
            except Exception as error:
                logger.error("Трасса: ошибка завершения %s", type(error).__name__)


class TracedLLM:
    def __init__(self, llm, component, model_name, budget=None):
        self.llm, self.component, self.model_name, self.budget = llm, component, model_name, budget

    async def _call(self, operation, *args, **kwargs):
        if self.budget is not None:
            self.budget.claim_llm()
        async with span(
            self.component,
            "model_decide" if operation == "decide" else "model_final",
            model_name=self.model_name,
        ) as metadata:
            result = await getattr(self.llm, operation)(*args, **kwargs)
            metadata.update(
                input_tokens=getattr(result, "input_tokens", None),
                output_tokens=getattr(result, "output_tokens", None),
            )
            return result

    async def decide(self, *args, **kwargs):
        return await self._call("decide", *args, **kwargs)

    async def complete(self, *args, **kwargs):
        return await self._call("complete", *args, **kwargs)
