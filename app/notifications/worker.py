"""Bounded reminder delivery. No LLM/MCP and no task/timer per stored reminder."""

import asyncio
import logging
from contextlib import nullcontext
from contextvars import Context

from aiogram.types import LinkPreviewOptions

from app.notifications.delivery_models import DeliveryFailure, classify_failure, valid_payload
from app.tracing.runtime import current_trace, span

logger = logging.getLogger(__name__)


class ReminderWorker:
    def __init__(self, repository, bot, *, traces=None, poll_seconds=5, briefing=None):
        if type(poll_seconds) not in (int, float) or not 0 < poll_seconds <= 60:
            raise ValueError("Invalid polling interval")
        self.repository, self.bot, self.traces = repository, bot, traces
        self.briefing = briefing
        self.poll_seconds = poll_seconds
        self._stop = asyncio.Event()
        self._running = asyncio.Semaphore(2)
        self.task = None

    def start(self):
        if self.task is not None:
            raise RuntimeError("Worker already started")
        self.task = asyncio.create_task(self.run(), context=Context())
        return self.task

    def _trace(self, owner, operation):
        return (
            self.traces.request(owner, component="reminder_delivery", operation=operation)
            if self.traces is not None
            else nullcontext()
        )

    async def tick(self):
        await self.repository.recover(limit=10)
        if self._stop.is_set():
            return
        leases = await self.repository.claim_preparations(limit=10)
        try:
            for lease in leases:
                if self._stop.is_set():
                    break
                async with self._trace(lease.owner, "prepare_notification"):
                    async with span("reminder_delivery", "prepare_notification"):
                        await self.repository.prepare(lease)
        finally:
            if self._stop.is_set() and leases:
                await self.repository.release_preparations(leases)
        if self._stop.is_set():
            return
        if self.briefing is not None:
            await self.briefing.tick()
        if self._stop.is_set():
            return
        candidates = await self.repository.candidates(limit=10)
        # Acquire a lease only after a slot is free, not while waiting behind other HTTP calls.
        await asyncio.gather(*(self._deliver(event, owner) for event, owner in candidates))

    async def _deliver(self, event, owner):
        async with self._running:
            if self._stop.is_set():
                return
            lease = None
            send_started = False
            message_id = None
            try:
                async with self._trace(owner, "claim_notification"):
                    async with span("reminder_delivery", "claim_notification"):
                        trace = current_trace.get()
                        lease = await self.repository.begin_send(
                            event, owner, trace_id=trace.trace_id if trace is not None else None
                        )
                    if lease is None:
                        return
                    if not valid_payload(lease.body):
                        await self.repository.failed(
                            lease, DeliveryFailure("INVALID_PAYLOAD", False)
                        )
                        return
                    try:
                        if not await self.repository.preflight(lease):
                            return
                        async with span("telegram", "send_message") as metadata:
                            if not await self.repository.preflight(lease):
                                metadata.update(status="cancelled", error_code="ACTION_CANCELLED")
                                return
                            send_started = True
                            async with asyncio.timeout(20):
                                result = await self.bot.send_message(
                                    chat_id=lease.destination,
                                    text=lease.body,
                                    parse_mode=None,
                                    link_preview_options=LinkPreviewOptions(is_disabled=True),
                                    allow_paid_broadcast=False,
                                    request_timeout=20,
                                )
                            message_id = getattr(result, "message_id", None)
                            if type(message_id) is not int or not 0 < message_id <= 2**63 - 1:
                                message_id = None
                                raise ValueError("Invalid acknowledgement")
                    except asyncio.CancelledError:
                        raise
                    except Exception as error:
                        failure = classify_failure(error)
                        await self.repository.failed(lease, failure)
                        logger.warning(
                            "reminder delivery code=%s error_type=%s",
                            failure.code,
                            type(error).__name__,
                        )
                        return
                    recorded = await self.repository.sent(lease, message_id)
                    if not recorded:
                        async with span("reminder_delivery", "claim_notification") as metadata:
                            metadata.update(status="failed", error_code="OUTCOME_UNKNOWN")
            except asyncio.CancelledError:
                if lease is not None:
                    await self._settle(lease, message_id, send_started)
                raise
            except Exception as error:
                # Storage failure must not trigger immediate new HTTP.
                logger.error("reminder worker error_type=%s", type(error).__name__)
                if lease is not None:
                    await self._settle(lease, message_id, send_started)

    async def _settle(self, lease, message_id, send_started):
        async def record():
            async with asyncio.timeout(5):
                if message_id is not None:
                    await self.repository.sent(lease, message_id)
                else:
                    await self.repository.failed(
                        lease, DeliveryFailure("PROCESS_INTERRUPTED", True, send_started)
                    )

        task = asyncio.create_task(record())
        try:
            await asyncio.shield(task)
        except (Exception, asyncio.CancelledError) as error:
            logger.error("reminder reconciliation error_type=%s", type(error).__name__)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def run(self):
        while not self._stop.is_set():
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.error("reminder poll error_type=%s", type(error).__name__)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.poll_seconds)
            except TimeoutError:
                pass

    async def close(self):
        self._stop.set()
        if self.task is None:
            return
        try:
            async with asyncio.timeout(35):
                await asyncio.shield(self.task)
        except TimeoutError:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
