"""Bound process-local generation admission; control commands stay available.

These limits contain bursts and queued work, not a public bot's shared daily
provider quota. Accepted attempts count even after an error or cancellation.
"""

import asyncio
import math
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

MAX_PENDING_PER_USER = 2
MAX_PENDING_TOTAL = 16
MAX_RUNNING = 4
REQUESTS_PER_MINUTE = 8
MAX_TRACKED_USERS = 1024
WINDOW_SECONDS = 60
MAX_CONTROL_PENDING = 64


class SlotAdmission:
    def __init__(self, release):
        self._release = release

    def release(self):
        if self._release is not None:
            release, self._release = self._release, None
            release()


class CommandSlots:
    """A separate, rate-free bound for commands waiting for the user lock."""

    def __init__(self):
        self._owners: dict[int, int] = {}
        self._total = 0

    def admit(self, owner):
        if self._owners.get(owner, 0) >= 2 or self._total >= 16:
            raise RequestLimitError(
                "Команда уже ожидает выполнения. Дождись ответа или используй /stop."
            )
        self._owners[owner] = self._owners.get(owner, 0) + 1
        self._total += 1

        def release():
            self._total -= 1
            count = self._owners[owner] - 1
            if count:
                self._owners[owner] = count
            else:
                del self._owners[owner]

        return SlotAdmission(release)


class ControlSlots:
    """Coalesce by owner/lane, never by arbitrary user-supplied arguments."""

    def __init__(self):
        self._active: set[tuple[int, str]] = set()

    def active(self, owner, lane):
        return (owner, lane) in self._active

    def admit(self, owner, lane, *, stop_reserve=False):
        key = (owner, lane)
        if key in self._active:
            return None
        if len(self._active) >= MAX_CONTROL_PENDING and not (
            lane == "stop" and stop_reserve and len(self._active) < MAX_CONTROL_PENDING + 16
        ):
            return None
        self._active.add(key)
        return SlotAdmission(lambda: self._active.discard(key))


class RequestLimitError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.user_message = message


@dataclass
class _UserWindow:
    accepted: deque = field(default_factory=deque)
    pending: int = 0
    last_notice: float = float("-inf")


class Admission:
    def __init__(self, limits, owner):
        self._limits, self._owner, self._released = limits, owner, False

    def release(self):
        if self._released:
            return
        self._released = True
        self._limits._users[self._owner].pending -= 1
        self._limits._pending -= 1


class RequestLimits:
    def __init__(self, *, clock=time.monotonic):
        self._clock = clock
        self._users: dict[int, _UserWindow] = {}
        self._pending = 0
        self._workers = asyncio.Semaphore(MAX_RUNNING)
        self._last_notice = float("-inf")

    def _prune(self, now):
        for owner, state in list(self._users.items()):
            while state.accepted and now - state.accepted[0] >= WINDOW_SECONDS:
                state.accepted.popleft()
            if not state.pending and not state.accepted:
                del self._users[owner]

    def admit(self, owner: int) -> Admission:
        """No await between checking and claiming slots in this event loop."""
        now = self._clock()
        self._prune(now)
        state = self._users.get(owner)
        if state is not None and state.pending >= MAX_PENDING_PER_USER:
            raise RequestLimitError(
                "У тебя уже есть активный и ожидающий запрос. Дождись ответа или используй /stop."
            )
        if state is not None and len(state.accepted) >= REQUESTS_PER_MINUTE:
            seconds = max(1, math.ceil(WINDOW_SECONDS - (now - state.accepted[0])))
            raise RequestLimitError(f"Слишком много запросов. Попробуй через {seconds} сек.")
        if self._pending >= MAX_PENDING_TOTAL or (
            state is None and len(self._users) >= MAX_TRACKED_USERS
        ):
            raise RequestLimitError("Бот сейчас занят. Подожди немного и отправь запрос снова.")
        if state is None:
            state = self._users[owner] = _UserWindow()
        state.accepted.append(now)
        state.pending += 1
        self._pending += 1
        return Admission(self, owner)

    def notify_rejection(self, owner: int) -> bool:
        """Coalesce rejection replies so flooding cannot create a reply queue."""
        now = self._clock()
        state = self._users.get(owner)
        if now - self._last_notice < 1 or (state is not None and now - state.last_notice < 5):
            return False
        self._last_notice = now
        if state is not None:
            state.last_notice = now
        return True

    @asynccontextmanager
    async def running(self):
        async with self._workers:
            yield
