import asyncio

import pytest

from app.request_limits import (
    MAX_PENDING_TOTAL,
    MAX_RUNNING,
    MAX_TRACKED_USERS,
    REQUESTS_PER_MINUTE,
    CommandSlots,
    ControlSlots,
    RequestLimitError,
    RequestLimits,
)


def test_command_queue_has_its_own_bounded_slots_without_generation_rate():
    slots = CommandSlots()
    first, second = slots.admit(1), slots.admit(1)
    with pytest.raises(RequestLimitError, match="Команда уже"):
        slots.admit(1)
    leases = [first, second]
    for owner in range(2, 16):
        leases.append(slots.admit(owner))
    with pytest.raises(RequestLimitError):
        slots.admit(200)
    first.release()
    first.release()
    replacement = slots.admit(200)
    for lease in leases:
        lease.release()
    replacement.release()
    assert slots._total == 0 and slots._owners == {}


def test_control_commands_coalesce_and_stop_has_a_bounded_reserved_lane():
    slots = ControlSlots()
    leases = [slots.admit(owner, "change") for owner in range(64)]
    assert slots.admit(0, "change") is None
    assert slots.admit(200, "callback") is None
    stops = [slots.admit(owner, "stop", stop_reserve=True) for owner in range(16)]
    assert all(stops)
    assert slots.admit(17, "stop", stop_reserve=True) is None
    assert len(slots._active) == 80
    for lease in leases + stops:
        lease.release()
        lease.release()
    assert slots._active == set()
    confirm = slots.admit(1, "callback")
    cancel = slots.admit(1, "cancel")
    assert confirm is not None and cancel is not None
    confirm.release()
    cancel.release()


def test_pending_admission_is_bounded_per_owner_and_global_and_release_is_idempotent():
    limits = RequestLimits()
    leases = [limits.admit(1), limits.admit(1)]
    with pytest.raises(RequestLimitError, match="/stop"):
        limits.admit(1)
    for owner in range(2, MAX_PENDING_TOTAL):
        if len(leases) < MAX_PENDING_TOTAL:
            leases.append(limits.admit(owner))
    with pytest.raises(RequestLimitError, match="занят"):
        limits.admit(999)
    leases[0].release()
    leases[0].release()
    replacement = limits.admit(999)
    with pytest.raises(RequestLimitError, match="занят"):
        limits.admit(1000)
    for lease in leases:
        lease.release()
    replacement.release()
    assert limits._pending == 0


def test_completed_or_cancelled_attempts_still_count_until_window_expires():
    now = [100.0]
    limits = RequestLimits(clock=lambda: now[0])
    for _ in range(REQUESTS_PER_MINUTE):
        limits.admit(1).release()
    with pytest.raises(RequestLimitError, match="60 сек"):
        limits.admit(1)
    limits.admit(2).release()
    now[0] += 59
    with pytest.raises(RequestLimitError, match="1 сек"):
        limits.admit(1)
    now[0] += 1
    limits.admit(1).release()


def test_rate_memory_is_bounded_without_evicting_unexpired_limits():
    now = [0.0]
    limits = RequestLimits(clock=lambda: now[0])
    for owner in range(MAX_TRACKED_USERS):
        limits.admit(owner).release()
    with pytest.raises(RequestLimitError, match="занят"):
        limits.admit(MAX_TRACKED_USERS)
    assert len(limits._users) == MAX_TRACKED_USERS
    now[0] += 60
    limits.admit(MAX_TRACKED_USERS).release()
    assert len(limits._users) == 1


def test_expiry_does_not_discard_live_admissions_and_rejection_notices_are_coalesced():
    now = [0.0]
    limits = RequestLimits(clock=lambda: now[0])
    lease = limits.admit(1)
    assert limits.notify_rejection(1)
    assert not limits.notify_rejection(1)
    assert not limits.notify_rejection(2)
    now[0] = 1
    assert limits.notify_rejection(2)
    now[0] = 2
    assert not limits.notify_rejection(1)
    now[0] = 5
    assert limits.notify_rejection(1)
    now[0] = 100
    limits.admit(2).release()
    lease.release()
    assert limits._pending == 0


async def test_execution_bound_and_cancellation_of_waiter_release_worker_permits():
    limits = RequestLimits()
    release, full = asyncio.Event(), asyncio.Event()
    active = maximum = 0

    async def work():
        nonlocal active, maximum
        async with limits.running():
            active += 1
            maximum = max(maximum, active)
            if active == MAX_RUNNING:
                full.set()
            try:
                await release.wait()
            finally:
                active -= 1

    tasks = [asyncio.create_task(work()) for _ in range(MAX_RUNNING + 2)]
    try:
        await asyncio.wait_for(full.wait(), 2)
        tasks[-1].cancel()
        await asyncio.gather(tasks[-1], return_exceptions=True)
        assert active == MAX_RUNNING
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
    assert maximum == MAX_RUNNING and active == 0
    async with limits.running():
        pass
