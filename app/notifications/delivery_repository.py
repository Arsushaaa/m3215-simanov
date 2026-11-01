"""Short transactions and lease/version fencing. No database connection crosses HTTP."""

from datetime import timedelta
from uuid import uuid4

from pydantic import TypeAdapter

from app.mcp.contracts import AwareDateTime
from app.notifications.delivery_models import DeliveryFailure, Lease, reminder_body, retry_delay

_AWARE = TypeAdapter(AwareDateTime)
TERMINAL = {"sent", "failed", "missed", "cancelled"}


class DeliveryRepository:
    def __init__(self, pool, *, clock=None):
        self.pool, self.clock = pool, clock

    async def _now(self, conn):
        return (
            _AWARE.validate_python(self.clock())
            if self.clock is not None
            else await conn.fetchval("SELECT clock_timestamp()")
        )

    @staticmethod
    def _limit(limit):
        if type(limit) is not int or not 1 <= limit <= 10:
            raise ValueError("Bounded batch required")

    @staticmethod
    def _lease(row):
        return Lease(
            row["id"],
            row["user_id"],
            row["lease_token"],
            row["lease_version"],
            row["body"],
            row["destination_chat_id"],
            row["attempts"],
        )

    async def _terminal(self, conn, row, now, state, code):
        await conn.execute(
            "UPDATE labbi_outbox SET state=$3,error_code=$4,finished_at=GREATEST($5,created_at),"
            "lease_until=NULL,lease_token=NULL WHERE id=$1 AND user_id=$2",
            row["id"],
            row["user_id"],
            state,
            code,
            now,
        )

    async def _known_receipt(self, conn, event):
        return await conn.fetchrow(
            "SELECT attempt,telegram_message_id,finished_at FROM labbi_delivery_attempts "
            "WHERE event_id=$1 AND outcome='sent' ORDER BY attempt LIMIT 1",
            event,
        )

    async def _acknowledge(self, conn, row, now, receipt):
        await conn.execute(
            "UPDATE labbi_outbox SET state='sent',telegram_message_id=$2,sent_at=$3,"
            "finished_at=GREATEST($4,created_at),lease_token=NULL,lease_until=NULL,error_code=NULL "
            "WHERE id=$1",
            row["id"],
            receipt["telegram_message_id"],
            receipt["finished_at"],
            now,
        )

    async def recover(self, *, limit=10):
        self._limit(limit)
        async with self.pool.acquire() as conn, conn.transaction():
            now = await self._now(conn)
            rows = await conn.fetch(
                "SELECT * FROM labbi_outbox WHERE kind='reminder' AND ("
                "(state IN ('pending','preparing','ready','retry_wait') AND deadline_at<$1) OR "
                "(state IN ('preparing','sending') AND lease_until<=$1)) "
                "ORDER BY (state='sending') DESC,(state='preparing') DESC,due_at,id "
                "LIMIT $2 FOR UPDATE SKIP LOCKED",
                now,
                limit,
            )
            for row in rows:
                if row["state"] == "sending":
                    known = await self._known_receipt(conn, row["id"])
                    if known is not None and known["attempt"] == row["attempts"]:
                        await self._acknowledge(conn, row, now, known)
                        continue
                    await conn.execute(
                        "UPDATE labbi_delivery_attempts SET outcome='unknown',finished_at=$3,"
                        "error_code='LEASE_EXPIRED' WHERE event_id=$1 AND attempt=$2 "
                        "AND outcome='running'",
                        row["id"],
                        row["attempts"],
                        now,
                    )
                    await conn.execute(
                        "UPDATE labbi_outbox SET uncertain=TRUE WHERE id=$1", row["id"]
                    )
                    known = await self._known_receipt(conn, row["id"])
                    if known is not None:
                        await self._acknowledge(conn, row, now, known)
                        continue
                    await self._retry_or_stop(
                        conn, row, now, DeliveryFailure("LEASE_EXPIRED", True, True)
                    )
                elif now > row["deadline_at"]:
                    await self._terminal(conn, row, now, "missed", "DEADLINE_EXPIRED")
                else:
                    await conn.execute(
                        "UPDATE labbi_outbox SET state='pending',lease_until=NULL,lease_token=NULL,"
                        "error_code='PREPARATION_INTERRUPTED' WHERE id=$1",
                        row["id"],
                    )
            return len(rows)

    async def claim_preparations(self, *, limit=10):
        self._limit(limit)
        async with self.pool.acquire() as conn, conn.transaction():
            now = await self._now(conn)
            rows = await conn.fetch(
                "SELECT * FROM labbi_outbox WHERE kind='reminder' AND state='pending' "
                "AND due_at<=$1 AND deadline_at>=$1 AND next_attempt_at<=$1 "
                "ORDER BY due_at,id LIMIT $2 FOR UPDATE SKIP LOCKED",
                now,
                limit,
            )
            leases = []
            for row in rows:
                updated = await conn.fetchrow(
                    "UPDATE labbi_outbox SET state='preparing',lease_token=$2,"
                    "lease_version=lease_version+1,"
                    "lease_until=$3::timestamptz+INTERVAL '150 seconds' WHERE id=$1 RETURNING *",
                    row["id"],
                    uuid4(),
                    now,
                )
                leases.append(self._lease(updated))
            return leases

    async def prepare(self, lease):
        async with self.pool.acquire() as conn, conn.transaction():
            now = await self._now(conn)
            row = await conn.fetchrow(
                "SELECT * FROM labbi_outbox WHERE id=$1 AND user_id=$2 FOR UPDATE",
                lease.event_id,
                lease.owner,
            )
            if not self._matches(row, lease, "preparing", now):
                return False
            if now > row["deadline_at"]:
                await self._terminal(conn, row, now, "missed", "DEADLINE_EXPIRED")
                return False
            source = await conn.fetchrow(
                "SELECT text,remind_at,timezone FROM labbi_reminders WHERE id=$1 AND user_id=$2",
                row["reminder_id"],
                lease.owner,
            )
            if (
                source is None
                or source["remind_at"] != row["due_at"]
                or source["timezone"] != row["timezone"]
            ):
                await self._terminal(conn, row, now, "failed", "INVALID_SOURCE")
                return False
            try:
                body = reminder_body(source["text"], row["due_at"], row["timezone"], now)
            except (ValueError, KeyError):
                await self._terminal(conn, row, now, "failed", "INVALID_PAYLOAD")
                return False
            await conn.execute(
                "UPDATE labbi_outbox SET state='ready',body=COALESCE(body,$2),lease_token=NULL,"
                "lease_until=NULL,error_code=NULL WHERE id=$1",
                lease.event_id,
                body,
            )
            return True

    async def candidates(self, *, limit=10):
        self._limit(limit)
        async with self.pool.acquire() as conn:
            now = await self._now(conn)
            rows = await conn.fetch(
                "SELECT id,user_id FROM labbi_outbox WHERE kind='reminder' "
                "AND state IN ('ready','retry_wait') AND due_at<=$1 AND next_attempt_at<=$1 "
                "AND deadline_at>=$1 "
                "ORDER BY due_at,id LIMIT $2",
                now,
                limit,
            )
            return [(r["id"], r["user_id"]) for r in rows]

    async def release_preparations(self, leases):
        if len(leases) > 10:
            raise ValueError("Bounded batch required")
        await self.pool.executemany(
            "UPDATE labbi_outbox SET state='pending',lease_token=NULL,lease_until=NULL "
            "WHERE id=$1 AND user_id=$2 AND lease_token=$3 AND lease_version=$4 "
            "AND state='preparing'",
            [(v.event_id, v.owner, v.token, v.version) for v in leases],
        )

    async def _lock(self, conn, event, owner):
        # Claims which also inspect a profile always use user -> profile -> outbox.
        await conn.fetchval("SELECT user_id FROM labbi_users WHERE user_id=$1 FOR UPDATE", owner)
        profile = await conn.fetchrow(
            "SELECT destination_chat_id FROM labbi_notification_profiles "
            "WHERE user_id=$1 FOR UPDATE",
            owner,
        )
        row = await conn.fetchrow(
            "SELECT * FROM labbi_outbox WHERE id=$1 AND user_id=$2 FOR UPDATE", event, owner
        )
        return profile, row

    async def begin_send(self, event, owner, *, trace_id=None):
        async with self.pool.acquire() as conn, conn.transaction():
            profile, row = await self._lock(conn, event, owner)
            now = await self._now(conn)
            if (
                row is None
                or row["kind"] != "reminder"
                or row["state"] not in {"ready", "retry_wait"}
            ):
                return None
            if now < row["due_at"] or now < row["next_attempt_at"]:
                return None
            if now > row["deadline_at"]:
                await self._terminal(conn, row, now, "missed", "DEADLINE_EXPIRED")
                return None
            if profile is None or profile["destination_chat_id"] != row["destination_chat_id"]:
                await self._terminal(conn, row, now, "cancelled", "DESTINATION_CHANGED")
                return None
            known = await self._known_receipt(conn, event)
            if known is not None:
                await self._acknowledge(conn, row, now, known)
                return None
            if row["attempts"] >= 3:
                await self._terminal(conn, row, now, "failed", "ATTEMPTS_EXHAUSTED")
                return None
            if trace_id is not None and not await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM labbi_traces WHERE id=$1 AND user_id=$2)",
                trace_id,
                owner,
            ):
                raise ValueError("Foreign delivery trace")
            row = await conn.fetchrow(
                "UPDATE labbi_outbox SET state='sending',attempts=attempts+1,lease_token=$2,"
                "lease_version=lease_version+1,lease_until=$3::timestamptz+INTERVAL '60 seconds' "
                "WHERE id=$1 RETURNING *",
                event,
                uuid4(),
                now,
            )
            await conn.execute(
                "INSERT INTO labbi_delivery_attempts(event_id,attempt,lease_version,"
                "trace_id,started_at,lease_token,outcome) VALUES($1,$2,$3,$4,$5,$6,'running')",
                event,
                row["attempts"],
                row["lease_version"],
                trace_id,
                now,
                row["lease_token"],
            )
            return self._lease(row)

    @staticmethod
    def _matches(row, lease, state, now):
        return (
            row is not None
            and row["state"] == state
            and row["lease_token"] == lease.token
            and row["lease_version"] == lease.version
            and row["lease_until"] > now
        )

    async def preflight(self, lease):
        async with self.pool.acquire() as conn, conn.transaction():
            profile, row = await self._lock(conn, lease.event_id, lease.owner)
            now = await self._now(conn)
            if not self._matches(row, lease, "sending", now):
                return False
            known = await self._known_receipt(conn, row["id"])
            if known is not None:
                await self._attempt(conn, lease, now, "failed", "ALREADY_DELIVERED")
                await self._acknowledge(conn, row, now, known)
                return False
            code = (
                "DEADLINE_EXPIRED"
                if now > row["deadline_at"]
                else "DESTINATION_CHANGED"
                if profile is None or profile["destination_chat_id"] != row["destination_chat_id"]
                else None
            )
            if code is not None:
                await self._terminal(
                    conn, row, now, "missed" if code == "DEADLINE_EXPIRED" else "cancelled", code
                )
                await self._attempt(conn, lease, now, "failed", code)
                return False
            return now >= row["due_at"]

    async def _attempt(self, conn, lease, now, outcome, code=None, message_id=None):
        await conn.execute(
            "UPDATE labbi_delivery_attempts SET outcome=$4,finished_at=GREATEST($5,started_at),"
            "error_code=$6,telegram_message_id=$7 WHERE event_id=$1 AND attempt=$2 "
            "AND lease_version=$3 AND lease_token=$8 "
            "AND (outcome<>'sent' OR ($4='sent' AND telegram_message_id=$7))",
            lease.event_id,
            lease.attempt,
            lease.version,
            outcome,
            now,
            code,
            message_id,
            lease.token,
        )

    async def sent(self, lease, message_id):
        if type(message_id) is not int or not 0 < message_id <= 2**63 - 1:
            raise ValueError("Invalid Telegram receipt")
        async with self.pool.acquire() as conn, conn.transaction():
            _, row = await self._lock(conn, lease.event_id, lease.owner)
            now = await self._now(conn)
            if not self._matches(row, lease, "sending", now):
                if row is not None:
                    await self._attempt(conn, lease, now, "sent", message_id=message_id)
                return False
            await self._attempt(conn, lease, now, "sent", message_id=message_id)
            await conn.execute(
                "UPDATE labbi_outbox SET state='sent',telegram_message_id=$3,sent_at=$4,"
                "finished_at=GREATEST($4,created_at),lease_token=NULL,lease_until=NULL,"
                "error_code=NULL "
                "WHERE id=$1 AND user_id=$2",
                lease.event_id,
                lease.owner,
                message_id,
                now,
            )
            return True

    async def failed(self, lease, failure):
        async with self.pool.acquire() as conn, conn.transaction():
            _, row = await self._lock(conn, lease.event_id, lease.owner)
            now = await self._now(conn)
            if not self._matches(row, lease, "sending", now):
                return False
            await self._attempt(
                conn,
                lease,
                now,
                "unknown" if failure.uncertain else "retry" if failure.transient else "failed",
                failure.code,
            )
            if failure.uncertain:
                await conn.execute(
                    "UPDATE labbi_outbox SET uncertain=TRUE WHERE id=$1", lease.event_id
                )
            known = await self._known_receipt(conn, row["id"])
            if known is not None:
                await self._acknowledge(conn, row, now, known)
                return True
            await self._retry_or_stop(conn, row, now, failure)
            return True

    async def _retry_or_stop(self, conn, row, now, failure):
        if not failure.transient:
            await self._terminal(conn, row, now, "failed", failure.code)
        elif now > row["deadline_at"]:
            await self._terminal(conn, row, now, "missed", "DEADLINE_EXPIRED")
        elif row["attempts"] >= 3:
            await self._terminal(conn, row, now, "failed", "ATTEMPTS_EXHAUSTED")
        else:
            delay = retry_delay(row["attempts"], failure.retry_after)
            if delay > (row["deadline_at"] - now).total_seconds():
                await self._terminal(conn, row, now, "missed", "RETRY_AFTER_DEADLINE")
                return
            await conn.execute(
                "UPDATE labbi_outbox SET state='retry_wait',error_code=$2,next_attempt_at=$3,"
                "lease_token=NULL,lease_until=NULL WHERE id=$1",
                row["id"],
                failure.code,
                now + timedelta(seconds=delay),
            )
