"""Durable daily identity and API reservations, committed before external calls."""

from datetime import UTC, datetime, time, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo

from app.notifications.delivery_repository import DeliveryRepository
from app.notifications.models import briefing_moment, next_briefing


class BriefingRepository(DeliveryRepository):
    async def schedule(self, *, limit=10):
        self._limit(limit)
        async with self.pool.acquire() as conn:
            now = await self._now(conn)
            profiles = await conn.fetch(
                "SELECT p.user_id FROM labbi_notification_profiles p LEFT JOIN "
                "labbi_briefing_cursors c USING(user_id) WHERE p.briefing_enabled AND "
                "(c.user_id IS NULL OR c.profile_version<>p.version OR c.next_at<=$1) "
                "ORDER BY c.next_at NULLS FIRST,p.user_id LIMIT $2",
                now,
                limit,
            )
        for item in profiles:
            async with self.pool.acquire() as conn, conn.transaction():
                owner = item["user_id"]
                await conn.fetchval(
                    "SELECT user_id FROM labbi_users WHERE user_id=$1 FOR UPDATE", owner
                )
                p = await conn.fetchrow(
                    "SELECT p.*,u.timezone FROM labbi_notification_profiles p JOIN labbi_users u "
                    "USING(user_id) WHERE user_id=$1 FOR UPDATE OF p",
                    owner,
                )
                now = await self._now(conn)
                if not p or not p["briefing_enabled"] or not p["timezone"] or not p["city"]:
                    continue
                day = now.astimezone(ZoneInfo(p["timezone"])).date()
                due = briefing_moment(day, p["timezone"])
                if p["consented_at"] <= due and now <= due + timedelta(hours=1):
                    await conn.execute(
                        "INSERT INTO labbi_outbox(user_id,kind,local_date,destination_chat_id,"
                        "profile_version,timezone,due_at,deadline_at,next_attempt_at) "
                        "VALUES($1,'briefing',$2,$3,$4,$5,$6,$7,$6) "
                        "ON CONFLICT(user_id,local_date) DO UPDATE SET state='pending',"
                        "finished_at=NULL,error_code=NULL,profile_version=EXCLUDED.profile_version,"
                        "destination_chat_id=EXCLUDED.destination_chat_id,timezone=EXCLUDED.timezone,"
                        "due_at=EXCLUDED.due_at,deadline_at=EXCLUDED.deadline_at,"
                        "next_attempt_at=EXCLUDED.next_attempt_at "
                        "WHERE labbi_outbox.state='cancelled' AND labbi_outbox.attempts=0 "
                        "AND labbi_outbox.body IS NULL AND labbi_outbox.mcp_reserved=0 "
                        "AND labbi_outbox.llm_reserved=0 AND EXCLUDED.due_at>$8",
                        owner,
                        day,
                        p["destination_chat_id"],
                        p["version"],
                        p["timezone"],
                        due,
                        due + timedelta(hours=1),
                        now,
                    )
                # Before 08:00 wake at today's 08:00; otherwise at the next local 08:00.
                cursor = next_briefing(now, p["timezone"])
                await conn.execute(
                    "INSERT INTO labbi_briefing_cursors VALUES($1,$2,$3) ON CONFLICT(user_id) "
                    "DO UPDATE SET "
                    "profile_version=EXCLUDED.profile_version,next_at=EXCLUDED.next_at",
                    owner,
                    p["version"],
                    cursor,
                )

    async def claim_daily(self, *, limit=2):
        self._limit(limit)
        async with self.pool.acquire() as conn:
            now = await self._now(conn)
            candidates = await conn.fetch(
                "SELECT id,user_id FROM labbi_outbox WHERE kind='briefing' AND state='pending' "
                "AND due_at<=$1 AND deadline_at>=$1 ORDER BY due_at,id LIMIT $2",
                now,
                limit,
            )
        result = []
        for candidate in candidates:
            async with self.pool.acquire() as conn, conn.transaction():
                profile, row = await self._lock(conn, candidate["id"], candidate["user_id"])
                now = await self._now(conn)
                if row["state"] != "pending" or not self._profile_valid(profile, row):
                    continue
                updated = await conn.fetchrow(
                    "UPDATE labbi_outbox SET state='preparing',lease_token=$2,"
                    "lease_version=lease_version+1,lease_until=$3::timestamptz+INTERVAL '150 "
                    "seconds' "
                    "WHERE id=$1 AND deadline_at>=$3 RETURNING *",
                    row["id"],
                    uuid4(),
                    now,
                )
                if updated:
                    result.append(self._lease(updated))
        return result

    async def snapshot(self, lease):
        async with self.pool.acquire() as conn, conn.transaction():
            p, row = await self._lock(conn, lease.event_id, lease.owner)
            now = await self._now(conn)
            if (
                not self._matches(row, lease, "preparing", now)
                or not self._profile_valid(p, row)
                or now > row["deadline_at"]
            ):
                return None
            return dict(row), dict(p), now

    async def reserve(self, lease, kind):
        column = {"llm": "llm_reserved", "mcp": "mcp_reserved"}.get(kind)
        if column is None:
            raise ValueError("Invalid API budget")
        maximum = 1 if kind == "llm" else 3
        async with self.pool.acquire() as conn, conn.transaction():
            p, row = await self._lock(conn, lease.event_id, lease.owner)
            now = await self._now(conn)
            if (
                not self._matches(row, lease, "preparing", now)
                or not self._profile_valid(p, row)
                or now > row["deadline_at"]
                or row[column] >= maximum
            ):
                raise ValueError("Unavailable daily budget")
            await conn.execute(
                f"UPDATE labbi_outbox SET {column}={column}+1 WHERE id=$1", lease.event_id
            )

    async def ready(self, lease, body):
        from app.notifications.delivery_models import valid_payload

        if not valid_payload(body):
            raise ValueError("Invalid daily payload")
        async with self.pool.acquire() as conn, conn.transaction():
            p, row = await self._lock(conn, lease.event_id, lease.owner)
            now = await self._now(conn)
            if (
                not self._matches(row, lease, "preparing", now)
                or not self._profile_valid(p, row)
                or now > row["deadline_at"]
            ):
                return False
            await conn.execute(
                "UPDATE labbi_outbox SET "
                "state='ready',body=$2,lease_token=NULL,lease_until=NULL WHERE id=$1",
                lease.event_id,
                body,
            )
            return True

    async def reminders(self, owner, day, zone):
        z = ZoneInfo(zone)
        start = datetime.combine(day, time.min, z).astimezone(UTC)
        end = datetime.combine(day + timedelta(days=1), time.min, z).astimezone(UTC)
        rows = await self.pool.fetch(
            "SELECT text,remind_at FROM labbi_reminders WHERE user_id=$1 AND "
            "remind_at>=$2 AND remind_at<$3 ORDER BY remind_at,id LIMIT 6",
            owner,
            start,
            end,
        )
        return [dict(r) for r in rows]

    async def focus(self, version, topic):
        return await self.pool.fetchrow(
            "SELECT body FROM labbi_focus_cache WHERE catalog_version=$1 AND topic=$2",
            version,
            topic,
        )

    async def claim_focus(self, version, topic):
        token = uuid4()
        found = await self.pool.fetchval(
            "INSERT INTO labbi_focus_cache(catalog_version,topic,token) "
            "VALUES($1,$2,$3) ON CONFLICT DO NOTHING RETURNING token",
            version,
            topic,
            token,
        )
        return found

    async def save_focus(self, version, topic, token, body):
        await self.pool.execute(
            "UPDATE labbi_focus_cache SET body=$4 WHERE catalog_version=$1 AND topic=$2 "
            "AND token=$3 AND body IS NULL",
            version,
            topic,
            token,
            body,
        )
