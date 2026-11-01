"""Profile/menu/outbox storage. All mutations use owner -> profile -> event locks."""

import hashlib
import json
import re
import secrets
from datetime import date, timedelta
from uuid import UUID
from zoneinfo import ZoneInfo

from app.confirmation_models import ConfirmationError
from app.notifications.models import CityCandidate, briefing_moment, next_briefing


def _owner(value):
    if type(value) is not int or not 0 < value <= 2**63 - 1:
        raise ValueError("Invalid trusted owner")


def _hash(token):
    if not isinstance(token, str) or re.fullmatch(r"[A-Za-z0-9_-]{22}", token) is None:
        raise ValueError("Invalid menu token")
    return hashlib.sha256(token.encode("ascii")).digest()


class NotificationRepository:
    def __init__(self, pool):
        self.pool = pool

    async def _lock(self, conn, owner):
        _owner(owner)
        await conn.execute(
            "INSERT INTO labbi_users(user_id) VALUES($1) ON CONFLICT DO NOTHING", owner
        )
        zone = await conn.fetchval(
            "SELECT timezone FROM labbi_users WHERE user_id=$1 FOR UPDATE", owner
        )
        await conn.execute(
            "INSERT INTO labbi_notification_profiles(user_id) VALUES($1) ON CONFLICT DO NOTHING",
            owner,
        )
        row = await conn.fetchrow(
            "SELECT * FROM labbi_notification_profiles WHERE user_id=$1 FOR UPDATE", owner
        )
        return {**dict(row), "timezone": zone}

    async def now(self):
        return await self.pool.fetchval("SELECT clock_timestamp()")

    async def profile(self, owner):
        _owner(owner)
        row = await self.pool.fetchrow(
            "SELECT p.*,u.timezone FROM labbi_notification_profiles p JOIN "
            "labbi_users u USING(user_id) WHERE p.user_id=$1",
            owner,
        )
        return dict(row) if row is not None else None

    async def register_destination(self, owner, chat_id):
        _owner(owner)
        _owner(chat_id)
        if owner != chat_id:
            raise ValueError("Only a verified private update registers a destination")
        async with self.pool.acquire() as conn, conn.transaction():
            await self._lock(conn, owner)
            await conn.execute(
                "UPDATE labbi_notification_profiles SET destination_chat_id=$2 WHERE user_id=$1",
                owner,
                chat_id,
            )
            # A fresh private update may attach still-live legacy reminders.
            await conn.execute(
                "INSERT INTO labbi_outbox(user_id,kind,reminder_id,destination_chat_id"
                ",profile_version,timezone,due_at,deadline_at,next_attempt_at) "
                "SELECT r.user_id,'reminder',r.id,p.destination_chat_id,p.version,r.ti"
                "mezone,r.remind_at,r.remind_at+INTERVAL '15 minutes',r.remind_at "
                "FROM labbi_reminders r JOIN labbi_notification_profiles p USING(user_id) "
                "WHERE r.user_id=$1 AND r.remind_at>=clock_timestamp()-INTERVAL '15 minutes' "
                "ON CONFLICT(reminder_id) DO NOTHING",
                owner,
            )

    async def clear_menu(self, owner):
        _owner(owner)
        await self.pool.execute("DELETE FROM labbi_city_menus WHERE user_id=$1", owner)

    async def start_menu(self, owner):
        token = secrets.token_urlsafe(16)
        async with self.pool.acquire() as conn, conn.transaction():
            profile = await self._lock(conn, owner)
            await conn.execute(
                "INSERT INTO labbi_city_menus(user_id,token_hash,profile_version,expires_at) "
                "VALUES($1,$2,$3,clock_timestamp()+INTERVAL '5 minutes') ON CONFLICT(user_id) "
                "DO UPDATE SET token_hash=EXCLUDED.token_hash,profile_version=EXCLUDED"
                ".profile_version,"
                "expires_at=EXCLUDED.expires_at,options='[]',selected=NULL",
                owner,
                _hash(token),
                profile["version"],
            )
        return token, profile["version"]

    async def fill_menu(self, owner, token, version, options):
        if len(options) > 50:
            raise ValueError("Too many choices")
        values = [CityCandidate.model_validate(o).model_dump(mode="json") for o in options]
        async with self.pool.acquire() as conn, conn.transaction():
            profile = await self._lock(conn, owner)
            if profile["version"] != version:
                return False
            result = await conn.fetchval(
                "UPDATE labbi_city_menus SET options=$4::jsonb,selected=NULL WHERE "
                "user_id=$1 AND token_hash=$2 "
                "AND profile_version=$3 AND expires_at>clock_timestamp() RETURNING user_id",
                owner,
                _hash(token),
                version,
                json.dumps(values, ensure_ascii=False, allow_nan=False),
            )
            return result is not None

    async def menu(self, owner, token):
        return await self._menu(self.pool, owner, token)

    async def _menu(self, source, owner, token, *, lock=False):
        _owner(owner)
        try:
            digest = _hash(token)
        except ValueError:
            return None
        row = await source.fetchrow(
            "SELECT m.* FROM labbi_city_menus m JOIN labbi_notification_profiles p "
            "USING(user_id) "
            "WHERE m.user_id=$1 AND m.token_hash=$2 AND m.profile_version=p.version "
            "AND m.expires_at>clock_timestamp()" + (" FOR UPDATE OF m" if lock else ""),
            owner,
            digest,
        )
        if row is None:
            return None
        data = dict(row)
        data["options"] = [CityCandidate.model_validate(o) for o in json.loads(data["options"])]
        return data

    async def choose(self, owner, token, index):
        if type(index) is not int or not 0 <= index < 50:
            return None
        async with self.pool.acquire() as conn, conn.transaction():
            await self._lock(conn, owner)
            menu = await self._menu(conn, owner, token, lock=True)
            if menu is None or index >= len(menu["options"]):
                return None
            await conn.execute(
                "UPDATE labbi_city_menus SET selected=$2 WHERE user_id=$1", owner, index
            )
            return menu["options"][index]

    async def confirm_city(self, owner, token, index):
        if type(index) is not int or not 0 <= index < 50:
            return None
        async with self.pool.acquire() as conn, conn.transaction():
            await self._lock(conn, owner)
            menu = await self._menu(conn, owner, token, lock=True)
            if menu is None or menu["selected"] != index or index >= len(menu["options"]):
                return None
            city = menu["options"][index]
            saved = await conn.fetchval(
                "UPDATE labbi_notification_profiles SET "
                "city=$2::jsonb,city_confirmed_at=clock_timestamp() WHERE user_id=$1 "
                "AND clock_timestamp()<$3 RETURNING user_id",
                owner,
                city.model_dump_json(),
                menu["expires_at"],
            )
            if saved is None:
                return None
            await conn.execute("DELETE FROM labbi_city_menus WHERE user_id=$1", owner)
            return city

    async def cancel_menu(self, owner, token):
        _owner(owner)
        try:
            digest = _hash(token)
        except ValueError:
            return False
        return (
            await self.pool.fetchval(
                "DELETE FROM labbi_city_menus WHERE user_id=$1 AND token_hash=$2 RETURNING user_id",
                owner,
                digest,
            )
            is not None
        )

    async def set_briefing(self, owner, enabled):
        if type(enabled) is not bool:
            raise ValueError("Explicit boolean required")
        async with self.pool.acquire() as conn, conn.transaction():
            profile = await self._lock(conn, owner)
            now = await conn.fetchval("SELECT clock_timestamp()")
            if enabled:
                if profile["timezone"] is None:
                    raise ConfirmationError(
                        "TIMEZONE_REQUIRED", "Сначала выбери часовой пояс: /timezone."
                    )
                if profile["city"] is None:
                    raise ConfirmationError(
                        "INVALID_ARGUMENTS", "Сначала подтверди город: /city <название>."
                    )
                if profile["destination_chat_id"] is None:
                    raise ConfirmationError(
                        "INVALID_ARGUMENTS",
                        "Открой личный чат бота и отправь /start для регистрации доставки.",
                    )
                moment = next_briefing(now, profile["timezone"])
            else:
                moment = None
            await conn.execute(
                "UPDATE labbi_notification_profiles SET "
                "briefing_enabled=$2,consented_at=CASE WHEN $2 AND NOT "
                "briefing_enabled "
                "THEN $3 ELSE consented_at END WHERE user_id=$1",
                owner,
                enabled,
                now,
            )
        return moment

    async def enqueue_briefing(self, owner, day: date, *, expected_version):
        if type(day) is not date or type(expected_version) is not int or expected_version < 0:
            raise ValueError("Invalid host event identity")
        async with self.pool.acquire() as conn, conn.transaction():
            profile = await self._lock(conn, owner)
            if (
                not profile["briefing_enabled"]
                or profile["version"] != expected_version
                or profile["city"] is None
                or profile["timezone"] is None
                or profile["destination_chat_id"] is None
            ):
                return None
            now = await conn.fetchval("SELECT clock_timestamp()")
            today = now.astimezone(ZoneInfo(profile["timezone"])).date()
            if not 0 <= (day - today).days <= 1:
                return None
            due = briefing_moment(day, profile["timezone"])
            if due < profile["consented_at"] or now > due + timedelta(hours=1):
                return None
            event = await conn.fetchval(
                "INSERT INTO labbi_outbox(user_id,kind,local_date,destination_chat_id,"
                "profile_version,timezone,due_at,deadline_at,next_attempt_at) "
                "VALUES($1,'briefing',$2,$3,$4,$5,$6::timestamptz,"
                "$6::timestamptz+INTERVAL '60 minutes',$6::timestamptz) "
                "ON CONFLICT(user_id,local_date) DO NOTHING RETURNING id",
                owner,
                day,
                profile["destination_chat_id"],
                profile["version"],
                profile["timezone"],
                due,
            )
            if event is None:
                event = await conn.fetchval(
                    "SELECT id FROM labbi_outbox WHERE user_id=$1 AND local_date=$2", owner, day
                )
                await conn.execute(
                    "UPDATE labbi_outbox SET state='pending',finished_at=NULL,"
                    "error_code=NULL,profile_version=$3,destination_chat_id=$4,timezone=$5,"
                    "due_at=$6::timestamptz,deadline_at=$6::timestamptz+INTERVAL '60 minutes',"
                    "next_attempt_at=$6::timestamptz "
                    "WHERE id=$1 AND user_id=$2 AND state='cancelled' AND attempts=0 "
                    "AND body IS NULL "
                    "AND llm_reserved=0 AND mcp_reserved=0 AND $6::timestamptz>clock_timestamp()",
                    event,
                    owner,
                    profile["version"],
                    profile["destination_chat_id"],
                    profile["timezone"],
                    due,
                )
            return event

    async def event(self, owner, event_id: UUID):
        _owner(owner)
        row = await self.pool.fetchrow(
            "SELECT * FROM labbi_outbox WHERE user_id=$1 AND id=$2", owner, event_id
        )
        return dict(row) if row is not None else None
