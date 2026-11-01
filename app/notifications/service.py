"""User-facing notification settings; stage four does not send messages by time."""

from app.notifications.city import CityUI, label
from app.notifications.models import CityCandidate, next_briefing
from app.time_display import format_datetime


class NotificationService:
    def __init__(self, repository, lookup):
        self.repository = repository
        self.city_ui = CityUI(repository, lookup)

    async def register_destination(self, owner, chat_id):
        await self.repository.register_destination(owner, chat_id)

    async def clear_menu(self, owner):
        await self.repository.clear_menu(owner)

    async def city(self, owner, query=None):
        return await self.city_ui.open(owner, query)

    async def city_callback(self, owner, data):
        return await self.city_ui.handle(owner, data)

    async def briefing(self, owner, enabled):
        moment = await self.repository.set_briefing(owner, enabled)
        if not enabled:
            return (
                "Ежедневная сводка выключена. Не начатые отправки отменены. "
                "Уже начатую отправку гарантированно отозвать нельзя."
            )
        profile = await self.repository.profile(owner)
        return (
            "Ежедневная сводка включена. Ближайшее плановое время: "
            + format_datetime(moment, profile["timezone"])
            + ".\n"
            "Служба отправки ещё не подключена; сейчас сохраняются настройки и задания."
        )

    async def status(self, owner):
        profile = await self.repository.profile(owner)
        if profile is None:
            return "Настройки ещё не созданы. Отправь /start в личном чате."
        city = CityCandidate.model_validate_json(profile["city"]) if profile["city"] else None
        lines = [
            "Настройки уведомлений:",
            "Город: " + (label(city) if city else "не выбран"),
            "Часовой пояс: " + (profile["timezone"] or "не выбран"),
            "Сводка в 08:00: " + ("включена" if profile["briefing_enabled"] else "выключена"),
            "Личный чат: "
            + (
                "зарегистрирован"
                if profile["destination_chat_id"] is not None
                else "отправь /start"
            ),
        ]
        if profile["briefing_enabled"] and profile["timezone"]:
            moment = next_briefing(await self.repository.now(), profile["timezone"])
            lines.append("Плановое время: " + format_datetime(moment, profile["timezone"]))
        lines.append(
            "Служба отправки ещё не подключена. /city — город, /timezone — пояс, "
            "/briefing_on и /briefing_off — согласие на сводку."
        )
        return "\n".join(lines)
