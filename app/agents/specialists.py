from app.agent import AgentService

RULES = {
    "schedule": "\nYou are ScheduleAgent. Handle university classes and reminder cards only. "
    "Weather and materials are outside your authority. Never claim a reminder was saved "
    "before the owner's confirmation callback. Do not replace an academic schedule with a plan.",
    "lifestyle": "\nYou are LifestyleAgent. Handle current weather and recommendations grounded "
    "in its result. No reminders, classes, material search or writes are available. "
    "Do not change pending cards. Ask a clarification for an out-of-scope request.",
}


class ScheduleAgent(AgentService):
    @staticmethod
    async def read_day(mcp, day):
        from app.mcp.contracts import ScheduleResult

        result = ScheduleResult.model_validate(
            await mcp.call_tool("get_schedule", {"date": day.isoformat()})
        ).root
        if result.status != "error" and result.date != day:
            raise ValueError("Wrong schedule day")
        return result

    def __init__(self, *args, **kwargs):
        super().__init__(*args, role_rules=RULES["schedule"], **kwargs)


class LifestyleAgent(AgentService):
    @staticmethod
    async def read_point(mcp, point):
        from app.mcp.contracts import PointWeatherResult

        result = PointWeatherResult.model_validate(
            await mcp.call_tool("get_weather", {"point": point.model_dump(mode="json")})
        ).root
        if result.status != "error" and result.point != point:
            raise ValueError("Wrong weather point")
        return result

    def __init__(self, *args, **kwargs):
        super().__init__(*args, role_rules=RULES["lifestyle"], **kwargs)
