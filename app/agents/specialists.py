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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, role_rules=RULES["schedule"], **kwargs)


class LifestyleAgent(AgentService):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, role_rules=RULES["lifestyle"], **kwargs)
