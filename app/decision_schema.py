"""Provider-facing hints; canonical models still validate every tool argument."""

from copy import deepcopy


def model_schema(schema, *, pending=False):
    result = deepcopy(schema)
    definitions = result.get("$defs", {})
    if not pending and "ReminderEvidence" in definitions:
        properties = definitions["ReminderEvidence"]["properties"]
        for name in ("text", "date", "time"):
            properties[name] = {"$ref": "#/$defs/UserQuote"}
        definitions.pop("PendingCardReference", None)

    def hints(value):
        if isinstance(value, dict):
            # Provider JSON decoders differ; runtime and MCP keep these rules.
            value.pop("pattern", None)
            # Pydantic's nested discriminator mapping is not a portable decoder
            # hint. Flatten nested alternatives; local validation keeps the
            # canonical discriminated union and rejects every invalid result.
            value.pop("discriminator", None)
            if "oneOf" in value:
                branches = []
                for branch in value["oneOf"]:
                    if set(branch) <= {"oneOf", "discriminator"} and "oneOf" in branch:
                        branches.extend(branch["oneOf"])
                    else:
                        branches.append(branch)
                value["oneOf"] = branches
            for child in value.values():
                hints(child)
        elif isinstance(value, list):
            for child in value:
                hints(child)

    hints(result)
    weather = result.get("$defs", {}).get("WeatherArguments", result)
    city = weather.get("properties", {}).get("city")
    if city is not None:
        city["description"] = (
            "City name IN RUSSIAN (на русском), Cyrillic only. Translate city and "
            "explicitly supplied country/region before returning this argument. "
            "Do not copy Latin names or invent a country."
        )
    return result
