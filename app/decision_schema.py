"""Provider-facing hints; canonical models still validate every tool argument."""

from copy import deepcopy


def model_schema(schema, *, pending=False, inline_refs=False, message_indices=None):
    result = deepcopy(schema)
    definitions = result.get("$defs", {})
    if message_indices is not None and "UserQuote" in definitions:
        index = definitions["UserQuote"]["properties"]["message_index"]
        allowed = sorted(
            {
                value
                for value in message_indices
                if type(value) is int and index["minimum"] <= value <= index["maximum"]
            }
        )
        if not allowed or 0 not in allowed:
            raise ValueError("The current user message must have index zero.")
        index["enum"] = allowed
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
            " For a short clarification retain the city and ALL country/region qualifiers "
            "the user explicitly supplied earlier; add the new qualifier without dropping any."
            " When choosing an offered candidate after a short clarification, return its "
            "COMPLETE Russian label: city, country, matching Russian region. Keep the offered "
            "Russian spelling; do not add administrative words or omit the country. "
            "An API-added Latin secondary administrative fragment is not a user qualifier: "
            "exclude it instead of copying Latin into city or inventing a translation."
        )
    if not inline_refs:
        return result

    def expand(value, ancestors=()):
        if isinstance(value, list):
            return [expand(child, ancestors) for child in value]
        if not isinstance(value, dict):
            return value
        siblings = {k: v for k, v in value.items() if k not in {"$defs", "$ref"}}
        if "$ref" in value:
            reference = value["$ref"]
            prefix = "#/$defs/"
            if (
                not isinstance(reference, str)
                or not reference.startswith(prefix)
                or reference in ancestors
                or reference[len(prefix) :] not in definitions
            ):
                raise ValueError("Only acyclic local schema references are supported.")
            resolved = expand(definitions[reference[len(prefix) :]], (*ancestors, reference))
            return {**resolved, **expand(siblings, ancestors)}
        return {key: expand(child, ancestors) for key, child in siblings.items()}

    return expand(result)
