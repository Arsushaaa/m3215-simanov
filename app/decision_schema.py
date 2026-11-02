"""Provider-facing hints; canonical models still validate every tool argument."""

from copy import deepcopy


def model_schema(
    schema, *, pending=False, inline_refs=False, message_indices=None, evidence_hints=False
):
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

    if evidence_hints and "ReminderEvidence" in definitions:
        fields = definitions["ReminderEvidence"]["properties"]
        for name, description in {
            "text": "Source of task text, not a time-only reply.",
            "date": "Source of the requested date.",
            "time": "Source of the precise clock time.",
        }.items():
            fields[name]["description"] = description

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
    # Keep the MCP contract's city description verbatim. A separate provider
    # hint used to request a complete candidate label and overrode the rule
    # against inventing country/region qualifiers.
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
