"""Wire schemas may evolve; host domain fields and write authority remain closed."""

import json

from app.decision_schema import model_schema


def input_shape(schema):
    """Compare parameter structure, not server-owned descriptions or constraints.

    Renamed/added fields require an explicit adapter. Restricting an existing
    field (length, enum, pattern, format) is supported and validated on each call.
    Local DTOs still enforce the host's stricter semantic and authority rules.
    """
    expanded = model_schema(schema, inline_refs=True)

    def shape(value):
        if not isinstance(value, dict):
            raise ValueError("Object schemas required")
        result = {k: value[k] for k in ("type", "additionalProperties") if k in value}
        if "properties" in value:
            result["properties"] = {k: shape(v) for k, v in value["properties"].items()}
            result["required"] = sorted(value.get("required", []))
        if "items" in value:
            result["items"] = shape(value["items"])
        for key in ("anyOf", "oneOf", "allOf"):
            if key in value:
                result[key] = sorted(
                    [shape(v) for v in value[key]], key=lambda v: json.dumps(v, sort_keys=True)
                )
        return result

    return shape(expanded)


def weather_city_schema(schema):
    """Expose only city weather to model; point remains a host-only operation."""
    expanded = model_schema(schema, inline_refs=True)
    branches = expanded.get("anyOf", expanded.get("oneOf", [expanded]))
    for branch in branches:
        if set(branch.get("properties", {})) == {"city"}:
            return branch
    raise ValueError("No supported city weather schema")
