from __future__ import annotations

import json
import re
from typing import Any


def parse_output(text: str, mode: str = "strict") -> tuple[dict[str, Any] | None, str | None, str | None]:
    """Parse model output while preserving whether any wrapper was removed."""
    stripped = (text or "").strip()
    if not stripped:
        return None, None, "empty_output"

    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj, "direct_json", None
        return None, "direct_json", "top_level_json_is_not_object"
    except json.JSONDecodeError as exc:
        if mode == "strict":
            return None, None, f"json_parse_error: {exc}"

    if mode != "tolerant":
        return None, None, f"json_parse_error: {exc}"

    # Some providers emit literal tabs/newlines inside JSON string values.
    # Recover these only in tolerant mode and retain the recovery method.
    try:
        obj = json.loads(stripped, strict=False)
        if isinstance(obj, dict):
            return obj, "direct_json_nonstandard_control_chars", None
    except json.JSONDecodeError:
        pass

    # Tolerant mode accepts common presentation wrappers but records the repair method.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.I | re.S)
    if fence:
        try:
            obj = json.loads(fence.group(1), strict=False)
            if isinstance(obj, dict):
                method = "json_code_fence"
                if "\t" in fence.group(1) or any(ord(char) < 0x20 and char not in "\r\n\t" for char in fence.group(1)):
                    method = "json_code_fence_nonstandard_control_chars"
                return obj, method, None
        except json.JSONDecodeError:
            pass

    # raw_decode finds the end of a complete nested object without guessing from
    # the last brace, so prose or braces inside the JSON cannot corrupt extraction.
    decoder = json.JSONDecoder(strict=False)
    for first in (match.start() for match in re.finditer(r"\{", stripped)):
        try:
            obj, _ = decoder.raw_decode(stripped[first:])
            if isinstance(obj, dict):
                return obj, "extracted_json_object", None
        except json.JSONDecodeError:
            continue

    return None, None, "no_parseable_json_object"


def validate_schema(parsed: dict[str, Any] | None, schema: dict[str, Any] | None) -> list[str]:
    if parsed is None or schema is None:
        return []
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise RuntimeError("jsonschema is required when output_schema_file is configured") from exc
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(parsed), key=lambda e: list(e.path))
    return [f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors]


def validate_context(
    parsed: dict[str, Any] | None,
    case: dict[str, Any],
    valid_task_ids: list[str],
    valid_kc_ids: list[str],
) -> list[str]:
    """Validate case-specific values that a static JSON Schema cannot enumerate."""
    if parsed is None:
        return []

    errors: list[str] = []
    allowed_tasks = set(valid_task_ids)
    allowed_kcs = set(valid_kc_ids)
    returned_task_ids: list[str] = []
    for index, task in enumerate(parsed.get("tasks", [])):
        if isinstance(task, dict):
            task_id = task.get("task_id")
            if isinstance(task_id, str):
                returned_task_ids.append(task_id)
            if task_id not in allowed_tasks:
                errors.append(f"tasks/{index}/task_id: unknown frozen task ID {task_id!r}")

    if len(returned_task_ids) != len(set(returned_task_ids)):
        errors.append("tasks: duplicate task_id values are not allowed")
    if returned_task_ids != sorted(returned_task_ids):
        errors.append("tasks: entries must be sorted by task_id")

    returned_kc_ids: list[str] = []
    for index, label in enumerate(parsed.get("labels", [])):
        if not isinstance(label, dict):
            continue
        kc_id = label.get("kc_id")
        if isinstance(kc_id, str):
            returned_kc_ids.append(kc_id)
        if kc_id not in allowed_kcs:
            errors.append(f"labels/{index}/kc_id: unknown frozen KC ID {kc_id!r}")
        for linked_index, task_id in enumerate(label.get("linked_task_ids", [])):
            if task_id not in allowed_tasks:
                errors.append(
                    f"labels/{index}/linked_task_ids/{linked_index}: unknown frozen task ID {task_id!r}"
                )
    if len(returned_kc_ids) != len(set(returned_kc_ids)):
        errors.append("labels: duplicate kc_id values are not allowed")
    if returned_kc_ids != sorted(returned_kc_ids):
        errors.append("labels: entries must be sorted by kc_id")
    return errors
