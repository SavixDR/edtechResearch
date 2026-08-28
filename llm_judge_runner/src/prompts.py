from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import load_json, sha256_text

PLACEHOLDER = "[PASTE YOUR EXACT FROZEN V3 PROMPT HERE]"


def load_prompt_variants(path: Path, project_root: Path) -> dict[str, dict[str, Any]]:
    raw = load_json(path)
    out: dict[str, dict[str, Any]] = {}
    for key, item in raw.items():
        file_path = Path(item["file"])
        if not file_path.is_absolute():
            file_path = project_root / file_path
        user_file_path = Path(item["user_file"])
        if not user_file_path.is_absolute():
            user_file_path = project_root / user_file_path
        out[key] = {**item, "resolved_file": file_path, "resolved_user_file": user_file_path}
    return out


def render_system_prompt(
    prompt_variant: dict[str, Any],
    assignment_id: str,
    frozen_tasks: str,
    task_kc_mapping: str,
    frozen_codebook: str,
    assignment_reference_material: str = "",
    human_validated_few_shot_examples: str = "",
) -> dict[str, Any]:
    path: Path = prompt_variant["resolved_file"]
    template = path.read_text(encoding="utf-8")
    if PLACEHOLDER in template:
        raise RuntimeError(
            f"Prompt file {path} still contains the placeholder. Paste your exact frozen V3 prompt before running live models."
        )
    rendered = template.replace("{{ASSIGNMENT_ID}}", assignment_id)
    rendered = rendered.replace("{{FROZEN_TASKS}}", frozen_tasks)
    rendered = rendered.replace("{{TASK_KC_MAPPING}}", task_kc_mapping)
    rendered = rendered.replace("{{FROZEN_CODEBOOK}}", frozen_codebook)
    rendered = rendered.replace("{{ASSIGNMENT_REFERENCE_MATERIAL}}", assignment_reference_material)
    rendered = rendered.replace("{{HUMAN_VALIDATED_FEW_SHOT_EXAMPLES}}", human_validated_few_shot_examples)
    return {
        "text": rendered,
        "template_text": template,
        "template_hash": sha256_text(template),
        "rendered_hash": sha256_text(rendered),
        "source_file": str(path),
    }


def render_user_prompt(
    prompt_variant: dict[str, Any],
    case: dict[str, Any],
    conversation_text: str,
    output_schema_text: str,
) -> dict[str, Any]:
    path: Path = prompt_variant["resolved_user_file"]
    template = path.read_text(encoding="utf-8")
    required = {"{{CONVERSATION}}", "{{OUTPUT_SCHEMA}}"}
    missing = sorted(token for token in required if token not in template)
    if missing:
        raise RuntimeError(f"User prompt file {path} is missing required placeholders: {missing}")

    replacements = {
        "{{ASSIGNMENT_ID}}": str(case["assignment_id"]),
        "{{CONVERSATION_ID}}": str(case["conversation_id"]),
        "{{CHUNK_ID}}": str(case["chunk_id"]),
        "{{BASE_CONVERSATION_ID}}": str(case["base_conversation_id"]),
        "{{CASE_ID}}": str(case["case_id"]),
        "{{TEST_ID}}": str(case["test_id"]),
        "{{VARIANT_ID}}": str(case["variant_id"]),
        "{{CONVERSATION}}": conversation_text,
        "{{OUTPUT_SCHEMA}}": output_schema_text,
    }
    rendered = template
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    return {
        "text": rendered,
        "template_text": template,
        "template_hash": sha256_text(template),
        "rendered_hash": sha256_text(rendered),
        "source_file": str(path),
    }
