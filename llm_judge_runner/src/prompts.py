from __future__ import annotations

import re
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
    required = {"{{DYNAMIC_CASE_INPUT}}", "{{OUTPUT_SCHEMA}}"}
    missing = sorted(token for token in required if token not in template)
    if missing:
        raise RuntimeError(f"User prompt file {path} is missing required placeholders: {missing}")

    stable_template, dynamic_template = template.split("{{DYNAMIC_CASE_INPUT}}", 1)
    stable_prefix = stable_template.replace("{{OUTPUT_SCHEMA}}", output_schema_text).rstrip()
    dynamic_case_input = "\n".join(
        [
            "Metadata:",
            f'assignment_id: {case["assignment_id"]}',
            f'conversation_id: {case["conversation_id"]}',
            f'chunk_id: {case["chunk_id"]}',
            f'base_conversation_id: {case["base_conversation_id"]}',
            f'case_id: {case["case_id"]}',
            f'test_id: {case["test_id"]}',
            f'variant_id: {case["variant_id"]}',
            "",
            "<conversation>",
            conversation_text,
            "</conversation>",
        ]
    )
    dynamic_suffix = dynamic_case_input + dynamic_template
    rendered = f"{stable_prefix}\n\n{dynamic_suffix}"
    if re.search(r"\{\{[A-Z0-9_]+\}\}", rendered):
        raise RuntimeError(f"User prompt file {path} contains an unresolved placeholder")
    return {
        "text": rendered,
        "stable_prefix_text": stable_prefix,
        "dynamic_suffix_text": dynamic_suffix,
        "template_text": template,
        "template_hash": sha256_text(template),
        "rendered_hash": sha256_text(rendered),
        "stable_prefix_hash": sha256_text(stable_prefix),
        "dynamic_suffix_hash": sha256_text(dynamic_suffix),
        "source_file": str(path),
    }
