from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import read_jsonl, sha256_json

REQUIRED_CASE_FIELDS = {
    "case_id",
    "base_conversation_id",
    "conversation_id",
    "assignment_id",
    "test_id",
    "variant_id",
    "variant_type",
    "is_baseline",
    "prompt_variant_id",
    "turns",
}


def normalize_speaker(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    if text in {"student", "user", "human", "learner"}:
        return "student"
    if text in {"assistant", "llm", "bot", "chatgpt", "gpt", "model"}:
        return "assistant"
    return text or "unknown"


def validate_case(case: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_CASE_FIELDS - set(case))
    if missing:
        raise ValueError(f"Case {case.get('case_id', '<unknown>')} missing fields: {missing}")
    for field in ("case_id", "conversation_id", "chunk_id"):
        value = case.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Case {case.get('case_id', '<unknown>')} {field} must be a non-empty string")
    chunk_index = case.get("chunk_index")
    chunk_count = case.get("chunk_count")
    if (chunk_index is None) != (chunk_count is None):
        raise ValueError(
            f"Case {case['case_id']} must provide both chunk_index and chunk_count, or omit both"
        )
    if chunk_index is not None:
        if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 1:
            raise ValueError(f"Case {case['case_id']} chunk_index must be a positive integer")
        if isinstance(chunk_count, bool) or not isinstance(chunk_count, int) or chunk_count < 1:
            raise ValueError(f"Case {case['case_id']} chunk_count must be a positive integer")
        if chunk_index > chunk_count:
            raise ValueError(f"Case {case['case_id']} chunk_index cannot exceed chunk_count")
    if not isinstance(case["turns"], list):
        raise ValueError(f"Case {case['case_id']} turns must be a list")
    for idx, turn in enumerate(case["turns"], start=1):
        if not isinstance(turn, dict) or "text" not in turn:
            raise ValueError(f"Case {case['case_id']} turn {idx} must be an object containing text")


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = read_jsonl(path)
    seen: set[str] = set()
    for case in cases:
        # Older whole-conversation case files did not have chunk_id.
        case.setdefault("chunk_id", str(case.get("conversation_id", "")))
        case.setdefault("student_id", None)
        case.setdefault("transformation_metadata", {})
        validate_case(case)
        case_id = str(case["case_id"])
        if case_id in seen:
            raise ValueError(f"Duplicate case_id: {case_id}")
        seen.add(case_id)
        case["case_hash"] = sha256_json({k: v for k, v in case.items() if k != "case_hash"})
    return cases


def render_conversation(case: dict[str, Any]) -> str:
    lines = [
        f"Conversation ID: {case['conversation_id']}",
        f"Chunk ID: {case['chunk_id']}",
        f"Base conversation ID: {case['base_conversation_id']}",
        f"Case ID: {case['case_id']}",
        f"Test ID: {case['test_id']}",
        f"Variant ID: {case['variant_id']}",
        "",
    ]
    for i, turn in enumerate(case["turns"], start=1):
        speaker = normalize_speaker(turn.get("speaker") or turn.get("role"))
        turn_index = turn.get("turn_index", i)
        text = str(turn.get("text", "")).strip()
        if not text:
            continue
        lines.append(f"TURN {i:03d} | original_turn_index={turn_index} | speaker={speaker}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip()
