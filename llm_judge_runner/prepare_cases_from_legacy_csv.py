from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any


def parse_rows(value: str) -> list[dict[str, Any]]:
    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(value)
            if isinstance(obj, list):
                return [x for x in obj if isinstance(x, dict)]
        except Exception:
            pass
    raise ValueError("Could not parse conversation_rows")


def first_present(row: dict[str, str], names: list[str]) -> str | None:
    for n in names:
        value = (row.get(n) or "").strip()
        if value:
            return value
    return None


def convert(input_csv: Path, output_jsonl: Path, assignment_id: str) -> None:
    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as out:
        for row_no, row in enumerate(rows, start=1):
            conv_id = first_present(row, ["conversation_id", "conversationId", "chat_id", "chatId", "conversation"])
            if not conv_id:
                conv_id = f"conversation_{row_no:06d}"
            student_id = first_present(row, ["student_id", "studentId", "user_id", "userId", "participant_id"])
            chunk_id = first_present(row, ["chunk_id", "chunkId"]) or conv_id
            turns: list[dict[str, Any]] = []

            if row.get("conversation_rows"):
                for i, raw in enumerate(parse_rows(row["conversation_rows"]), start=1):
                    base = raw.get("interactionCount", i)
                    prompt = str(raw.get("prompt") or "").strip()
                    response = str(raw.get("response") or "").strip()
                    if prompt:
                        turns.append({"turn_index": f"{base}.student", "speaker": "student", "text": prompt})
                    if response:
                        turns.append({"turn_index": f"{base}.assistant", "speaker": "assistant", "text": response})
            else:
                prompt = first_present(row, ["prompt", "Prompt", "student_prompt", "student_message", "question"])
                response = first_present(row, ["response", "Response", "assistant_response", "assistant_message", "answer"])
                if prompt:
                    turns.append({"turn_index": "1.student", "speaker": "student", "text": prompt})
                if response:
                    turns.append({"turn_index": "1.assistant", "speaker": "assistant", "text": response})

            case = {
                "case_id": f"{assignment_id}::{conv_id}::T1::base",
                "base_conversation_id": conv_id,
                "conversation_id": conv_id,
                "chunk_id": chunk_id,
                "assignment_id": assignment_id,
                "student_id": student_id,
                "test_id": "T1",
                "variant_id": "base",
                "variant_type": "baseline",
                "is_baseline": True,
                "prompt_variant_id": "canonical",
                "transformation_metadata": {"source_file": str(input_csv)},
                "turns": turns,
            }
            out.write(json.dumps(case, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input_csv")
    p.add_argument("output_jsonl")
    p.add_argument("--assignment-id", required=True, help="e.g. f24_a3")
    args = p.parse_args()
    convert(Path(args.input_csv), Path(args.output_jsonl), args.assignment_id)
