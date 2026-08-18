"""
Preprocess StudyChat nested conversation data for KC mapping and student-knowledge analysis.

Input:  student_chat_conversations_nested.json
Output: a folder of normalized CSV, Parquet (when pyarrow is installed), JSONL, and QA files.

Privacy design:
- Raw userId/chatId are pseudonymized with an HMAC-SHA256 salt by default.
- Student-authored prompts and assistant responses remain separate. Later knowledge/concern
  scoring should use student prompts as evidence and use assistant responses only as context.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
INPUT_FILE = Path("student_chat_conversations_nested.json")
OUTPUT_DIR = Path("preprocessed_student_chats")

# Set this before running. Example (PowerShell):
#   $env:STUDENT_ID_SALT = "a-long-private-random-string"
# The same salt gives stable pseudonymous IDs across future runs.
# ID_HASH_SALT = os.environ.get("STUDENT_ID_SALT", "CHANGE_THIS_PRIVATE_SALT")
ID_HASH_SALT = "fgsdaaskasksajkfashjhasdfhaskjfhaskjfh"  # Replace with a long, private, random string before sharing/output use.
KEEP_RAW_IDS = True  # Keep False for research data preparation.

# Set this only if every record in this file belongs to one known semester.
# It prevents a generic topic such as "a2" being confused with another semester's "a2".
SOURCE_TERM = "f24"  # Change to None if the file combines multiple semesters.
TOPICS_TO_KEEP = None  # Example: {"a2"}; None keeps every topic.

# -----------------------------------------------------------------------------
# Loading and normalisation helpers
# -----------------------------------------------------------------------------
def ensure_text(value: Any) -> str:
    """Convert null/non-string values safely to text without losing meaningful content."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def clean_text(value: Any) -> str:
    """Light-touch cleaning: preserve code and wording; normalize invisible characters/newlines."""
    text = unicodedata.normalize("NFKC", ensure_text(value))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def stable_id(value: Any, namespace: str) -> str:
    """Create a non-reversible, stable pseudonymous identifier."""
    payload = f"{namespace}:{ensure_text(value)}".encode("utf-8")
    digest = hmac.new(ID_HASH_SALT.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return f"{namespace}_{digest[:20]}"


def epoch_ms_to_utc(value: Any) -> str | None:
    """Convert epoch milliseconds to ISO-8601 UTC. Return None for invalid/missing values."""
    if value is None or value == "":
        return None
    try:
        dt = datetime.fromtimestamp(float(value) / 1000, tz=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def parse_label(value: Any) -> Tuple[str, str, str, str]:
    """Return label path, top-level label, sublabel, and justification."""
    if isinstance(value, dict):
        path = clean_text(value.get("label", ""))
        justification = clean_text(value.get("justification", ""))
    else:
        path = clean_text(value)
        justification = ""

    parts = [p.strip() for p in path.split(">") if p.strip()]
    top = parts[0] if parts else ""
    sub = ">".join(parts[1:]) if len(parts) > 1 else ""
    return path, top, sub, justification


def has_code(text: str) -> bool:
    """A conservative indicator for code-like content; it does not alter the source text."""
    patterns = [r"```", r"^\s*(def|class|if|for|while|import|from|return)\b", r"[{}];", r"\bself\.\w+"]
    return any(re.search(pattern, text, flags=re.MULTILINE) for pattern in patterns)


def as_records(payload: Any) -> List[Dict[str, Any]]:
    """Accept a list, a common wrapper object, or JSON Lines input."""
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("data", "records", "items", "chats", "conversations"):
            if isinstance(payload.get(key), list):
                return [x for x in payload[key] if isinstance(x, dict)]
        # A single chat record is also accepted.
        if "conversation_rows" in payload:
            return [payload]
    raise ValueError("Expected a JSON list of chat records or a wrapper containing one.")


def load_nested_json(path: Path) -> List[Dict[str, Any]]:
    raw = path.read_text(encoding="utf-8-sig").strip()
    if not raw:
        raise ValueError(f"Input file is empty: {path}")

    try:
        return as_records(json.loads(raw))
    except json.JSONDecodeError:
        # Useful only when the source has one JSON object per line (JSONL/NDJSON).
        records: List[Dict[str, Any]] = []
        for line_number, line in enumerate(raw.splitlines(), start=1):
            line = line.strip().rstrip(",")
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"The file is not valid JSON or JSONL. First failing line: {line_number}."
                ) from exc
            if isinstance(item, dict):
                records.append(item)
        if not records:
            raise ValueError("No JSON records could be read from the input file.")
        return records


def sort_turns(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort reliably even when interactionCount/timestamp is missing or malformed."""
    indexed = []
    for original_position, row in enumerate(rows):
        row = row if isinstance(row, dict) else {}
        raw_count = row.get("interactionCount")
        raw_timestamp = row.get("timestamp")
        try:
            count_key = int(raw_count)
        except (TypeError, ValueError):
            count_key = 10**12
        try:
            timestamp_key = float(raw_timestamp)
        except (TypeError, ValueError):
            timestamp_key = float("inf")
        indexed.append((count_key, timestamp_key, original_position, row))
    indexed.sort(key=lambda x: (x[0], x[1], x[2]))
    return [item[-1] for item in indexed]


def format_context(turns: List[Dict[str, Any]]) -> str:
    """Context is for resolving references only; it is not student-knowledge evidence."""
    blocks = []
    for t in turns:
        blocks.append(f"STUDENT:\n{t['prompt_clean']}")
        if t["response_clean"]:
            blocks.append(f"ASSISTANT:\n{t['response_clean']}")
    return "\n\n".join(blocks)


def format_student_only(turns: List[Dict[str, Any]]) -> str:
    """Student-only transcript for scoring knowledge/concern without assistant-answer leakage."""
    return "\n\n".join(
        f"STUDENT TURN {t['turn_index']}:\n{t['prompt_clean']}" for t in turns if t["prompt_clean"]
    )


def write_table(df: pd.DataFrame, stem: str, output_dir: Path, report: Dict[str, Any]) -> None:
    """Always write CSV; write Parquet as well when the optional engine is available."""
    csv_path = output_dir / f"{stem}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    try:
        df.to_parquet(output_dir / f"{stem}.parquet", index=False)
        report["parquet_written"].append(stem)
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        report["parquet_not_written_reason"] = str(exc)


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE.resolve()}. Put the script in the same folder as the JSON "
            "or edit INPUT_FILE at the top of this script."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_nested_json(INPUT_FILE)

    report: Dict[str, Any] = {
        "input_file": str(INPUT_FILE),
        "source_term": SOURCE_TERM,
        "raw_chat_records_read": len(records),
        "chat_records_kept": 0,
        "interactions_written": 0,
        "privacy": {
            "raw_ids_kept": KEEP_RAW_IDS,
            "ids_pseudonymized_with_hmac_sha256": not KEEP_RAW_IDS,
            "warning": "Replace CHANGE_THIS_PRIVATE_SALT before sharing/output use."
            if ID_HASH_SALT == "CHANGE_THIS_PRIVATE_SALT"
            else None,
        },
        "parquet_written": [],
        "data_quality": {
            "missing_user_id": 0,
            "missing_chat_id": 0,
            "missing_topic": 0,
            "chats_without_conversation_rows": 0,
            "missing_prompt": 0,
            "missing_response": 0,
            "invalid_or_missing_timestamp": 0,
            "duplicate_interaction_counts_within_chat": 0,
            "num_interactions_mismatch": 0,
            "chat_total_interaction_count_unexpected": 0,
        },
        "topic_counts": {},
        "source_label_counts": {},
    }

    chat_rows: List[Dict[str, Any]] = []
    interaction_rows: List[Dict[str, Any]] = []
    analysis_turns: List[Dict[str, Any]] = []
    chat_student_evidence: List[Dict[str, Any]] = []

    for source_chat_position, chat in enumerate(records):
        raw_topic = clean_text(chat.get("topic", ""))
        topic = raw_topic.lower()
        if TOPICS_TO_KEEP is not None and topic not in {x.lower() for x in TOPICS_TO_KEEP}:
            continue

        raw_user_id = chat.get("userId")
        raw_chat_id = chat.get("chatId")
        if not ensure_text(raw_user_id):
            report["data_quality"]["missing_user_id"] += 1
        if not ensure_text(raw_chat_id):
            report["data_quality"]["missing_chat_id"] += 1
        if not topic:
            report["data_quality"]["missing_topic"] += 1

        user_id = ensure_text(raw_user_id) if KEEP_RAW_IDS else stable_id(raw_user_id, "student")
        chat_id = ensure_text(raw_chat_id) if KEEP_RAW_IDS else stable_id(raw_chat_id, "chat")
        assignment_id = f"{SOURCE_TERM}_{topic}" if SOURCE_TERM and topic else topic

        raw_turns = chat.get("conversation_rows")
        if not isinstance(raw_turns, list):
            raw_turns = []
            report["data_quality"]["chats_without_conversation_rows"] += 1
        ordered_turns = sort_turns(raw_turns)

        interaction_counts = [t.get("interactionCount") for t in ordered_turns]
        valid_counts = [str(x) for x in interaction_counts if x is not None]
        if len(valid_counts) != len(set(valid_counts)):
            report["data_quality"]["duplicate_interaction_counts_within_chat"] += 1

        n_turns = len(ordered_turns)
        try:
            reported_num_interactions = int(chat.get("num_interactions"))
            num_interactions_status = "matches_actual" if reported_num_interactions == n_turns else "mismatch"
            if num_interactions_status == "mismatch":
                report["data_quality"]["num_interactions_mismatch"] += 1
        except (TypeError, ValueError):
            reported_num_interactions = None
            num_interactions_status = "missing_or_invalid"

        numeric_counts = []
        for count in interaction_counts:
            try:
                numeric_counts.append(int(count))
            except (TypeError, ValueError):
                pass
        max_interaction_count = max(numeric_counts) if numeric_counts else None
        try:
            reported_total = int(chat.get("chatTotalInteractionCount"))
            if reported_total == n_turns:
                total_status = "equals_number_of_turns"
            elif max_interaction_count is not None and reported_total == max_interaction_count:
                total_status = "equals_max_zero_based_interaction_index"
            else:
                total_status = "unexpected"
                report["data_quality"]["chat_total_interaction_count_unexpected"] += 1
        except (TypeError, ValueError):
            reported_total = None
            total_status = "missing_or_invalid"

        normalised_turns: List[Dict[str, Any]] = []
        for turn_index, source_turn in enumerate(ordered_turns):
            prompt_raw = ensure_text(source_turn.get("prompt"))
            response_raw = ensure_text(source_turn.get("response"))
            prompt_clean = clean_text(prompt_raw)
            response_clean = clean_text(response_raw)
            if not prompt_clean:
                report["data_quality"]["missing_prompt"] += 1
            if not response_clean:
                report["data_quality"]["missing_response"] += 1

            timestamp_raw = source_turn.get("timestamp")
            timestamp_utc = epoch_ms_to_utc(timestamp_raw)
            if timestamp_raw not in (None, "") and timestamp_utc is None:
                report["data_quality"]["invalid_or_missing_timestamp"] += 1

            label_path, label_top, label_sub, label_justification = parse_label(source_turn.get("llm_label"))
            if label_path:
                report["source_label_counts"][label_path] = report["source_label_counts"].get(label_path, 0) + 1

            interaction_id = f"{chat_id}_t{turn_index:04d}"
            row = {
                "interaction_id": interaction_id,
                "student_id": user_id,
                "chat_id": chat_id,
                "assignment_id": assignment_id,
                "topic": topic,
                "chat_title": clean_text(chat.get("chatTitle", "")),
                "source_chat_position": source_chat_position,
                "turn_index": turn_index,
                "interaction_count_raw": source_turn.get("interactionCount"),
                "timestamp_epoch_ms": timestamp_raw,
                "timestamp_utc": timestamp_utc,
                "is_first_turn_in_chat": turn_index == 0,
                # Preserve raw content for restricted audit/reproducibility use.
                "prompt_raw": prompt_raw,
                "response_raw": response_raw,
                # Use prompt_clean as primary student evidence in later KC/concern analysis.
                "prompt_clean": prompt_clean,
                "response_clean": response_clean,
                "prompt_char_count": len(prompt_clean),
                "response_char_count": len(response_clean),
                "student_prompt_has_code": has_code(prompt_clean),
                "assistant_response_has_code": has_code(response_clean),
                "source_llm_label_path": label_path,
                "source_llm_label_top": label_top,
                "source_llm_label_sub": label_sub,
                "source_llm_label_justification": label_justification,
            }
            if KEEP_RAW_IDS:
                row["source_user_id"] = ensure_text(raw_user_id)
                row["source_chat_id"] = ensure_text(raw_chat_id)
            normalised_turns.append(row)
            interaction_rows.append(row)

        # Make context only after normalising all turns, so each record has preceding context.
        for index, row in enumerate(normalised_turns):
            history = normalised_turns[:index]
            context_before = format_context(history)
            prior_student_only = format_student_only(history)
            analysis_turns.append(
                {
                    "analysis_unit_id": row["interaction_id"],
                    "student_id": row["student_id"],
                    "chat_id": row["chat_id"],
                    "assignment_id": row["assignment_id"],
                    "topic": row["topic"],
                    "chat_title": row["chat_title"],
                    "turn_index": row["turn_index"],
                    "timestamp_utc": row["timestamp_utc"],
                    # Score understanding from this field. Do not treat assistant answers as student knowledge.
                    "student_evidence_current_turn": row["prompt_clean"],
                    "prior_student_evidence": prior_student_only,
                    # Context may clarify pronouns/references, but must not be scored as student evidence.
                    "conversation_context_before_reference_only": context_before,
                    "assistant_response_after_turn_reference_only": row["response_clean"],
                    "source_llm_label_path": row["source_llm_label_path"],
                }
            )

        chat_rows.append(
            {
                "student_id": user_id,
                "chat_id": chat_id,
                "assignment_id": assignment_id,
                "topic": topic,
                "chat_title": clean_text(chat.get("chatTitle", "")),
                "chat_start_time_raw": chat.get("chatStartTime"),
                "source_chat_position": source_chat_position,
                "actual_number_of_turns": n_turns,
                "reported_num_interactions": reported_num_interactions,
                "reported_num_interactions_status": num_interactions_status,
                "reported_chat_total_interaction_count": reported_total,
                "reported_chat_total_interaction_count_status": total_status,
                "first_turn_timestamp_utc": normalised_turns[0]["timestamp_utc"] if normalised_turns else None,
                "last_turn_timestamp_utc": normalised_turns[-1]["timestamp_utc"] if normalised_turns else None,
                "student_turns_with_code": sum(r["student_prompt_has_code"] for r in normalised_turns),
                "source_label_paths_in_chat": json.dumps(
                    sorted({r["source_llm_label_path"] for r in normalised_turns if r["source_llm_label_path"]}),
                    ensure_ascii=False,
                ),
            }
        )

        chat_student_evidence.append(
            {
                "chat_analysis_unit_id": f"{chat_id}_all_student_turns",
                "student_id": user_id,
                "chat_id": chat_id,
                "assignment_id": assignment_id,
                "topic": topic,
                "chat_title": clean_text(chat.get("chatTitle", "")),
                "number_of_student_turns": n_turns,
                # Suitable for a chat-level evidence synthesis; assistant output excluded deliberately.
                "student_only_chat_evidence": format_student_only(normalised_turns),
                "interaction_ids": [r["interaction_id"] for r in normalised_turns],
            }
        )

        report["topic_counts"][topic or "<missing>"] = report["topic_counts"].get(topic or "<missing>", 0) + 1

    chats_df = pd.DataFrame(chat_rows)
    interactions_df = pd.DataFrame(interaction_rows)
    if not chats_df.empty:
        chats_df = chats_df.sort_values(["assignment_id", "student_id", "chat_id"], kind="stable")
    if not interactions_df.empty:
        interactions_df = interactions_df.sort_values(["assignment_id", "student_id", "chat_id", "turn_index"], kind="stable")

    report["chat_records_kept"] = len(chats_df)
    report["interactions_written"] = len(interactions_df)

    write_table(chats_df, "chats", OUTPUT_DIR, report)
    write_table(interactions_df, "interactions", OUTPUT_DIR, report)

    with (OUTPUT_DIR / "analysis_turns.jsonl").open("w", encoding="utf-8") as f:
        for item in analysis_turns:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with (OUTPUT_DIR / "chat_student_evidence.jsonl").open("w", encoding="utf-8") as f:
        for item in chat_student_evidence:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with (OUTPUT_DIR / "preprocessing_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Preprocessing complete.")
    print(f"Chats: {len(chats_df):,}")
    print(f"Interactions: {len(interactions_df):,}")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")
    print("Files: chats.csv, interactions.csv, analysis_turns.jsonl, "
          "chat_student_evidence.jsonl, preprocessing_report.json")


if __name__ == "__main__":
    main()
