# teacher_label_kcs.py

import os
import re
import json
import time
import hashlib
import ast
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Iterable, Optional, Tuple

import pandas as pd
try:
    import tiktoken
except ImportError:
    tiktoken = None

from openai import OpenAI
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# 0. CONFIG
# ============================================================

# Pilot comparison models. Every model receives the same prompt and chunks.
MODELS_TO_COMPARE = [
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
]

# Full batch processing uses only this model after the pilot comparison.
BATCH_MODEL = "gpt-5.6-terra"
REASONING_EFFORT = "medium"

# ---------------- USER-EDITABLE SELECTION ----------------
# Assignment names use the short form ("a3", not "f24_a3"). A2 is skipped by
# default because it has already been labelled. Remove/add items whenever needed.
ASSIGNMENTS_TO_PROCESS = ["a2", "a3", "a4", "a5", "a6", "a7"]
SKIP_ASSIGNMENTS = {"a2"}

# Empty list = choose SAMPLE_SIZE conversations randomly from that assignment.
# To process exact chats instead, paste their chatId values into the relevant list.
CONVERSATION_IDS_BY_ASSIGNMENT = {
    "a3": [],
    "a4": [],
    "a5": [],
    "a6": [],
    "a7": [],
}

# These IDs are excluded after either random or explicit selection.
SKIP_CONVERSATION_IDS_BY_ASSIGNMENT = {
    "a3": set(),
    "a4": set(),
    "a5": set(),
    "a6": set(),
    "a7": set(),
}

SAMPLE_SIZE = 5
RANDOM_SEED = 20260719  # Same value reproduces exactly the same random chats.

# KC filenames are not fully consistent, so keep the paths explicit and editable.
KC_JSON_PATHS = {
    "a2": Path("KC/f24_a2/f24_a2_kc_model.json"),
    "a3": Path("KC/f24_a3/f24_a3_corrected_kc_model.json"),
    "a4": Path("KC/f24_a4/f24_a4_corrected_kc_model.json"),
    "a5": Path("KC/f24_a5/f24_a5_corrected_kc_model.json"),
    "a6": Path("KC/f24_a6/f24_a6_corrected_complete_kc_model.json"),
    "a7": Path("KC/f24_a7/f24_a7_corrected_kc_model.json"),
}
SAMPLES_DIR = Path("student_chat_conversations_preprocessing/samples")

# The parent process starts one isolated child run per enabled assignment. This
# keeps each codebook, request file, pilot output, and batch record separate.
_assignment_from_parent = os.environ.get("TEACHER_KC_ASSIGNMENT")
if __name__ == "__main__" and not _assignment_from_parent:
    enabled = [a for a in ASSIGNMENTS_TO_PROCESS if a not in SKIP_ASSIGNMENTS]
    if not enabled:
        raise SystemExit("No assignments enabled. Edit ASSIGNMENTS_TO_PROCESS/SKIP_ASSIGNMENTS.")
    print(f"Assignments selected: {', '.join(enabled)}")
    for assignment_name in enabled:
        child_env = os.environ.copy()
        child_env["TEACHER_KC_ASSIGNMENT"] = assignment_name
        subprocess.run([sys.executable, str(Path(__file__).resolve())], env=child_env, check=True)
    raise SystemExit(0)

CURRENT_ASSIGNMENT = _assignment_from_parent or next(
    (a for a in ASSIGNMENTS_TO_PROCESS if a not in SKIP_ASSIGNMENTS), "a3"
)
if CURRENT_ASSIGNMENT not in KC_JSON_PATHS:
    raise ValueError(f"No KC path configured for {CURRENT_ASSIGNMENT!r}")

TARGET_ASSIGNMENT_ID = f"f24_{CURRENT_ASSIGNMENT}"
KC_JSON_PATH = KC_JSON_PATHS[CURRENT_ASSIGNMENT]
CONVERSATION_FILE = SAMPLES_DIR / f"{CURRENT_ASSIGNMENT}_samples.csv"

OUTPUT_DIR = Path("teacher_label_outputs") / CURRENT_ASSIGNMENT
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FROZEN_CODEBOOK_TXT = OUTPUT_DIR / "frozen_codebook_compact.txt"
REQUESTS_JSONL = OUTPUT_DIR / "batch_requests.jsonl"
PILOT_OUTPUT_JSONL = OUTPUT_DIR / "pilot_comparison_outputs.jsonl"
PILOT_SUMMARY_CSV = OUTPUT_DIR / "pilot_comparison_summary.csv"
FINAL_LABELS_JSONL = OUTPUT_DIR / "teacher_labels_validated.jsonl"
ERRORS_JSONL = OUTPUT_DIR / "teacher_labels_errors.jsonl"
BATCH_INFO_JSON = OUTPUT_DIR / "batch_info.json"

# Safety settings
DRY_RUN_ONLY = False
# Keep True for the first run. It prepares files and estimates tokens.
# After inspecting REQUESTS_JSONL, set to False to submit the batch.

RUN_PILOT = True

SUBMIT_BATCH = False
# Set True only after pilot outputs look good.

# If a single conversation is too long, split into chunks.
MAX_CONVERSATION_TOKENS = 12000
CHUNK_OVERLAP_TURNS = 2

# Output control
MAX_OUTPUT_TOKENS = 8000

VALID_STATUS = [
    "difficulty_evidence",
    "misconception",
    "information_seeking",
    "resolved_understanding",
    "neutral_mention",
    "insufficient_evidence",
]

# Optional dollar estimate.
# Put your current OpenAI dashboard/API pricing here if you want dollar estimates.
INPUT_USD_PER_1M = 2.50
OUTPUT_USD_PER_1M = 15


# ============================================================
# 1. BASIC UTILITIES
# ============================================================

def ensure_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_encoder(model: str):
    if tiktoken is None:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        # Custom model aliases are not known to tiktoken. Avoid an implicit
        # network download and use the conservative character estimate instead.
        print(f"WARNING: no tiktoken mapping for {model}; using the character token estimate.")
        return None


ENCODER = get_encoder(BATCH_MODEL)


def count_tokens(text: str) -> int:
    if ENCODER is None:
        # Rough fallback: English-ish approximation.
        return max(1, len(text) // 4)
    return len(ENCODER.encode(text))


# ============================================================
# 2. LOAD AND COMPACT THE FROZEN KC CODEBOOK
# ============================================================

def load_kc_model(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compact_list(values: Any, max_items: int = 8) -> str:
    if not values:
        return "-"
    if isinstance(values, str):
        return values
    if isinstance(values, list):
        vals = [ensure_text(v) for v in values if ensure_text(v)]
        if len(vals) > max_items:
            vals = vals[:max_items] + [f"...({len(values) - max_items} more)"]
        return "; ".join(vals) if vals else "-"
    return ensure_text(values)


def compact_difficulty_signals(signals: Any, max_items: int = 4) -> str:
    if not signals:
        return "-"
    out = []
    for sig in signals[:max_items]:
        if isinstance(sig, dict):
            signal_type = ensure_text(sig.get("signal_type", ""))
            description = ensure_text(sig.get("description", ""))
            interpretation = ensure_text(sig.get("interpretation", ""))
            sufficient = sig.get("sufficient_alone", None)
            line = f"{signal_type}: {description}"
            if interpretation:
                line += f" | interpretation: {interpretation}"
            if sufficient is not None:
                line += f" | sufficient_alone={sufficient}"
            out.append(line)
        else:
            out.append(ensure_text(sig))
    if len(signals) > max_items:
        out.append(f"...({len(signals) - max_items} more)")
    return "; ".join(out) if out else "-"


def build_frozen_codebook_text(kc_model: Dict[str, Any]) -> Tuple[str, List[str]]:
    assignment = kc_model.get("assignment", {})
    kcs = kc_model.get("knowledge_components", [])

    lines = []
    lines.append("FROZEN KC CODEBOOK")
    lines.append("Do not invent KC IDs. Select KC IDs only from this codebook.")
    lines.append("")
    assignment_id = ensure_text(assignment.get("assignment_id")) or TARGET_ASSIGNMENT_ID
    lines.append(f"Assignment ID: {assignment_id}")
    lines.append(f"Assignment title: {assignment.get('assignment_title', '')}")
    lines.append(f"Assignment objective: {assignment.get('objective', '')}")
    lines.append("")
    lines.append("KC ENTRIES")
    lines.append("")

    kc_ids = []

    for kc in kcs:
        kc_id = ensure_text(kc.get("kc_id"))
        if not kc_id:
            continue

        kc_ids.append(kc_id)

        lines.append(f"KC_ID: {kc_id}")
        lines.append(f"Name: {ensure_text(kc.get('canonical_name'))}")
        lines.append(f"Definition: {ensure_text(kc.get('definition'))}")
        lines.append(
            "Type/Granularity/Level: "
            f"{ensure_text(kc.get('kc_type'))} | "
            f"{ensure_text(kc.get('granularity'))} | "
            f"{ensure_text(kc.get('requirement_level'))}"
        )
        lines.append(f"Aliases: {compact_list(kc.get('aliases'))}")
        lines.append(f"Student must know/do: {compact_list(kc.get('student_must_know_or_do'))}")
        lines.append(f"Difficulty signals: {compact_difficulty_signals(kc.get('possible_difficulty_signals'))}")
        lines.append(f"Parent KC: {ensure_text(kc.get('parent_kc_id')) or '-'}")
        lines.append(f"Prerequisite KCs: {compact_list(kc.get('prerequisite_kc_ids'))}")
        lines.append(f"Commonly confused with: {compact_list(kc.get('commonly_confused_with_kc_ids'))}")
        lines.append(f"Related task IDs: {compact_list(kc.get('related_task_ids'))}")
        lines.append("---")

    return "\n".join(lines).strip(), kc_ids


kc_model = load_kc_model(KC_JSON_PATH)
FROZEN_CODEBOOK, VALID_KC_IDS = build_frozen_codebook_text(kc_model)
CODEBOOK_HASH = sha256_text(FROZEN_CODEBOOK)

FROZEN_CODEBOOK_TXT.write_text(FROZEN_CODEBOOK, encoding="utf-8")

print(f"Loaded {len(VALID_KC_IDS)} KCs")
print(f"Pilot comparison models: {', '.join(MODELS_TO_COMPARE)}")
print(f"Selected batch model: {BATCH_MODEL}")
print(f"Reasoning effort: {REASONING_EFFORT}")
print(f"Codebook hash: {CODEBOOK_HASH}")
print(f"Compact codebook saved to: {FROZEN_CODEBOOK_TXT}")
print(f"Approx codebook tokens: {count_tokens(FROZEN_CODEBOOK):,}")


# ============================================================
# 3. READ CONVERSATIONS
# ============================================================

CONV_ID_COL_CANDIDATES = [
    "conversation_id", "conversationId", "chat_id", "chatId",
    "session_id", "thread_id", "dialogue_id", "conversation"
]

STUDENT_ID_COL_CANDIDATES = [
    "student_id", "studentId", "user_id", "userId", "participant_id"
]

TURN_COL_CANDIDATES = [
    "interactionCount", "interaction_count", "turn_id", "turn_index",
    "message_index", "index", "timestamp"
]

ROLE_COL_CANDIDATES = [
    "role", "speaker", "author", "sender", "LLM_Label"
]

CONTENT_COL_CANDIDATES = [
    "content", "message", "text", "utterance", "body", "full_conversation"
]

PROMPT_COL_CANDIDATES = [
    "prompt", "Prompt", "student_prompt", "student_message", "user_prompt", "question"
]

RESPONSE_COL_CANDIDATES = [
    "response", "Response", "assistant_response", "assistant_message", "answer"
]


def filter_to_target_assignment(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not TARGET_ASSIGNMENT_ID:
        return df

    if "assignment_id" not in df.columns:
        # Per-assignment sample files use `topic` (a3, a4, ...) rather than an
        # assignment_id column. Their filename already scopes the assignment.
        if "topic" in df.columns:
            topics = set(df["topic"].map(ensure_text))
            if topics - {CURRENT_ASSIGNMENT}:
                raise ValueError(f"Unexpected topics in {path}: {sorted(topics)}")
        return df

    original_count = len(df)
    assignment_ids = df["assignment_id"].map(ensure_text)
    filtered = df.loc[assignment_ids == TARGET_ASSIGNMENT_ID].copy()

    print(
        f"Filtered conversations to assignment_id={TARGET_ASSIGNMENT_ID}: "
        f"{len(filtered):,}/{original_count:,} rows kept"
    )

    if filtered.empty:
        raise ValueError(f"No rows found for assignment_id={TARGET_ASSIGNMENT_ID!r} in {path}")

    return filtered


def first_existing(columns: Iterable[str], candidates: List[str]) -> Optional[str]:
    colset = set(columns)
    for c in candidates:
        if c in colset:
            return c
    return None


def normalize_role(role: Any) -> str:
    r = ensure_text(role).lower()
    if r in {"user", "student", "human", "learner"}:
        return "student"
    if r in {"assistant", "llm", "bot", "chatgpt", "gpt"}:
        return "assistant"
    if "student" in r or "user" in r:
        return "student"
    if "assistant" in r or "llm" in r or "gpt" in r:
        return "assistant"
    return ensure_text(role) or "unknown"


def parse_conversation_rows(value: Any) -> List[Dict[str, Any]]:
    """Parse the list stored in samples/*.csv without executing arbitrary code."""
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    text = ensure_text(value)
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, list):
                return [row for row in parsed if isinstance(row, dict)]
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    raise ValueError("Could not parse conversation_rows as a list of turn dictionaries")


def select_conversation_rows(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Apply explicit IDs/skips, or reproducibly select SAMPLE_SIZE random chats."""
    id_col = first_existing(df.columns, CONV_ID_COL_CANDIDATES)
    if id_col is None:
        raise ValueError(f"No conversation/chat ID column found in {path}")

    available = df.copy()
    available["_selection_id"] = available[id_col].map(ensure_text)
    skipped = set(SKIP_CONVERSATION_IDS_BY_ASSIGNMENT.get(CURRENT_ASSIGNMENT, set()))
    available = available.loc[~available["_selection_id"].isin(skipped)].copy()

    requested = CONVERSATION_IDS_BY_ASSIGNMENT.get(CURRENT_ASSIGNMENT, [])
    if requested:
        requested = list(dict.fromkeys(map(ensure_text, requested)))
        missing = sorted(set(requested) - set(available["_selection_id"]))
        if missing:
            raise ValueError(f"Configured chat IDs not found for {CURRENT_ASSIGNMENT}: {missing}")
        order = {chat_id: i for i, chat_id in enumerate(requested)}
        selected = available.loc[available["_selection_id"].isin(requested)].copy()
        selected["_selection_order"] = selected["_selection_id"].map(order)
        selected = selected.sort_values("_selection_order", kind="stable")
        selection_mode = "explicit"
    else:
        if SAMPLE_SIZE < 1:
            raise ValueError("SAMPLE_SIZE must be at least 1")
        if len(available) < SAMPLE_SIZE:
            raise ValueError(
                f"Requested {SAMPLE_SIZE} chats for {CURRENT_ASSIGNMENT}, "
                f"but only {len(available)} remain after skips"
            )
        assignment_number = int(re.sub(r"\D", "", CURRENT_ASSIGNMENT) or 0)
        selected = available.sample(
            n=SAMPLE_SIZE,
            random_state=RANDOM_SEED + assignment_number,
            replace=False,
        )
        selection_mode = "random"

    manifest_columns = [
        c for c in [id_col, "userId", "topic", "semester", "interaction_stratum"]
        if c in selected.columns
    ]
    selected[manifest_columns].to_csv(OUTPUT_DIR / "selected_chats.csv", index=False)
    selection_info = {
        "assignment": CURRENT_ASSIGNMENT,
        "assignment_id": TARGET_ASSIGNMENT_ID,
        "mode": selection_mode,
        "sample_size": len(selected),
        "random_seed": RANDOM_SEED if selection_mode == "random" else None,
        "skipped_conversation_ids": sorted(skipped),
        "selected_conversation_ids": selected["_selection_id"].tolist(),
        "source_file": str(path),
    }
    (OUTPUT_DIR / "selection_manifest.json").write_text(
        json.dumps(selection_info, indent=2), encoding="utf-8"
    )
    print(
        f"Selected {len(selected)} {CURRENT_ASSIGNMENT} chats ({selection_mode}): "
        + ", ".join(selection_info["selected_conversation_ids"])
    )
    return selected.drop(columns=["_selection_id", "_selection_order"], errors="ignore")


def dataframe_to_conversations(df: pd.DataFrame, source_file: Path) -> List[Dict[str, Any]]:
    columns = list(df.columns)

    conv_col = first_existing(columns, CONV_ID_COL_CANDIDATES)
    student_col = first_existing(columns, STUDENT_ID_COL_CANDIDATES)
    turn_col = first_existing(columns, TURN_COL_CANDIDATES)
    role_col = first_existing(columns, ROLE_COL_CANDIDATES)
    content_col = first_existing(columns, CONTENT_COL_CANDIDATES)
    prompt_col = first_existing(columns, PROMPT_COL_CANDIDATES)
    response_col = first_existing(columns, RESPONSE_COL_CANDIDATES)
    full_conversation_col = "full_conversation" if "full_conversation" in columns else None
    conversation_rows_col = "conversation_rows" if "conversation_rows" in columns else None

    if conv_col is None:
        print("WARNING: No conversation ID column found. Treating entire file as one conversation.")
        df["_conversation_id_fallback"] = "conversation_000001"
        conv_col = "_conversation_id_fallback"

    conversations = []

    # sort=False preserves the random/explicit selection order in pilot outputs.
    for conv_id, g in df.groupby(conv_col, dropna=False, sort=False):
        if turn_col and turn_col in g.columns:
            g = g.sort_values(turn_col, kind="stable")

        turns = []
        student_id = None

        if student_col and student_col in g.columns:
            vals = [ensure_text(v) for v in g[student_col].dropna().unique()]
            student_id = vals[0] if vals else None

        # Case A: one sample row contains a serialized list of prompt/response turns.
        if conversation_rows_col:
            for _, row in g.iterrows():
                for turn_i, raw_turn in enumerate(parse_conversation_rows(row.get(conversation_rows_col))):
                    turn_base = ensure_text(raw_turn.get("interactionCount")) or str(turn_i)
                    prompt = ensure_text(raw_turn.get("prompt"))
                    response = ensure_text(raw_turn.get("response"))
                    if prompt:
                        turns.append({
                            "turn_index": f"{turn_base}.student",
                            "speaker": "student",
                            "text": prompt,
                        })
                    if response:
                        turns.append({
                            "turn_index": f"{turn_base}.assistant",
                            "speaker": "assistant",
                            "text": response,
                        })

        # Case B: one row is an already-rendered transcript.
        elif full_conversation_col:
            for row_i, row in g.iterrows():
                text = ensure_text(row.get(full_conversation_col))
                if not text:
                    continue
                turns.append({
                    "turn_index": ensure_text(row.get(turn_col)) if turn_col else str(row_i),
                    "speaker": "transcript",
                    "text": text,
                })

        # Case C: one row has student prompt and assistant response.
        elif prompt_col and response_col:
            for row_i, row in g.iterrows():
                turn_base = ensure_text(row.get(turn_col)) if turn_col else str(row_i)
                prompt = ensure_text(row.get(prompt_col))
                response = ensure_text(row.get(response_col))

                if prompt:
                    turns.append({
                        "turn_index": f"{turn_base}.student",
                        "speaker": "student",
                        "text": prompt,
                    })
                if response:
                    turns.append({
                        "turn_index": f"{turn_base}.assistant",
                        "speaker": "assistant",
                        "text": response,
                    })

        # Case D: one row is one message turn.
        elif role_col and content_col:
            for row_i, row in g.iterrows():
                text = ensure_text(row.get(content_col))
                if not text:
                    continue
                turns.append({
                    "turn_index": ensure_text(row.get(turn_col)) if turn_col else str(row_i),
                    "speaker": normalize_role(row.get(role_col)),
                    "text": text,
                })

        # Case E: fallback: concatenate text-like columns per row.
        else:
            text_cols = []
            for c in columns:
                if df[c].dtype == "object" and c not in {conv_col, student_col}:
                    text_cols.append(c)

            for row_i, row in g.iterrows():
                parts = []
                for c in text_cols:
                    value = ensure_text(row.get(c))
                    if value:
                        parts.append(f"{c}: {value}")
                if parts:
                    turns.append({
                        "turn_index": ensure_text(row.get(turn_col)) if turn_col else str(row_i),
                        "speaker": "unknown",
                        "text": "\n".join(parts),
                    })

        conversations.append({
            "conversation_id": ensure_text(conv_id),
            "student_id": student_id,
            "turns": turns,
            "source_file": str(source_file),
        })

    return conversations


def load_conversations(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
        df = filter_to_target_assignment(df, path)
        df = select_conversation_rows(df, path)
        return dataframe_to_conversations(df, path)

    if suffix == ".jsonl":
        rows = list(read_jsonl(path))

        # If each line is already one conversation with turns.
        if rows and isinstance(rows[0].get("turns"), list):
            conversations = []
            for i, row in enumerate(rows):
                conversations.append({
                    "conversation_id": ensure_text(row.get("conversation_id") or row.get("chat_id") or f"conversation_{i+1:06d}"),
                    "student_id": ensure_text(row.get("student_id") or row.get("user_id")) or None,
                    "turns": row.get("turns", []),
                    "source_file": str(path),
                })
            return conversations

        # Otherwise treat JSONL as a table.
        df = pd.DataFrame(rows)
        df = filter_to_target_assignment(df, path)
        return dataframe_to_conversations(df, path)

    raise ValueError(f"Unsupported conversation file type: {path}")


def render_conversation(conv: Dict[str, Any], turns: Optional[List[Dict[str, Any]]] = None) -> str:
    turns = turns if turns is not None else conv["turns"]

    lines = []
    lines.append(f"Conversation ID: {conv.get('conversation_id')}")
    if conv.get("student_id"):
        lines.append(f"Student ID: {conv.get('student_id')}")
    lines.append("")

    for i, turn in enumerate(turns, start=1):
        speaker = normalize_role(turn.get("speaker") or turn.get("role") or turn.get("author") or "unknown")
        turn_index = ensure_text(turn.get("turn_index") or turn.get("interactionCount") or i)
        text = ensure_text(turn.get("text") or turn.get("content") or turn.get("message") or turn.get("utterance"))
        if not text:
            continue
        lines.append(f"TURN {i:03d} | original_turn_index={turn_index} | speaker={speaker}")
        lines.append(text)
        lines.append("")

    return "\n".join(lines).strip()


def chunk_conversation(conv: Dict[str, Any]) -> List[Dict[str, Any]]:
    turns = conv.get("turns", [])
    full_text = render_conversation(conv, turns)
    if count_tokens(full_text) <= MAX_CONVERSATION_TOKENS:
        return [{
            "conversation_id": conv["conversation_id"],
            "student_id": conv.get("student_id"),
            "source_file": conv.get("source_file"),
            "chunk_id": "chunk_001_of_001",
            "chunk_index": 1,
            "num_chunks": 1,
            "turns": turns,
            "conversation_text": full_text,
        }]

    chunks = []
    current = []

    for turn in turns:
        proposed = current + [turn]
        proposed_text = render_conversation(conv, proposed)

        if current and count_tokens(proposed_text) > MAX_CONVERSATION_TOKENS:
            chunks.append(current)
            overlap = current[-CHUNK_OVERLAP_TURNS:] if CHUNK_OVERLAP_TURNS > 0 else []
            current = overlap + [turn]
        else:
            current = proposed

    if current:
        chunks.append(current)

    out = []
    n = len(chunks)
    for idx, chunk_turns in enumerate(chunks, start=1):
        out.append({
            "conversation_id": conv["conversation_id"],
            "student_id": conv.get("student_id"),
            "source_file": conv.get("source_file"),
            "chunk_id": f"chunk_{idx:03d}_of_{n:03d}",
            "chunk_index": idx,
            "num_chunks": n,
            "turns": chunk_turns,
            "conversation_text": render_conversation(conv, chunk_turns),
        })
    return out


conversations = load_conversations(CONVERSATION_FILE)
chunks = []
for conv in conversations:
    if not conv.get("turns"):
        continue
    chunks.extend(chunk_conversation(conv))

print(f"Loaded conversations: {len(conversations):,}")
print(f"Prepared chunks/requests: {len(chunks):,}")


# ============================================================
# 4. STRUCTURED OUTPUT SCHEMA
# ============================================================

def make_output_schema(valid_kc_ids: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "conversation_id": {"type": "string"},
            "chunk_id": {"type": "string"},
            "assignment_id": {"type": "string", "enum": [TARGET_ASSIGNMENT_ID]},
            "labels": {
                "type": "array",
                "description": "Only KCs directly evidenced by the student's own words. Use an empty array when there is no direct KC evidence.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "kc_id": {"type": "string", "enum": valid_kc_ids},
                        "status": {"type": "string", "enum": VALID_STATUS},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "evidence": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 2,
                            "items": {"type": "string"},
                            "description": "Short verbatim quote(s) from the student only."
                        },
                        "note": {
                            "type": "string",
                            "description": "One concise reason for the label; include misconception detail here only when status is misconception."
                        }
                    },
                    "required": [
                        "kc_id",
                        "status",
                        "confidence",
                        "evidence",
                        "note"
                    ]
                }
            },
            "flags": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "no_relevant_kc_evidence",
                        "too_short",
                        "too_ambiguous",
                        "mostly_code_generation_request",
                        "requires_code_or_output_not_present",
                        "possible_academic_integrity_issue"
                    ]
                },
                "description": "Only include applicable flags; otherwise use an empty array."
            }
        },
        "required": [
            "conversation_id",
            "chunk_id",
            "assignment_id",
            "labels",
            "flags"
        ]
    }


OUTPUT_SCHEMA = make_output_schema(VALID_KC_IDS)


# ============================================================
# 5. PROMPT CONSTRUCTION
# ============================================================

DEVELOPER_INSTRUCTIONS = f"""
Act as an evidence-conservative Knowledge Component annotator. Identify only KCs
directly evidenced by the student's own words, code, outputs, or questions. Return
only the structured JSON required by the schema.

Rules:
- Use KC IDs only from the frozen codebook. Never invent IDs.
- Treat assistant statements, explanations, and generated code as context, not as
  evidence of the student's knowledge. Do not infer understanding merely because
  the assistant supplied a correct answer.
- Label only KCs directly evidenced by the student. Do not label a parent,
  prerequisite, sibling, or related KC unless separate student evidence directly
  supports it.
- If a child KC is directly evidenced, label the parent only when the parent is
  also explicit.
- If a broad request for full code does not reveal a specific KC, omit the label.
- Emit each KC at most once. Use the status best supported by the student's most
  informative evidence in this chunk.
- Use at most 2 short verbatim student evidence quotes per KC.
- Put one short reason in note. Include misconception detail there only when
  status="misconception".
- When speaker attribution is unclear, do not treat uncertain text as student
  evidence; use the too_ambiguous flag when applicable.
- Use labels=[] and flags=["no_relevant_kc_evidence"] when only assistant evidence
  exists or no KC is directly evidenced.

Statuses: {", ".join(VALID_STATUS)}

{FROZEN_CODEBOOK}
""".strip()


def build_user_message(chunk: Dict[str, Any]) -> str:
    return f"""
Label the following conversation chunk.

Metadata:
assignment_id: {TARGET_ASSIGNMENT_ID}
conversation_id: {chunk["conversation_id"]}
student_id: {chunk.get("student_id")}
chunk_id: {chunk["chunk_id"]}

Conversation:
{chunk["conversation_text"]}
""".strip()


def make_response_body(chunk: Dict[str, Any], model: str) -> Dict[str, Any]:
    return {
        "model": model,
        "input": [
            {
                "role": "developer",
                "content": DEVELOPER_INSTRUCTIONS
            },
            {
                "role": "user",
                "content": build_user_message(chunk)
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "kc_teacher_label",
                "strict": True,
                "schema": OUTPUT_SCHEMA
            }
        },
        "reasoning": {
            "effort": REASONING_EFFORT
        },
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "prompt_cache_key": f"kc-codebook-{CODEBOOK_HASH[:16]}",
        "prompt_cache_options": {
            "ttl": "30m"
        },
        "store": False
    }


def make_batch_line(chunk: Dict[str, Any]) -> Dict[str, Any]:
    custom_id = f"{chunk['conversation_id']}::{chunk['chunk_id']}"
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": make_response_body(chunk, BATCH_MODEL)
    }


# ============================================================
# 6. TOKEN ESTIMATION
# ============================================================

def estimate_request_tokens(chunk: Dict[str, Any]) -> int:
    body = make_response_body(chunk, BATCH_MODEL)
    # Approximation. Real API accounting may differ slightly.
    return count_tokens(stable_json(body))


def estimate_all(chunks: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for ch in chunks:
        conv_tokens = count_tokens(ch["conversation_text"])
        total_tokens = estimate_request_tokens(ch)
        rows.append({
            "conversation_id": ch["conversation_id"],
            "student_id": ch.get("student_id"),
            "chunk_id": ch["chunk_id"],
            "conversation_tokens": conv_tokens,
            "estimated_total_input_tokens": total_tokens,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "num_turns": len(ch.get("turns", [])),
        })
    return pd.DataFrame(rows)


estimate_df = estimate_all(chunks)
estimate_path = OUTPUT_DIR / "token_estimate.csv"
estimate_df.to_csv(estimate_path, index=False)

print(f"Token estimate saved to: {estimate_path}")
print(f"Estimated total input tokens: {estimate_df['estimated_total_input_tokens'].sum():,}")
print(f"Estimated median input tokens/request: {int(estimate_df['estimated_total_input_tokens'].median()):,}")
print(f"Maximum possible output tokens: {estimate_df['max_output_tokens'].sum():,}")

if INPUT_USD_PER_1M is not None:
    estimated_input_cost = estimate_df["estimated_total_input_tokens"].sum() / 1_000_000 * INPUT_USD_PER_1M
    print(f"Estimated synchronous input cost before cache/batch: ${estimated_input_cost:,.2f}")

if OUTPUT_USD_PER_1M is not None:
    estimated_max_output_cost = estimate_df["max_output_tokens"].sum() / 1_000_000 * OUTPUT_USD_PER_1M
    print(f"Estimated maximum output cost before batch discounts: ${estimated_max_output_cost:,.2f}")

if INPUT_USD_PER_1M is not None and OUTPUT_USD_PER_1M is not None:
    estimated_max_total_cost = estimated_input_cost + estimated_max_output_cost
    print(f"Estimated maximum total cost before cache/batch: ${estimated_max_total_cost:,.2f}")

if estimate_df["estimated_total_input_tokens"].max() > 120_000:
    print("WARNING: At least one request is very large. Consider reducing MAX_CONVERSATION_TOKENS.")


# ============================================================
# 7. CREATE BATCH REQUEST FILE
# ============================================================

with REQUESTS_JSONL.open("w", encoding="utf-8") as f:
    for ch in chunks:
        f.write(json.dumps(make_batch_line(ch), ensure_ascii=False) + "\n")

print(f"Batch request file written to: {REQUESTS_JSONL}")
print(f"Batch request file size: {REQUESTS_JSONL.stat().st_size / (1024 * 1024):.2f} MB")

if REQUESTS_JSONL.stat().st_size > 190 * 1024 * 1024:
    raise RuntimeError("Batch file is close to or above the 200MB limit. Split into multiple batch files.")


# ============================================================
# 8. OUTPUT EXTRACTION AND VALIDATION
# ============================================================

def extract_response_text(response_body: Dict[str, Any]) -> str:
    if "output_text" in response_body and response_body["output_text"]:
        return response_body["output_text"]

    texts = []
    for item in response_body.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                if "text" in content:
                    texts.append(content["text"])
    return "\n".join(texts).strip()


def validate_label_object(obj: Dict[str, Any]) -> List[str]:
    errors = []

    if obj.get("assignment_id") != TARGET_ASSIGNMENT_ID:
        errors.append(f"Unexpected assignment_id: {obj.get('assignment_id')}")

    labels = obj.get("labels", [])
    if not isinstance(labels, list):
        errors.append("labels is not a list")
        return errors

    valid_ids = set(VALID_KC_IDS)
    valid_status = set(VALID_STATUS)

    for i, label in enumerate(labels):
        kc_id = label.get("kc_id")
        status = label.get("status")

        if kc_id not in valid_ids:
            errors.append(f"Invalid kc_id at label {i}: {kc_id}")
        if status not in valid_status:
            errors.append(f"Invalid status at label {i}: {status}")

        evidence = label.get("evidence", [])
        if not evidence:
            errors.append(f"Missing evidence for {kc_id}")

    return errors


def parse_and_validate_response(
    custom_id: str,
    response_body: Dict[str, Any],
    model: str,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    text = extract_response_text(response_body)
    if not text:
        return None, ["No output text found"]

    try:
        obj = json.loads(text)
    except Exception as e:
        return None, [f"JSON parse error: {e}", text[:1000]]

    errors = validate_label_object(obj)

    obj["_custom_id"] = custom_id
    obj["_teacher_model"] = model
    obj["_codebook_hash"] = CODEBOOK_HASH
    obj["_created_by_pipeline"] = "teacher_label_kcs.py"

    return obj, errors


# ============================================================
# 9. PILOT RUN
# ============================================================

def nested_usage_value(usage: Dict[str, Any], *keys: str) -> int:
    value: Any = usage
    for key in keys:
        if not isinstance(value, dict):
            return 0
        value = value.get(key, 0)
    return int(value or 0)


def run_pilot(chunks: List[Dict[str, Any]], models: List[str]) -> None:
    client = OpenAI()
    if PILOT_OUTPUT_JSONL.exists():
        PILOT_OUTPUT_JSONL.unlink()

    summary_rows = []
    for model in models:
        for ch in chunks:
            base_id = f"{ch['conversation_id']}::{ch['chunk_id']}"
            custom_id = f"{model}::{base_id}"
            body = make_response_body(ch, model)

            print(f"\nRunning pilot: model={model} request={base_id}")
            try:
                response = client.responses.create(**body)
                response_dict = response.model_dump()
                obj, errors = parse_and_validate_response(custom_id, response_dict, model)
                usage = response_dict.get("usage", {})
                api_error = None
                response_id = response_dict.get("id")
            except Exception as exc:
                obj = None
                errors = [f"API request failed: {type(exc).__name__}: {exc}"]
                usage = {}
                api_error = str(exc)
                response_id = None

            row = {
                "model": model,
                "conversation_id": ch["conversation_id"],
                "chunk_id": ch["chunk_id"],
                "custom_id": custom_id,
                "parsed": obj,
                "validation_errors": errors,
                "usage": usage,
                "raw_response_id": response_id,
                "api_error": api_error,
            }
            append_jsonl(PILOT_OUTPUT_JSONL, row)

            summary_rows.append({
                "model": model,
                "conversation_id": ch["conversation_id"],
                "chunk_id": ch["chunk_id"],
                "api_succeeded": api_error is None,
                "validation_error_count": len(errors),
                "label_count": len(obj.get("labels", [])) if obj else 0,
                "input_tokens": nested_usage_value(usage, "input_tokens"),
                "cached_input_tokens": nested_usage_value(
                    usage, "input_tokens_details", "cached_tokens"
                ),
                "output_tokens": nested_usage_value(usage, "output_tokens"),
                "total_tokens": nested_usage_value(usage, "total_tokens"),
            })

            print("Validation errors:", errors)
            print("Usage:", usage)

    pd.DataFrame(summary_rows).to_csv(PILOT_SUMMARY_CSV, index=False)

    print(f"\nPilot outputs saved to: {PILOT_OUTPUT_JSONL}")
    print(f"Pilot comparison summary saved to: {PILOT_SUMMARY_CSV}")


configured_pilot_requests = len(chunks) * len(MODELS_TO_COMPARE)
print(
    f"Configured pilot comparison: {len(chunks)} chunks from "
    f"{len(conversations)} selected chats x "
    f"{len(MODELS_TO_COMPARE)} models = {configured_pilot_requests} API requests"
)

if RUN_PILOT and not DRY_RUN_ONLY:
    run_pilot(chunks, MODELS_TO_COMPARE)
else:
    print("Pilot not run because DRY_RUN_ONLY=True or RUN_PILOT=False.")


# ============================================================
# 10. SUBMIT BATCH
# ============================================================

def submit_batch(requests_path: Path) -> Dict[str, Any]:
    client = OpenAI()

    print("Uploading batch input file...")
    uploaded_file = client.files.create(
        file=requests_path.open("rb"),
        purpose="batch"
    )

    print("Creating batch...")
    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={
            "project": "kc_teacher_labeling",
            "assignment": CURRENT_ASSIGNMENT,
            "codebook_hash": CODEBOOK_HASH[:32],
            "model": BATCH_MODEL,
        }
    )

    info = {
        "input_file_id": uploaded_file.id,
        "batch_id": batch.id,
        "status": batch.status,
        "codebook_hash": CODEBOOK_HASH,
        "model": BATCH_MODEL,
        "requests_file": str(requests_path),
    }

    BATCH_INFO_JSON.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Batch submitted. Info saved to: {BATCH_INFO_JSON}")
    print(json.dumps(info, indent=2))

    return info


if SUBMIT_BATCH and not DRY_RUN_ONLY:
    submit_batch(REQUESTS_JSONL)
else:
    print("Batch not submitted. Set DRY_RUN_ONLY=False and SUBMIT_BATCH=True after inspecting pilot/request file.")


# ============================================================
# 11. DOWNLOAD AND PARSE BATCH RESULTS
# ============================================================

def download_file_content(client: OpenAI, file_id: str, out_path: Path) -> None:
    content = client.files.content(file_id)
    try:
        content.write_to_file(str(out_path))
    except AttributeError:
        data = getattr(content, "content", None)
        if data is None:
            data = bytes(content)
        out_path.write_bytes(data)


def poll_batch(batch_id: str, sleep_seconds: int = 60) -> Dict[str, Any]:
    client = OpenAI()

    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"Batch {batch_id} status: {batch.status}")

        if batch.status in {"completed", "failed", "expired", "cancelled"}:
            return batch.model_dump()

        time.sleep(sleep_seconds)


def parse_batch_output(batch_output_jsonl: Path) -> None:
    if FINAL_LABELS_JSONL.exists():
        FINAL_LABELS_JSONL.unlink()
    if ERRORS_JSONL.exists():
        ERRORS_JSONL.unlink()

    valid_count = 0
    error_count = 0

    for line in batch_output_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        row = json.loads(line)
        custom_id = row.get("custom_id", "")

        if row.get("error"):
            append_jsonl(ERRORS_JSONL, {
                "custom_id": custom_id,
                "error": row.get("error"),
                "raw": row,
            })
            error_count += 1
            continue

        response_body = row.get("response", {}).get("body", {})
        obj, errors = parse_and_validate_response(custom_id, response_body, BATCH_MODEL)

        if errors:
            append_jsonl(ERRORS_JSONL, {
                "custom_id": custom_id,
                "validation_errors": errors,
                "parsed": obj,
                "raw_response_body": response_body,
            })
            error_count += 1
        else:
            append_jsonl(FINAL_LABELS_JSONL, obj)
            valid_count += 1

    print(f"Valid parsed labels: {valid_count}")
    print(f"Errors: {error_count}")
    print(f"Final labels saved to: {FINAL_LABELS_JSONL}")
    print(f"Errors saved to: {ERRORS_JSONL}")


def download_and_parse_completed_batch(batch_id: str) -> None:
    client = OpenAI()
    batch = client.batches.retrieve(batch_id)
    batch_dict = batch.model_dump()

    if batch.status != "completed":
        raise RuntimeError(f"Batch is not completed. Current status: {batch.status}")

    output_file_id = batch_dict.get("output_file_id")
    error_file_id = batch_dict.get("error_file_id")

    if output_file_id:
        out_path = OUTPUT_DIR / f"{batch_id}_output.jsonl"
        download_file_content(client, output_file_id, out_path)
        print(f"Downloaded output file to: {out_path}")
        parse_batch_output(out_path)

    if error_file_id:
        err_path = OUTPUT_DIR / f"{batch_id}_api_errors.jsonl"
        download_file_content(client, error_file_id, err_path)
        print(f"Downloaded API error file to: {err_path}")


# To use after submission:
# info = json.loads(BATCH_INFO_JSON.read_text())
# final_status = poll_batch(info["batch_id"], sleep_seconds=60)
# download_and_parse_completed_batch(info["batch_id"])
