"""Group nested StudyChat conversations into one JSON array per semester/topic.

Default output layout:

    categorized_student_chats/
        f24/
            a1/
                f24_a1.json
            a2/
                f24_a2.json
        s25/
            a1/
                s25_a1.json

Each output file is an array of normalized conversation objects. Chat metadata
is kept once at the top level, while each item in ``conversation_rows`` keeps
only interaction-specific fields.
The semester is inferred from the ``semester`` field in ``conversation_rows``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(__file__).with_name("student_chat_conversations_nested.json")
DEFAULT_OUTPUT = Path(__file__).with_name("categorized_student_chats")
ALLOWED_SEMESTERS = {"f24", "s25"}
CHAT_FIELDS = (
    "userId",
    "chatId",
    "topic",
    "chatTitle",
    "chatStartTime",
    "chatTotalInteractionCount",
    "num_interactions",
)
INTERACTION_FIELDS = (
    "interactionCount",
    "timestamp",
    "prompt",
    "response",
    "llm_label",
)


def safe_path_part(value: Any, fallback: str) -> str:
    """Return a filesystem-safe directory or filename component."""
    text = str(value).strip()
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text)
    text = text.rstrip(" .")
    return text or fallback


def load_conversations(input_path: Path) -> list[dict[str, Any]]:
    """Load and validate the top-level JSON array."""
    with input_path.open("r", encoding="utf-8-sig") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("Expected the input JSON's top level to be an array.")

    invalid_positions = [i for i, item in enumerate(payload) if not isinstance(item, dict)]
    if invalid_positions:
        raise ValueError(
            "Every array item must be a JSON object. Invalid positions: "
            f"{invalid_positions[:10]}"
        )
    return payload


def conversation_semester(conversation: dict[str, Any], position: int) -> str:
    """Infer one semester from a conversation's rows and validate consistency."""
    rows = conversation.get("conversation_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Conversation {position} has no conversation_rows.")

    semesters = {
        str(row.get("semester", "")).strip().lower()
        for row in rows
        if isinstance(row, dict) and str(row.get("semester", "")).strip()
    }

    if len(semesters) != 1:
        chat_id = conversation.get("chatId", "<missing chatId>")
        raise ValueError(
            f"Conversation {position} ({chat_id}) must have exactly one semester; "
            f"found {sorted(semesters) or 'none'}."
        )

    semester = semesters.pop()
    if semester not in ALLOWED_SEMESTERS:
        raise ValueError(
            f"Conversation {position} has unsupported semester {semester!r}; "
            f"expected one of {sorted(ALLOWED_SEMESTERS)}."
        )
    return semester


def normalize_conversation(
    conversation: dict[str, Any], position: int
) -> dict[str, Any]:
    """Remove repeated chat metadata and message history from interaction rows."""
    rows = conversation.get("conversation_rows")
    if not isinstance(rows, list):
        raise ValueError(f"Conversation {position} has invalid conversation_rows.")

    normalized_rows = []
    for row_position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(
                f"Conversation {position}, row {row_position} is not a JSON object."
            )
        normalized_rows.append({field: row.get(field) for field in INTERACTION_FIELDS})

    normalized = {field: conversation.get(field) for field in CHAT_FIELDS}
    normalized["conversation_rows"] = normalized_rows
    return normalized


def split_conversations(
    input_path: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Counter[tuple[str, str]]:
    """Write one JSON array for each semester/topic combination."""
    conversations = load_conversations(input_path)
    counts: Counter[tuple[str, str]] = Counter()
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for position, conversation in enumerate(conversations):
        semester = conversation_semester(conversation, position)
        topic = safe_path_part(conversation.get("topic", ""), "unknown_topic").lower()
        grouped[(semester, topic)].append(normalize_conversation(conversation, position))
        counts[(semester, topic)] += 1

    if dry_run:
        return counts

    for (semester, topic), topic_conversations in sorted(grouped.items()):
        topic_dir = output_dir / semester / topic
        output_path = topic_dir / f"{semester}_{topic}.json"

        if topic_dir.exists():
            legacy_files = [
                path for path in topic_dir.glob("*.json") if path.name != output_path.name
            ]
            if legacy_files and not overwrite:
                raise FileExistsError(
                    f"{topic_dir} contains {len(legacy_files)} legacy per-chat JSON files. "
                    "Use --overwrite to remove them and write the grouped file."
                )
            if output_path.exists() and not overwrite:
                raise FileExistsError(
                    f"Output already exists: {output_path}. Use --overwrite to replace it."
                )
            if overwrite:
                for legacy_file in legacy_files:
                    legacy_file.unlink()

        topic_dir.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(topic_conversations, file, ensure_ascii=False, indent=2)
            file.write("\n")

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split StudyChat conversations by semester and topic."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"input JSON array (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"root output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace grouped files and remove legacy per-chat JSON files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and report counts without writing files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = split_conversations(
        args.input,
        args.output_dir,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    action = "Would write" if args.dry_run else "Wrote"
    total = sum(counts.values())
    print(f"{action} {total} conversations in {len(counts)} grouped files:")
    for (semester, topic), count in sorted(counts.items()):
        print(f"  {semester}/{topic}: {count}")
    if not args.dry_run:
        print(f"Output directory: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
