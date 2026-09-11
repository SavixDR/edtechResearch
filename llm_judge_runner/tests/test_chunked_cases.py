from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.cases import load_cases
from src.result_writer import FLAT_FIELDS
from src.runner import execution_key


def chunk(case_id: str, chunk_id: str, chunk_index: int) -> dict[str, object]:
    return {
        "case_id": case_id,
        "base_conversation_id": "chat_123",
        "conversation_id": "chat_123",
        "chunk_id": chunk_id,
        "chunk_index": chunk_index,
        "chunk_count": 2,
        "assignment_id": "f24_a2",
        "test_id": "T0",
        "variant_id": "base",
        "variant_type": "baseline",
        "is_baseline": True,
        "prompt_variant_id": "canonical",
        "turns": [{"speaker": "student", "text": f"Content for {chunk_id}"}],
    }


class ChunkedCasesTest(unittest.TestCase):
    def test_chunks_remain_linked_but_have_distinct_execution_keys(self) -> None:
        rows = [
            chunk("f24_a2::chat_123::chunk_001::T0::base", "chat_123::chunk_001", 1),
            chunk("f24_a2::chat_123::chunk_002::T0::base", "chat_123::chunk_002", 2),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cases.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            loaded = load_cases(path)

        self.assertEqual({case["conversation_id"] for case in loaded}, {"chat_123"})
        self.assertEqual([case["chunk_index"] for case in loaded], [1, 2])
        keys = {execution_key("gpt_5_6_sol", case["case_id"], 1) for case in loaded}
        self.assertEqual(len(keys), 2)
        self.assertIn("chunk_index", FLAT_FIELDS)
        self.assertIn("chunk_count", FLAT_FIELDS)

    def test_incomplete_chunk_position_is_rejected(self) -> None:
        row = chunk("case_1", "chat_123::chunk_001", 1)
        del row["chunk_count"]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cases.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "both chunk_index and chunk_count"):
                load_cases(path)


if __name__ == "__main__":
    unittest.main()
