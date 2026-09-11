from __future__ import annotations

import unittest

from prepare_chunked_conversations import build_chunked_cases, validate_reconstruction


def make_case(case_id: str, exchange_sizes: list[int]) -> dict[str, object]:
    turns: list[dict[str, str]] = []
    for index, size in enumerate(exchange_sizes):
        student_size = size // 2
        turns.extend(
            [
                {
                    "turn_index": f"{index}.student",
                    "speaker": "student",
                    "text": "s" * student_size,
                },
                {
                    "turn_index": f"{index}.assistant",
                    "speaker": "assistant",
                    "text": "a" * (size - student_size),
                },
            ]
        )
    conversation_id = case_id.split("::", 2)[1]
    return {
        "case_id": case_id,
        "base_conversation_id": conversation_id,
        "conversation_id": conversation_id,
        "chunk_id": conversation_id,
        "assignment_id": "f24_a2",
        "student_id": "student",
        "test_id": "T0",
        "variant_id": "base",
        "variant_type": "baseline",
        "is_baseline": True,
        "prompt_variant_id": "canonical",
        "transformation_metadata": {},
        "turns": turns,
    }


class PrepareChunkedConversationsTest(unittest.TestCase):
    def test_split_preserves_complete_exchanges_and_reconstructs_exactly(self) -> None:
        source = make_case("f24_a2::conversation::T0::base", [7, 7, 4])
        chunks, summary = build_chunked_cases([source], max_characters=10)
        validate_reconstruction([source], chunks, max_characters=10)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(summary["split_conversations"], 1)
        self.assertEqual([chunk["chunk_index"] for chunk in chunks], [1, 2, 3])
        self.assertTrue(all(len(chunk["turns"]) == 2 for chunk in chunks))
        self.assertEqual(chunks[0]["chunk_id"], "conversation::chunk_001")
        self.assertEqual(
            chunks[0]["case_id"], "f24_a2::conversation::chunk_001::T0::base"
        )

    def test_singleton_preserves_existing_identifiers(self) -> None:
        source = make_case("f24_a2::conversation::T0::base", [4, 4])
        chunks, summary = build_chunked_cases([source], max_characters=10)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(summary["singleton_conversations"], 1)
        self.assertEqual(chunks[0]["case_id"], source["case_id"])
        self.assertEqual(chunks[0]["chunk_id"], source["chunk_id"])
        self.assertEqual(chunks[0]["chunk_index"], 1)
        self.assertEqual(chunks[0]["chunk_count"], 1)

    def test_oversized_exchange_is_rejected_instead_of_split(self) -> None:
        source = make_case("f24_a2::conversation::T0::base", [11])
        with self.assertRaisesRegex(ValueError, "will not split a student-assistant exchange"):
            build_chunked_cases([source], max_characters=10)


if __name__ == "__main__":
    unittest.main()
