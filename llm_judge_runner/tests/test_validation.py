import unittest

from src.validation import parse_output


class ParseOutputTest(unittest.TestCase):
    def test_strict_mode_rejects_markdown_fence(self) -> None:
        parsed, method, error = parse_output("```json\n{\"tasks\": []}\n```", "strict")

        self.assertIsNone(parsed)
        self.assertIsNone(method)
        self.assertTrue(error.startswith("json_parse_error:"))

    def test_tolerant_mode_accepts_markdown_fence(self) -> None:
        parsed, method, error = parse_output("```json\n{\"tasks\": []}\n```", "tolerant")

        self.assertEqual(parsed, {"tasks": []})
        self.assertEqual(method, "json_code_fence")
        self.assertIsNone(error)

    def test_tolerant_mode_extracts_nested_json_from_prose(self) -> None:
        parsed, method, error = parse_output(
            "Here is the result:\n{\"tasks\": [{\"task_id\": \"T1\"}], \"flags\": []}\nDone.",
            "tolerant",
        )

        self.assertEqual(parsed, {"tasks": [{"task_id": "T1"}], "flags": []})
        self.assertEqual(method, "extracted_json_object")
        self.assertIsNone(error)

    def test_tolerant_mode_does_not_invent_data_for_invalid_json(self) -> None:
        parsed, method, error = parse_output("```json\n{\"tasks\": [}\n```", "tolerant")

        self.assertIsNone(parsed)
        self.assertIsNone(method)
        self.assertEqual(error, "no_parseable_json_object")

    def test_tolerant_mode_accepts_literal_control_chars_in_strings(self) -> None:
        parsed, method, error = parse_output(
            '{"tasks": [], "labels": [], "flags": [], "evidence": "bp\t-\tblood pressure"}',
            "tolerant",
        )

        self.assertEqual(parsed["evidence"], "bp\t-\tblood pressure")
        self.assertEqual(method, "direct_json_nonstandard_control_chars")
        self.assertIsNone(error)


if __name__ == "__main__":
    unittest.main()