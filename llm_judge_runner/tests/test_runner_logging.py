import unittest

from src.runner import format_completion_outcome


class CompletionOutcomeTest(unittest.TestCase):
    def test_api_error_includes_provider_reason(self) -> None:
        message = format_completion_outcome(
            status="api_error",
            api_error="APITimeoutError: Request timed out.",
            parse_error=None,
            validation_errors=[],
            finish_reason=None,
            usage={},
            max_output_tokens=8000,
        )

        self.assertIn("API failure", message)
        self.assertIn("timed out", message)

    def test_parse_error_includes_parse_reason(self) -> None:
        message = format_completion_outcome(
            status="parse_error",
            api_error=None,
            parse_error="empty_output",
            validation_errors=[],
            finish_reason="length",
            usage={"completion_tokens": 8000},
            max_output_tokens=8000,
        )

        self.assertIn("output parsing failed: empty_output", message)

    def test_length_finish_reason_explains_output_limit(self) -> None:
        message = format_completion_outcome(
            status="ok",
            api_error=None,
            parse_error=None,
            validation_errors=[],
            finish_reason="length",
            usage={"completion_tokens": 8000},
            max_output_tokens=8000,
        )

        self.assertIn("output limit/truncation", message)
        self.assertIn("output_tokens=8000", message)

    def test_schema_error_includes_first_validation_error(self) -> None:
        message = format_completion_outcome(
            status="schema_error",
            api_error=None,
            parse_error=None,
            validation_errors=["tasks/0/task_id: unknown frozen task ID 'T9'"],
            finish_reason="stop",
            usage={},
            max_output_tokens=8000,
        )

        self.assertIn("validation failed (1 error(s))", message)
        self.assertIn("unknown frozen task ID", message)


if __name__ == "__main__":
    unittest.main()