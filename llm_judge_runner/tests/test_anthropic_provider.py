from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.config import ModelConfig
from src.providers.anthropic_provider import AnthropicMessagesProvider


def _model(extra: dict | None = None) -> ModelConfig:
    return ModelConfig(
        key="claude_opus_5",
        display_name="Claude Opus 5",
        provider="anthropic",
        adapter="anthropic_messages",
        model_id="claude-opus-5",
        country="US",
        enabled=True,
        api_key_env="ANTHROPIC_API_KEY",
        reasoning_effort="high",
        extra=extra or {},
    )


def _response() -> SimpleNamespace:
    return SimpleNamespace(
        id="msg_test",
        model="claude-opus-5",
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text='{"tasks":[],"labels":[],"flags":[]}')],
        usage=SimpleNamespace(
            model_dump=lambda: {
                "input_tokens": 20,
                "cache_creation_input_tokens": 1000,
                "cache_read_input_tokens": 0,
                "output_tokens": 10,
            }
        ),
    )


def _stream(response: SimpleNamespace | None = None) -> MagicMock:
    stream = MagicMock()
    stream.__enter__.return_value = stream
    stream.__iter__.return_value = iter([SimpleNamespace(type="message_start")])
    stream.get_final_message.return_value = response or _response()
    return stream


class AnthropicMessagesProviderTest(unittest.TestCase):
    def test_places_five_minute_cache_breakpoint_on_system_prompt(self) -> None:
        client = MagicMock()
        client.messages.stream.return_value = _stream()

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}), patch(
            "src.providers.anthropic_provider.Anthropic", return_value=client
        ) as anthropic:
            result = AnthropicMessagesProvider().generate(
                _model({"cache_control": {"type": "ephemeral", "ttl": "5m"}}),
                "stable system prompt",
                "dynamic conversation",
                8000,
                300,
            )

        anthropic.assert_called_once_with(
            api_key="test-key",
            timeout=300,
            max_retries=0,
        )
        request = client.messages.stream.call_args.kwargs
        self.assertEqual(
            request["system"],
            [
                {
                    "type": "text",
                    "text": "stable system prompt",
                    "cache_control": {"type": "ephemeral", "ttl": "5m"},
                }
            ],
        )
        self.assertEqual(
            request["messages"],
            [{"role": "user", "content": "dynamic conversation"}],
        )
        self.assertNotIn("cache_control", request)
        client.messages.create.assert_not_called()
        self.assertEqual(result.provider_response_id, "msg_test")
        self.assertEqual(result.finish_reason, "end_turn")
        self.assertEqual(result.usage["cache_creation_input_tokens"], 1000)
        self.assertEqual(
            result.provider_metadata,
            {
                "api_family": "messages",
                "streaming": True,
                "sdk_max_retries": 0,
            },
        )

    def test_uses_plain_system_prompt_when_cache_is_not_configured(self) -> None:
        client = MagicMock()
        client.messages.stream.return_value = _stream()

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}), patch(
            "src.providers.anthropic_provider.Anthropic", return_value=client
        ):
            AnthropicMessagesProvider().generate(
                _model(),
                "stable system prompt",
                "dynamic conversation",
                8000,
                300,
            )

        self.assertEqual(
            client.messages.stream.call_args.kwargs["system"],
            "stable system prompt",
        )

    def test_rejects_invalid_cache_ttl_before_calling_api(self) -> None:
        client = MagicMock()

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}), patch(
            "src.providers.anthropic_provider.Anthropic", return_value=client
        ):
            with self.assertRaisesRegex(ValueError, "must be '5m' or '1h'"):
                AnthropicMessagesProvider().generate(
                    _model({"cache_control": {"type": "ephemeral", "ttl": "2h"}}),
                    "stable system prompt",
                    "dynamic conversation",
                    8000,
                    300,
                )

        client.messages.stream.assert_not_called()


if __name__ == "__main__":
    unittest.main()
