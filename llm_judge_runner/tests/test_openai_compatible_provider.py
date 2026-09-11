from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.config import ModelConfig
from src.providers.openai_compatible import OpenAICompatibleChatProvider


def _model(*, stream: bool, sdk_max_retries: int | None) -> ModelConfig:
    return ModelConfig(
        key="deepseek_v4_pro_0813",
        display_name="DeepSeek-V4 Pro-0813",
        provider="DeepSeek",
        adapter="openai_compatible_chat",
        model_id="deepseek-v4-pro",
        country="China",
        enabled=True,
        api_key_env="DEEPSEEK_API_KEY",
        base_url_env="DEEPSEEK_BASE_URL",
        reasoning_effort="high",
        stream=stream,
        sdk_max_retries=sdk_max_retries,
        extra={"thinking": {"type": "enabled"}},
    )


def _qwen_model() -> ModelConfig:
    return ModelConfig(
        key="qwen_3_8_max_0902",
        display_name="Qwen3.8-Max-0902",
        provider="Alibaba",
        adapter="openai_compatible_chat",
        model_id="qwen3.8-max-0902",
        requested_version_label="Qwen3.8-Max-0902",
        country="China",
        enabled=True,
        api_key_env="DASHSCOPE_API_KEY",
        base_url_env="QWEN_BASE_URL",
        reasoning_effort=None,
        temperature=None,
        stream=True,
        sdk_max_retries=0,
        extra={"enable_thinking": True},
    )


def _choice(
    *, content: str | None = None, reasoning: str | None = None,
    finish_reason: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        delta=SimpleNamespace(content=content, reasoning_content=reasoning),
        finish_reason=finish_reason,
    )


class OpenAICompatibleChatProviderTest(unittest.TestCase):
    def test_deepseek_stream_reconstructs_content_and_captures_final_usage(self) -> None:
        usage = SimpleNamespace(
            model_dump=lambda: {
                "prompt_tokens": 110,
                "prompt_cache_hit_tokens": 90,
                "prompt_cache_miss_tokens": 20,
                "completion_tokens": 15,
                "completion_tokens_details": {"reasoning_tokens": 8},
                "total_tokens": 125,
            }
        )
        stream = iter(
            [
                SimpleNamespace(
                    id="chat_test",
                    model="deepseek-v4-pro",
                    choices=[_choice(reasoning="internal reasoning")],
                    usage=None,
                ),
                SimpleNamespace(
                    id="chat_test",
                    model="deepseek-v4-pro",
                    choices=[_choice(content='{"tasks":[]')],
                    usage=None,
                ),
                SimpleNamespace(
                    id="chat_test",
                    model="deepseek-v4-pro",
                    choices=[_choice(content=',"labels":[],"flags":[]}')],
                    usage=None,
                ),
                SimpleNamespace(
                    id="chat_test",
                    model="deepseek-v4-pro",
                    choices=[_choice(finish_reason="stop")],
                    usage=usage,
                ),
            ]
        )
        client = MagicMock()
        client.chat.completions.create.return_value = stream

        with patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "test-key",
                "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
            },
        ), patch(
            "src.providers.openai_compatible.OpenAI", return_value=client
        ) as openai:
            result = OpenAICompatibleChatProvider().generate(
                _model(stream=True, sdk_max_retries=0),
                "stable system prompt",
                "dynamic conversation",
                48000,
                700,
            )

        openai.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.deepseek.com",
            timeout=700,
            max_retries=0,
        )
        request = client.chat.completions.create.call_args.kwargs
        self.assertTrue(request["stream"])
        self.assertEqual(request["stream_options"], {"include_usage": True})
        self.assertEqual(request["max_tokens"], 48000)
        self.assertEqual(request["reasoning_effort"], "high")
        self.assertEqual(
            request["extra_body"], {"thinking": {"type": "enabled"}}
        )
        self.assertEqual(
            result.raw_text, '{"tasks":[],"labels":[],"flags":[]}'
        )
        self.assertNotIn("internal reasoning", result.raw_text)
        self.assertEqual(result.provider_response_id, "chat_test")
        self.assertEqual(result.provider_model, "deepseek-v4-pro")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.usage["prompt_cache_hit_tokens"], 90)
        self.assertEqual(
            result.provider_metadata,
            {
                "api_family": "chat_completions",
                "streaming": True,
                "sdk_max_retries": 0,
            },
        )

    def test_non_streaming_path_remains_available(self) -> None:
        response = SimpleNamespace(
            id="chat_non_stream",
            model="deepseek-v4-pro",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"tasks":[]}'),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                model_dump=lambda: {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                }
            ),
        )
        client = MagicMock()
        client.chat.completions.create.return_value = response

        with patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "test-key",
                "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
            },
        ), patch(
            "src.providers.openai_compatible.OpenAI", return_value=client
        ) as openai:
            result = OpenAICompatibleChatProvider().generate(
                _model(stream=False, sdk_max_retries=None),
                "system",
                "user",
                8000,
                300,
            )

        openai.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.deepseek.com",
            timeout=300,
        )
        request = client.chat.completions.create.call_args.kwargs
        self.assertFalse(request["stream"])
        self.assertNotIn("stream_options", request)
        self.assertEqual(result.raw_text, '{"tasks":[]}')
        self.assertEqual(
            result.provider_metadata,
            {"api_family": "chat_completions", "streaming": False},
        )

    def test_qwen_streams_and_captures_implicit_cache_usage(self) -> None:
        usage = SimpleNamespace(
            model_dump=lambda: {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "prompt_tokens_details": {"cached_tokens": 80},
            }
        )
        stream = iter(
            [
                SimpleNamespace(
                    id="chat_qwen",
                    model="qwen3.8-max-0902",
                    choices=[_choice(reasoning="internal reasoning")],
                    usage=None,
                ),
                SimpleNamespace(
                    id="chat_qwen",
                    model="qwen3.8-max-0902",
                    choices=[_choice(content='{"tasks":[],"labels":[],"flags":[]}')],
                    usage=None,
                ),
                # Qwen's final usage-only chunk may contain no choices.
                SimpleNamespace(
                    id="chat_qwen",
                    model="qwen3.8-max-0902",
                    choices=[],
                    usage=usage,
                ),
            ]
        )
        client = MagicMock()
        client.chat.completions.create.return_value = stream

        with patch.dict(
            os.environ,
            {
                "DASHSCOPE_API_KEY": "test-key",
                "QWEN_BASE_URL": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            },
        ), patch(
            "src.providers.openai_compatible.OpenAI", return_value=client
        ) as openai:
            result = OpenAICompatibleChatProvider().generate(
                _qwen_model(),
                "stable system prompt",
                "dynamic conversation",
                131072,
                700,
            )

        openai.assert_called_once_with(
            api_key="test-key",
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            timeout=700,
            max_retries=0,
        )
        request = client.chat.completions.create.call_args.kwargs
        self.assertTrue(request["stream"])
        self.assertEqual(request["stream_options"], {"include_usage": True})
        self.assertEqual(request["max_completion_tokens"], 131072)
        self.assertNotIn("max_tokens", request)
        self.assertNotIn("reasoning_effort", request)
        self.assertEqual(request["extra_body"], {"enable_thinking": True})
        self.assertEqual(result.raw_text, '{"tasks":[],"labels":[],"flags":[]}')
        self.assertEqual(result.usage["prompt_tokens_details"]["cached_tokens"], 80)


if __name__ == "__main__":
    unittest.main()
