from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from ..config import ModelConfig
from .base import ProviderResult


def _dump_usage(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return dict(usage) if isinstance(usage, dict) else {"repr": str(usage)}


class OpenAICompatibleChatProvider:
    def generate(self, model: ModelConfig, system_prompt: str, user_prompt: str,
                 max_output_tokens: int, timeout_seconds: int) -> ProviderResult:
        api_key = os.getenv(model.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing environment variable {model.api_key_env}")
        if not model.base_url_env:
            raise RuntimeError(f"Model {model.key} has no base_url_env configured")
        base_url = os.getenv(model.base_url_env)
        if not base_url:
            raise RuntimeError(f"Missing environment variable {model.base_url_env}")

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": timeout_seconds,
        }
        if model.sdk_max_retries is not None:
            client_kwargs["max_retries"] = model.sdk_max_retries
        client = OpenAI(**client_kwargs)
        kwargs: dict[str, Any] = {
            "model": model.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": model.stream,
        }
        if model.stream:
            # Compatible providers report complete request usage on the last SSE
            # chunk when stream_options.include_usage is enabled.
            kwargs["stream_options"] = {"include_usage": True}
        if model.provider.lower() in {"moonshot ai", "alibaba"}:
            kwargs["max_completion_tokens"] = max_output_tokens
        else:
            kwargs["max_tokens"] = max_output_tokens
        if model.temperature is not None:
            kwargs["temperature"] = model.temperature

        extra_body = dict(model.extra or {})

        if model.reasoning_effort:
            if model.provider.lower() == "alibaba":
                extra_body["reasoning_effort"] = model.reasoning_effort
            else:
                kwargs["reasoning_effort"] = model.reasoning_effort

        if extra_body:
            kwargs["extra_body"] = extra_body

        response = client.chat.completions.create(**kwargs)
        metadata: dict[str, Any] = {
            "api_family": "chat_completions",
            "streaming": model.stream,
        }
        if model.sdk_max_retries is not None:
            metadata["sdk_max_retries"] = model.sdk_max_retries

        if model.stream:
            content_parts: list[str] = []
            usage: dict[str, Any] = {}
            response_id: str | None = None
            provider_model: str | None = None
            finish_reason = ""

            for chunk in response:
                response_id = getattr(chunk, "id", None) or response_id
                provider_model = getattr(chunk, "model", None) or provider_model
                chunk_usage = _dump_usage(getattr(chunk, "usage", None))
                if chunk_usage:
                    usage = chunk_usage

                for choice in getattr(chunk, "choices", None) or []:
                    delta = getattr(choice, "delta", None)
                    content = getattr(delta, "content", None) if delta is not None else None
                    if content:
                        content_parts.append(content)
                    # Thinking-mode reasoning arrives separately as
                    # delta.reasoning_content. Consume the stream fully, but do
                    # not mix private reasoning into the judge's JSON answer.
                    choice_finish_reason = getattr(choice, "finish_reason", None)
                    if choice_finish_reason:
                        finish_reason = str(choice_finish_reason)

            return ProviderResult(
                raw_text="".join(content_parts),
                provider_response_id=response_id,
                provider_model=provider_model,
                usage=usage,
                finish_reason=finish_reason,
                provider_metadata=metadata,
            )

        choice = response.choices[0]
        message = choice.message
        return ProviderResult(
            raw_text=message.content or "",
            provider_response_id=getattr(response, "id", None),
            provider_model=getattr(response, "model", None),
            usage=_dump_usage(getattr(response, "usage", None)),
            finish_reason=str(getattr(choice, "finish_reason", "") or ""),
            provider_metadata=metadata,
        )
