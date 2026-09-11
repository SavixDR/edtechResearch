from __future__ import annotations

import math
import os
from typing import Any


def offline_text_tokens(text: str, chars_per_token: float = 4.0) -> int:
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be greater than zero")
    return math.ceil(len(text) / chars_per_token)


def offline_request_tokens(system_prompt: str, user_prompt: str, chars_per_token: float = 4.0) -> int:
    # Eight tokens is a deliberately small allowance for the two message wrappers.
    return offline_text_tokens(system_prompt + user_prompt, chars_per_token) + 8


def openai_request_tokens(
    *,
    reference_model: str,
    system_prompt: str,
    user_prompt: str,
    api_key_env: str = "OPENAI_API_KEY",
) -> int:
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The OpenAI SDK is required for provider-count mode") from exc

    client = OpenAI(api_key=api_key)
    try:
        response: Any = client.responses.input_tokens.count(
            model=reference_model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except AttributeError as exc:
        raise RuntimeError(
            "The installed OpenAI SDK does not expose responses.input_tokens.count; upgrade the SDK"
        ) from exc
    return int(response.input_tokens)
