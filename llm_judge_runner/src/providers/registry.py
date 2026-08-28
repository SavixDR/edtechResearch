from __future__ import annotations

def get_provider(adapter: str):
    if adapter == "openai_responses":
        from .openai_provider import OpenAIResponsesProvider
        return OpenAIResponsesProvider()
    if adapter == "anthropic_messages":
        from .anthropic_provider import AnthropicMessagesProvider
        return AnthropicMessagesProvider()
    if adapter == "gemini_generate_content":
        from .gemini_provider import GeminiGenerateContentProvider
        return GeminiGenerateContentProvider()
    if adapter == "openai_compatible_chat":
        from .openai_compatible import OpenAICompatibleChatProvider
        return OpenAICompatibleChatProvider()
    raise ValueError(f"Unknown adapter: {adapter}")
