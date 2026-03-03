from __future__ import annotations

import abc
import os
import asyncio
from typing import Dict, List, Optional

ChatMessages = List[Dict[str, str]]


NUM_CONCURRENT_QUERIES = 32

sem = asyncio.Semaphore(NUM_CONCURRENT_QUERIES) # ！！过大的并行度会导致vllm engine的KV Cache 被discard，降低速度，尤其是在超长上下文的agent中。请根据vllm在超长序列(32k-128k)下的可用并行度设定

class LLMClient(abc.ABC):
    """Minimal chat-completion interface required by the WebWalker agent."""

    @abc.abstractmethod
    async def complete(self, messages: ChatMessages, **kwargs) -> str:
        """Return the assistant reply for the provided conversation history."""

    def name(self) -> str:
        """Human-readable identifier used for logging/debugging."""
        return self.__class__.__name__  # pragma: no cover


class OpenAIChatClient(LLMClient):
    """Wrapper around the `openai` Python SDK."""

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_kwargs: Optional[Dict[str, object]] = None,
    ) -> None:
        from openai import AsyncOpenAI, RateLimitError
        import backoff

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("An OpenAI API key must be provided.")
        client_kwargs: Dict[str, object] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = AsyncOpenAI(**client_kwargs)
        self.model = model
        self.query = backoff.on_exception(backoff.expo, RateLimitError)(self.client.chat.completions.create)
        
        self.default_kwargs = default_kwargs or {}

    async def complete(self, messages: ChatMessages, **kwargs) -> str:
        payload = {**self.default_kwargs, **kwargs}
        async with sem:
            response = await self.query(
                model=self.model,
                messages=messages,
                **payload,
            )
        content = response.choices[0].message.content
        assert content, f"Model returned empty content {content} from messages {messages}."
        return content

    def name(self) -> str:
        return self.model


def build_llm_from_env() -> LLMClient:
    """
    Convenience factory that inspects environment variables.

    An OpenAI-compatible endpoint must be provided via env vars; otherwise an error is raised.
    """

    model = (
        os.environ.get("OPENAI_MODEL")
        or os.environ.get("OPENAI_MODEL_NAME")
        or os.environ.get("OPENAI_CHAT_MODEL")
    )
    if not model:
        raise ValueError(
            "No OpenAI-compatible model configured. "
            "Set OPENAI_MODEL or OPENAI_MODEL_NAME (optionally OPENAI_MODEL_SERVER)."
        )

    base_url = os.environ.get("OPENAI_MODEL_SERVER") or os.environ.get("OPENAI_BASE_URL")
    default_kwargs: Dict[str, object] = {}
    temperature = os.environ.get("OPENAI_TEMPERATURE")
    if temperature:
        default_kwargs["temperature"] = float(temperature)
    max_tokens = os.environ.get("OPENAI_MAX_OUTPUT_TOKENS")
    if max_tokens:
        default_kwargs["max_tokens"] = int(max_tokens)
    return OpenAIChatClient(model=model, base_url=base_url, default_kwargs=default_kwargs)
