"""Warpped LLM client

Controllable LLM call concorrency
Parse from and to `ChatMessage`

TODO:
- [x] support openai compatible engines
- [ ] support non server LLM engines
"""
from __future__ import annotations

import abc
import asyncio
from contextlib import nullcontext
from typing import Any, Dict, Iterable, Optional, Sequence, TYPE_CHECKING

from webagent.llm.parser import get_parser_cls
from webagent.llm.parser import ParsingError

if TYPE_CHECKING:
    from webagent.llm.chat_types import ChatMessage

class Client:
    @abc.abstractmethod
    async def complete(self, messages: Iterable[ChatMessage], **kwargs) -> ChatMessage:
        pass

class OpenAIClient(Client):
    """Wrapper around the `openai` Python SDK."""
    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_kwargs: Dict[str, object] | None = None,
        retry = False,
        retry_errors: Sequence[type[Exception]] | None = None,
        concurrency_limit: int | None = None
    ) -> None:
        """Initialize an OpenAI chat-completions client wrapper.

        Args:
            model: Model name passed to `chat.completions.create`.
            api_key: OpenAI-compatible API key. Uses environment defaults when `None`.
            base_url: Optional base URL for OpenAI-compatible providers.
            default_kwargs: Default request parameters merged into every `complete` call.
            retry: Whether to enable exponential backoff retries around `complete`.
            retry_errors: Exception classes that should trigger a retry. When `retry`
                is enabled and this is `None`, `RateLimitError` is used.
            concurrency_limit: Maximum number of concurrent LLM requests. `None`
                means no explicit semaphore limit.
        """
        from openai import AsyncOpenAI, RateLimitError
        import backoff
        self.client = AsyncOpenAI(
            base_url = base_url,
            api_key = api_key
        )
        self.model = model
        self.default_kwargs = default_kwargs or {}
        
        self.llm_concurrency_lock = asyncio.Semaphore(concurrency_limit) if concurrency_limit else nullcontext()
        
        if retry:
            if not retry_errors:
                retry_errors = (RateLimitError,)
            self.complete = backoff.on_exception(
                backoff.expo, tuple(retry_errors), max_time=30
            )(self.complete)

    async def complete(self, messages: Iterable[dict[str,Any]], **kwargs) -> dict[str,Any]:
        """Send chat messages to the model and return the assistant message object.

        Args:
            messages: OpenAI-format chat messages.
            **kwargs: Per-call request parameters that override `default_kwargs`.

        Returns:
            The first choice message from the API response.
        """
        payload = {**self.default_kwargs, **kwargs}
        async with self.llm_concurrency_lock:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages, # type: ignore
                **payload,
            )
        return response.choices[0].message
