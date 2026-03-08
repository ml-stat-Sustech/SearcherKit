"""Warpped LLM client

Controllable LLM call concorrency
Parse from and to `ChatMessage`

TODO:
- [ ] support openai compatible engines
- [ ] support non server LLM engines
"""
from __future__ import annotations

import abc
import asyncio
from contextlib import nullcontext
from typing import Any, Dict, Iterable, Optional, TYPE_CHECKING

from webagent.llm.parser import get_parser_cls

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
        parser: str | None = None,
        parser_kwargs: dict[str, Any] | None = None,
        drop_thinking = True,
        concurrency_limit: int | None = None
    ) -> None:
        from openai import AsyncOpenAI, RateLimitError
        import backoff
        self.client = AsyncOpenAI(
            base_url = base_url,
            api_key = api_key
        )
        self.model = model
        self.query = backoff.on_exception(backoff.expo, RateLimitError)(self.client.chat.completions.create)
        self.default_kwargs = default_kwargs or {}
        self.parser = get_parser_cls(parser or model)(**(parser_kwargs or {}))
        self.drop_thinking = drop_thinking
        
        self.llm_concurrency_lock = asyncio.Semaphore(concurrency_limit) if concurrency_limit else nullcontext()

    async def complete(self, messages: Iterable[ChatMessage], **kwargs) -> ChatMessage:
        payload = {**self.default_kwargs, **kwargs}
        async with self.llm_concurrency_lock:
            response = await self.query(
                model=self.model,
                messages=self.parser.to_model(messages), # type: ignore
                **payload,
            )
        msg = response.choices[0].message
        return self.parser.from_model([msg])[0]
