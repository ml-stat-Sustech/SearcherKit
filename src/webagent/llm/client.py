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
import random
from contextlib import nullcontext
from typing import Any, Awaitable, Callable, Dict, Iterable, Optional, Union, TYPE_CHECKING, overload
from dataclasses import dataclass

from openai import AsyncOpenAI

from webagent.log import get_logger
from webagent.commons.retry import wrap_async

if TYPE_CHECKING:
    from openai.types.completion_usage import CompletionUsage
    from webagent.commons.retry import RetryPolicy

logger = get_logger(__name__)

@dataclass
class OpenAIConfig:
    api_key: str | None
    base_url: str | list[str] | None = None
    concurrency_limit: int | None = None
    extra_client_kwargs: dict[str, object] | None = None

@dataclass
class ClientConfig:
    type: str = "openai"
    model: str = ""
    retry_policy: RetryPolicy | None = None
    default_kwargs: dict[str, object] | None = None
    openai: OpenAIConfig | None = None

    def __post_init__(self):
        assert self.model

def get_client(config: ClientConfig) -> "Client":
    if "openai" == config.type.lower():
        return OpenAIClient(config=config)
    raise ValueError(f"Cannot infer client type from name: {config.type}")

class Client:
    @abc.abstractmethod
    async def complete(self, messages: Iterable[dict[str,Any]], session_id: int | None = None,**kwargs) -> dict[str,Any]:
        pass
    
    @abc.abstractmethod
    async def complete_with_usage(
        self,
        messages: Iterable[dict[str, Any]],
        session_id: int | None = None,
        **kwargs,
    ) -> tuple[dict[str, Any], CompletionUsage | None]:
        pass

class OpenAIClient(Client):
    """Wrapper around the `openai` Python SDK."""
    @overload
    def __init__(
        self,
        *,
        config: ClientConfig,
    ) -> None:
        """Initialize an OpenAI chat-completions client wrapper.

        Args:
            config: OpenAI client configuration.
        """
        ...

    @overload
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | list[str] | None = None,
        retry_policy: RetryPolicy | None = None,
        concurrency_limit: int | None = None,
        default_kwargs: dict[str, object] | None = None,
        **extra_client_kwargs
    ) -> None:
        """Initialize an OpenAI chat-completions client wrapper.

        Args:
            model: Model name passed to `chat.completions.create`.
            api_key: OpenAI-compatible API key. Uses environment defaults when `None`.
            base_url: Optional base URL for OpenAI-compatible providers.
            default_kwargs: Default request parameters merged into every `complete` call.
            retry_policy: Retry policy for API requests. If `None`, requests are sent
                without retries.
            concurrency_limit: Maximum number of concurrent LLM requests. `None`
                means no explicit semaphore limit.
            **extra_client_kwargs: Extra keyword arguments passed to `openai.Client`.
        """
        ...

    def __init__(
        self,
        *,
        config: ClientConfig | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | list[str] | None = None,
        retry_policy: RetryPolicy | None = None,
        concurrency_limit: int | None = None,
        default_kwargs: dict[str, object] | None = None,
        **extra_client_kwargs
    ) -> None:
        
        if config is not None:
            assert config.openai, "Please specify openai config when using openai client"
            self.__init__(
                model=config.model,
                api_key=config.openai.api_key,
                base_url=config.openai.base_url,
                retry_policy=config.retry_policy,
                concurrency_limit=config.openai.concurrency_limit,
                default_kwargs=config.default_kwargs,
                **(config.openai.extra_client_kwargs or {}),
            )
            return


        if isinstance(base_url, str) or base_url is None:
            base_urls = [base_url]
        else:
            base_urls = base_url

        self.clients = [AsyncOpenAI(
            base_url = url,
            api_key = api_key,
            **extra_client_kwargs
        ) for url in base_urls]

        assert model
        self.model = model
        self.default_kwargs = default_kwargs or {}
        
        self.llm_concurrency_lock = asyncio.Semaphore(concurrency_limit) if concurrency_limit else nullcontext()
        self._create_completion: Callable[
            [list[dict[str, Any]], dict[str, Any], Optional[int]],
            Awaitable[Any],
        ] = self._create_completion_no_retry
        if retry_policy is not None:
            self._create_completion = wrap_async(
                self._create_completion_no_retry,
                policy=retry_policy,
                op_name="openai.chat.completions.create",
            )

    async def complete(self, messages: Iterable[dict[str,Any]], session_id: int | None = None,**kwargs) -> dict[str,Any]:
        """Send chat messages to the model and return the assistant message object.

        Args:
            messages: OpenAI-format chat messages.
            **kwargs: Per-call request parameters that override `default_kwargs`.

        Returns:
            The first choice message from the API response.
        """
        return (await self.complete_with_usage(messages, session_id, **kwargs))[0]

    async def _create_completion_no_retry(
        self,
        messages: list[dict[str, Any]],
        payload: dict[str, Any],
        session_id: int | None = None,
    ) -> Any:
        logger.debug(
            "Submitting LLM completion model=%s messages=%s",
            self.model,
            len(messages),
        )
        if not session_id:
            session_id = random.randint(0, len(self.clients))
        async with self.llm_concurrency_lock:
            return await self.clients[session_id % len(self.clients)].chat.completions.create(
                model=self.model,
                messages=messages, # type: ignore
                **payload,
            )

    async def complete_with_usage(
        self,
        messages: Iterable[dict[str, Any]],
        session_id: int | None = None,
        **kwargs,
    ) -> tuple[dict[str, Any], CompletionUsage | None]:
        message_list = messages if isinstance(messages, list) else list(messages)
        payload = {**self.default_kwargs, **kwargs}
        resp = await self._create_completion(message_list, payload, session_id)
        logger.debug(
            "LLM completion finished model=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
            self.model,
            getattr(resp.usage, "prompt_tokens", None),
            getattr(resp.usage, "completion_tokens", None),
            getattr(resp.usage, "total_tokens", None),
        )
        return resp.choices[0].message.model_dump(), resp.usage
