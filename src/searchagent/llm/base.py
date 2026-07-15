"""Provider-neutral LLM client interfaces and configuration."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterable

from searchagent.common.retry import RetryPolicy
from searchagent.common.messages import ChatMessage

if TYPE_CHECKING:
    from openai.types.completion_usage import CompletionUsage


@dataclass
class LLMResult:
    message: ChatMessage
    usage: dict[str, Any] = field(default_factory=dict)
    raw: Any | None = None


@dataclass(slots=True)
class LLMStreamChunk:
    """One chunk from a streaming LLM completion.

    ``message`` is set on the terminal chunk and contains the complete
    OpenAI-compatible assistant message shape expected by parsers.
    """

    content_delta: str = ""
    thinking_delta: str = ""
    raw_delta: Any | None = None
    message: dict[str, Any] | None = None
    usage: Any | None = None
    done: bool = False


@dataclass
class ClientConfig:
    type: str = "openai"
    model: str = ""
    api_key: str | None = None
    base_url: Any | None = None
    concurrency_limit: int | None = None
    extra_client_kwargs: dict[str, Any] | None = None
    retry_policy: RetryPolicy | None = None
    default_kwargs: dict[str, Any] | None = None


class Client(abc.ABC):
    @abc.abstractmethod
    async def complete(
        self,
        messages: Iterable[dict[str, Any]],
        session_id: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return one assistant message in OpenAI chat-completion message shape."""

    @abc.abstractmethod
    async def complete_with_usage(
        self,
        messages: Iterable[dict[str, Any]],
        session_id: int | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], "CompletionUsage | None"]:
        """Return one assistant message and provider usage metadata when available."""

    async def stream_complete_with_usage(
        self,
        messages: Iterable[dict[str, Any]],
        session_id: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Yield a streaming completion.

        Providers that do not implement native streaming fall back to the existing
        non-streaming call and emit a single terminal chunk. Batch execution keeps
        using ``complete_with_usage`` directly.
        """
        message, usage = await self.complete_with_usage(messages, session_id=session_id, **kwargs)
        yield LLMStreamChunk(message=message, usage=usage, done=True)


LLMClient = Client


def get_client(config: ClientConfig) -> Client:
    provider = config.type.lower()
    if provider == "openai":
        from searchagent.llm.openai import OpenAIClient

        return OpenAIClient(config=config)
    if provider == "anthropic":
        from searchagent.llm.anthropic import AnthropicClient

        return AnthropicClient(config=config)
    if provider in {"vllm", "vllm_server"}:
        from searchagent.llm.vllm import VllmClient

        return VllmClient(config=config)
    raise ValueError(f"Cannot infer client type from name: {config.type}")
