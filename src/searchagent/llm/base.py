"""Provider-neutral LLM client interfaces and configuration."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

from searchagent.common.retry import RetryPolicy
from searchagent.common.messages import ChatMessage

if TYPE_CHECKING:
    from openai.types.completion_usage import CompletionUsage


@dataclass
class LLMResult:
    message: ChatMessage
    usage: dict[str, Any] = field(default_factory=dict)
    raw: Any | None = None


@dataclass
class OpenAIConfig:
    api_key: str | None = None
    base_url: Any | None = None
    concurrency_limit: int | None = None
    extra_client_kwargs: dict[str, Any] | None = None


@dataclass
class AnthropicConfig:
    api_key: str | None = None
    base_url: str | None = None
    concurrency_limit: int | None = None
    extra_client_kwargs: dict[str, Any] | None = None


@dataclass
class DashScopeConfig(OpenAIConfig):
    base_url: Any | None = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@dataclass
class VllmConfig(OpenAIConfig):
    base_url: Any | None = "http://127.0.0.1:8000/v1"


@dataclass
class OllamaConfig(OpenAIConfig):
    api_key: str | None = "ollama"
    base_url: Any | None = "http://127.0.0.1:11434/v1"


@dataclass
class TransformersConfig:
    device: str | None = None
    dtype: str | None = None
    model_kwargs: dict[str, Any] | None = None
    generation_kwargs: dict[str, Any] | None = None


@dataclass
class ClientConfig:
    type: str = "openai"
    model: str = ""
    retry_policy: RetryPolicy | None = None
    default_kwargs: dict[str, Any] | None = None
    openai: OpenAIConfig | None = None
    anthropic: AnthropicConfig | None = None
    dashscope: DashScopeConfig | None = None
    vllm: VllmConfig | None = None
    ollama: OllamaConfig | None = None
    transformers: TransformersConfig | None = None


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


LLMClient = Client


def get_client(config: ClientConfig) -> Client:
    provider = config.type.lower()
    if provider == "openai":
        from searchagent.llm.openai import OpenAIClient

        return OpenAIClient(config=config)
    if provider == "anthropic":
        from searchagent.llm.anthropic import AnthropicClient

        return AnthropicClient(config=config)
    if provider == "dashscope":
        from searchagent.llm.dashscope import DashScopeClient

        return DashScopeClient(config=config)
    if provider in {"vllm", "vllm_server"}:
        from searchagent.llm.vllm import VllmClient

        return VllmClient(config=config)
    if provider == "ollama":
        from searchagent.llm.ollama import OllamaClient

        return OllamaClient(config=config)
    if provider in {"transformers", "hf", "huggingface"}:
        from searchagent.llm.transformers import TransformersClient

        return TransformersClient(config=config)
    raise ValueError(f"Cannot infer client type from name: {config.type}")
