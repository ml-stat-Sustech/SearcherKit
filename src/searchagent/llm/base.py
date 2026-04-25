"""Provider-neutral LLM interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from searchagent.common.messages import ChatMessage


@dataclass(slots=True)
class LLMResult:
    message: ChatMessage
    usage: Mapping[str, Any] = field(default_factory=dict)
    raw: Any | None = None


class LLMClient(Protocol):
    async def complete(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResult:
        """Return one assistant message from a provider-specific backend."""
