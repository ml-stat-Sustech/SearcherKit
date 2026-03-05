from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class ToolCall:
    """Provider-agnostic tool/function invocation emitted by an assistant."""
    id: str
    name: str
    arguments: Mapping[str, Any] | str

@dataclass(slots=True)
class ChatMessage:
    """
    Internal chat message model.

    Use this in domain code; convert at provider boundaries via parser functions.
    """

    role: Role
    content: str | None = None
    thinking: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    extensions: dict[str, Any] = field(default_factory=dict)

ChatMessages = list[ChatMessage]

def system(text: str, **kwargs: Any) -> ChatMessage:
    return ChatMessage(role="system", content=text, **kwargs)


def user(text: str, **kwargs: Any) -> ChatMessage:
    return ChatMessage(role="user", content=text, **kwargs)


def assistant(text: str | None = None, **kwargs: Any) -> ChatMessage:
    return ChatMessage(role="assistant", content=text, **kwargs)


def tool(text: str, **kwargs: Any) -> ChatMessage:
    return ChatMessage(role="tool", content=text, **kwargs)
