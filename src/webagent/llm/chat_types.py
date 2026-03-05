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
class TextPart:
    type: Literal["text"] = "text"
    text: str = ""



@dataclass(slots=True)
class ChatMessage:
    """
    Internal chat message model.

    Use this in domain code; convert at provider boundaries via adapter functions.
    """

    role: Role
    content: str | list[TextPart] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    extensions: dict[str, Any] = field(default_factory=dict)

ChatMessages = list[ChatMessage]

def system(text: str, **kwargs: Any) -> ChatMessage:
    return ChatMessage(role="system", content=text, **kwargs)


def user(text: str, **kwargs: Any) -> ChatMessage:
    return ChatMessage(role="user", content=text, **kwargs)


def assistant(text: str | None = None, **kwargs: Any) -> ChatMessage:
    return ChatMessage(role="assistant", content=text, **kwargs)


def tool(text: str, *, tool_call_id: str, **kwargs: Any) -> ChatMessage:
    return ChatMessage(role="tool", content=text, tool_call_id=tool_call_id, **kwargs)
