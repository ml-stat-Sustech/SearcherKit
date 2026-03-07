from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, TYPE_CHECKING

Role = Literal["system", "user", "assistant", "tool"]

if TYPE_CHECKING:
    from webagent.tools.tool import Tool

@dataclass(slots=True)
class ToolCall:
    """Provider-agnostic tool/function invocation emitted by an assistant."""
    id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(slots=True)
class SystemMessage:
    role: Literal["system"] = "system"
    content: str | None = None
    tools: list[Tool] | None = None
    extensions: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class UserMessage:
    role: Literal["user"] = "user"
    content: str | None = None
    extensions: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AssistantMessage:
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    thinking: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    extensions: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolMessage:
    role: Literal["tool"] = "tool"
    content: str | None = None
    extensions: dict[str, Any] = field(default_factory=dict)

ChatMessage = SystemMessage | UserMessage | AssistantMessage | ToolMessage

def system(text: str, **kwargs: Any) -> SystemMessage:
    return SystemMessage(content=text, **kwargs)


def user(text: str, **kwargs: Any) -> UserMessage:
    return UserMessage(content=text, **kwargs)


def assistant(text: str | None = None, **kwargs: Any) -> AssistantMessage:
    return AssistantMessage(content=text, **kwargs)


def tool(text: str, **kwargs: Any) -> ToolMessage:
    return ToolMessage(content=text, **kwargs)
