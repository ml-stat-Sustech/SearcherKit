from __future__ import annotations

from dataclasses import dataclass
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
    content: str
    role: Literal["system"] = "system"
    tools: list[Tool] | None = None
    extensions: dict[str, Any] | None = None


@dataclass(slots=True)
class UserMessage:
    content: str
    role: Literal["user"] = "user"
    extensions: dict[str, Any] | None = None


@dataclass(slots=True)
class AssistantMessage:
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    thinking: str | None = None
    tool_calls: list[ToolCall] | None = None
    extensions: dict[str, Any] | None = None


@dataclass(slots=True)
class ToolMessage:
    tool_responses: list[str]
    role: Literal["tool"] = "tool"
    extensions: dict[str, Any] | None = None

ChatMessage = SystemMessage | UserMessage | AssistantMessage | ToolMessage

def system(text: str, tools: list[Tool] | None = None, extensions: dict[str, Any] | None = None) -> SystemMessage:
    return SystemMessage(content=text, tools=tools, extensions=extensions)

def user(text: str, extensions: dict[str, Any] | None = None) -> UserMessage:
    return UserMessage(content=text, extensions=extensions)

def assistant(text: str | None, thinking: str | None = None, tool_calls: list[ToolCall] | None = None, extensions: dict[str, Any] | None = None) -> AssistantMessage:
    return AssistantMessage(content=text, thinking=thinking, tool_calls=tool_calls, extensions=extensions)

def tool(tool_responses: list[str], extensions: dict[str, Any] | None = None) -> ToolMessage:
    return ToolMessage(tool_responses=tool_responses, extensions=extensions)
