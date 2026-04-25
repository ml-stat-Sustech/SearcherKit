"""Provider-agnostic message structures."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

Role = Literal["system", "user", "assistant", "tool"]

@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: Mapping[str, Any]
    id: str | None = None
    result: str | None = None
    
@dataclass(slots=True)
class Tool:
    name: str
    description: str | None
    inputSchema: Mapping[str, Any] | None # mcp tool schema


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
    tool_responses: list[ToolCall]
    role: Literal["tool"] = "tool"
    extensions: dict[str, Any] | None = None

ChatMessage = SystemMessage | UserMessage | AssistantMessage | ToolMessage

def system(text: str, tools: list[Tool] | None = None, extensions: dict[str, Any] | None = None) -> SystemMessage:
    return SystemMessage(content=text, tools=tools, extensions=extensions)

def user(text: str, extensions: dict[str, Any] | None = None) -> UserMessage:
    return UserMessage(content=text, extensions=extensions)

def assistant(text: str | None, thinking: str | None = None, tool_calls: list[ToolCall] | None = None, extensions: dict[str, Any] | None = None) -> AssistantMessage:
    return AssistantMessage(content=text, thinking=thinking, tool_calls=tool_calls, extensions=extensions)

def tool(tool_responses: list[ToolCall], extensions: dict[str, Any] | None = None) -> ToolMessage:
    return ToolMessage(tool_responses=tool_responses, extensions=extensions)
