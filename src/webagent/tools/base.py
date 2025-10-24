from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..agents.base import AgentState


@dataclass
class ToolCall:
    """Normalized representation of a tool invocation requested by the agent."""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Result returned by an executed tool."""

    call: ToolCall
    output: str


class BaseTool(abc.ABC):
    """Minimal base class for runtime tool implementations."""

    name: str
    description: str
    arguments_schema: Optional[Dict[str, Any]] = None

    @abc.abstractmethod
    def run(self, call: ToolCall, state: "AgentState") -> str:
        """Execute the tool with the provided arguments."""
