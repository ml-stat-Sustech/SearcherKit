from __future__ import annotations

import abc
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, overload

from jsonschema import validate, ValidationError

from webagent.commons.log import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Unified Tool configuration
# ---------------------------------------------------------------------------

@dataclass
class ToolConfig:
    """Single configuration schema shared by every tool entry."""

    # Common fields
    name: str = ""
    description: str | None = None
    inputSchema: dict[str, Any] | None = None
    raise_argument_validation_error: bool = False

    # MCP connection
    endpoint: str | None = None
    transport: str = "streamable-http"
    auth_header: str | None = None
    max_concurrency: int | None = None
    enable_trace_logging: bool = True
    raise_on_fatal: bool = True
    mcp_tool_name: str = ""

    # MCPTool specific
    response_char_limit: int | None = None

    # Factory extension
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tool abstract base class
# ---------------------------------------------------------------------------

class BaseTool(abc.ABC):
    name: str
    description: str | None
    inputSchema: Mapping[str, Any] | None
    raise_argument_validation_error: bool

    @overload
    def __init__(self, *, config: ToolConfig) -> None: ...

    @overload
    def __init__(
        self,
        name: str,
        description: str | None = None,
        inputSchema: Mapping[str, Any] | None = None,
        *,
        raise_argument_validation_error: bool = False,
    ) -> None: ...

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        inputSchema: Mapping[str, Any] | None = None,
        *,
        raise_argument_validation_error: bool = False,
        config: ToolConfig | None = None,
        **kwargs: Any,
    ) -> None:
        if config:
            self.name = config.name
            self.description = config.description
            self.inputSchema = config.inputSchema
            self.raise_argument_validation_error = config.raise_argument_validation_error
        else:
            assert name
            self.name = name
            self.description = description
            self.inputSchema = inputSchema
            self.raise_argument_validation_error = raise_argument_validation_error

    async def init(self, *args: Any, **kwargs: Any) -> None:
        """Initialize tool resources."""

    async def run(self, **kwargs: Any) -> str:
        """Execute the tool with the provided arguments."""
        if self.inputSchema is None:
            return await self._run(**kwargs)
        try:
            validate(instance=kwargs, schema=self.inputSchema)
        except ValidationError as exc:
            logger.warning(
                "Tool %s arguments validation failed: %s",
                self.name,
                exc,
            )
            if self.raise_argument_validation_error:
                raise
            return (
                f"[Tool] invalid type for tool call argument.\n"
                f"Problem:{exc!r}\n\n"
                f"Argument type should be:\n{json.dumps(self.inputSchema)}"
            )
        return await self._run(**kwargs)

    @abc.abstractmethod
    async def _run(self, arguments: dict[str, Any]) -> str:
        """Subclasses implement actual tool execution."""
        raise NotImplementedError

    def as_openai_tool(self) -> Mapping[str, Any]:
        return to_openai_tool(self.name, self.description, self.inputSchema)


def to_openai_tool(
    name: str,
    description: str | None = None,
    inputSchema: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    description = description or ""
    if inputSchema is None:
        parameters: Mapping[str, Any] = {}
    elif isinstance(inputSchema, Mapping):
        parameters = inputSchema
    else:
        parameters = {}
    if not description:
        logger.warning("Tool %s has no description", name)
    if not parameters:
        logger.warning("Tool %s has no arguments schema", name)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }
