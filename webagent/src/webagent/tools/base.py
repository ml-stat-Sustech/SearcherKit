from __future__ import annotations

import abc
import json
from collections.abc import Mapping
from typing import Any

from jsonschema import validate, ValidationError

from webagent.log import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tool 抽象基类
# ---------------------------------------------------------------------------

class BaseTool(abc.ABC):
    name: str
    description: str | None
    inputSchema: Mapping[str, Any] | None
    raise_argument_validation_error: bool

    def __init__(self, name, description: str | None = None, inputSchema:  Mapping[str, Any] | None = None, *, raise_argument_validation_error: bool = False) -> None:
        self.name = name
        self.description = description
        self.inputSchema = inputSchema
        self.raise_argument_validation_error = raise_argument_validation_error

    async def init(self, *args: Any, **kwargs: Any) -> None:
        """Initialize tool resources."""

    async def run(self, **kwargs: Any) -> str:
        """Execute the tool with the provided arguments."""
        if self.inputSchema is None:
            return await self._run(kwargs)
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
            return f"[Tool] invalid type for tool call argument.\nProblem:{exc!r}\n\nArgument type should be:\n{json.dumps(self.inputSchema)}"
            # return "[Search] Invalid request format: 'query' must be a string, not an array" # TODO
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
    arguments_schema: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    description = description or ""
    if arguments_schema is None:
        parameters: Mapping[str, Any] = {}
    elif isinstance(arguments_schema, Mapping):
        parameters = arguments_schema
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
