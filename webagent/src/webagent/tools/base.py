from __future__ import annotations

import abc
import json
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ValidationError

from webagent.log import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tool 抽象基类
# ---------------------------------------------------------------------------

class BaseTool(abc.ABC):
    name: str
    description: str | None
    arguments_schema: type[BaseModel] | None
    raise_argument_validation_error: bool

    def __init__(self, name, description: str | None = None, arguments_schema: type[BaseModel] | None = None, *, raise_argument_validation_error: bool = False) -> None:
        self.name = name
        self.description = description
        self.arguments_schema = arguments_schema
        self.raise_argument_validation_error = raise_argument_validation_error

    async def init(self, *args: Any, **kwargs: Any) -> None:
        """Initialize tool resources."""

    async def run(self, **kwargs: Any) -> str:
        """Execute the tool with the provided arguments."""
        if self.arguments_schema is None:
            return await self._run(kwargs)
        try:
            model = self.arguments_schema.model_validate(kwargs)
        except ValidationError as exc:
            logger.warning(
                "Tool %s arguments validation failed: %s",
                self.name,
                exc,
            )
            if self.raise_argument_validation_error:
                raise
            return f"[Tool] invalid type for argument:\n{kwargs!r}\rProblem:{exc!r}\n\nArgument type should be:\n{json.dumps(self.arguments_schema.model_json_schema())}"
        return await self._run(model.model_dump())

    @abc.abstractmethod
    async def _run(self, arguments: dict[str, Any]) -> str:
        """Subclasses implement actual tool execution."""
        raise NotImplementedError

    # def dump_metadata(self) -> dict[str, Any]:
    #     """Return a JSON-serializable metadata dict for logging."""
    #     return {
    #         "name": self.name,
    #         "description": self.description,
    #         "arguments_schema": (
    #             self.arguments_schema.model_json_schema()
    #             if self.arguments_schema is not None
    #             else None
    #         ),
    #         "type": self.__class__.__name__,
    #     }
    
    def as_openai_tool(self) -> Mapping[str, Any]:
        return self.to_openai_tool(self.name, self.description, self.arguments_schema)
    
def to_openai_tool(
    name: str,
    description: str | None = None,
    arguments_schema: type[BaseModel] | Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    description = description or ""
    if arguments_schema is None:
        parameters: Mapping[str, Any] = {}
    elif isinstance(arguments_schema, Mapping):
        parameters = arguments_schema
    else:
        parameters = arguments_schema.model_json_schema()
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
