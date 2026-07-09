from __future__ import annotations

import abc
import inspect
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, overload

from jsonschema import validate, ValidationError

from searchagent.common.json_schema import schema_from_signature
from searchagent.common.retry import RetryConfig
from searchagent.tools.summarizer import Summarizer
from searchagent.common.log import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Unified Tool configuration
# ---------------------------------------------------------------------------

@dataclass
class SummarizerConfig:
    model: str = ""
    api_key: str | None = None
    base_url: str | None = None
    max_chars: int = 400000
    timeout: float = 3600
    max_concurrency: int | None = None
    default_kwargs: dict[str, Any] | None = None
    retry_config: RetryConfig | None = None


@dataclass
class ToolConfig:
    """Single configuration schema shared by every tool entry."""

    # Common fields
    type: str | None = None
    target: str | None = None
    name: str = ""
    description: str | None = None
    inputSchema: dict[str, Any] | None = None
    argument_mapping: dict[str, str] = field(default_factory=dict)
    raise_argument_validation_error: bool = False

    # MCP connection
    endpoint: str | None = None
    transport: str = "streamable-http"
    auth_header: str | None = None
    max_concurrency: int | None = None
    raise_on_fatal: bool = True
    mcp_tool_name: str = ""

    # MCPTool specific
    response_char_limit: int | None = None

    # Source-backed tools
    source: list[str] = field(default_factory=list)

    # Summary tools
    summarizer: SummarizerConfig | None = None
    summary_goal_key: str = "query"

    # Factory extension
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tool abstract base class
# ---------------------------------------------------------------------------

class BaseTool(abc.ABC):
    name: str
    description: str | None
    inputSchema: Mapping[str, Any]
    argument_mapping: Mapping[str, str]
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
        argument_mapping: Mapping[str, str] | None = None,
        raise_argument_validation_error: bool = False,
        summarizer: Summarizer | None = None,
        summary_goal_key = "query",
        **kwargs: Any,
    ) -> None: ...

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        inputSchema: Mapping[str, Any] | None = None,
        *,
        argument_mapping: Mapping[str, str] | None = None,
        raise_argument_validation_error: bool = False,
        summarizer: Summarizer | None = None,
        config: ToolConfig | None = None,
        summary_goal_key = "query",
        **kwargs: Any,
    ) -> None:
        self.summarizer = None
        if config:
            if not config.name:
                raise ValueError("tool config requires a model-visible name")
            self.name = config.name
            self.description = config.description or self.get_default_description()
            self.argument_mapping = dict(config.argument_mapping)
            self._uses_default_input_schema = config.inputSchema is None
            self.inputSchema = self._resolve_input_schema(config.inputSchema)
            self.raise_argument_validation_error = config.raise_argument_validation_error
            if config.summarizer is not None:
                self._configure_summarizer(config=config.summarizer)
                self.summary_goal_key = config.summary_goal_key
        else:
            if not name:
                raise ValueError("tool requires a model-visible name")
            self.name = name
            self.description = description or self.get_default_description()
            self.argument_mapping = dict(argument_mapping or {})
            self._uses_default_input_schema = inputSchema is None
            self.inputSchema = self._resolve_input_schema(inputSchema)
            self.raise_argument_validation_error = raise_argument_validation_error
            if summarizer is not None:
                self._configure_summarizer(summarizer=summarizer)
                self.summary_goal_key = summary_goal_key

    def _configure_summarizer(
        self,
        *,
        config: SummarizerConfig | None = None,
        summarizer: Summarizer | None = None,
    ) -> None:
        if summarizer is not None:
            self.summarizer = summarizer
            return
        if config is not None:
            self.summarizer = Summarizer(config=config)
            return
        raise ValueError("Provide either a summarizer or a summarizer config to configure the tool's summarizer")

    @property
    def summary_enabled(self) -> bool:
        return self.summarizer is not None

    def format_summary(
        self,
        *,
        goal: str,
        evidence: str,
        summary: str,
    ) -> str:
        heading = f"The useful information for query {goal} as follows:"
        lines = [
            heading,
            "",
            "Evidence in page:",
            evidence or "No evidence extracted.",
            "",
            "Summary:",
            summary or "No summary available.",
            "",
        ]
        return "\n".join(lines).strip()

    async def init(self, *args: Any, **kwargs: Any) -> None:
        """Initialize tool resources."""

    async def close(self) -> None:
        """Release tool resources."""

    async def run(self, **kwargs: Any) -> str:
        """Execute the tool with the provided arguments."""
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
        mapped_kwargs = map_arguments(kwargs, self.argument_mapping)
        tool_result = await self._run(**mapped_kwargs)
        if self.summarizer:
            goal = mapped_kwargs.get(self.summary_goal_key, "")
            evidence, summary = await self.summarizer.summarize(goal=goal, content=tool_result)
            tool_result = self.format_summary(goal=goal, evidence=evidence, summary=summary)
        return tool_result

    def get_default_description(self) -> str | None:
        return inspect.getdoc(self._run) or inspect.getdoc(type(self))

    def _resolve_input_schema(self, configured_schema: Mapping[str, Any] | None) -> Mapping[str, Any]:
        if configured_schema is not None:
            return configured_schema
        default_schema = schema_from_signature(self._run)
        if default_schema is None:
            return {}
        return map_to_model_visible_schema(
            default_schema,
            self.argument_mapping,
        )

    @abc.abstractmethod
    async def _run(self, **kwargs: Any) -> str:
        """Subclasses implement actual tool execution."""
        raise NotImplementedError

    def as_openai_tool(self) -> Mapping[str, Any]:
        return to_openai_tool(self.name, self.description, self.inputSchema)


def map_arguments(
    arguments: Mapping[str, Any],
    argument_mapping: Mapping[str, str],
) -> dict[str, Any]:
    """Translate model-visible argument names to implementation argument names."""
    mapped: dict[str, Any] = {}
    for name, value in arguments.items():
        implementation_name = argument_mapping.get(name, name)
        if implementation_name == "":
            raise ValueError(
                f"argument_mapping for {name!r} must name an implementation argument"
            )
        if implementation_name in mapped:
            raise ValueError(
                "argument_mapping maps multiple model-visible arguments "
                f"to implementation argument {implementation_name!r}"
            )
        mapped[implementation_name] = value
    return mapped


def map_to_model_visible_schema(
    schema: Mapping[str, Any],
    argument_mapping: Mapping[str, str],
) -> Mapping[str, Any]:
    """Translate implementation argument names in a default schema to model-visible names."""
    if not argument_mapping:
        return schema
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        return schema

    model_properties = dict(properties)
    required = schema.get("required")
    required_set = set(required) if isinstance(required, list) else None

    for model_name, implementation_name in argument_mapping.items():
        if implementation_name == "":
            raise ValueError(
                f"argument_mapping for {model_name!r} must name an implementation argument"
            )
        if implementation_name not in properties:
            raise ValueError(
                f"argument_mapping target {implementation_name!r} is not present "
                "in the default input schema"
            )

        property_schema = model_properties.pop(implementation_name)
        if model_name in model_properties:
            raise ValueError(
                f"argument_mapping maps implementation argument {implementation_name!r} "
                f"to model-visible argument {model_name!r}, which already exists"
            )
        model_properties[model_name] = property_schema

        if required_set is not None and implementation_name in required_set:
            required_set.remove(implementation_name)
            required_set.add(model_name)

    model_schema = dict(schema)
    model_schema["properties"] = model_properties
    if required_set is not None:
        model_schema["required"] = list(required_set)
    return model_schema


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
