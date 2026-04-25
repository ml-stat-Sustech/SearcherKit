"""Tool factory for built-in and MCP-backed agent tools."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Callable

from searchagent.sources import DataSource
from searchagent.tools.base import BaseTool, ToolConfig
from searchagent.tools.mcp import MCPTool
from searchagent.tools.search import SearchTool, VisitTool


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_RegistryEntry = type[BaseTool] | Callable[[ToolConfig], BaseTool]
_REGISTRY: dict[str, _RegistryEntry] = {}


def register_tool(name: str, entry: _RegistryEntry) -> _RegistryEntry:
    """Register a tool implementation under a type/kind name."""
    if isinstance(entry, type):
        if not issubclass(entry, BaseTool):
            raise TypeError(f"{entry} is not a subclass of BaseTool")
    elif not callable(entry):
        raise TypeError(f"entry must be a class or callable, got {type(entry)}")
    _REGISTRY[name] = entry
    return entry


# Built-ins
register_tool("search", SearchTool)
register_tool("visit", VisitTool)
register_tool("mcp", MCPTool)


def _tool_type(cfg: ToolConfig) -> str:
    """Return the implementation type while preserving old name-based configs."""

    explicit = cfg.type or cfg.kind
    if explicit:
        return explicit
    if cfg.endpoint:
        return "mcp"
    return cfg.name


def _coerce_tool_config(cfg: ToolConfig | Mapping[str, Any]) -> ToolConfig:
    if isinstance(cfg, ToolConfig):
        return cfg
    if isinstance(cfg, Mapping):
        return ToolConfig(**dict(cfg))
    items = getattr(cfg, "items", None)
    if callable(items):
        return ToolConfig(**{str(key): value for key, value in items()})
    raise TypeError(f"tool config must be ToolConfig or mapping, got {type(cfg)}")


def _normalize_tool_config(cfg: ToolConfig, tool_type: str) -> ToolConfig:
    if cfg.name:
        return cfg
    return replace(cfg, name=tool_type)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_tool(
    cfg: ToolConfig | Mapping[str, Any],
    *,
    sources: Mapping[str, DataSource] | None = None,
) -> BaseTool:
    """Instantiate a tool from *cfg*.

    Looks up *cfg.type* / *cfg.kind* in the registry. For compatibility, configs
    without a type still use *cfg.name* as the lookup key, and configs with an
    endpoint default to the ``mcp`` implementation.
    If the entry is a class, instantiates it with ``config=cfg``.
    If the entry is a callable, calls it with *cfg* and returns the result.
    """
    tool_cfg = _coerce_tool_config(cfg)
    tool_type = _tool_type(tool_cfg)
    tool_cfg = _normalize_tool_config(tool_cfg, tool_type)
    entry = _REGISTRY.get(tool_type)
    if entry is None:
        raise ValueError(
            f"Unknown tool type: {tool_type!r} for tool {tool_cfg.name!r}. "
            f"Available: {list(_REGISTRY.keys())}"
        )

    if isinstance(entry, type) and issubclass(entry, BaseTool):
        if issubclass(entry, (SearchTool, VisitTool)):
            return entry(config=tool_cfg, sources=sources)
        return entry(config=tool_cfg)

    if callable(entry):
        return entry(tool_cfg)

    raise TypeError(f"Invalid registry entry for {tool_type!r}")
