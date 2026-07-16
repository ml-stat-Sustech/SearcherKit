"""Tool factory for built-in and MCP-backed agent tools."""

from __future__ import annotations

from typing import Callable

from searcherkit.common.config import import_from_path

from .base import BaseTool, ToolConfig
from .mcp import MCPTool
from .multi_source_search import MultiSourceSearchTool
from .multi_source_visit import MultiSourceVisitTool
from .search import SearchTool
from .visit import VisitTool


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
register_tool("multi_source_search", MultiSourceSearchTool)
register_tool("multi_source_visit", MultiSourceVisitTool)
register_tool("mcp", MCPTool)

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_tool(
    cfg: ToolConfig,
) -> BaseTool:
    """Instantiate a tool from *cfg*.

    Looks up *cfg.type* in the registry. For the ``custom`` type, imports the
    class at ``cfg.target`` and passes ``cfg.extra`` as constructor arguments.
    If the entry is a class, instantiates it with ``config=cfg``.
    If the entry is a callable, calls it with *cfg* and returns the result.
    """
    if cfg.type == "custom":
        if not cfg.target:
            raise ValueError(
                "tool type 'custom' requires a non-empty 'target' field "
                "(e.g. pkg://my.package:MyTool or file:///path/to/file.py:MyTool)"
            )
        tool_class = import_from_path(cfg.target)
        if not isinstance(tool_class, type) or not issubclass(tool_class, BaseTool):
            raise TypeError(
                f"custom tool class {tool_class!r} must be a subclass of BaseTool"
            )
        return tool_class(config=cfg, **cfg.extra)

    entry = _REGISTRY.get(cfg.type) if cfg.type else None
    if entry is None:
        raise ValueError(
            f"Unknown tool type: {cfg.type!r} for tool {cfg.name!r}. "
            f"Available: {list(_REGISTRY.keys())}"
        )

    if isinstance(entry, type) and issubclass(entry, BaseTool):
        return entry(config=cfg)

    if callable(entry):
        return entry(cfg)

    raise TypeError(f"Invalid registry entry for {cfg.type!r}")
