"""Tool factory for built-in and MCP-backed agent tools."""

from __future__ import annotations

from typing import Callable

from searchagent.tools.base import BaseTool, ToolConfig
from searchagent.tools.mcp import MCPTool
from searchagent.tools.search import SearchTool
from searchagent.tools.visit import VisitTool


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

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_tool(
    cfg: ToolConfig,
) -> BaseTool:
    """Instantiate a tool from *cfg*.

    Looks up *cfg.type* / *cfg.kind* in the registry. For compatibility, configs
    without a type still use *cfg.name* as the lookup key, and configs with an
    endpoint default to the ``mcp`` implementation.
    If the entry is a class, instantiates it with ``config=cfg``.
    If the entry is a callable, calls it with *cfg* and returns the result.
    """
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
