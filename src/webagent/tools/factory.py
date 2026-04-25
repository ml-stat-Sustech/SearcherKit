"""Tool factory: unified ToolConfig + registry for all tools."""

from __future__ import annotations

from typing import Any, Callable

from webagent.tools.base import BaseTool, ToolConfig
from webagent.tools.mcp import MCPTool


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_RegistryEntry = type[BaseTool] | Callable[[ToolConfig], BaseTool]
_REGISTRY: dict[str, _RegistryEntry] = {}


def register_tool(name: str, entry: _RegistryEntry) -> _RegistryEntry:
    """Register a tool class or builder function under *name*."""
    if isinstance(entry, type):
        if not issubclass(entry, BaseTool):
            raise TypeError(f"{entry} is not a subclass of BaseTool")
    elif not callable(entry):
        raise TypeError(f"entry must be a class or callable, got {type(entry)}")
    _REGISTRY[name] = entry
    return entry


# Built-ins
register_tool("search", MCPTool)
register_tool("visit", MCPTool)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_tool(cfg: ToolConfig) -> BaseTool:
    """Instantiate a tool from *cfg*.

    Looks up *cfg.name* in the registry.
    If the entry is a class, instantiates it with ``config=cfg``.
    If the entry is a callable, calls it with *cfg* and returns the result.
    """
    entry = _REGISTRY.get(cfg.name)
    if entry is None:
        raise ValueError(
            f"Unknown tool name: {cfg.name!r}. "
            f"Available: {list(_REGISTRY.keys())}"
        )

    if isinstance(entry, type) and issubclass(entry, BaseTool):
        return entry(config=cfg)

    if callable(entry):
        return entry(cfg)

    raise TypeError(f"Invalid registry entry for {cfg.name!r}")
