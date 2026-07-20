"""Status bar content builder for the terminal UI."""

from __future__ import annotations

from dataclasses import dataclass

from searcherkit.agent.search_agent import SearchAgentConfig
from searcherkit.runtime.interactive_selection import (
    SelectionState,
    active_model_label,
    active_source_label,
)
from searcherkit.runtime.interactive import InteractiveQueryConfig, resolve_record_dir


@dataclass(frozen=True)
class StatusBarContent:
    """Text content for the bottom status bar and splash selection panel."""

    model_label: str
    source_label: str
    tool_label: str
    record_label: str


def configured_tools_label(agent_config: SearchAgentConfig) -> str:
    """Return a short label for configured tools (names, comma-separated)."""
    names: list[str] = []
    for tool in agent_config.tools:
        name = (tool.name or tool.type or "").strip()
        if name:
            names.append(name)
    return ", ".join(names) if names else "none"


def build_status_bar(
    config: InteractiveQueryConfig,
    session_state: SelectionState,
) -> StatusBarContent:
    """Build status-bar labels from configuration and session state."""

    return StatusBarContent(
        model_label=active_model_label(config.agent.llm_client, session_state.active_model),
        source_label=active_source_label(config.agent, session_state.active_source),
        tool_label=configured_tools_label(config.agent),
        record_label=str(resolve_record_dir(config)),
    )
