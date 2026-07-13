"""Status bar content builder for the terminal UI."""

from __future__ import annotations

from dataclasses import dataclass

from searchagent.runtime.interactive_selection import (
    SelectionState,
    active_model_label,
    active_source_label,
)
from searchagent.runtime.interactive import InteractiveQueryConfig, resolve_record_dir


@dataclass(frozen=True)
class StatusBarContent:
    """Text content for the bottom status bar."""

    model_label: str
    source_label: str
    record_label: str


def build_status_bar(
    config: InteractiveQueryConfig,
    session_state: SelectionState,
) -> StatusBarContent:
    """Build status-bar labels from configuration and session state."""

    return StatusBarContent(
        model_label=active_model_label(config.agent.llm_client, session_state.active_model),
        source_label=active_source_label(config.agent, session_state.active_source),
        record_label=str(resolve_record_dir(config)),
    )
