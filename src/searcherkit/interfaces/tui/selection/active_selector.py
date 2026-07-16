from __future__ import annotations

from dataclasses import dataclass

from searcherkit.runtime.interactive_selection import (
    ActiveSource,
    ModelOption,
    SelectionState,
    SourceOption,
    parse_model_command,
    parse_source_command,
    selectable_source_options,
)
from searcherkit.sources import SourceConfig


@dataclass(frozen=True)
class SelectableItem:
    """A candidate shown in the slash command menu."""

    command_name: str
    label: str


class ModelSelector:
    """Selector for Active Model commands."""

    def __init__(self, options: list[ModelOption], discovery_message: str = "") -> None:
        self._options = list(options)
        self._discovery_message = discovery_message

    @property
    def prefix(self) -> str:
        return "models"

    @property
    def discovery_message(self) -> str:
        return self._discovery_message

    @property
    def display_name(self) -> str:
        return "Model"

    def list_items(self) -> list[SelectableItem]:
        return [
            SelectableItem(command_name=option.command_name, label=option.label)
            for option in self._options
        ]

    def parse_command(self, command: str) -> ModelOption | None:
        return parse_model_command(command, self._options)

    def apply(self, session_state: SelectionState, option: ModelOption) -> str:
        session_state.active_model = option
        return option.label


class SourceSelector:
    """Selector for Active Source commands."""

    def __init__(self, sources: list[SourceConfig]) -> None:
        self._sources = list(sources)

    @property
    def prefix(self) -> str:
        return "sources"

    @property
    def display_name(self) -> str:
        return "Source"

    def list_items(self) -> list[SelectableItem]:
        return [
            SelectableItem(command_name=option.command_name, label=option.label)
            for option in selectable_source_options(self._sources)
        ]

    def parse_command(self, command: str) -> SourceOption | None:
        return parse_source_command(command, self._sources)

    def apply(self, session_state: SelectionState, option: SourceOption) -> str:
        session_state.active_source = ActiveSource(
            kind="active",
            name=option.name,
            source_type=option.source_type,
            tool_sources=(option.name,),
        )
        return option.label
