from __future__ import annotations

from searcherkit.runtime.interactive_selection import (
    apply_active_source,
    infer_active_source,
    parse_source_command,
    selectable_source_options,
)
from searcherkit.agent.search_agent import SearchAgentConfig
from searcherkit.sources import SourceConfig
from searcherkit.tools.base import ToolConfig


def test_selectable_source_options_use_safe_name_and_type() -> None:
    options = selectable_source_options(
        [
            SourceConfig(name="wiki", type="elasticsearch"),
            SourceConfig(name="bad name", type="file"),
            SourceConfig(name="custom", type="custom:backend"),
        ]
    )

    assert [option.command_name for option in options] == ["sources:wiki(elasticsearch)"]


def test_parse_source_command_requires_exact_generated_command() -> None:
    sources = [SourceConfig(name="wiki", type="elasticsearch")]

    assert parse_source_command("sources:wiki(elasticsearch)", sources).name == "wiki"
    assert parse_source_command("sources wiki", sources) is None
    assert parse_source_command("sources:wiki(file)", sources) is None


def test_infer_active_source_from_source_backed_tools() -> None:
    config = SearchAgentConfig(
        sources=[SourceConfig(name="wiki", type="file")],
        tools=[
            ToolConfig(type="search", name="search", source=["wiki"]),
            ToolConfig(type="visit", name="visit", source=["wiki"]),
            ToolConfig(type="mcp", name="external", source=["other"]),
        ],
    )

    state = infer_active_source(config)

    assert state.kind == "active"
    assert state.label == "wiki(file)"
    assert state.as_event_data() == {
        "state": "active",
        "name": "wiki",
        "type": "file",
        "tool_sources": ["wiki"],
    }


def test_infer_active_source_reports_mixed_and_none_states() -> None:
    mixed = infer_active_source(
        SearchAgentConfig(
            sources=[SourceConfig(name="wiki", type="file"), SourceConfig(name="bcp", type="file")],
            tools=[
                ToolConfig(type="search", name="search", source=["wiki"]),
                ToolConfig(type="visit", name="visit", source=["bcp"]),
            ],
        )
    )
    none = infer_active_source(
        SearchAgentConfig(
            sources=[SourceConfig(name="wiki", type="file")],
            tools=[ToolConfig(type="mcp", name="external", source=["wiki"])],
        )
    )

    assert mixed.kind == "mixed"
    assert mixed.tool_sources == ("bcp", "wiki")
    assert none.kind == "none"


def test_apply_active_source_changes_only_built_in_source_backed_tools() -> None:
    config = SearchAgentConfig(
        tools=[
            ToolConfig(type="search", name="search", source=["wiki"]),
            ToolConfig(type="visit", name="visit", source=["wiki"]),
            ToolConfig(type="search", name="search_all", source=["wiki", "bcp"]),
            ToolConfig(type="mcp", name="external", source=["wiki"]),
        ]
    )

    updated = apply_active_source(config, "bcp")

    assert updated == 3
    assert {tool.name: tool.source for tool in config.tools} == {
        "search": ["bcp"],
        "visit": ["bcp"],
        "search_all": ["bcp"],
        "external": ["wiki"],
    }
