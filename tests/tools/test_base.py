from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from searchagent.sources import Document, SourceConfig, add_source_cfg
from searchagent.sources.memory import MemorySource
from searchagent.common.json_schema import schema_from_signature
from searchagent.tools import ToolConfig, build_tool
from searchagent.tools.base import map_arguments, map_to_model_visible_schema
from searchagent.tools.search import SearchTool


ToolFactory = Callable[..., SearchTool]


def _documents() -> list[Document]:
    return [
        Document(
            id="doc-1",
            title="SearchAgent Runtime",
            text="SearchAgent provides source-backed tools.",
            url="https://example.test/runtime",
        ),
    ]


def _custom_search_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "q": {"type": "string"},
        },
        "required": ["q"],
        "additionalProperties": False,
    }


def _source_name(test_name: str) -> str:
    return f"tools-base-{test_name}"


def _add_memory_source(name: str) -> None:
    add_source_cfg(
        name,
        SourceConfig(type="memory", name=name, documents=_documents()),
    )


def _direct_tool(*, test_name: str) -> SearchTool:
    return SearchTool(
        MemorySource(documents=_documents()),
        name="search",
        inputSchema=_custom_search_schema(),
        argument_mapping={"q": "query"},
    )


def _config_tool(*, test_name: str) -> SearchTool:
    source_name = _source_name(test_name)
    _add_memory_source(source_name)
    tool = build_tool(
        ToolConfig(
            type="search",
            name="search",
            source=[source_name],
            inputSchema=_custom_search_schema(),
            argument_mapping={"q": "query"},
        )
    )
    assert isinstance(tool, SearchTool)
    return tool


def _direct_default_mapping_tool(*, test_name: str) -> SearchTool:
    return SearchTool(
        MemorySource(documents=_documents()),
        name="search",
        argument_mapping={"q": "query"},
    )


def _config_default_mapping_tool(*, test_name: str) -> SearchTool:
    source_name = _source_name(test_name)
    _add_memory_source(source_name)
    tool = build_tool(
        ToolConfig(
            type="search",
            name="search",
            source=[source_name],
            argument_mapping={"q": "query"},
        )
    )
    assert isinstance(tool, SearchTool)
    return tool


@pytest.mark.parametrize("tool_factory", [_direct_tool, _config_tool])
def test_tool_argument_mapping(tool_factory: ToolFactory) -> None:
    async def run() -> None:
        tool = tool_factory(test_name="argument-mapping")

        result = await tool.run(q="source-backed tools")

        assert "SearchAgent Runtime" in result
        assert "source-backed tools" in result

    asyncio.run(run())


@pytest.mark.parametrize("tool_factory", [_direct_default_mapping_tool, _config_default_mapping_tool])
def test_argument_mapping_translates_default_input_schema(tool_factory: ToolFactory) -> None:
    async def run() -> None:
        tool = tool_factory(test_name="default-schema-argument-mapping")

        assert tool.inputSchema["properties"]["q"]["type"] == "string"
        assert "query" not in tool.inputSchema["properties"]
        assert tool.inputSchema["properties"]["top_k"]["type"] == "integer"
        assert tool.inputSchema["required"] == ["q"]

        result = await tool.run(q="source-backed tools")

        assert "SearchAgent Runtime" in result

    asyncio.run(run())


def test_input_schema_is_derived_from_run_signature() -> None:
    tool = SearchTool(
        MemorySource(documents=_documents()),
        name="search",
    )

    assert tool.description == "Search the configured data source."
    assert tool.inputSchema is not None
    assert tool.inputSchema["properties"]["query"]["type"] == "string"
    assert tool.inputSchema["properties"]["top_k"]["type"] == "integer"
    assert tool.inputSchema["required"] == ["query"]
    assert tool.inputSchema["additionalProperties"] is False


def test_schema_from_signature_supports_optional_types() -> None:
    async def run(*, goal: str | None = None) -> str:
        return goal or ""

    schema = schema_from_signature(run)

    assert schema is not None
    assert schema["properties"]["goal"]["type"] == ["string", "null"]
    assert "required" not in schema


def test_argument_mapping_key_is_left_to_runtime_schema_validation() -> None:
    async def run() -> None:
        tool = SearchTool(
            MemorySource(documents=_documents()),
            name="search",
            inputSchema=_custom_search_schema(),
            argument_mapping={"query": "query"},
        )

        result = await tool.run(query="source-backed tools")

        assert result.startswith("[Tool] invalid type for tool call argument.")

    asyncio.run(run())


def test_argument_mapping_target_must_be_named() -> None:
    async def run() -> None:
        tool = SearchTool(
            MemorySource(documents=_documents()),
            name="search",
            inputSchema=_custom_search_schema(),
            argument_mapping={"q": ""},
        )

        with pytest.raises(ValueError, match="must name an implementation argument"):
            await tool.run(q="source-backed tools")

    asyncio.run(run())


def test_argument_mapping_rejects_collisions() -> None:
    schema = {
        "type": "object",
        "properties": {
            "q": {"type": "string"},
            "query": {"type": "string"},
        },
        "required": ["q"],
        "additionalProperties": False,
    }

    async def run() -> None:
        tool = SearchTool(
            MemorySource(documents=_documents()),
            name="search",
            inputSchema=schema,
            argument_mapping={"q": "query"},
        )

        with pytest.raises(ValueError, match="maps multiple model-visible arguments"):
            await tool.run(q="source-backed tools", query="duplicate")

    asyncio.run(run())


def test_map_arguments_helper_translates_model_visible_names() -> None:
    assert map_arguments(
        {"q": "source-backed tools", "top_k": 3},
        {"q": "query"},
    ) == {
        "query": "source-backed tools",
        "top_k": 3,
    }


def test_map_to_model_visible_schema_translates_properties_and_required() -> None:
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer"},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    mapped_schema = map_to_model_visible_schema(schema, {"q": "query"})

    assert mapped_schema["properties"] == {
        "top_k": {"type": "integer"},
        "q": {"type": "string"},
    }
    assert mapped_schema["required"] == ["q"]
    assert mapped_schema["additionalProperties"] is False


def test_map_to_model_visible_schema_rejects_property_collisions() -> None:
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer"},
        },
        "required": ["query"],
    }

    with pytest.raises(ValueError, match="already exists"):
        map_to_model_visible_schema(schema, {"top_k": "query"})


def test_input_schema_override_is_not_matched_against_run_signature() -> None:
    schema = {
        "type": "object",
        "properties": {
            "top_k": {"type": "integer"},
        },
        "required": [],
        "additionalProperties": False,
    }

    tool = SearchTool(
        MemorySource(documents=_documents()),
        name="search",
        inputSchema=schema,
    )

    assert tool.inputSchema is schema
