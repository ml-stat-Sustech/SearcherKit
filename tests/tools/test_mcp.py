from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from searcherkit.tools.mcp import MCPTool


def test_mcp_tool_defers_default_metadata_until_init(
    caplog: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(logging.getLogger("searcherkit"), "propagate", True)
    caplog.set_level(logging.WARNING, logger="searcherkit.tools.mcp")

    tool = MCPTool(
        name="lookup",
        endpoint="http://example.test/mcp",
        argument_mapping={"q": "query"},
    )

    assert tool.description is None
    assert tool.inputSchema == {}
    assert "metadata will be applied after init() runs" in caplog.text

    tool._apply_tool_metadata(
        SimpleNamespace(
            description="Lookup a document.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        )
    )

    assert tool.description == "Lookup a document."
    assert tool.inputSchema == {
        "type": "object",
        "properties": {
            "top_k": {"type": "integer"},
            "q": {"type": "string"},
        },
        "required": ["q"],
        "additionalProperties": False,
    }


def test_mcp_tool_keeps_configured_schema_when_metadata_is_applied() -> None:
    configured_schema = {
        "type": "object",
        "properties": {
            "configured": {"type": "string"},
        },
        "required": ["configured"],
        "additionalProperties": False,
    }
    tool = MCPTool(
        name="lookup",
        endpoint="http://example.test/mcp",
        inputSchema=configured_schema,
    )

    tool._apply_tool_metadata(
        SimpleNamespace(
            description="Lookup a document.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        )
    )

    assert tool.inputSchema is configured_schema
