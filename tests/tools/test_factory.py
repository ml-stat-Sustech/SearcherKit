from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from searcherkit.tools import BaseTool, ToolConfig, build_tool


CUSTOM_TOOL_FILE = Path(__file__).resolve().parents[1] / "fixtures" / "custom_tool.py"


def _target(attribute: str) -> str:
    return f"file://{CUSTOM_TOOL_FILE}:{attribute}"


def test_build_custom_tool_passes_extra_constructor_arguments() -> None:
    tool = build_tool(
        ToolConfig(
            type="custom",
            target=_target("CustomTool"),
            name="custom",
            extra={"prefix": "result", "count": 3},
        )
    )

    assert isinstance(tool, BaseTool)
    assert tool.prefix == "result"
    assert tool.count == 3
    assert asyncio.run(tool.run(query="example")) == (
        "result:example",
        {"count": 3},
    )


def test_build_custom_tool_from_omegaconf_config() -> None:
    cfg = OmegaConf.merge(
        OmegaConf.structured(ToolConfig),
        {
            "type": "custom",
            "target": _target("CustomTool"),
            "name": "custom",
            "extra": {"prefix": "configured", "count": 5},
        },
    )
    tool_config = OmegaConf.to_object(cfg)

    assert isinstance(tool_config, ToolConfig)
    tool = build_tool(tool_config)

    assert tool.prefix == "configured"
    assert tool.count == 5


def test_build_custom_tool_requires_target() -> None:
    with pytest.raises(ValueError, match="requires a non-empty 'target'"):
        build_tool(ToolConfig(type="custom", name="custom"))


def test_build_custom_tool_requires_base_tool_subclass() -> None:
    with pytest.raises(TypeError, match="must be a subclass of BaseTool"):
        build_tool(
            ToolConfig(
                type="custom",
                target=_target("NOT_A_TOOL"),
                name="custom",
            )
        )
