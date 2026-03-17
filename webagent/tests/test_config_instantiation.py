from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from webagent.llm.client import OpenAIClient
from webagent.runtime.agent_runner import AgentRunner
from webagent.tools import MCPTool
from webagent.utils.retry import RetryPolicy


def _load_cfg() -> Any:
    repo_root = Path(__file__).resolve().parents[1]
    conf_dir = repo_root / "src" / "webagent" / "conf"
    with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
        return compose(config_name="config")


def test_agent_components_from_config() -> None:
    cfg = _load_cfg()
    runner = AgentRunner(agent_config=cfg.agent)
    agent = runner.build_agent()

    assert isinstance(agent.client, OpenAIClient)
    assert isinstance(agent.tool_dict, dict)
    assert isinstance(agent.llm_retry_policy, RetryPolicy)
    assert isinstance(agent.tool_retry_policy, RetryPolicy)

    tool_cfgs = OmegaConf.to_container(cfg.agent.tools, resolve=True) if cfg.agent.tools else []
    for tool_cfg in tool_cfgs:
        tool_name = tool_cfg.get("name") or tool_cfg.get("mcp_tool_name")
        assert tool_name, "tool name must be provided in config"
        assert tool_name in agent.tool_dict
        tool = agent.tool_dict[tool_name]
        assert isinstance(tool, MCPTool)
        assert getattr(tool, "name", None) == tool_name

        expected_desc = tool_cfg.get("description")
        expected_args = tool_cfg.get("arguments_schema", tool_cfg.get("arguments"))
        assert getattr(tool, "description", None) == expected_desc
        assert getattr(tool, "arguments_schema", None) == expected_args
