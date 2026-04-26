from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from searchagent.llm.base import ClientConfig, DashScopeConfig, OllamaConfig, VllmConfig, get_client
from searchagent.llm.dashscope import DashScopeClient
from searchagent.llm.ollama import OllamaClient
from searchagent.llm.openai import OpenAIClient
from searchagent.llm.parsers import ParserConfig, get_parser
from searchagent.llm.vllm import VllmClient
from searchagent.runtime.agent_runner import AgentRunner
from searchagent.tools import BaseTool
from searchagent.common.retry import RetryPolicy


def _load_cfg() -> Any:
    repo_root = Path(__file__).resolve().parents[1]
    conf_dir = repo_root / "src" / "searchagent" / "config"
    with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
        return compose(config_name="config")


def test_agent_components_from_config() -> None:
    cfg = _load_cfg()
    runner = AgentRunner(config=cfg)
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
        assert isinstance(tool, BaseTool)
        assert getattr(tool, "name", None) == tool_name

        expected_desc = tool_cfg.get("description")
        expected_args = tool_cfg.get("inputSchema")
        assert getattr(tool, "description", None) == expected_desc
        assert getattr(tool, "inputSchema", None) == expected_args


def test_parser_target_config_instantiates_custom_parser() -> None:
    websailor_parser = get_parser(
        ParserConfig(
            type="websailor",
            target="pkg://searchagent.llm.parsers.websailor:WebSailorParser",
        )
    )
    webexplorer_parser = get_parser(
        ParserConfig(
            type="webexplorer",
            target="pkg://searchagent.llm.parsers.webexplorer:WebExplorerParser",
        )
    )

    assert type(websailor_parser).__name__ == "WebSailorParser"
    assert type(webexplorer_parser).__name__ == "WebExplorerParser"


def test_openai_compatible_provider_configs_instantiate_clients() -> None:
    providers = [
        (
            ClientConfig(
                type="dashscope",
                model="qwen",
                dashscope=DashScopeConfig(api_key="test"),
            ),
            DashScopeClient,
        ),
        (
            ClientConfig(
                type="vllm",
                model="qwen",
                vllm=VllmConfig(api_key="test"),
            ),
            VllmClient,
        ),
        (ClientConfig(type="ollama", model="qwen", ollama=OllamaConfig()), OllamaClient),
    ]

    for config, expected_type in providers:
        assert isinstance(get_client(config), expected_type)
