from __future__ import annotations

from dataclasses import dataclass, field

from areal.api.cli_args import GRPOConfig

from searchagent.agent.search_agent import SearchAgentConfig


@dataclass
class AgentConfig(SearchAgentConfig):
    raise_repeat_tool_call: bool = field(default=True)
    max_queries_per_tool_call: int | None = field(default=None)


@dataclass
class WorkFlowConfig:
    agent: AgentConfig = field(default_factory=AgentConfig)
    reward: str = field(default="f1", metadata={"choices": ["f1", "llm_as_judge"]})
    overlong_penalty_margin: int = field(default=5000)


@dataclass
class SearchAgentTrainingConfig(GRPOConfig):
    workflow: WorkFlowConfig = field(default_factory=WorkFlowConfig)
    eval_workflow: WorkFlowConfig = field(default_factory=WorkFlowConfig)
    dynamic_filter_fn: str | None = field(
        default="searchagent.training.rewards.should_accept"
    )
