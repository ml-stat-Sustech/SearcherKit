from __future__ import annotations

from dataclasses import dataclass, field

from searcherkit.agent import SearchAgentConfig


@dataclass
class AgentConfig(SearchAgentConfig):
    raise_repeat_tool_call: bool = field(default=True)


@dataclass
class WorkFlowConfig:
    agent: AgentConfig = field(default_factory=AgentConfig)
    reward: str = field(default="f1", metadata={"choices": ["f1", "llm_as_judge"]})
    overlong_penalty_margin: int = field(default=5000)
    answer_pattern: str = field(default=r"\\boxed\{(?P<answer>[^}]*)\}")


@dataclass
class SearchAgentTrainingConfig:
    workflow: WorkFlowConfig = field(default_factory=WorkFlowConfig)
    eval_workflow: WorkFlowConfig = field(default_factory=WorkFlowConfig)
    dynamic_filter_fn: str | None = field(
        default="searcherkit.training.rewards.should_accept"
    )
