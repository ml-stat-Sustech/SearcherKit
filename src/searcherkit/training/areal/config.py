from __future__ import annotations

from dataclasses import dataclass, field

from areal.api.cli_args import GRPOConfig, PPOActorConfig

from searcherkit.training.config import WorkFlowConfig


@dataclass
class SearchAgentPPOActorConfig(PPOActorConfig):
    enable_igpo_reward: bool = field(
        default=False,
        metadata={"help": "Enable the IGPO information-gain reward."},
    )


@dataclass
class SearchAgentTrainingConfig(GRPOConfig):
    actor: SearchAgentPPOActorConfig = field(default_factory=SearchAgentPPOActorConfig)
    workflow: WorkFlowConfig = field(default_factory=WorkFlowConfig)
    eval_workflow: WorkFlowConfig = field(default_factory=WorkFlowConfig)
    dynamic_filter_fn: str | None = field(
        default="searcherkit.training.rewards.should_accept"
    )
