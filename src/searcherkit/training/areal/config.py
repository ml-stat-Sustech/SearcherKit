from __future__ import annotations

from dataclasses import dataclass, field

from areal.api.cli_args import GRPOConfig, PPOActorConfig

from searcherkit.training.config import WorkFlowConfig


@dataclass
class SearchAgentPPOActorConfig(PPOActorConfig):
    importance_sampling_level: str = field(
        default="token",
        metadata={
            "help": (
                "PPO ratio granularity: token, sequence (GSPO), or step. "
                "Step aggregates each contiguous span selected by loss_mask."
            ),
            "choices": ["token", "sequence", "step"],
        },
    )
    enable_igpo_reward: bool = field(
        default=False,
        metadata={"help": "Enable the IGPO reward."},
    )
    punish_last_turn: bool = field(
        default=False,
        metadata={
            "help": "Assign -1 advantage to bad final turns before normalization."
        },
    )


@dataclass
class SearchAgentTrainingConfig(GRPOConfig):
    actor: SearchAgentPPOActorConfig = field(default_factory=SearchAgentPPOActorConfig)
    workflow: WorkFlowConfig = field(default_factory=WorkFlowConfig)
    eval_workflow: WorkFlowConfig = field(default_factory=WorkFlowConfig)
    dynamic_filter_fn: str | None = field(
        default="searcherkit.training.rewards.should_accept"
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.actor.enable_igpo_reward or self.actor.punish_last_turn
        ) and self.critic is not None:
            raise ValueError(
                "critic must be disabled when enable_igpo_reward or "
                "punish_last_turn is enabled"
            )
