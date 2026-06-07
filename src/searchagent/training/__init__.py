"""Training support for SearchAgent.

Heavy training dependencies such as AReaL, slime, and Megatron are imported by
the concrete submodules that need them.
"""

from typing import Any

__all__ = [
    "AgentConfig",
    "SearchAgentTrainingConfig",
    "WorkFlowConfig",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from searchagent.training.config import (
            AgentConfig,
            SearchAgentTrainingConfig,
            WorkFlowConfig,
        )

        return {
            "AgentConfig": AgentConfig,
            "SearchAgentTrainingConfig": SearchAgentTrainingConfig,
            "WorkFlowConfig": WorkFlowConfig,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
