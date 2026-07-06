from __future__ import annotations

from typing import Any

from areal.api.engine_api import InferenceEngine
from areal.experimental.openai.types import InteractionWithTokenLogpReward

from searchagent.training.areal.workflow import ARealSearchAgentWorkflow


class ARealIGPOSearchAgentWorkflow(ARealSearchAgentWorkflow):
    """AReaL workflow variant that carries ground-truth answers for IGPO."""

    async def arun_episode(
        self,
        engine: InferenceEngine,
        data: dict[str, Any],
    ) -> dict[str, Any] | None | dict[str, InteractionWithTokenLogpReward]:
        trajectory = await super().arun_episode(engine, data)
        if not isinstance(trajectory, dict):
            return trajectory

        ground_truth = data.get("answer", "")
        for interaction in trajectory.values():
            if not hasattr(interaction, "to_tensor_dict"):
                continue
            tensor_dict = interaction.to_tensor_dict()
            tensor_dict["ground_truth"] = [ground_truth]
            interaction._cache = tensor_dict
        return trajectory
