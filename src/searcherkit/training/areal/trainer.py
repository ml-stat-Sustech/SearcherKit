from __future__ import annotations

from areal import PPOTrainer
from areal.utils.environ import is_single_controller


class SearchAgentPPOTrainer(PPOTrainer):
    def _create_train_engine(self, actor_config, alloc):
        if alloc.backend != "fsdp":
            raise ValueError(
                f"SearchAgentPPOTrainer only supports fsdp backend, got {alloc.backend}"
            )

        from searcherkit.training.areal.fsdp_actor import FSDPSearchAgentPPOActor

        if is_single_controller():
            actor = FSDPSearchAgentPPOActor.as_controller(actor_config, self.scheduler)
        else:
            actor = FSDPSearchAgentPPOActor(config=actor_config)
        actor.create_process_group(parallel_strategy=alloc.parallel)
        return actor
