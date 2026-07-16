from __future__ import annotations

from areal import PPOTrainer
from areal.utils.environ import is_single_controller


class IGPOTrainer(PPOTrainer):
    def _create_train_engine(self, actor_config, alloc):
        if alloc.backend != "fsdp":
            raise ValueError(
                f"IGPOTrainer only supports fsdp backend, got {alloc.backend}"
            )

        from searcherkit.training.areal.igpo.fsdp_igpo_actor import FSDPIGPOActor

        if is_single_controller():
            actor = FSDPIGPOActor.as_controller(actor_config, self.scheduler)
        else:
            actor = FSDPIGPOActor(config=actor_config)
        actor.create_process_group(parallel_strategy=alloc.parallel)
        return actor
