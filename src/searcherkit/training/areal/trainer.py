from __future__ import annotations

from areal import PPOTrainer
from areal.utils.environ import is_single_controller


class SearchAgentPPOTrainer(PPOTrainer):
    def _create_train_engine(self, actor_config, alloc):
        if alloc.backend == "fsdp":
            from searcherkit.training.areal.fsdp_actor import FSDPSearchAgentPPOActor

            actor_cls = FSDPSearchAgentPPOActor
        elif alloc.backend == "megatron":
            from searcherkit.training.areal.megatron_actor import (
                MegatronSearchAgentPPOActor,
            )

            actor_cls = MegatronSearchAgentPPOActor
        else:
            raise ValueError(
                "SearchAgentPPOTrainer only supports fsdp and megatron backends, "
                f"got {alloc.backend}"
            )

        if is_single_controller():
            actor = actor_cls.as_controller(actor_config, self.scheduler)
        else:
            actor = actor_cls(config=actor_config)
        actor.create_process_group(parallel_strategy=alloc.parallel)
        return actor
