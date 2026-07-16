from __future__ import annotations

from areal.engine import FSDPPPOActor

from searcherkit.training.areal.igpo.igpo_actor import IGPOActor


class FSDPIGPOActor(FSDPPPOActor):
    def __init__(self, config):
        super(FSDPPPOActor, self).__init__(config)
        self.actor = IGPOActor(config, self)
