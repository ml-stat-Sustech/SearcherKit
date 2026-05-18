from areal.engine import FSDPPPOActor
from .igpo_actor import IGPOActor


class FSDPIGPOActor(FSDPPPOActor):
    def __init__(self, config):
        super().__init__(config)
        self.actor = IGPOActor(config, self)
