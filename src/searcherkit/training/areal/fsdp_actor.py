from __future__ import annotations

from areal.engine import FSDPPPOActor

from searcherkit.training.areal.actor import SearchAgentPPOActor


class FSDPSearchAgentPPOActor(FSDPPPOActor):
    def __init__(self, config):
        super(FSDPPPOActor, self).__init__(config)
        self.actor = SearchAgentPPOActor(config, self)
