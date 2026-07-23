from __future__ import annotations

from areal.engine import MegatronPPOActor

from searcherkit.training.areal.actor import SearchAgentPPOActor


class MegatronSearchAgentPPOActor(MegatronPPOActor):
    def __init__(self, config):
        super(MegatronPPOActor, self).__init__(config)
        self.actor = SearchAgentPPOActor(config, self)
