from __future__ import annotations

import asyncio

import hydra
from omegaconf import DictConfig

import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

from webagent.runtime.agent_runner import AgentRunner


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    ageng_cfg = cfg.get("agent")
    runner = AgentRunner(agent_config=ageng_cfg)
    # TODO

if __name__ == "__main__":
    main()
