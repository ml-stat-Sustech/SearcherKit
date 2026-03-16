from __future__ import annotations

import asyncio
import sys

import hydra
from omegaconf import DictConfig

import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

from webagent.log import get_logger, setup_logger
from webagent.runtime.agent_runner import AgentRunner
from webagent.runtime.evaluate import evaluate_main

logger = get_logger(__name__)

async def _run(cfg: DictConfig) -> None:
    agent_cfg = cfg.get("agent")
    runner = AgentRunner(agent_config=agent_cfg, max_concurrency=cfg.get("max_concurrency"))
    await runner.run(cfg=cfg)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logger()
    asyncio.run(_run(cfg))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        sys.argv.pop(1)
        evaluate_main()
    else:
        main()
