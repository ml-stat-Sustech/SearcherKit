from __future__ import annotations

import argparse
import asyncio
from typing import Sequence

from omegaconf import OmegaConf

from searcherkit.cli.config import compose_config
from searcherkit.common.log import setup_logger
from searcherkit.runtime.runner import AgentRunner, RunConfig


async def _run_config(cfg) -> None:
    cfg = OmegaConf.merge(OmegaConf.structured(RunConfig), cfg)
    config = OmegaConf.to_object(cfg)
    if not isinstance(config, RunConfig):
        raise ValueError("Invalid config. Please check for extra or missing fields")
    async with AgentRunner(config=config) as runner:
        await runner.run(cfg=config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="searcher run",
        description="Run a SearcherKit batch from a config file.",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="Directory containing the Hydra config. Defaults to the packaged searcherkit config directory.",
    )
    parser.add_argument("--config-name", default="config", help="Config name without .yaml")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Set log level to DEBUG for verbose output.",
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra-style overrides, for example agent.llm_client.model=Qwen3-8B",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = compose_config(
        config_path=args.config_path,
        config_name=args.config_name,
        overrides=args.overrides,
    )
    setup_logger(level="DEBUG" if args.debug else None)
    asyncio.run(_run_config(cfg))
    return 0
