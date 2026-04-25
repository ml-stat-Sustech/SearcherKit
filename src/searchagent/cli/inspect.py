from __future__ import annotations

import argparse
from typing import Sequence

from omegaconf import OmegaConf

from searchagent.cli.config import compose_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="searchagent inspect",
        description="Inspect SearchAgent configuration and environment.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    config = subparsers.add_parser("config", help="Print the composed config")
    config.add_argument("--config-path", default=None)
    config.add_argument("--config-name", default="config")
    config.add_argument("overrides", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "config":
        cfg = compose_config(
            config_path=args.config_path,
            config_name=args.config_name,
            overrides=args.overrides,
        )
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return 0
    raise ValueError(f"unknown inspect command: {args.command}")
