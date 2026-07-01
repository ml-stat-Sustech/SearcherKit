"""CLI for plugin data conversion commands."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from searchagent.plugins.conversion.config import load_config
from searchagent.plugins.conversion.convert import convert_file


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Convert plugin datasets into training formats.")
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--config-name", default="openseeker_ms_swift")
    parser.add_argument("--max-records", type=int, default=None, help="Override max_records for this invocation")
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra-style overrides",
    )
    return parser


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> int:
    args = build_parser(prog=prog).parse_args(argv)
    config = load_config(
        config_path=args.config_path,
        config_name=args.config_name,
        overrides=args.overrides,
    )
    if args.max_records is not None:
        config.max_records = args.max_records
    stats = convert_file(
        config.input_path,
        config.output_path,
        max_records=config.max_records,
    )
    print(json.dumps(stats.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0
