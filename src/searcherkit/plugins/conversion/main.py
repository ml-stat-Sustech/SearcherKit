"""CLI for plugin data conversion commands."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from searcherkit.plugins.conversion.convert import convert_file


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Convert plugin datasets into training formats.")
    parser.add_argument("input_path", help="Input OpenSeeker JSONL dataset path.")
    parser.add_argument("output_path", help="Output MS-Swift JSONL dataset path.")
    parser.add_argument("--max-records", type=int, default=0, help="Maximum records to convert; 0 means no limit.")
    return parser


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> int:
    args = build_parser(prog=prog).parse_args(argv)
    stats = convert_file(
        args.input_path,
        args.output_path,
        max_records=args.max_records,
    )
    print(json.dumps(stats.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0
