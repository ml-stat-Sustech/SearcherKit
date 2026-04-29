from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Sequence

import uvloop

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="searchagent",
        description="Run SearchAgent, evaluate outputs, and manage bundled plugins.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run", help="Run an agent config")
    subparsers.add_parser("evaluate", help="Evaluate saved run records")
    subparsers.add_parser("plugins", help="List or run plugin utilities")
    subparsers.add_parser("inspect", help="Inspect config and environment")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)
    if not argv or argv[0] in {"-h", "--help"}:
        build_parser().parse_args(argv)
        return 0

    command = argv[0]
    remaining = argv[1:]

    if command == "run":
        from searchagent.cli.run import main as run_main

        return run_main(remaining)
    if command == "evaluate":
        from searchagent.cli.evaluate import main as evaluate_main

        return evaluate_main(remaining)
    if command == "plugins":
        from searchagent.cli.plugins import main as plugins_main

        return plugins_main(remaining)
    if command == "inspect":
        from searchagent.cli.inspect import main as inspect_main

        return inspect_main(remaining)
    parser = build_parser()
    parser.error(f"unknown command: {command}")
    return 2
