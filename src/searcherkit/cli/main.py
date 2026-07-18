from __future__ import annotations

import argparse
import asyncio
import sys
import warnings
from typing import Sequence


def _configure_event_loop() -> None:
    """Use uvloop when available, otherwise retain asyncio's default loop."""
    try:
        import uvloop
    except ImportError:
        warnings.warn(
            "uvloop is not installed; using Python's default asyncio event loop. "
            "Install the optional uvloop dependency for improved performance.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="searcher",
        description="Run SearcherKit, evaluate outputs, and manage bundled plugins.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run", help="Run an agent config")
    subparsers.add_parser("evaluate", help="Evaluate saved run records")
    subparsers.add_parser("plugins", help="List or run plugin utilities")
    subparsers.add_parser("inspect", help="Recursively validate config fields against structured types")
    subparsers.add_parser("tui", help="Run ad hoc queries in a terminal UI")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _configure_event_loop()
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)
    if not argv or argv[0] in {"-h", "--help"}:
        build_parser().parse_args(argv)
        return 0

    command = argv[0]
    remaining = argv[1:]

    if command == "run":
        from searcherkit.cli.run import main as run_main

        return run_main(remaining)
    if command == "evaluate":
        from searcherkit.cli.evaluate import main as evaluate_main

        return evaluate_main(remaining)
    if command == "plugins":
        from searcherkit.cli.plugins import main as plugins_main

        return plugins_main(remaining)
    if command == "inspect":
        from searcherkit.cli.inspect import main as inspect_main

        return inspect_main(remaining)
    if command == "tui":
        from searcherkit.interfaces.tui.cli import main as tui_main

        return tui_main(remaining)
    parser = build_parser()
    parser.error(f"unknown command: {command}")
    return 2
