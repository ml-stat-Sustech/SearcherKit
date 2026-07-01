"""Command dispatcher for OpenSeeker BrowseComp Plus SFT evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OpenSeeker-style BrowseComp Plus SFT evaluation.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("generate", help="Generate answers")
    subparsers.add_parser("judge", help="Judge generated answers")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "generate":
        import generate

        return generate.main(argv[1:])
    if argv and argv[0] == "judge":
        import judge

        return judge.main(argv[1:])
    parser = build_parser()
    parser.parse_args(argv)
    return 2


if __name__ == "__main__":
    sys.exit(main())
