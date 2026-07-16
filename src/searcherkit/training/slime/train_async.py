from __future__ import annotations

import argparse
import os
import sys

from searcherkit.training.slime.args import (
    add_searcherkit_slime_arguments,
    add_searcherkit_slime_fallback_arguments,
)


def _build_fallback_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m searcherkit.training.slime.train_async",
        description=(
            "SearcherKit slime async training entry. Full slime/Megatron/SGLang "
            "dependencies are required to parse and run the complete training CLI."
        ),
    )
    add_searcherkit_slime_fallback_arguments(parser)
    parser.add_argument(
        "--show-missing-slime-deps",
        action="store_true",
        help="Show this fallback help when slime runtime dependencies are unavailable.",
    )
    return parser


def _handle_missing_slime_deps(exc: ModuleNotFoundError, argv: list[str], runtime_name: str):
    if any(arg in {"-h", "--help", "--show-missing-slime-deps"} for arg in argv):
        print(
            "Full slime CLI dependencies are not importable in this environment "
            f"({exc.name}). Showing SearcherKit-specific fallback help.\n",
            file=sys.stderr,
        )
        parser = _build_fallback_parser()
        parser.parse_args(argv)
        raise SystemExit(0) from exc
    raise RuntimeError(
        "slime training dependencies are not installed or not on PYTHONPATH. "
        "Install the THUDM/slime runtime requirements, including sglang_router, "
        f"sglang, Ray, and Megatron, before running {runtime_name}."
    ) from exc


def _parse_args(argv: list[str] | None):
    current_argv = sys.argv[1:] if argv is None else argv
    try:
        from slime.utils.arguments import parse_args
    except ModuleNotFoundError as exc:
        _handle_missing_slime_deps(exc, current_argv, "async training")
    if argv is None:
        try:
            return parse_args(add_searcherkit_slime_arguments)
        except ModuleNotFoundError as exc:
            _handle_missing_slime_deps(exc, current_argv, "async training")

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *argv]
        try:
            return parse_args(add_searcherkit_slime_arguments)
        except ModuleNotFoundError as exc:
            _handle_missing_slime_deps(exc, current_argv, "async training")
    finally:
        sys.argv = old_argv


def train(args) -> None:
    import ray
    from train_async import train as train_async

    if args.colocate:
        raise ValueError("Colocation is not supported for async slime training.")

    if not ray.is_initialized():
        ray.init(
            address=os.environ.get("RAY_ADDRESS", "auto"),
            ignore_reinit_error=True,
        )

    train_async(args)

def main(argv: list[str] | None = None) -> None:
    train(_parse_args(argv))


if __name__ == "__main__":
    main()
