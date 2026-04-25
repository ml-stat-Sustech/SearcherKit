from __future__ import annotations

from typing import Sequence

from searchagent.runtime.evaluate import evaluate_main


def main(argv: Sequence[str] | None = None) -> int:
    evaluate_main(argv, prog="searchagent evaluate")
    return 0
