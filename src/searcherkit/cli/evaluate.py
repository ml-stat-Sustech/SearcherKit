from __future__ import annotations

from typing import Sequence

from searcherkit.runtime.evaluate import evaluate_main


def main(argv: Sequence[str] | None = None) -> int:
    evaluate_main(argv, prog="searcher evaluate")
    return 0
