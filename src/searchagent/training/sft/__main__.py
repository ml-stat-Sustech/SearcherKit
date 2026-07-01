"""Command line entry for SFT training."""

from __future__ import annotations

import sys

from searchagent.training.sft.train import main


if __name__ == "__main__":
    sys.exit(main())
