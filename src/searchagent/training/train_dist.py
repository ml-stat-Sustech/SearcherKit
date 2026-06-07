from __future__ import annotations

import sys


def main(args: list[str]) -> None:
    from searchagent.training.train_dist_slime import main as slime_main

    slime_main(args)


if __name__ == "__main__":
    main(sys.argv[1:])
