from __future__ import annotations

import sys


def main(args: list[str]) -> None:
    from searcherkit.training.slime.train_dist import main as slime_main

    slime_main(args)


if __name__ == "__main__":
    main(sys.argv[1:])
