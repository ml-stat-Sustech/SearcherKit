from __future__ import annotations

from typing import Any

from great_docs.core import GreatDocs

from logging import getLogger

logger =  getLogger()
def _insert_before_reference(
    self: GreatDocs,
    navbar_items: list[Any],
    link: dict[str, Any],
) -> None:
    """Insert a navbar item before the API reference in any language."""
    del self

    for index, item in enumerate(navbar_items):
        if isinstance(item, dict) and item.get("href") == "reference/index.qmd":
            navbar_items.insert(index, link)
            return

    navbar_items.append(link)


GreatDocs._insert_before_reference = _insert_before_reference

from great_docs.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
