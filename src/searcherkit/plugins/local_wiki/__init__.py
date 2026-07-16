"""Local wiki indexing and retrieval toolkit."""

from __future__ import annotations

from .source import WikiDumpSource, preprocess_wiki_page

__all__ = ["WikiDumpSource", "preprocess_wiki_page"]
