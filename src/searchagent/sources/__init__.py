"""Data source abstractions and backend adapters."""

from __future__ import annotations

from .base import DataSource, Document, SearchResult, SourceConfig
from .factory import build_source

__all__ = [
    "DataSource",
    "Document",
    "SearchResult",
    "SourceConfig",
    "build_source",
]
