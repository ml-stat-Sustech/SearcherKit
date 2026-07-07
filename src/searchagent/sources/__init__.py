"""Data source abstractions and backend adapters."""

from __future__ import annotations

from .base import DataSource, Document, SearchResult, SourceConfig, SourceError
from .factory import build_source, add_source_cfg
from .local_file import LocalFileSource

__all__ = [
    "DataSource",
    "Document",
    "SearchResult",
    "SourceConfig",
    "SourceError",
    "LocalFileSource",
    "build_source",
    "add_source_cfg",
]
