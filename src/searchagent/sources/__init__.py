"""Data source abstractions and backend adapters."""

from __future__ import annotations

from .base import DataSource, Document, SearchResult, SourceConfig
from .elasticsearch import ElasticsearchSource
from .factory import build_source
from .memory import MemorySource

__all__ = [
    "DataSource",
    "Document",
    "ElasticsearchSource",
    "MemorySource",
    "SearchResult",
    "SourceConfig",
    "build_source",
]
