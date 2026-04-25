"""Data source abstractions and backend adapters."""

from __future__ import annotations

from .base import DataSource, Document, SearchResult
from .elasticsearch import ElasticsearchSource
from .factory import SourceConfig, build_source, build_sources
from .memory import MemorySource

__all__ = [
    "DataSource",
    "Document",
    "ElasticsearchSource",
    "MemorySource",
    "SearchResult",
    "SourceConfig",
    "build_source",
    "build_sources",
]
