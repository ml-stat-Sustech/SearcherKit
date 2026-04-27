"""Factories for named data source adapters."""

from __future__ import annotations

from searchagent.common.config import import_from_path

from .base import DataSource, SourceConfig


def build_source(config: SourceConfig) -> DataSource:
    """Build a single data source from *config*.

    Dispatches on ``config.type`` using the source registry.  When *type* is
    ``"custom"``, the class is loaded from ``config.target`` and verified
    against the :class:`DataSource` contract.
    """
    source_type = config.type
    if not source_type:
        raise ValueError(
            "source config must have a non-empty 'type' field"
        )
    if source_type == "elasticsearch":
        from .elasticsearch import ElasticsearchSource

        return ElasticsearchSource(config=config)
    if source_type == "memory":
        from .memory import MemorySource

        return MemorySource(config=config)
    if source_type == "custom":
        if not config.target:
            raise ValueError(
                "source type 'custom' requires a non-empty 'target' field "
                "(e.g. pkg://my.package:MySource or file:///path/to/file.py:MySource)"
            )
        cls = import_from_path(config.target)
        if not issubclass(cls, DataSource):
            raise TypeError(
                f"custom source class {cls!r} must be a subclass of DataSource"
            )
    return cls(config=config)
