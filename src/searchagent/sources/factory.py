"""Factories for named data source adapters."""

from __future__ import annotations
from typing import overload

from searchagent.common.config import import_from_path

from .base import DataSource, SourceConfig

_SOURCE_CFG_MAP = {}

def add_source_cfg(name: str, source_cfg: SourceConfig):
    _SOURCE_CFG_MAP[name] = source_cfg

def get_source_cfg(name: str) -> SourceConfig:
    return _SOURCE_CFG_MAP[name]

@overload
def build_source(name: str):
    ...

@overload
def build_source(*, config: SourceConfig) -> DataSource:
    """Build a single data source from *config*.

    Dispatches on ``config.type`` using the source registry.  When *type* is
    ``"custom"``, the class is loaded from ``config.target`` and verified
    against the :class:`DataSource` contract.
    """
    ...

def build_source(name: str | None = None, *, config: SourceConfig | None = None) -> DataSource:
    if name:
        if name not in _SOURCE_CFG_MAP:
            raise ValueError(
                f"unknown source name {name!r}; available sources: "
                f"{sorted(_SOURCE_CFG_MAP.keys())}"
            )
        return build_source(config=_SOURCE_CFG_MAP[name])
    if not config:
        raise ValueError(
            "must specify either name or config to build a source"
        )
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
    if source_type == "web":
        from .web import WebSource

        return WebSource(config=config)
    if source_type == "local_file":
        from .local_file import LocalFileSource

        return LocalFileSource(config=config)
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
