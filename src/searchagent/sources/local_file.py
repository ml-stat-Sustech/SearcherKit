"""Local filesystem data source for filename search and file fetch."""

from __future__ import annotations

import os
import re
from pathlib import Path
from re import Pattern
from typing import overload

from .base import DataSource, Document, SearchResult, SourceConfig, SourceError


class LocalFileSource(DataSource):
    """Search local files by relative filename and fetch file contents."""

    @overload
    def __init__(self, *, config: SourceConfig) -> None: ...

    @overload
    def __init__(
        self,
        *,
        root_path: str | os.PathLike[str],
        encoding: str = "utf-8",
        errors: str = "replace",
    ) -> None: ...

    def __init__(
        self,
        *,
        root_path: str | os.PathLike[str] | None = None,
        encoding: str = "utf-8",
        errors: str = "replace",
        config: SourceConfig | None = None,
    ) -> None:
        if config is not None:
            root_path = root_path or config.root_path
            encoding = config.local_file_encoding
            errors = config.local_file_errors

        if not root_path:
            raise ValueError("LocalFileSource requires root_path")
        root = Path(root_path).expanduser().resolve()
        if not root.exists():
            raise ValueError(f"LocalFileSource root_path does not exist: {root}")
        if not root.is_dir():
            raise ValueError(f"LocalFileSource root_path must be a directory: {root}")
        if not encoding:
            raise ValueError("local file encoding must be non-empty")
        if not errors:
            raise ValueError("local file error handler must be non-empty")

        self.root_path = root
        self.encoding = encoding
        self.errors = errors

    async def search(self, query: str, *, top_k: int = 10) -> list[SearchResult]:
        if not query:
            raise ValueError("query must be non-empty")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        pattern = self._compile_query(query)
        results: list[SearchResult] = []
        for path in self._iter_files():
            relative_id = path.relative_to(self.root_path).as_posix()
            if not pattern.search(relative_id):
                continue
            document = Document(
                id=relative_id,
                title=path.name,
                text=self._read_text_or_cannot_open(path),
                metadata={
                    "path": relative_id,
                    "source": "local_file",
                },
            )
            results.append(
                SearchResult(
                    document=document,
                    score=1.0,
                    snippet=relative_id,
                    metadata={"source": "local_file"},
                )
            )
            if len(results) >= top_k:
                break
        return results

    async def fetch(self, document_id: str, *, goal: str | None = None) -> Document:
        if not document_id:
            raise ValueError("document_id must be non-empty")
        path = self._resolve_document_path(document_id)
        text = self._read_text_or_cannot_open(path)
        relative_id = path.relative_to(self.root_path).as_posix()
        return Document(
            id=relative_id,
            title=path.name,
            text=text,
            metadata={
                "path": relative_id,
                "source": "local_file",
            },
        )

    def _iter_files(self) -> list[Path]:
        return sorted(
            path
            for path in self.root_path.rglob("*")
            if path.is_file() and self._is_within_root(path.resolve())
        )

    def _compile_query(self, query: str) -> Pattern[str]:
        try:
            return re.compile(query, re.IGNORECASE)
        except re.error as exc:
            raise ValueError(f"invalid filename search pattern: {query!r}") from exc

    def _read_text_or_cannot_open(self, path: Path) -> str:
        try:
            return path.read_text(encoding=self.encoding, errors=self.errors)
        except (OSError, UnicodeError):
            return f"Cannot open {path.name}"

    def _resolve_document_path(self, document_id: str) -> Path:
        candidate = Path(document_id)
        if candidate.is_absolute():
            path = candidate.expanduser().resolve()
        else:
            path = (self.root_path / candidate).resolve()

        try:
            path.relative_to(self.root_path)
        except ValueError as exc:
            raise SourceError(f"local file document is outside root_path: {document_id!r}") from exc
        if not path.is_file():
            raise SourceError(f"local file document not found: {document_id!r}")
        return path

    def _is_within_root(self, path: Path) -> bool:
        try:
            path.relative_to(self.root_path)
        except ValueError:
            return False
        return True
