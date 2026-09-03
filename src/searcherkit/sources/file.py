"""Filesystem data source for full-text search and file fetch."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import overload

from .base import DataSource, Document, SearchResult, SourceConfig, SourceError


_FILE_ERROR_HANDLER = "replace"


class FileSource(DataSource):
    """Search file contents with ripgrep and fetch file contents."""

    @overload
    def __init__(self, *, config: SourceConfig) -> None: ...

    @overload
    def __init__(
        self,
        *,
        root_path: str | os.PathLike[str],
        encoding: str = "utf-8",
    ) -> None: ...

    def __init__(
        self,
        *,
        root_path: str | os.PathLike[str] | None = None,
        encoding: str = "utf-8",
        config: SourceConfig | None = None,
    ) -> None:
        if config is not None:
            root_path = root_path or config.root_path
            encoding = config.file_encoding

        if not root_path:
            raise ValueError("FileSource requires root_path")
        root = Path(root_path).expanduser().resolve()
        if not root.exists():
            raise ValueError(f"FileSource root_path does not exist: {root}")
        if not root.is_dir():
            raise ValueError(f"FileSource root_path must be a directory: {root}")
        if not encoding:
            raise ValueError("file encoding must be non-empty")

        self.root_path = root
        self.encoding = encoding

    async def search(self, query: str, *, top_k: int = 10) -> list[SearchResult]:
        if not query:
            raise ValueError("query must be non-empty")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        matches = await self._search_with_rg(query, top_k=top_k)
        results: list[SearchResult] = []
        for relative_id, hit_lines in matches.items():
            path = self._resolve_document_path(relative_id)
            document = Document(
                id=relative_id,
                title=path.name,
                text=self._read_text_or_cannot_open(path),
                metadata={
                    "path": relative_id,
                    "source": "file",
                },
            )
            results.append(
                SearchResult(
                    document=document,
                    score=1.0,
                    snippet="\n".join(hit_lines),
                    metadata={"source": "file"},
                )
            )
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
                "source": "file",
            },
        )

    async def _search_with_rg(self, query: str, *, top_k: int) -> dict[str, list[str]]:
        if shutil.which("rg") is None:
            raise SourceError("rg executable not found; install ripgrep to use file source search")

        try:
            proc = await asyncio.create_subprocess_exec(
                "rg",
                "--json",
                "--smart-case",
                "--line-number",
                "--",
                query,
                ".",
                cwd=self.root_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise SourceError("failed to run rg for file source search") from exc
        stdout, stderr = await proc.communicate()

        if proc.returncode == 1:
            return {}
        if proc.returncode != 0:
            message = stderr.decode(self.encoding, errors=_FILE_ERROR_HANDLER).strip()
            if not message:
                message = f"rg failed with exit code {proc.returncode}"
            raise SourceError(message)

        results: dict[str, list[str]] = {}
        for raw_line in stdout.splitlines():
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise SourceError("rg returned invalid JSON output") from exc
            if event.get("type") != "match":
                continue
            data = event.get("data", {})
            path = data.get("path", {}).get("text")
            line = data.get("lines", {}).get("text")
            line_number = data.get("line_number")
            if not isinstance(path, str) or not isinstance(line, str):
                continue
            if not isinstance(line_number, int):
                continue

            relative_id = self._relative_rg_path(path)
            if relative_id not in results and len(results) >= top_k:
                continue
            results.setdefault(relative_id, []).append(
                f"{relative_id}:{line_number}:{line.rstrip()}"
            )
        return results

    def _relative_rg_path(self, path_text: str) -> str:
        path = Path(path_text)
        if path.is_absolute():
            resolved = path.expanduser().resolve()
        else:
            resolved = (self.root_path / path).resolve()
        try:
            return resolved.relative_to(self.root_path).as_posix()
        except ValueError as exc:
            raise SourceError(f"rg returned path outside root_path: {path_text!r}") from exc

    def _read_text_or_cannot_open(self, path: Path) -> str:
        try:
            return path.read_text(encoding=self.encoding, errors=_FILE_ERROR_HANDLER)
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
            raise SourceError(f"file document is outside root_path: {document_id!r}") from exc
        if not path.is_file():
            raise SourceError(f"file document not found: {document_id!r}")
        return path
