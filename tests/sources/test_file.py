import asyncio
from collections.abc import Callable

import pytest

from searchagent.sources import DataSource, SourceConfig, SourceError, build_source
from searchagent.sources.file import FileSource


def _write_files(root_path) -> None:
    (root_path / "README.md").write_text("Project overview", encoding="utf-8")
    nested = root_path / "docs"
    nested.mkdir()
    (nested / "SearchAgent Notes.txt").write_text(
        "Nested SearchAgent notes\nAnother searchagent line",
        encoding="utf-8",
    )
    (nested / "other.txt").write_text("Other text", encoding="utf-8")


def _direct_source(root_path) -> DataSource:
    return FileSource(root_path=root_path)


def _config_source(root_path) -> DataSource:
    return build_source(
        config=SourceConfig(
            type="file",
            root_path=str(root_path),
        )
    )


@pytest.mark.parametrize("source_factory", [_direct_source, _config_source])
def test_search_matches_file_contents_in_subfolders(
    tmp_path,
    source_factory: Callable[..., DataSource],
) -> None:
    _write_files(tmp_path)

    async def run() -> None:
        source = source_factory(tmp_path)

        results = await asyncio.wait_for(
            source.search("searchagent", top_k=5),
            timeout=5,
        )

        assert [result.document.id for result in results] == [
            "docs/SearchAgent Notes.txt"
        ]
        assert results[0].document.title == "SearchAgent Notes.txt"
        assert results[0].document.text == (
            "Nested SearchAgent notes\nAnother searchagent line"
        )
        assert results[0].snippet == (
            "docs/SearchAgent Notes.txt:1:Nested SearchAgent notes\n"
            "docs/SearchAgent Notes.txt:2:Another searchagent line"
        )
        assert results[0].metadata == {"source": "file"}

    asyncio.run(run())


def test_search_does_not_match_filename_only(tmp_path) -> None:
    _write_files(tmp_path)

    async def run() -> None:
        source = FileSource(root_path=tmp_path)

        results = await asyncio.wait_for(
            source.search("README", top_k=5),
            timeout=5,
        )

        assert results == []

    asyncio.run(run())


def test_search_limits_results_by_file_count(tmp_path) -> None:
    (tmp_path / "first.txt").write_text("shared needle", encoding="utf-8")
    (tmp_path / "second.txt").write_text("shared needle", encoding="utf-8")

    async def run() -> None:
        source = FileSource(root_path=tmp_path)

        results = await asyncio.wait_for(
            source.search("needle", top_k=1),
            timeout=5,
        )

        assert len(results) == 1
        assert results[0].document.id in {"first.txt", "second.txt"}
        assert results[0].metadata == {"source": "file"}

    asyncio.run(run())


def test_search_raises_source_error_when_rg_is_missing(tmp_path, monkeypatch) -> None:
    _write_files(tmp_path)
    monkeypatch.setattr("searchagent.sources.file.shutil.which", lambda name: None)

    async def run() -> None:
        source = FileSource(root_path=tmp_path)

        with pytest.raises(SourceError, match="rg executable not found"):
            await asyncio.wait_for(source.search("SearchAgent"), timeout=5)

    asyncio.run(run())


@pytest.mark.parametrize("source_factory", [_direct_source, _config_source])
def test_fetch_opens_file_within_root(
    tmp_path,
    source_factory: Callable[..., DataSource],
) -> None:
    _write_files(tmp_path)

    async def run() -> None:
        source = source_factory(tmp_path)

        document = await source.fetch("docs/SearchAgent Notes.txt")

        assert document.id == "docs/SearchAgent Notes.txt"
        assert document.title == "SearchAgent Notes.txt"
        assert document.text == "Nested SearchAgent notes\nAnother searchagent line"
        assert document.metadata == {
            "path": "docs/SearchAgent Notes.txt",
            "source": "file",
        }

    asyncio.run(run())


def test_fetch_rejects_paths_outside_root_with_source_error(tmp_path) -> None:
    _write_files(tmp_path)
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside", encoding="utf-8")

    async def run() -> None:
        source = FileSource(root_path=tmp_path)

        with pytest.raises(SourceError, match="outside root_path"):
            await source.fetch("../outside.txt")

    asyncio.run(run())


def test_fetch_returns_cannot_open_when_file_cannot_be_read(tmp_path, monkeypatch) -> None:
    _write_files(tmp_path)
    source = FileSource(root_path=tmp_path)

    def fail_read_text(
        self,
        *,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        raise OSError("permission denied")

    monkeypatch.setattr(type(tmp_path / "README.md"), "read_text", fail_read_text)

    async def run() -> None:
        document = await source.fetch("README.md")

        assert document.text == "Cannot open README.md"

    asyncio.run(run())


def test_search_returns_cannot_open_when_file_cannot_be_read(
    tmp_path,
    monkeypatch,
) -> None:
    _write_files(tmp_path)
    source = FileSource(root_path=tmp_path)

    def fail_read_text(
        self,
        *,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        raise OSError("permission denied")

    monkeypatch.setattr(type(tmp_path / "README.md"), "read_text", fail_read_text)

    async def run() -> None:
        results = await asyncio.wait_for(
            source.search("Project overview", top_k=1),
            timeout=5,
        )

        assert results[0].document.id == "README.md"
        assert results[0].document.text == "Cannot open README.md"

    asyncio.run(run())


def test_fetch_raises_source_error_when_file_does_not_exist(tmp_path) -> None:
    _write_files(tmp_path)

    async def run() -> None:
        source = FileSource(root_path=tmp_path)

        with pytest.raises(SourceError, match="document not found"):
            await source.fetch("missing.txt")

    asyncio.run(run())
