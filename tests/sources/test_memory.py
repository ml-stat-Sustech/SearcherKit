import asyncio

from searchagent.sources import Document, SourceConfig, build_source
from searchagent.sources.base import DataSource
from searchagent.sources.memory import MemorySource


def _documents() -> list[Document]:
    return [
        Document(id="doc-1", title="SearchAgent", text="pluggable runtime"),
        Document(id="doc-2", title="Other", text="unrelated text"),
    ]


def _direct_source() -> DataSource:
    return MemorySource(documents=_documents())


def _config_source() -> DataSource:
    return build_source(
        config=SourceConfig(
            type="memory",
            documents=_documents(),
        )
    )


def test_search() -> None:
    async def run() -> None:
        for source_factory in (_direct_source, _config_source):
            source = source_factory()

            results = await source.search("searchagent runtime", top_k=1)

            assert len(results) == 1
            assert results[0].document.id == "doc-1"
            assert results[0].score == 2.0

    asyncio.run(run())


def test_fetch() -> None:
    async def run() -> None:
        for source_factory in (_direct_source, _config_source):
            source = source_factory()

            document = await source.fetch("doc-1")

            assert document.id == "doc-1"
            assert document.title == "SearchAgent"
            assert document.text == "pluggable runtime"

    asyncio.run(run())
