from __future__ import annotations

from pathlib import Path

from searchagent.plugins.browsecomp_plus import (
    BrowseCompPlusSource,
    preprocess_browsecomp_plus_record,
)
from searchagent.plugins.indexing import apply_embedding_prompt
from searchagent.plugins.local_wiki import WikiDumpSource, preprocess_wiki_page

FIXTURES = Path(__file__).parent / "fixtures" / "plugin_sources"


def test_preprocess_wiki_page_extracts_text_links_and_url() -> None:
    document = preprocess_wiki_page(
        title="Ada Lovelace",
        wikitext="'''Ada''' wrote about [[Analytical Engine|the engine]].\n== References ==\nignored",
    )

    assert document is not None
    assert document.id == "Ada_Lovelace"
    assert document.title == "Ada Lovelace"
    assert "the engine" in document.text
    assert "References" not in document.text
    assert document.links == [{"text": "the engine", "target": "Analytical Engine"}]


def test_wiki_dump_source_reads_mediawiki_xml() -> None:
    documents = list(WikiDumpSource(FIXTURES / "wiki.xml").iter_documents())

    assert len(documents) == 1
    assert documents[0].title == "Search Agent"
    assert documents[0].text == "Useful Benchmark page."


def test_preprocess_browsecomp_plus_record_normalizes_fields() -> None:
    document = preprocess_browsecomp_plus_record(
        {
            "docid": "doc-1",
            "title": "BrowseComp Plus",
            "contents": "A benchmark corpus.",
            "url": "https://example.test/doc-1",
            "extra": "kept",
        }
    )

    assert document is not None
    assert document.id == "doc-1"
    assert document.text == "A benchmark corpus."
    assert document.metadata["source"] == "browsecomp_plus"
    assert document.metadata["extra"] == "kept"


def test_browsecomp_plus_source_reads_jsonl() -> None:
    documents = list(BrowseCompPlusSource(FIXTURES / "bcp.jsonl").iter_documents())

    assert len(documents) == 1
    assert documents[0].url == "browsecomp-plus://1"


def test_apply_embedding_prompt_supports_qwen3() -> None:
    prompted = apply_embedding_prompt("content", "qwen3")

    assert "retrieve relevant passages" in prompted
    assert prompted.endswith("Passage:content")
