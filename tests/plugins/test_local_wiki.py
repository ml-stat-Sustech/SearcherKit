from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from searcherkit.plugins.indexing import IndexDocument
from searcherkit.plugins.local_wiki import WikiDumpSource, preprocess_wiki_page
from searcherkit.plugins.local_wiki import deploy_elasticsearch


FIXTURE_DIR = Path("tests/fixtures/plugin_sources")
WIKI_DUMP_PATH = FIXTURE_DIR / "wiki.xml"


def test_preprocess_wiki_page_extracts_text_links_and_url() -> None:
    document = preprocess_wiki_page(
        title="Ada Lovelace",
        wikitext="'''Ada''' wrote about [[Analytical Engine|the engine]].\n== References ==\nignored",
    )

    assert document is not None
    assert document.id == "Ada_Lovelace"
    assert document.title == "Ada Lovelace"
    assert document.url == "https://en.wikipedia.org/wiki/Ada_Lovelace"
    assert "the engine" in document.text
    assert "References" not in document.text
    assert document.links == [{"text": "the engine", "target": "Analytical Engine"}]
    assert document.metadata == {"source": "wiki"}


def test_wiki_dump_source_reads_mediawiki_xml() -> None:
    documents = list(WikiDumpSource(WIKI_DUMP_PATH).iter_documents())

    assert len(documents) == 1
    assert documents[0].id == "Search_Agent"
    assert documents[0].title == "Search Agent"
    assert documents[0].text == "Useful Benchmark page."
    assert documents[0].links == [{"text": "Benchmark", "target": "Benchmark"}]


def test_local_wiki_deploy_passes_documents_to_elasticsearch(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    captured: dict[str, Any] = {}

    def fake_deploy_to_elasticsearch(
        *,
        documents: Iterable[IndexDocument],
        es_host: str,
        index_name: str,
        embedding_model_name: str | None,
        embedding_dim: int | None,
        prompt_strategy: str,
        overwrite: bool,
        batch_size: int,
        embedding_batch_size: int,
        max_text_chars: int,
        shards: int,
        replicas: int,
    ) -> int:
        captured.update(
            {
                "documents": list(documents),
                "es_host": es_host,
                "index_name": index_name,
                "embedding_model_name": embedding_model_name,
                "embedding_dim": embedding_dim,
                "prompt_strategy": prompt_strategy,
                "overwrite": overwrite,
                "batch_size": batch_size,
                "embedding_batch_size": embedding_batch_size,
                "max_text_chars": max_text_chars,
                "shards": shards,
                "replicas": replicas,
            }
        )
        return len(captured["documents"])

    monkeypatch.setattr(deploy_elasticsearch, "deploy_to_elasticsearch", fake_deploy_to_elasticsearch)

    deploy_elasticsearch.main(
        [
            "--wiki_dump_path",
            str(WIKI_DUMP_PATH),
            "--es_host",
            "http://localhost:9200",
            "--index_name",
            "wiki-test",
            "--limit",
            "1",
            "--overwrite",
        ]
    )

    assert [document.title for document in captured["documents"]] == ["Search Agent"]
    assert captured["es_host"] == "http://localhost:9200"
    assert captured["index_name"] == "wiki-test"
    assert captured["embedding_model_name"] is None
    assert captured["embedding_dim"] is None
    assert captured["prompt_strategy"] == "none"
    assert captured["overwrite"] is True
    assert "Indexed 1 wiki documents into wiki-test" in capsys.readouterr().out
