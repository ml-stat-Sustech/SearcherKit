from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from searchagent.plugins.browsecomp_plus import BrowseCompPlusSource, preprocess_browsecomp_plus_record
from searchagent.plugins.browsecomp_plus import deploy_elasticsearch
from searchagent.plugins.indexing import IndexDocument


FIXTURE_DIR = Path("tests/fixtures/plugin_sources")
BCP_PATH = FIXTURE_DIR / "bcp.jsonl"


def test_preprocess_browsecomp_plus_record_normalizes_fields() -> None:
    document = preprocess_browsecomp_plus_record(
        {
            "docid": "doc-1",
            "text": "---\ntitle: BrowseComp Plus\n---\nA benchmark corpus.",
            "url": "https://example.test/doc-1",
        }
    )

    assert document is not None
    assert document.id == "doc-1"
    assert document.text == "A benchmark corpus."
    assert document.title == "BrowseComp Plus"
    assert document.url == "https://example.test/doc-1"
    assert document.links == []
    assert document.metadata["source"] == "browsecomp_plus"


def test_browsecomp_plus_source_reads_jsonl(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf_datasets"))

    documents = list(BrowseCompPlusSource(BCP_PATH).iter_documents())

    assert len(documents) == 1
    assert documents[0].id == "1"
    assert documents[0].title == "First Document"
    assert documents[0].text == "First document content."
    assert documents[0].url == "https://example.test/1"


def test_browsecomp_plus_deploy_passes_documents_to_elasticsearch(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    captured: dict[str, Any] = {}

    def fake_iter_records(self: BrowseCompPlusSource) -> Iterator[Mapping[str, Any]]:
        assert self.dataset_path == str(BCP_PATH)
        assert self.split == "validation"
        yield {
            "docid": "doc-1",
            "text": "---\ntitle: BrowseComp Plus\n---\nA benchmark corpus.",
            "url": "https://example.test/doc-1",
        }

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

    monkeypatch.setattr(BrowseCompPlusSource, "_iter_records", fake_iter_records)
    monkeypatch.setattr(deploy_elasticsearch, "deploy_to_elasticsearch", fake_deploy_to_elasticsearch)

    deploy_elasticsearch.main(
        [
            "--dataset_path",
            str(BCP_PATH),
            "--split",
            "validation",
            "--es_host",
            "http://localhost:9200",
            "--index_name",
            "bcp-test",
            "--limit",
            "1",
            "--overwrite",
        ]
    )

    assert [document.id for document in captured["documents"]] == ["doc-1"]
    assert captured["es_host"] == "http://localhost:9200"
    assert captured["index_name"] == "bcp-test"
    assert captured["embedding_model_name"] is None
    assert captured["embedding_dim"] is None
    assert captured["prompt_strategy"] == "none"
    assert captured["overwrite"] is True
    assert "Indexed 1 BrowseComp Plus documents into bcp-test" in capsys.readouterr().out
