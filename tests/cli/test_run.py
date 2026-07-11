from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from aioresponses import aioresponses
from yarl import URL

from searchagent.cli import main as cli_main


LLM_BASE_URL = "https://llm.example.test/v1"
EMBEDDING_BASE_URL = "https://embedding.example.test/v1"
SUMMARY_BASE_URL = "https://summary.example.test/v1"
ES_BASE_URL = "http://localhost:9200"
INDEX = "cli-docs"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "cli"


def _chat_response(
    *,
    content: str,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
) -> httpx.Response:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
        "reasoning_content": "mock thinking",
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-cli",
            "object": "chat.completion",
            "created": 1_700_000_000,
            "model": "mock-chat",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            },
        },
    )


def _embedding_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, 0.2, 0.3],
                }
            ],
            "model": "mock-embedding",
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
        },
    )


def _summary_response() -> httpx.Response:
    return _chat_response(
        content=json.dumps(
            {
                "evidence": "SearchAgent is a pluggable runtime.",
                "summary": "SearchAgent supports pluggable search-agent runs.",
            }
        )
    )


def _search_response() -> dict[str, Any]:
    return {
        "hits": {
            "hits": [
                {
                    "_index": INDEX,
                    "_id": "doc-1",
                    "_score": 1.0,
                    "_source": {
                        "title": "SearchAgent",
                        "text": "SearchAgent is a pluggable search-agent runtime.",
                        "url": "https://example.test/searchagent",
                    },
                }
            ]
        }
    }


def _elastic_headers() -> dict[str, str]:
    return {"x-elastic-product": "Elasticsearch"}


def _write_dataset(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "question": "What runtime is this test about?",
                "answer": "SearchAgent",
            }
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def dataset_path(tmp_path: Path) -> Path:
    dataset_path = tmp_path / "questions.jsonl"
    _write_dataset(dataset_path)
    return dataset_path


@pytest.fixture
def output_path(tmp_path: Path) -> Path:
    return tmp_path / "outputs"


def _config_args(command: str, dataset_path: Path, output_path: Path) -> list[str]:
    return [
        command,
        "--config-path",
        str(CONFIG_PATH),
        "--config-name",
        "test_config",
        f"dataloader.source={dataset_path.as_posix()}",
        f"output_path={output_path.as_posix()}",
    ]


def test_run(
    dataset_path: Path,
    output_path: Path,
) -> None:
    tool_call = {
        "id": "call_search",
        "type": "function",
        "function": {
            "name": "search",
            "arguments": json.dumps(
                {
                    "query": "SearchAgent runtime",
                    "top_k": 1,
                }
            ),
        },
    }
    search_url = f"{ES_BASE_URL}/{INDEX}/_search"

    with (
        respx.mock(assert_all_called=True) as router,
        aioresponses() as mocked,
    ):
        llm_route = router.post(f"{LLM_BASE_URL}/chat/completions").mock(
            side_effect=[
                _chat_response(
                    content="I should search first.",
                    tool_calls=[tool_call],
                    finish_reason="tool_calls",
                ),
                _chat_response(content=r"\boxed{SearchAgent}"),
            ]
        )
        embedding_route = router.post(f"{EMBEDDING_BASE_URL}/embeddings").mock(
            return_value=_embedding_response()
        )
        summary_route = router.post(f"{SUMMARY_BASE_URL}/chat/completions").mock(
            return_value=_summary_response()
        )
        mocked.post(search_url, payload=_search_response(), headers=_elastic_headers())

        exit_code = cli_main.main(_config_args("run", dataset_path, output_path))

    assert exit_code == 0
    assert len(llm_route.calls) == 2
    assert len(embedding_route.calls) == 1
    assert len(summary_route.calls) == 1

    first_llm_payload = json.loads(llm_route.calls[0].request.content)
    assert first_llm_payload["model"] == "mock-chat"
    assert first_llm_payload["tools"][0]["function"]["name"] == "search"
    assert first_llm_payload["messages"][1]["content"] == (
        "Question: What runtime is this test about?"
    )

    embedding_payload = json.loads(embedding_route.calls[0].request.content)
    assert embedding_payload["model"] == "mock-embedding"
    assert embedding_payload["input"] == "query: SearchAgent runtime"
    assert embedding_payload["encoding_format"] == "float"

    es_request = mocked.requests[("POST", URL(search_url))][0]
    es_payload = json.loads(es_request.kwargs["data"])
    assert es_payload["size"] == 1
    assert es_payload["knn"] == {
        "field": "text_vector",
        "query_vector": [0.1, 0.2, 0.3],
        "k": 1,
        "num_candidates": 1,
    }
    assert "query" not in es_payload

    summary_payload = json.loads(summary_route.calls[0].request.content)
    assert summary_payload["model"] == "mock-summary"
    assert summary_payload["response_format"] == {"type": "json_object"}
    assert "SearchAgent is a pluggable search-agent runtime." in (
        summary_payload["messages"][0]["content"]
    )

    summary = json.loads((output_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["total"] == 1
    assert summary["completed"] == 1
    assert summary["failed"] == 0

    record = json.loads(
        (output_path / "history" / "000000.json").read_text(encoding="utf-8")
    )
    assert record["input"] == "What runtime is this test about?"
    assert record["answer"] == "SearchAgent"
    assert record["history"][-1]["content"] == r"\boxed{SearchAgent}"
    tool_message = next(message for message in record["history"] if message["role"] == "tool")
    assert "SearchAgent supports pluggable search-agent runs." in (
        next(iter(tool_message["tool_responses"].values()))
    )
