from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from searcherkit.cli import main as cli_main


EVALUATE_BASE_URL = "https://judge.example.test/v1"


def _judge_response() -> httpx.Response:
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
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "extracted_final_answer": "SearcherKit",
                                "correct_answer": "SearcherKit",
                                "reasoning": (
                                    "The final answer matches the expected answer."
                                ),
                                "correct": True,
                                "confidence": 100,
                            }
                        ),
                        "reasoning_content": "mock thinking",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            },
        },
    )


def test_evaluate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "history"
    output_dir = tmp_path / "eval"
    input_dir.mkdir()
    (input_dir / "000000.json").write_text(
        json.dumps(
            {
                "input": "What runtime is this test about?",
                "answer": "SearcherKit",
                "history": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": r"\boxed{SearcherKit}"},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", EVALUATE_BASE_URL)

    with respx.mock(assert_all_called=True) as router:
        judge_route = router.post(f"{EVALUATE_BASE_URL}/chat/completions").mock(
            return_value=_judge_response()
        )

        exit_code = cli_main.main(
            [
                "evaluate",
                str(input_dir),
                str(output_dir),
                "--max-concurrency",
                "1",
            ]
        )

    assert exit_code == 0
    assert len(judge_route.calls) == 1
    judge_payload = json.loads(judge_route.calls[0].request.content)
    assert judge_payload["model"] == "qwen3-32b"
    assert "SearcherKit" in judge_payload["messages"][1]["content"]

    result = json.loads((output_dir / "000000.json").read_text(encoding="utf-8"))
    assert result["correct"] is True
    assert result["confidence"] == 100
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["evaluated"] == 1
    assert summary["correct"] == 1
    assert summary["accuracy"] == 1.0
