from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from webagent.log import get_logger, setup_logger
from webagent.agent.agent import Agent
from webagent.agent.single_turn_agent import SingleTurnAgent
from webagent.llm.client import OpenAIClient
from webagent.llm.parser import QwenParser

logger = get_logger(__name__)

_ANSWER_PATTERN = re.compile(r"<answer>(?P<answer>.*?)</answer>", re.DOTALL)


def _extract_answer_content(content: Any, *, record_path: Path) -> str:
    if not isinstance(content, str):
        raise ValueError(f"record {record_path} last assistant message missing string content")

    matched = _ANSWER_PATTERN.search(content)
    if matched is None:
        raise ValueError(f"record {record_path} assistant content missing <answer>...</answer> block")

    answer = matched.group("answer").strip()
    if not answer:
        raise ValueError(f"record {record_path} assistant <answer> block is empty")
    return answer

def build_agent() -> Agent:
    system_prompt = """Output exactly one valid JSON object and nothing else.

Schema:
{
  "extracted_final_answer": "string",
  "correct_answer": "string",
  "reasoning": "string",
  "correct": true,
  "confidence": 100
}
"""
    return SingleTurnAgent(
        llm_client=OpenAIClient(
            model="qwen3-32b",
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            default_kwargs={
                "temperature": 0.7,
                "top_p": 0.8,
                "max_tokens": 4096,
                "extra_body": {
                    "top_k": 20,
                    "chat_template_kwargs": {
                        "enable_thinking": False,
                    },
                },
            },
        ),
        parser=QwenParser(upstream_parsed=False),
        system_prompt=system_prompt
    )

def _build_judge_prompt(question: str, response: str, correct_answer: Any) -> str:
    return """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. 

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.

correct: Output true if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Output false otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.
""".strip().format(
        question=question,
        response=response,
        correct_answer=correct_answer,
    )


def _serialize_history(history: list[Any]) -> list[Any]:
    serialized: list[Any] = []
    for message in history:
        if hasattr(message, "__dataclass_fields__"):
            serialized.append(asdict(message))
        else:
            serialized.append(message)
    return serialized


def _save_evaluation_result(
    path: Path,
    *,
    question: str,
    response: str,
    correct_answer: Any,
    history: list[Any],
) -> None:
    path.write_text(
        json.dumps(
            {
                "question": question,
                "response": response,
                "correct_answer": correct_answer,
                "history": _serialize_history(history),
            },
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )


def _normalize_judge_history(history: list[Any], *, record_path: Path) -> list[Any]:
    serialized = _serialize_history(history)
    if not serialized:
        raise ValueError(f"evaluation for {record_path} returned empty history")

    last_message = serialized[-1]
    if not isinstance(last_message, dict):
        raise ValueError(f"evaluation for {record_path} returned invalid last history item")
    if last_message.get("role") != "assistant":
        raise ValueError(f"evaluation for {record_path} did not end with assistant")

    content = last_message.get("content")
    if not isinstance(content, str):
        last_message["content"] = json.dumps(
            {
                "parse_error": True,
                "error": "judge_output_missing_string_content",
            },
            ensure_ascii=False,
        )
        return serialized

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        last_message["content"] = json.dumps(
            {
                "parse_error": True,
                "error": "judge_output_invalid_json",
                "raw_content": content,
            },
            ensure_ascii=False,
        )
        return serialized

    last_message["content"] = json.dumps(parsed, ensure_ascii=False)
    return serialized


def _parse_judge_result(record_path: Path) -> dict[str, Any]:
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    history = payload.get("history", [])
    if not isinstance(history, list) or not history:
        raise ValueError(f"evaluation record {record_path} missing non-empty history")

    last_message = history[-1]
    if not isinstance(last_message, dict):
        raise ValueError(f"evaluation record {record_path} last history item is not an object")
    if last_message.get("role") != "assistant":
        raise ValueError(f"evaluation record {record_path} last message role is not assistant")

    content = last_message.get("content")
    if not isinstance(content, str):
        raise ValueError(f"evaluation record {record_path} last assistant message missing string content")

    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"evaluation record {record_path} assistant content is not valid JSON") from exc

    if not isinstance(result, dict):
        raise ValueError(f"evaluation record {record_path} assistant JSON is not an object")
    if result.get("parse_error") is True:
        raise ValueError(f"evaluation record {record_path} judge output has parse_error")
    if not isinstance(result["correct"], bool):
        raise ValueError(f"evaluation record {record_path} field 'correct' must be a boolean")
    if not isinstance(result["confidence"], int) or isinstance(result["confidence"], bool):
        raise ValueError(f"evaluation record {record_path} field 'confidence' must be an integer")
    return result


def _collect_stats(output_dir: str | os.PathLike[str]) -> dict[str, Any]:
    total = 0
    valid = 0
    invalid = 0
    correct = 0
    incorrect = 0
    confidence_sum = 0

    for record_path in sorted(Path(output_dir).glob("*.json")):
        if record_path.name == "summary.json":
            continue
        total += 1
        try:
            judge_result = _parse_judge_result(record_path)
        except ValueError as exc:
            invalid += 1
            logger.warning("Invalid evaluation result path=%s error=%s", record_path, exc)
            continue

        valid += 1
        confidence_sum += judge_result["confidence"]
        if judge_result["correct"]:
            correct += 1
        else:
            incorrect += 1

    return {
        "evaluated": total,
        "valid": valid,
        "invalid": invalid,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": (correct / valid) if valid else 0.0,
        "avg_confidence": (confidence_sum / valid) if valid else 0.0,
    }


async def _evaluate_one(
    *,
    record_path: Path,
    output_path: Path,
) -> Path:
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    history = payload.get("history", [])
    if not isinstance(history, list) or not history:
        raise ValueError(f"record {record_path} missing non-empty history")

    last_message = history[-1]
    if not isinstance(last_message, dict):
        raise ValueError(f"record {record_path} last history item is not an object")
    if last_message.get("role") != "assistant":
        raise ValueError(f"record {record_path} last message role is not assistant")

    question = payload.get("input", "")
    response = _extract_answer_content(last_message.get("content"), record_path=record_path)
    correct_answer = payload.get("answer")
    prompt = _build_judge_prompt(question, response, correct_answer)

    agent = build_agent()
    judge_history = await agent.run(prompt)
    _save_evaluation_result(
        output_path,
        question=question,
        response=response,
        correct_answer=correct_answer,
        history=_normalize_judge_history(judge_history, record_path=record_path),
    )
    return output_path


async def _run_evaluate(
    *,
    input_dir: str,
    output_dir: str,
    max_concurrency: int | None,
) -> None:
    src_dir = Path(input_dir)
    dst_dir = Path(output_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None
    record_paths = sorted(src_dir.glob("*.json"))

    async def _run_limited(record_path: Path, output_path: Path) -> Path:
        if semaphore is None:
            return await _evaluate_one(record_path=record_path, output_path=output_path)
        async with semaphore:
            return await _evaluate_one(record_path=record_path, output_path=output_path)

    tasks: list[asyncio.Task[Path]] = []
    pbar = tqdm(total=0)

    for record_path in record_paths:
        output_path = dst_dir / record_path.name
        if output_path.exists():
            continue

        task = asyncio.create_task(_run_limited(record_path, output_path))

        def _on_done(_done_task: asyncio.Future[Path]) -> None:
            pbar.update(1)

        task.add_done_callback(_on_done)
        tasks.append(task)

    pbar.reset(total=len(tasks))
    results = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []

    failed_requests = 0
    for result in results:
        if isinstance(result, Exception):
            failed_requests += 1
            logger.exception("Evaluation task failed", exc_info=result)

    stats = _collect_stats(dst_dir)
    stats["source_total"] = len(record_paths)
    stats["submitted"] = len(tasks)
    stats["skipped_existing"] = len(record_paths) - len(tasks)
    stats["failed_requests"] = failed_requests

    summary_path = dst_dir / "summary.json"
    summary_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def evaluate_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved agent run records with an LLM judge.")
    parser.add_argument("input_dir", help="Directory containing agent run record JSON files.")
    parser.add_argument("output_dir", help="Directory to write evaluation result JSON files.")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent judge requests.",
    )
    args = parser.parse_args()

    setup_logger()
    asyncio.run(
        _run_evaluate(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            max_concurrency=args.max_concurrency,
        )
    )
