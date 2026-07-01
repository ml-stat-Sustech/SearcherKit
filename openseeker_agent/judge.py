"""A/B judge for OpenSeeker-style BCP SFT evaluation outputs."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from config import OpenSeekerBCPJudgeConfig, load_openseeker_bcp_run_config
from prompts import JUDGE_PROMPT_BC_EN


LABEL_RE = re.compile(r"^\s*([AB])\b")
TOOL_CALL_MARK = "<|start|>functions."
API_ERROR_TYPES: tuple[type[BaseException], ...] = ()


def _scorer_clients() -> tuple[list[Any], str]:
    from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

    global API_ERROR_TYPES
    API_ERROR_TYPES = (APIConnectionError, APIError, APITimeoutError)

    scorer_urls_str = os.getenv("SCORER_URLS", "YOUR_SCORER_URL")
    scorer_api_key = os.getenv("SCORER_API_KEY", "YOUR_API_KEY")
    model_name = os.getenv("SCORER_MODEL_NAME", "YOUR_SCORER_MODEL_NAME")
    urls = [url.strip() for url in scorer_urls_str.split(",") if url.strip()]
    if not urls or urls == ["YOUR_SCORER_URL"]:
        raise ValueError("SCORER_URLS is required")
    if model_name == "YOUR_SCORER_MODEL_NAME":
        raise ValueError("SCORER_MODEL_NAME is required")
    return [OpenAI(api_key=scorer_api_key, base_url=url) for url in urls], model_name


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def count_tool_calls(item: Mapping[str, Any]) -> int:
    tool_calls = item.get("tool_calls")
    if isinstance(tool_calls, int) and tool_calls >= 0:
        return tool_calls
    full_traj = item.get("full_traj")
    if isinstance(full_traj, str) and full_traj:
        return full_traj.count(TOOL_CALL_MARK)
    return 0


def parse_judge_label(raw: str | None) -> int | None:
    if raw is None:
        return None
    text = str(raw).strip()
    match = LABEL_RE.match(text)
    if match:
        return 1 if match.group(1) == "A" else 0
    if "</think>" in text:
        after_tag = text.split("</think>", 1)[-1].strip()
        match = LABEL_RE.match(after_tag)
        if match:
            return 1 if match.group(1) == "A" else 0
    return None


def is_clean_01(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value in (0, 1)
    if isinstance(value, float):
        as_int = int(value)
        return as_int in (0, 1) and abs(value - as_int) < 1e-9
    return False


def get_llm_response(clients: list[Any], model_name: str, messages: list[dict[str, str]]) -> str:
    client = random.choice(clients)
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
        stream=False,
        extra_body={"skip_special_tokens": False},
    )
    content = response.choices[0].message.content or ""
    return content.split("<|message|>")[-1].split("<|return|>")[0].strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Judge OpenSeeker-style BCP result_tool*.jsonl outputs.")
    parser.add_argument("--config", type=Path, default=None, help="Path to SFT run YAML config")
    parser.add_argument("--data_path", type=Path, default=None)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--save_path", type=Path, default=None)
    return parser


def resolve_config(args: argparse.Namespace) -> argparse.Namespace:
    values: dict[str, Any] = asdict(OpenSeekerBCPJudgeConfig())
    if args.config:
        run_config = load_openseeker_bcp_run_config(args.config)
        values.update(asdict(run_config.judge))
        if values.get("data_path") is None:
            values["data_path"] = run_config.generate.out_dir / f"result_tool{run_config.generate.tool_count_max}.jsonl"
    for key, value in vars(args).items():
        if key == "config":
            continue
        if value is not None:
            values[key] = value
    if values.get("data_path") is None:
        raise ValueError("--data_path is required")
    return argparse.Namespace(**values)


def main(argv: list[str] | None = None) -> int:
    args = resolve_config(build_parser().parse_args(argv))
    clients, model_name = _scorer_clients()

    save_path = args.save_path or Path(str(args.data_path).replace(".jsonl", "_eval.jsonl"))
    save_path.parent.mkdir(parents=True, exist_ok=True)

    all_data = read_jsonl(args.data_path)
    data = list(all_data)
    if args.limit > 0:
        data = data[: args.limit]

    item_idx2scored: dict[int, dict[str, Any]] = {}
    if save_path.exists():
        for obj in read_jsonl(save_path):
            if obj.get("type") == "summary":
                continue
            item_idx = obj.get("item_index")
            if isinstance(item_idx, int):
                item_idx2scored[item_idx] = obj

    need_missing: list[tuple[int, dict[str, Any]]] = []
    need_unknown: list[tuple[int, dict[str, Any]]] = []
    for index, item in enumerate(data):
        previous = item_idx2scored.get(index)
        if previous is None:
            need_missing.append((index, item))
            continue
        if not is_clean_01(previous.get("score")):
            need_unknown.append((index, item))

    data_to_eval = need_missing + need_unknown
    print(
        f"[eval] loaded={len(data)}, already_scored={len(item_idx2scored)}, "
        f"to_eval={len(data_to_eval)} (missing={len(need_missing)}, unknown={len(need_unknown)}), "
        f"save_path={save_path}, no_dedup=True, unknown_always_retry=True"
    )

    file_lock = threading.Lock()

    def score_one_item(item_idx: int, item: dict[str, Any]) -> dict[str, Any]:
        query = item["query"]
        answer = item["answer"]
        response = item["final_response"]
        messages = [
            {"role": "system", "content": "Judge the response objectively."},
            {
                "role": "user",
                "content": JUDGE_PROMPT_BC_EN.format(
                    question=query,
                    correct_answer=answer,
                    response=response,
                ),
            },
        ]
        raw = get_llm_response(clients, model_name, messages)
        label = parse_judge_label(raw)
        out: dict[str, Any] = {
            "type": "item",
            "item_index": item_idx,
            "query": query,
            "answer": answer,
            "final_response": response,
            "judge_raw": raw,
            "score": label if label is not None else raw,
        }
        out["is_correct"] = True if label == 1 else (False if label == 0 else None)
        out["tool_calls"] = count_tool_calls(item)
        return out

    def worker(pair: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        item_idx, item = pair
        try:
            scored = score_one_item(item_idx, item)
        except API_ERROR_TYPES + (KeyError, TypeError, ValueError) as exc:
            print(f"\033[91mError: {exc}\033[0m")
            scored = {
                "type": "item",
                "item_index": item_idx,
                "query": item.get("query"),
                "answer": item.get("answer"),
                "final_response": item.get("final_response"),
                "judge_raw": "EVAL_ERROR",
                "score": "EVAL_ERROR",
                "is_correct": None,
                "eval_error": True,
                "tool_calls": count_tool_calls(item),
            }

        with file_lock:
            with save_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(scored, ensure_ascii=False) + "\n")
        return scored

    if data_to_eval:
        from tqdm import tqdm

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            new_scored = list(tqdm(executor.map(worker, data_to_eval), total=len(data_to_eval), desc="Scoring"))
        for obj in new_scored:
            item_idx = obj.get("item_index")
            if isinstance(item_idx, int) and is_clean_01(obj.get("score")):
                item_idx2scored[item_idx] = obj

    processed_data: list[dict[str, Any]] = []
    for index, item in enumerate(all_data):
        scored_item = item_idx2scored.get(index)
        if scored_item is not None:
            processed_data.append(scored_item)

    if args.top_k is not None and isinstance(args.top_k, int) and args.top_k > 0:
        processed_data = processed_data[: args.top_k]

    correct_num = 0
    wrong_num = 0
    unknown_num = 0
    incomplete_num = 0
    reach_max_tool_num = 0
    reach_max_token_num = 0
    tool_calls_all: list[int] = []
    tool_calls_correct: list[int] = []

    for item in processed_data:
        score = item.get("score")
        if isinstance(score, str):
            print(score)
            unknown_num += 1
        elif score == 1:
            correct_num += 1
        elif score == 0:
            wrong_num += 1

        tool_calls = item.get("tool_calls")
        if isinstance(tool_calls, int) and tool_calls >= 0:
            tool_calls_all.append(tool_calls)
            if score == 1:
                tool_calls_correct.append(tool_calls)

        full_traj = item.get("full_traj")
        if isinstance(full_traj, str) and full_traj and not full_traj.endswith("<|return|>"):
            incomplete_num += 1

        final_response = item.get("final_response") or ""
        if isinstance(final_response, str) and "I have used too many tools, so I will conclude my answer." in final_response:
            reach_max_tool_num += 1
        if isinstance(final_response, str) and (
            "The max context length has been reached." in final_response
            or "I have used too many tokens, so I will conclude my answer." in final_response
        ):
            reach_max_token_num += 1

    print(f"correct_num: {correct_num}, wrong_num: {wrong_num}, unknown_num: {unknown_num}")
    print(
        f"incomplete_num: {incomplete_num}, reach_max_tool_num: {reach_max_tool_num}, "
        f"reach_max_token_num: {reach_max_token_num}"
    )
    denom = correct_num + wrong_num
    accuracy = (correct_num / denom) if denom > 0 else 0.0
    print(f"accuracy: {accuracy}")

    summary = {
        "type": "summary",
        "data_path": str(args.data_path),
        "save_path": str(save_path),
        "total_items": len(processed_data),
        "correct_num": correct_num,
        "wrong_num": wrong_num,
        "unknown_num": unknown_num,
        "accuracy": accuracy,
        "avg_tool_calls_correct": (sum(tool_calls_correct) / len(tool_calls_correct)) if tool_calls_correct else 0.0,
        "min_tool_calls": min(tool_calls_all) if tool_calls_all else None,
        "max_tool_calls": max(tool_calls_all) if tool_calls_all else None,
        "incomplete_num": incomplete_num,
        "reach_max_tool_num": reach_max_tool_num,
        "reach_max_token_num": reach_max_token_num,
    }

    tmp_path = Path(str(save_path) + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
        for item in processed_data:
            if not is_clean_01(item.get("score")):
                continue
            if item.get("type") != "item":
                item = dict(item)
                item["type"] = "item"
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    tmp_path.replace(save_path)
    return 0
