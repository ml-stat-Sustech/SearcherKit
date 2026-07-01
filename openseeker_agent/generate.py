"""Generate BrowseComp Plus answers with the OpenSeeker SFT eval loop."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
import sys
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import requests

from config import OpenSeekerBCPGenerateConfig, load_openseeker_bcp_run_config


class Tee:
    """Duplicate writes to multiple file-like objects."""

    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        written = 0
        for stream in self.streams:
            try:
                written = stream.write(value)
            except OSError:
                continue
        return written

    def flush(self) -> None:
        for stream in self.streams:
            try:
                stream.flush()
            except OSError:
                continue


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            data.append(obj)
    return data


def normalize_dataset_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        query = row.get("query")
        if not isinstance(query, str) or not query.strip():
            query = row.get("question")
        if not isinstance(query, str) or not query.strip():
            continue
        out: dict[str, Any] = {"query": query.strip()}
        if "answer" in row:
            out["answer"] = row["answer"]
        normalized.append(out)
    return normalized


def write_normalized_dataset(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_queries_without_answer(save_path: Path, all_queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not save_path.exists():
        return all_queries

    query_to_has_answer: dict[str, bool] = {}
    with save_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, Mapping):
                continue
            query = obj.get("query")
            if isinstance(query, str) and query:
                final_response = obj.get("final_response", "")
                query_to_has_answer[query] = bool(final_response and str(final_response).strip())

    missing: list[dict[str, Any]] = []
    for item in all_queries:
        query = item.get("query", "")
        if isinstance(query, str) and query and not query_to_has_answer.get(query, False):
            missing.append(item)
    return missing


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_metrics(result_jsonl: Path) -> dict[str, Any]:
    if not result_jsonl.exists():
        return {"count": 0}

    count = 0
    tool_calls: list[float] = []
    context_chars: list[float] = []
    elapsed_seconds: list[float] = []

    with result_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, Mapping):
                continue
            count += 1
            tc = _safe_float(obj.get("tool_calls"))
            cc = _safe_float(obj.get("context_chars"))
            es = _safe_float(obj.get("elapsed_seconds"))
            if tc is not None:
                tool_calls.append(tc)
            if cc is not None:
                context_chars.append(cc)
            if es is not None:
                elapsed_seconds.append(es)

    def mean(values: list[float]) -> float | None:
        return (sum(values) / len(values)) if values else None

    return {
        "count": count,
        "tool_calls": {"mean": mean(tool_calls), "min": min(tool_calls) if tool_calls else None, "max": max(tool_calls) if tool_calls else None},
        "context_chars": {"mean": mean(context_chars), "min": min(context_chars) if context_chars else None, "max": max(context_chars) if context_chars else None},
        "elapsed_seconds": {"mean": mean(elapsed_seconds), "min": min(elapsed_seconds) if elapsed_seconds else None, "max": max(elapsed_seconds) if elapsed_seconds else None},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run strict OpenSeeker-style BCP generation.")
    parser.add_argument("--config", type=Path, default=None, help="Path to SFT run YAML config")
    parser.add_argument("--dataset_path", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--normalized_dataset", type=Path, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--tool_count_max", type=int, default=None)
    parser.add_argument("--max_worker", type=int, default=None)
    parser.add_argument("--pool_no_progress_timeout", type=int, default=None)
    parser.add_argument("--pool_restart_rounds", type=int, default=None)
    parser.add_argument("--max_retry_rounds", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--print_stream", action="store_true", default=None)
    parser.add_argument("--sequential", action="store_true", default=None)
    parser.add_argument("--run-log-path", type=Path, default=None)
    parser.add_argument("--no-run-log", action="store_true", default=None)
    return parser


def resolve_config(args: argparse.Namespace) -> argparse.Namespace:
    defaults = OpenSeekerBCPGenerateConfig(dataset_path=Path(), out_dir=Path())
    values = asdict(defaults)
    if args.config:
        values.update(asdict(load_openseeker_bcp_run_config(args.config).generate))
    for key, value in vars(args).items():
        if key == "config":
            continue
        if value is not None:
            values[key] = value
    if values.get("dataset_path") in (None, Path()):
        raise ValueError("--dataset_path is required")
    if values.get("out_dir") in (None, Path()):
        raise ValueError("--out_dir is required")
    return argparse.Namespace(**values)


async def run(args: argparse.Namespace) -> None:
    run_start_ts = time.time()
    rows = normalize_dataset_rows(read_jsonl(args.dataset_path))
    if args.limit != -1:
        rows = rows[: args.limit]
    print(f">> Loaded {len(rows)} questions from {args.dataset_path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = args.normalized_dataset or (args.out_dir / "input.normalized.jsonl")
    write_normalized_dataset(rows, normalized_path)
    print(f">> Normalized dataset: {normalized_path}")

    save_path = args.out_dir / f"result_tool{args.tool_count_max}.jsonl"
    log_path = args.out_dir / f"result_tool{args.tool_count_max}.log.txt"
    metric_path = args.out_dir / f"result_tool{args.tool_count_max}_metrics.json"

    print(
        json.dumps(
            {
                "max_tokens": args.max_tokens,
                "tool_count_max": args.tool_count_max,
                "max_worker": args.max_worker,
                "dataset_path": str(args.dataset_path),
                "normalized_dataset": str(normalized_path),
                "save_path": str(save_path),
                "error_log_path": str(log_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    before = len(rows)
    rows = get_queries_without_answer(save_path, rows)
    skipped = before - len(rows)
    if skipped:
        print(f">> Dedup (valid answers only): {before} -> {len(rows)} (skipped={skipped})")

    lock = asyncio.Lock()

    async def process_one(data: dict[str, Any]) -> bool:
        query = str(data.get("query", ""))
        started = time.time()
        print(f">> START query={query[:120]!r}")
        try:
            from llm_tool import solve_query_with_tools

            result = await asyncio.to_thread(
                solve_query_with_tools,
                query,
                max_tokens=args.max_tokens,
                tool_count_max=args.tool_count_max,
                print_stream=args.print_stream,
                return_full_traj=True,
            )
        except (RuntimeError, ValueError, TypeError, requests.exceptions.RequestException, OSError, ImportError) as exc:
            err = traceback.format_exc()
            print(f"\033[91m>> FAILED query={query[:120]!r}\033[0m")
            print(f"\033[91m>> Error type: {type(exc).__name__}\033[0m")
            print(f"\033[91m>> Error message: {exc}\033[0m")
            print(f"\033[91m>> Full traceback:\n{err}\033[0m")
            async with lock:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(f">> Error in processing query: {query}\n")
                    handle.write(f">> Error type: {type(exc).__name__}\n")
                    handle.write(f">> Error message: {exc}\n")
                    handle.write(err + "\n")
            return False

        out = dict(data)
        out["final_response"] = result.get("answer", "")
        out["tool_calls"] = result.get("tool_calls", None)
        out["elapsed_seconds"] = result.get("elapsed_seconds", None)
        out["context_chars"] = result.get("context_chars", None)
        out["context_est_tokens"] = result.get("context_est_tokens", None)
        out["full_traj"] = result.get("full_traj", "")
        out["trace"] = result.get("trace", "")
        out["wall_seconds"] = time.time() - started

        async with lock:
            with save_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(out, ensure_ascii=False) + "\n")
                handle.flush()
        print(
            f">> DONE  query={query[:120]!r} "
            f"wall={out['wall_seconds']:.2f}s tool_calls={out.get('tool_calls')} ctx_chars={out.get('context_chars')}"
        )
        return True

    def finalize() -> None:
        metrics = compute_metrics(save_path)
        metrics["run_total_seconds"] = time.time() - run_start_ts
        metrics["run_started_at_unix"] = run_start_ts
        metrics["run_finished_at_unix"] = time.time()
        metric_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f">> Metrics saved to {metric_path}")
        print(f">> Avg tool_calls: {(metrics.get('tool_calls') or {}).get('mean')}")
        print(f">> Avg context_chars: {(metrics.get('context_chars') or {}).get('mean')}")
        print(f">> Total run wall time (seconds): {metrics.get('run_total_seconds')}")

    async def run_processing_round(data_to_process: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
        if args.sequential:
            completed_ok: set[str] = set()
            failed_final: set[str] = set()
            for item in data_to_process:
                query = item.get("query", "")
                ok = await process_one(item)
                if isinstance(query, str) and query:
                    if ok:
                        completed_ok.add(query)
                    else:
                        failed_final.add(query)
            return completed_ok, failed_final

        print("\n" + "=" * 100 + "\n>> Start to process the test data...")
        remaining = list(data_to_process)
        rounds_total = max(0, int(args.pool_restart_rounds)) + 1
        completed_ok: set[str] = set()
        failed_final: set[str] = set()

        for round_idx in range(rounds_total):
            if not remaining:
                break
            print(f">> Pool round {round_idx + 1}/{rounds_total}: remaining={len(remaining)}")
            semaphore = asyncio.Semaphore(args.max_worker)
            task2item: dict[asyncio.Task[tuple[str, bool]], dict[str, Any]] = {}

            async def process_with_semaphore(item: dict[str, Any]) -> tuple[str, bool]:
                async with semaphore:
                    query = str(item.get("query", ""))
                    ok = await process_one(item)
                    return query, ok

            tasks: list[asyncio.Task[tuple[str, bool]]] = []
            for item in remaining:
                task = asyncio.create_task(process_with_semaphore(item))
                tasks.append(task)
                task2item[task] = item

            pending: set[asyncio.Task[tuple[str, bool]]] = set(tasks)
            last_progress = time.time()
            done_count = 0
            round_ok: set[str] = set()
            round_fail: set[str] = set()

            while pending:
                done, pending = await asyncio.wait(pending, timeout=5, return_when=asyncio.FIRST_COMPLETED)
                if not done:
                    if time.time() - last_progress > args.pool_no_progress_timeout:
                        print(
                            f"\n>> No progress for {args.pool_no_progress_timeout}s. "
                            f"Restarting pool; carry over remaining={len(pending)} tasks..."
                        )
                        break
                    continue

                last_progress = time.time()
                for task in done:
                    item = task2item.get(task, {})
                    query = item.get("query", "")
                    try:
                        q_result, ok = task.result()
                    except (RuntimeError, ValueError, TypeError, OSError, asyncio.CancelledError):
                        if isinstance(query, str) and query:
                            round_fail.add(query)
                        continue
                    if isinstance(q_result, str) and q_result:
                        query = q_result
                    if isinstance(query, str) and query:
                        if ok:
                            round_ok.add(query)
                        else:
                            round_fail.add(query)
                done_count += len(done)
                if done_count % 10 == 0:
                    print(f">> Progress (round {round_idx + 1}): {done_count}/{len(tasks)}")

            to_retry: list[dict[str, Any]] = []
            for task in pending:
                item = task2item.get(task)
                if item is not None:
                    to_retry.append(item)
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

            completed_ok |= round_ok
            failed_final |= round_fail
            remaining = to_retry

        if remaining:
            print(f">> WARNING: still remaining after {rounds_total} rounds: {len(remaining)} (likely stuck).")
            async with lock:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(f">> WARNING: remaining tasks after rounds_total={rounds_total}: {len(remaining)}\n")
                    for item in remaining:
                        handle.write(f"  - {item.get('query', '')}\n")
        return completed_ok, failed_final

    total_all = len(rows)
    all_completed_ok: set[str] = set()
    all_failed_final: set[str] = set()
    current_data = list(rows)
    retry_round = 0

    while current_data and (args.max_retry_rounds == 0 or retry_round < args.max_retry_rounds):
        if retry_round == 0:
            print(f"\n>> Initial processing round: {len(current_data)} queries")
        else:
            print(f"\n>> Retry round {retry_round}: processing {len(current_data)} queries without answers")

        round_ok, round_fail = await run_processing_round(current_data)
        all_completed_ok |= round_ok
        all_failed_final |= round_fail

        if args.max_retry_rounds > 0:
            current_data = get_queries_without_answer(save_path, rows)
            if current_data:
                retry_round += 1
                print(f">> Found {len(current_data)} queries without answers, will retry in next round")
                if retry_round >= args.max_retry_rounds:
                    print(f">> Reached max_retry_rounds={args.max_retry_rounds}, stopping retry")
                    break
            else:
                print(">> All queries have answers! Stopping retry loop.")
                break
        else:
            break

    if current_data and args.max_retry_rounds > 0:
        print(f">> WARNING: After {retry_round} retry rounds, {len(current_data)} queries still don't have answers.")
        async with lock:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f">> WARNING: After {retry_round} retry rounds, {len(current_data)} queries still don't have answers:\n")
                for item in current_data:
                    handle.write(f"  - {item.get('query', '')}\n")

    print(f">> Done. Saved to {save_path}")
    print(f">> Summary: total={total_all} ok_written={len(all_completed_ok)} failed_not_written={len(all_failed_final)}")
    if args.max_retry_rounds > 0:
        print(f">> Retry rounds executed: {retry_round}")
    finalize()


def main(argv: list[str] | None = None) -> int:
    args = resolve_config(build_parser().parse_args(argv))
    run_log_path = args.run_log_path or (args.out_dir / f"result_tool{args.tool_count_max}.run.log")
    run_log_handle = None
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    try:
        if not args.no_run_log:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            run_log_handle = run_log_path.open("a", encoding="utf-8")
            sys.stdout = Tee(orig_stdout, run_log_handle)
            sys.stderr = Tee(orig_stderr, run_log_handle)
            print(f">> Run log: {run_log_path}")
        asyncio.run(run(args))
    finally:
        sys.stdout, sys.stderr = orig_stdout, orig_stderr
        if run_log_handle is not None:
            run_log_handle.flush()
            run_log_handle.close()
    return 0
