from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import IO, List, Optional

from datasets import load_dataset

from .llm import EchoLLM
from .common.runtime import (
    emit,
    ensure_parent_directory,
    run_single_query,
)


_LOG_FILE_HANDLE: Optional[IO[str]] = None


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run agents against a Hugging Face dataset.")
    parser.add_argument(
        "--agent",
        choices=["webwalker", "webdancer"],
        default="webwalker",
        help="Agent to invoke (default: webwalker).",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Hugging Face dataset name (e.g. callanwu/WebWalkerQA).",
    )
    parser.add_argument(
        "--dataset-split",
        default="train",
        help="Dataset split to load (default: train).",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Destination jsonl file for saving question-answer pairs.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Limit the number of samples processed from the dataset.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Maximum number of agent actions to perform (defaults depend on agent).",
    )
    parser.add_argument(
        "--llm",
        choices=["auto", "echo"],
        default="auto",
        help="LLM backend to use (auto=environment-based, echo=deterministic fallback).",
    )
    parser.add_argument("--log-file", help="Path to a log file for saving the CLI output.")
    args = parser.parse_args(argv)

    llm_client = EchoLLM() if args.llm == "echo" else None

    ensure_parent_directory(args.output_path)

    log_handle: Optional[IO[str]] = None
    log_path: Optional[str] = None
    global _LOG_FILE_HANDLE
    if args.log_file:
        raw_log = args.log_file
        if raw_log.endswith(os.sep) or os.path.isdir(raw_log):
            log_directory = raw_log.rstrip(os.sep) or "."
            os.makedirs(log_directory, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(log_directory, f"run_{timestamp}.log")
        else:
            log_path = raw_log
            ensure_parent_directory(log_path)

        log_handle = open(log_path, "a", encoding="utf-8")
        _LOG_FILE_HANDLE = log_handle

    def emit_func(text: str = "", *, end: str = "\n") -> None:
        emit(text, end=end, log_handle=_LOG_FILE_HANDLE)

    try:
        emit_func(f"🚀 Agent: {args.agent}")
        emit_func(f"📦 Dataset: {args.dataset_name}")
        emit_func(f"🔀 Split: {args.dataset_split}")
        emit_func("=" * 80)

        dataset = load_dataset(
            path=args.dataset_name,
            split=args.dataset_split,
        )

        total_samples = len(dataset)
        if args.max_samples:
            total_samples = min(total_samples, args.max_samples)

        with open(args.output_path, "w", encoding="utf-8") as out_handle:
            for index, item in enumerate(dataset, start=1):
                if args.max_samples and index > args.max_samples:
                    break

                query_value = item.get("question")
                website_value = item.get("root_url")

                if not query_value:
                    emit_func(f"⚠️  Sample {index} skipped: missing 'question' field.")
                    continue

                if args.agent == "webwalker" and not website_value:
                    emit_func(f"⚠️  Sample {index} skipped: missing 'root_url' field.")
                    continue

                emit_func(f"▶️  Processing sample {index}/{total_samples}")
                emit_func(f"❓ Query: {query_value}")
                if args.agent == "webwalker":
                    emit_func(f"🌐 Root website: {website_value}")
                emit_func("-" * 80)

                try:
                    pred_answer = run_single_query(
                        args,
                        query=query_value,
                        website=website_value,
                        llm_client=llm_client,
                        emit_func=emit_func,
                        verbose=True,
                    )
                except Exception as exc:
                    emit_func(f"❗ Error while processing sample {index}: {exc}")
                    pred_answer = ""
                emit_func(f"🎯 Final Answer: {pred_answer or '[empty]'}")
                emit_func("-" * 80)

                record = {
                    "question": query_value,
                    "pred": pred_answer,
                    "answer": item.get("answer"),
                    "root_url": website_value,
                    "info": item.get("info"),
                }
                out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_handle.flush()
        abs_output = os.path.abspath(args.output_path)
        emit_func(f"✅ Completed run. Predictions saved to: {abs_output}")
        if log_path:
            emit_func(f"📝 Run log saved to: {os.path.abspath(log_path)}")
    except KeyboardInterrupt:
        emit_func("⛔ Session interrupted by user.")
    finally:
        if log_handle:
            log_handle.close()
        _LOG_FILE_HANDLE = None


if __name__ == "__main__":
    main()
