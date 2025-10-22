from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import IO, List, Optional

from datasets import load_dataset

from .common.runtime import (
    emit,
    ensure_parent_directory,
    run_single_query,
)
from .evaluate.evl import eval_result


_LOG_FILE_HANDLE: Optional[IO[str]] = None


def _default_eval_output_path(prediction_path: str) -> str:
    base, ext = os.path.splitext(prediction_path)
    if ext:
        return f"{base}_eval{ext}"
    return f"{prediction_path}_eval.jsonl"


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
    parser.add_argument("--log-file", help="Path to a log file for saving the CLI output.")
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Run evaluation on the generated predictions once the dataset sweep completes.",
    )
    parser.add_argument(
        "--eval-output-path",
        help="Destination jsonl file for saving evaluation scores (defaults to <output_path> with '_eval' suffix).",
    )
    parser.add_argument(
        "--judge-dataset",
        default="webwalker",
        help="Dataset identifier used for evaluation prompts and ground truth lookup.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Override the evaluation LLM model (defaults to environment configuration).",
    )
    parser.add_argument(
        "--judge-prompt",
        default=None,
        help="Override the evaluation prompt template.",
    )
    parser.add_argument(
        "--use-separate-judge-llm",
        action="store_true",
        help="Use a dedicated judge LLM during runtime (configured via OPENAI_JUDGE_* env vars).",
    )
    parser.add_argument(
        "--force-rejudge",
        action="store_true",
        help="Re-evaluate all questions even if they already exist in the evaluation output file.",
    )
    args = parser.parse_args(argv)

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
            processed_count = 0
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
                processed_count += 1
                emit_func(f"▶️  Processing sample {processed_count}/{total_samples}")
                emit_func(f"❓ Query: {query_value}")
                if args.agent == "webwalker":
                    emit_func(f"🌐 Root website: {website_value}")
                emit_func("-" * 80)

                try:
                    pred_answer = run_single_query(
                        args,
                        query=query_value,
                        website=website_value,
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
        try:
            with open(args.output_path, "r", encoding="utf-8") as verify_handle:
                lines = verify_handle.readlines()
            emit_func(f"📊 Records written so far: {len(lines)}")
            if lines:
                emit_func(f"🔚 Last record: {lines[-1].strip()}")
        except Exception as verify_exc:  # noqa: BLE001
            emit_func(f"⚠️  Unable to read back output file: {verify_exc}")
        abs_output = os.path.abspath(args.output_path)
        emit_func(f"✅ Completed run. Predictions saved to: {abs_output}")
        if log_path:
            emit_func(f"📝 Run log saved to: {os.path.abspath(log_path)}")

        # Evaluation
        if args.run_eval:
            eval_output_path = args.eval_output_path or _default_eval_output_path(args.output_path)
            ensure_parent_directory(eval_output_path)
            emit_func("=" * 80)
            emit_func("🧪 Running evaluation with LLM judge...")
            try:
                eval_result(
                    args.output_path,
                    eval_output_path,
                    dataset=args.judge_dataset,
                    judge_model=args.judge_model,
                    judge_prompt=args.judge_prompt,
                    skip_existing=not args.force_rejudge,
                )
                abs_eval_output = os.path.abspath(eval_output_path)
                emit_func(f"📈 Evaluation scores saved to: {abs_eval_output}")
                report_base, _ = os.path.splitext(eval_output_path)
                report_path = f"{report_base}_report.json"
                if os.path.exists(report_path):
                    abs_report = os.path.abspath(report_path)
                    emit_func(f"🧾 Evaluation summary saved to: {abs_report}")
                    try:
                        with open(report_path, "r", encoding="utf-8") as report_handle:
                            report_data = json.load(report_handle)
                        emit_func(f"🔢 Overall accuracy: {report_data.get('overall')}")
                    except Exception as report_exc:  # noqa: BLE001
                        emit_func(f"⚠️  Unable to read evaluation summary: {report_exc}")
            except Exception as eval_exc:  # noqa: BLE001
                emit_func(f"❗ Evaluation failed: {eval_exc}")

    except KeyboardInterrupt:
        emit_func("⛔ Session interrupted by user.")
    finally:
        if log_handle:
            log_handle.close()
        _LOG_FILE_HANDLE = None


if __name__ == "__main__":
    main()
