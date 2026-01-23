from __future__ import annotations

import argparse
import asyncio
import os
import traceback
from datetime import datetime
from typing import IO, Any, Dict, List, Optional, Callable

from .common.runtime import (
    emit,
    ensure_parent_directory,
    run_single_query,
)
from .common.utils import (
    apply_local_wiki_env_flags,
    default_eval_output_path,
    extract_sample_fields,
    load_existing_questions,
    write_prediction_record,
)
from .dataset_utils import load_dataset_records
from .evaluate.evl import run_llm_judge_evaluation


_LOG_FILE_HANDLE: Optional[IO[str]] = None

NUM_CONCURRENT_QUERIES = 16

sem = asyncio.Semaphore(NUM_CONCURRENT_QUERIES) # ！！过大的并行度会导致vllm engine的KV Cache 被discard，降低速度，尤其是在超长上下文的agent中。请根据vllm在超长序列(32k-128k)下的可用并行度设定

async def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run agents against a Hugging Face dataset.")
    parser.add_argument(
        "--agent",
        choices=["webwalker", "webdancer", "rag", "vanilla"],
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
        "--use-separate-judge-llm",
        action="store_true",
        help="Use a dedicated judge LLM during runtime (configured via OPENAI_JUDGE_* env vars).",
    )
    parser.add_argument(
        "--force-rejudge",
        action="store_true",
        help="Re-evaluate all questions even if they already exist in the evaluation output file.",
    )
    parser.add_argument(
        "--use-local-wiki-tools",
        action="store_true",
        help="Enable local wiki search/visit tools for WebDancer (sets WEBDANCER_USE_LOCAL_WIKI=1).",
    )
    parser.add_argument("--local-wiki-index", help="Name of the Elasticsearch index to query (LOCAL_WIKI_INDEX).")
    parser.add_argument(
        "--local-wiki-es-host",
        help="Override Elasticsearch endpoint for local wiki (LOCAL_WIKI_ES_HOST).",
    )
    parser.add_argument(
        "--local-wiki-es-timeout",
        help="Set Elasticsearch request timeout in seconds (LOCAL_WIKI_ES_TIMEOUT).",
    )
    parser.add_argument(
        "--local-wiki-retriever",
        choices=["bm25", "dense", "hybrid"],
        help="Choose retrieval strategy for local wiki (LOCAL_WIKI_RETRIEVER).",
    )
    parser.add_argument(
        "--local-wiki-model-name",
        help="Model name for dense/hybrid local wiki retrievers (LOCAL_WIKI_MODEL_NAME).",
    )
    args = parser.parse_args(argv)

    apply_local_wiki_env_flags(args)

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
        # Processing datasets
        existing_questions, existing_warning = load_existing_questions(args.output_path)
        if existing_questions:
            emit_func(f"🔁 Found {len(existing_questions)} existing records in {args.output_path}")
        else:
            emit_func(f"🔁 No existing records in {args.output_path}")
        if existing_warning:
            emit_func(existing_warning)

        agent_model_name = os.environ.get("OPENAI_MODEL") or os.environ.get("OPENAI_MODEL_NAME")
        judge_model_name = None
        if args.use_separate_judge_llm:
            judge_model_name = os.environ.get("OPENAI_JUDGE_MODEL") or agent_model_name
        else:
            judge_model_name = agent_model_name

        emit_func(f"🚀 Agent: {args.agent}")
        emit_func(f"📦 Dataset: {args.dataset_name}")
        emit_func(f"🔀 Split: {args.dataset_split}")
        if agent_model_name:
            emit_func(f"🤖 Agent LLM: {agent_model_name}")
        if judge_model_name:
            emit_func(f"⚖️ Judge LLM: {judge_model_name}")
        emit_func("=" * 80)
        if args.agent == "webdancer":
            wiki_flag = os.environ.get("WEBDANCER_USE_LOCAL_WIKI")
            if wiki_flag:
                emit_func("📚 Local wiki tools: ENABLED")
                index_name = os.environ.get("LOCAL_WIKI_INDEX")
                if index_name:
                    emit_func(f"🗂️ Local wiki index: {index_name}")
                retriever_name = os.environ.get("LOCAL_WIKI_RETRIEVER")
                if retriever_name:
                    emit_func(f"🔍 Local wiki retriever: {retriever_name}")
        elif args.agent == "webwalker":
            emit_func("📚 Local wiki tools: disabled (pass --use-local-wiki-tools to enable).")
        elif args.agent == "rag":
            emit_func("📚 Local wiki tools: not applicable (single-round search agent).")
        else:
            emit_func("📚 Local wiki tools: not applicable.")

        dataset = load_dataset_records(args.dataset_name, args.dataset_split)
        total_samples = len(dataset)
        if args.max_samples:
            total_samples = min(total_samples, args.max_samples)

        # Inference
        with open(args.output_path, "a", encoding="utf-8") as out_handle:
            processed_count = 0
            tasks = []
            contexts = []
            for index, item in enumerate(dataset, start=1):
                if args.max_samples and index > args.max_samples:
                    break
                item_dict = dict(item)
                (
                    question_value,
                    answer_value,
                    website_value_override,
                    info_value,
                    metadata_value,
                ) = extract_sample_fields(item_dict)

                if question_value is None:
                    emit_func(f"⚠️  Sample {index} skipped: unable to locate a question field.")
                    continue

                query_value = question_value
                website_value = website_value_override or item_dict.get("root_url")

                if not isinstance(query_value, str):
                    query_value = str(query_value)

                if args.agent == "webwalker" and not website_value:
                    emit_func(f"⚠️  Sample {index} skipped: missing 'root_url' field.")
                    continue

                if query_value in existing_questions:
                    emit_func(f"⏭️  Sample {index} skipped: question already processed.")
                    continue

                processed_count += 1
                emit_func(f"▶️ Processing sample {processed_count}/{total_samples}")
                emit_func(f"❓ Query: {query_value}")
                task_id_value = None
                for key in ("task_id", "Task ID", "TaskID"):
                    if metadata_value and key in metadata_value:
                        task_id_value = metadata_value[key]
                        break
                    if key in item_dict:
                        task_id_value = item_dict[key]
                        break
                if task_id_value:
                    emit_func(f"🆔 Task ID: {task_id_value}")
                level_value = None
                if metadata_value and "Level" in metadata_value:
                    level_value = metadata_value["Level"]
                elif "Level" in item_dict:
                    level_value = item_dict["Level"]
                if level_value:
                    emit_func(f"⭐ Level: {level_value}")
                if args.agent == "webwalker":
                    emit_func(f"🌐 Root website: {website_value}")
                emit_func("-" * 80)
                
                async def run_single(
                    args,
                    query: str,
                    website: str,
                    emit_func: Callable[[str], None],
                    verbose: bool = False,
                ):
                    async with sem:
                        return await run_single_query(
                            args,
                            query=query,
                            website=website,
                            emit_func=emit_func,
                            verbose=verbose,
                        )
                
                pred_answer_coroutine = run_single(
                    args,
                    query=query_value,
                    website=website_value,
                    emit_func=emit_func,
                    verbose=True,
                )
                tasks.append(pred_answer_coroutine) 
                contexts.append({
                    "query_value":  query_value,
                    "answer_value" : answer_value,
                    "website_value": website_value,
                    "info_value": info_value,
                    "metadata_value": metadata_value or None,
                    "fallback_task_id": item_dict.get("task_id"),
                })
                
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            e = None
            for item, pred_answer in zip(contexts, results):
                if isinstance(pred_answer, Exception):
                    emit_func(f"❗ Error while processing sample {index}: {pred_answer}\n{traceback.format_exception(pred_answer)[0]}")
                    e = pred_answer
                    continue
                assert isinstance(pred_answer, str)
                emit_func(f"🎯 Final Answer: {pred_answer or '[empty]'}")
                emit_func("-" * 80)
                
                query_value = item["query_value"]
                answer_value = item["answer_value"]
                website_value = item["website_value"]
                info_value = item["info_value"]
                metadata_value = item["metadata_value"]
                fallback_task_id = item["fallback_task_id"]

                write_prediction_record(
                    out_handle,
                    question=query_value,
                    prediction=pred_answer,
                    answer=answer_value,
                    root_url=website_value,
                    info=info_value,
                    metadata=metadata_value or None,
                    fallback_task_id=fallback_task_id,
                )
                existing_questions.add(query_value)
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

        if e:
            print("At least one error occurred during the run. Check log for details.\nLatest error:")
            raise e

        # Evaluation
        print('====================================== Evaluate ======================================')
        if args.run_eval:
            eval_output_path = args.eval_output_path or default_eval_output_path(args.output_path)
            ensure_parent_directory(eval_output_path)
            await run_llm_judge_evaluation(
                input_path=args.output_path,
                output_path=eval_output_path,
                skip_existing=not args.force_rejudge,
                use_separate_judge_llm=args.use_separate_judge_llm,
                emit_func=emit_func,
            )

    except KeyboardInterrupt:
        emit_func("⛔ Session interrupted by user.")
    finally:
        if log_handle:
            log_handle.close()
        _LOG_FILE_HANDLE = None
    

if __name__ == "__main__":
    asyncio.run(main())
