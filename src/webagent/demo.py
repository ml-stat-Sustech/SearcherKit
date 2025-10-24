from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import IO, List, Optional

from .common.runtime import (
    emit,
    ensure_parent_directory,
    run_single_query,
)
from .common.utils import apply_local_wiki_env_flags


_LOG_FILE_HANDLE: Optional[IO[str]] = None


def _emit(text: str = "", *, end: str = "\n") -> None:
    emit(text, end=end, log_handle=_LOG_FILE_HANDLE)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run WebAgent on a single query.")
    parser.add_argument(
        "--agent",
        choices=["webwalker", "webdancer"],
        default="webwalker",
        help="Agent to invoke (default: webwalker).",
    )
    parser.add_argument("--query", required=True, help="Question to answer.")
    parser.add_argument(
        "--website",
        help="Root website to explore (required for webwalker).",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Maximum number of agent actions to perform (defaults depend on agent).",
    )
    parser.add_argument(
        "--use-separate-judge-llm",
        action="store_true",
        help="Use judge-specific OPENAI_JUDGE_* environment variables instead of the default model.",
    )
    parser.add_argument("--log-file", help="Path to a log file for saving the CLI output.")
    parser.add_argument(
        "--use-local-wiki-tools",
        action="store_true",
        help="Enable local wiki search/visit tools for WebDancer (sets WEBDANCER_USE_LOCAL_WIKI=1).",
    )
    parser.add_argument("--local-wiki-index", help="Name of the Elasticsearch index to query (LOCAL_WIKI_INDEX).")
    parser.add_argument("--local-wiki-es-host", help="Override Elasticsearch endpoint (LOCAL_WIKI_ES_HOST).")
    parser.add_argument("--local-wiki-es-timeout", help="Set Elasticsearch timeout in seconds (LOCAL_WIKI_ES_TIMEOUT).")
    parser.add_argument("--local-wiki-retriever", choices=["bm25", "dense", "hybrid"], help="Retriever strategy.")
    parser.add_argument("--local-wiki-model-name", help="Model name for dense/hybrid retrievers.")
    args = parser.parse_args(argv)

    if args.agent == "webwalker" and not args.website:
        parser.error("--website is required when --agent webwalker is selected.")

    apply_local_wiki_env_flags(args)

    log_handle: Optional[IO[str]] = None
    global _LOG_FILE_HANDLE
    if args.log_file:
        log_path = args.log_file
        if log_path.endswith(os.sep) or os.path.isdir(log_path):
            log_directory = log_path.rstrip(os.sep) or "."
            os.makedirs(log_directory, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(log_directory, f"demo_{timestamp}.log")
        ensure_parent_directory(log_path)
        log_handle = open(log_path, "a", encoding="utf-8")
        _LOG_FILE_HANDLE = log_handle

    agent_model_name = os.environ.get("OPENAI_MODEL") or os.environ.get("OPENAI_MODEL_NAME")
    if args.use_separate_judge_llm:
        judge_model_name = os.environ.get("OPENAI_JUDGE_MODEL") or agent_model_name
    else:
        judge_model_name = agent_model_name

    _emit(f"🚀 Agent: {args.agent}")
    if agent_model_name:
        _emit(f"🤖 Agent LLM: {agent_model_name}")
    if judge_model_name:
        _emit(f"⚖️ Judge LLM: {judge_model_name}")
    if args.agent == "webwalker":
        _emit(f"🌐 Root website: {args.website}")
    _emit(f"❓ Query: {args.query}")

    if args.agent == "webdancer":
        wiki_flag = os.environ.get("WEBDANCER_USE_LOCAL_WIKI")
        if wiki_flag:
            _emit("📚 Local wiki tools: ENABLED")
            index_name = os.environ.get("LOCAL_WIKI_INDEX")
            if index_name:
                _emit(f"🗂️ Local wiki index: {index_name}")
            retriever_name = os.environ.get("LOCAL_WIKI_RETRIEVER")
            if retriever_name:
                _emit(f"🔍 Local wiki retriever: {retriever_name}")
        else:
            _emit("📚 Local wiki tools: disabled (pass --use-local-wiki-tools to enable).")

    _emit("=" * 80)

    try:
        final_answer = run_single_query(
            args,
            query=args.query,
            website=args.website,
            emit_func=_emit,
            verbose=True,
        )
        if not final_answer:
            _emit("⚠️  No final answer produced.")
    except KeyboardInterrupt:
        _emit("⛔ Session interrupted by user.")
    except Exception as exc:  # noqa: BLE001
        _emit(f"❗ Error while running agent: {exc}")
    finally:
        if log_handle:
            log_handle.close()
        _LOG_FILE_HANDLE = None


if __name__ == "__main__":
    main()
