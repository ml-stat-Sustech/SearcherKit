from __future__ import annotations

import argparse
import os
from typing import IO, List, Optional

from .agents.base import AgentEvent, Message
from .tools.base import ToolResult
from .common import AgentRunContext, create_agent


_LOG_FILE_HANDLE: Optional[IO[str]] = None


def _emit(text: str = "", *, end: str = "\n") -> None:
    print(text, end=end)
    if _LOG_FILE_HANDLE:
        _LOG_FILE_HANDLE.write(text)
        _LOG_FILE_HANDLE.write(end)
        _LOG_FILE_HANDLE.flush()


def _render_receive(event: AgentEvent) -> None:
    message: Message = event.payload
    _emit("🧑‍💻 User prompt ready:")
    _emit(message.content)
    _emit("-" * 80)


def _render_decision(event: AgentEvent) -> None:
    decision = event.payload
    if decision.kind == "tool":
        _emit(f"🛠️  Decided to call tool: {decision.tool_call.name} with {decision.tool_call.arguments}")
    elif decision.kind == "final":
        _emit("✅ Decided to finalize the answer.")
    else:
        _emit("💬 Continuing conversation without tool usage.")


def _render_tool_result(event: AgentEvent) -> None:
    result: ToolResult = event.payload
    _emit(f"📄 Tool `{result.call.name}` output:")
    _emit(result.output)
    _emit("-" * 80)


def _render_response(event: AgentEvent) -> None:
    _emit("🤖 Follow-up LLM response:")
    _emit(event.payload)
    _emit("-" * 80)


def _render_final(event: AgentEvent) -> None:
    _emit("🎯 Final Answer:")
    _emit(event.payload)


def _render_event(event: AgentEvent) -> None:
    if event.stage == "receive":
        _render_receive(event)
    elif event.stage == "decision":
        _render_decision(event)
    elif event.stage == "tool_result":
        _render_tool_result(event)
    elif event.stage == "response":
        _render_response(event)
    elif event.stage == "final":
        _render_final(event)


def _build_context(args: argparse.Namespace) -> AgentRunContext:
    return create_agent(
        args.agent,
        query=args.query,
        website=args.website,
        max_rounds=args.max_rounds,
    )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run Agent Factory CLI (WebWalker/WebDancer).")
    parser.add_argument(
        "--agent",
        choices=["webwalker", "webdancer"],
        default="webwalker",
        help="Agent to invoke (default: webwalker).",
    )
    parser.add_argument("--query", required=True, help="Question to answer.")
    parser.add_argument("--website", help="Root website to explore (required for webwalker).")
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Maximum number of agent actions to perform (defaults depend on agent).",
    )
    parser.add_argument("--log-file", help="Path to a log file for saving the CLI output.")
    args = parser.parse_args(argv)

    if args.agent == "webwalker" and not args.website:
        parser.error("--website is required when --agent webwalker is selected.")

    context = _build_context(args)
    log_handle: Optional[IO[str]] = None
    global _LOG_FILE_HANDLE
    if args.log_file:
        log_dir = os.path.dirname(args.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_handle = open(args.log_file, "a", encoding="utf-8")
        _LOG_FILE_HANDLE = log_handle

    _emit(f"🚀 Agent: {args.agent}")
    if args.agent == "webwalker":
        _emit(f"🌐 Root website: {context.request.website}")
    _emit(f"❓ Query: {args.query}")
    _emit("=" * 80)

    try:
        for event in context.agent.run(context.request):
            _render_event(event)
    except KeyboardInterrupt:
        _emit("⛔ Session interrupted by user.")
    finally:
        if log_handle:
            log_handle.close()
        _LOG_FILE_HANDLE = None


if __name__ == "__main__":
    main()
