from __future__ import annotations

import os
import json
from typing import Callable, IO, Optional, TYPE_CHECKING

from .builder import AgentRunContext, create_agent

if TYPE_CHECKING:  # pragma: no cover
    from ..agents.base import AgentDecision, AgentEvent, Message
    from ..llm import LLMClient
    from ..tools.base import ToolResult


def emit(text: str = "", *, end: str = "\n", log_handle: Optional[IO[str]] = None) -> None:
    """Print text to stdout and mirror it into the optional log file handle."""
    print(text, end=end)
    if log_handle:
        log_handle.write(text)
        log_handle.write(end)
        log_handle.flush()


def _format_arguments(arguments: Optional[dict]) -> str:
    if not arguments:
        return "{}"
    try:
        return json.dumps(arguments, ensure_ascii=False)
    except TypeError:
        return repr(arguments)


def render_event(event: "AgentEvent", *, emit_func: Callable[[str], None]) -> None:
    """
    Render key details for each agent event when verbose logging is enabled.

    Shows incoming messages, assistant replies, tool usage, and final answers.
    """
    stage = event.stage

    if stage == "receive":
        message: "Message" = event.payload
        emit_func(f"👤 {message.role.capitalize()} message:")
        emit_func(message.content)
        return

    if stage == "response":
        emit_func("🤖 Assistant response:")
        emit_func(str(event.payload))
        return

    if stage == "decision":
        decision: "AgentDecision" = event.payload
        if decision.kind == "tool" and decision.tool_call:
            emit_func(
                f"🧭 Decision: call tool '{decision.tool_call.name}' "
                f"with args {_format_arguments(decision.tool_call.arguments)}"
            )
        elif decision.kind == "final":
            emit_func("🧭 Decision: produce final answer.")
        else:
            emit_func(f"🧭 Decision: {decision.kind}")
        return

    if stage == "tool_result":
        result: "ToolResult" = event.payload
        emit_func(f"🛠 Tool '{result.call.name}' output:")
        emit_func(result.output)
        return

    if stage == "final":
        emit_func("🎯 Final Answer:")
        emit_func(str(event.payload))
        return

    emit_func(f"ℹ️ Event [{stage}]: {event.payload}")


def build_context(
    args,
    *,
    query: str,
    website: Optional[str],
    llm_client: Optional["LLMClient"] = None,
) -> AgentRunContext:
    """Instantiate the requested agent with the provided query and website."""
    return create_agent(
        args.agent,
        query=query,
        website=website,
        max_rounds=args.max_rounds,
        llm=llm_client,
    )


def ensure_parent_directory(path: str) -> None:
    """Create parent directories for the given path if necessary."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def run_single_query(
    args,
    *,
    query: str,
    website: Optional[str],
    emit_func: Callable[[str], None],
    verbose: bool = True,
) -> str:
    """Execute one agent run and return the final answer."""
    context = build_context(args, query=query, website=website)
    final_answer = ""

    for event in context.agent.run(context.request):
        if verbose:
            render_event(event, emit_func=emit_func)
        if event.stage == "final":
            final_answer = event.payload

    return final_answer
