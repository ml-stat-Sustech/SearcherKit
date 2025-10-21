from __future__ import annotations

import os
from typing import Callable, IO, Optional, TYPE_CHECKING

from .builder import AgentRunContext, create_agent

if TYPE_CHECKING:  # pragma: no cover
    from ..agents.base import AgentEvent, Message
    from ..llm import LLMClient
    from ..tools.base import ToolResult


def emit(text: str = "", *, end: str = "\n", log_handle: Optional[IO[str]] = None) -> None:
    """Print text to stdout and mirror it into the optional log file handle."""
    print(text, end=end)
    if log_handle:
        log_handle.write(text)
        log_handle.write(end)
        log_handle.flush()


def render_event(event: "AgentEvent", *, emit_func: Callable[[str], None]) -> None:
    """Render only the final event; suppress intermediate details."""
    if event.stage == "final":
        emit_func("🎯 Final Answer:")
        emit_func(event.payload)


def build_context(
    args,
    *,
    query: str,
    website: Optional[str],
    llm_client: Optional["LLMClient"],
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
    llm_client: Optional["LLMClient"],
    emit_func: Callable[[str], None],
    verbose: bool = True,
) -> str:
    """Execute one agent run and return the final answer."""
    context = build_context(args, query=query, website=website, llm_client=llm_client)
    final_answer = ""

    for event in context.agent.run(context.request):
        if verbose:
            render_event(event, emit_func=emit_func)
        if event.stage == "final":
            final_answer = event.payload

    return final_answer
