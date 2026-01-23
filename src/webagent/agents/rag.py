from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .base import AgentDecision, AgentState, BaseAgent, Message
from ..llm.client import LLMClient
from ..agents.prompts import (
    build_rag_context_prompt,
    build_rag_system_prompt,
    build_rag_user_prompt,
)
from ..tools import build_rag_tools
from ..tools.base import BaseTool, ToolCall, ToolResult


@dataclass
class RAGRequest:
    """Payload passed to the single-round RAG agent."""

    query: str


class RAGAgent(BaseAgent):
    """
    Retrieval-augmented agent that performs a single batched search before composing the answer.

    The agent always triggers one `search` tool call, feeds the observation back to the model,
    and expects the follow-up reply to contain the final answer.
    """

    def __init__(
        self,
        llm: LLMClient,
        *,
        tools: Optional[Iterable[BaseTool]] = None,
        max_steps: int = 8,
    ) -> None:
        default_tools = list(tools) if tools is not None else list(build_rag_tools().values())
        if not default_tools:
            raise ValueError("RAGAgent requires at least one search-capable tool.")
        super().__init__(tools=default_tools, max_steps=max(max_steps, 2))
        self.llm = llm
        # self.local_mode = use_local_wiki_tools()
        self.local_mode = False
        self.max_local_visits = 3 if self.local_mode else 0
        self.search_tool_name = self._find_tool_name("search")
        if not self.search_tool_name:
            raise ValueError("RAGAgent requires a registered tool named (or containing) 'search'.")
        self.visit_tool_name = self._find_tool_name("visit")

    # Stage 1 -----------------------------------------------------------------
    def handle_user_message(self, user_input: RAGRequest) -> AgentState:
        state = AgentState(user_input=user_input)
        state.messages.append(Message(role="system", content=build_rag_system_prompt()))
        state.messages.append(Message(role="user", content=build_rag_user_prompt(user_input.query)))
        state.scratchpad["query"] = user_input.query
        state.scratchpad["search_issued"] = False
        state.scratchpad["search_tool_name"] = self.search_tool_name
        if self.visit_tool_name:
            state.scratchpad["visit_tool_name"] = self.visit_tool_name
        return state

    # Stage 2+ ----------------------------------------------------------------
    def generate_step_response(self, state: AgentState) -> str:
        if state.scratchpad.pop("skip_next_llm", False):
            return ""
        return self.llm.complete(self._messages_to_dicts(state.messages))

    # Stage 3 -----------------------------------------------------------------
    def decide_next_action(self, state: AgentState) -> AgentDecision:
        if state.steps_taken == 1 and not state.scratchpad.get("search_issued"):
            tool_name = state.scratchpad.get("search_tool_name") or self.search_tool_name
            query = str(state.scratchpad.get("query") or state.user_input.query)
            state.scratchpad["search_issued"] = True
            return AgentDecision(
                kind="tool",
                tool_call=ToolCall(
                    name=tool_name,
                    arguments={"query": [query]},
                ),
            )

        visit_queue = state.scratchpad.get("visit_queue")
        if isinstance(visit_queue, list) and visit_queue:
            visit_tool = state.scratchpad.get("visit_tool_name") or self.visit_tool_name
            if visit_tool and visit_tool in self.tools:
                next_target = visit_queue.pop(0)
                if not visit_queue:
                    state.scratchpad.pop("visit_queue", None)
                else:
                    state.scratchpad["visit_queue"] = visit_queue
                arguments = {
                    "url": [next_target],
                    "goal": str(state.scratchpad.get("query") or state.user_input.query),
                }
                return AgentDecision(
                    kind="tool",
                    tool_call=ToolCall(name=visit_tool, arguments=arguments),
                )

        if not state.messages or state.messages[-1].role != "assistant":
            return AgentDecision(kind="continue")

        latest = state.messages[-1].content
        final = self._extract_answer(latest)
        if final:
            return AgentDecision(kind="final", message=final)

        if state.scratchpad.get("search_completed") and latest.strip():
            return AgentDecision(kind="final", message=latest.strip())

        return AgentDecision(kind="continue")

    # Stage 5 -----------------------------------------------------------------
    def process_tool_result(self, state: AgentState, result: ToolResult) -> Optional[List[object]]:
        query = str(state.scratchpad.get("query") or "")
        search_name = state.scratchpad.get("search_tool_name") or self.search_tool_name
        visit_name = state.scratchpad.get("visit_tool_name") or self.visit_tool_name

        if result.call.name == search_name:
            if self.local_mode and visit_name and visit_name in self.tools:
                candidates = self._extract_visit_targets(result.output)
                if candidates:
                    if self.max_local_visits:
                        candidates = candidates[: self.max_local_visits]
                    state.scratchpad["visit_queue"] = candidates
                    state.scratchpad["collected_visits"] = []
                    state.scratchpad["visits_completed"] = 0
                    state.scratchpad["skip_next_llm"] = True
                    state.scratchpad["raw_search_output"] = result.output
                    return None
            context_message = build_rag_context_prompt(query, result.output)
            state.messages.append(Message(role="user", content=context_message))
            state.scratchpad["search_completed"] = True
            return None

        if visit_name and result.call.name == visit_name:
            collected = state.scratchpad.setdefault("collected_visits", [])
            collected.append(result.output.strip())
            completed = int(state.scratchpad.get("visits_completed", 0)) + 1
            state.scratchpad["visits_completed"] = completed

            queue_obj = state.scratchpad.get("visit_queue")
            if isinstance(queue_obj, list):
                if self.max_local_visits and completed >= self.max_local_visits:
                    queue_obj.clear()
                if queue_obj:
                    state.scratchpad["skip_next_llm"] = True
                    return None
                state.scratchpad.pop("visit_queue", None)

            raw_search = state.scratchpad.get("raw_search_output")
            sources: List[str] = []
            if isinstance(raw_search, str) and raw_search.strip():
                sources.append(raw_search.strip())
            sources.extend(block for block in collected if block)
            combined = "\n\n".join(sources).strip() or result.output
            context_message = build_rag_context_prompt(query, combined)
            state.messages.append(Message(role="user", content=context_message))
            state.scratchpad["search_completed"] = True
            state.scratchpad.pop("collected_visits", None)
            state.scratchpad.pop("raw_search_output", None)
            state.scratchpad.pop("skip_next_llm", None)
            return None

        context_message = build_rag_context_prompt(query, result.output)
        state.messages.append(Message(role="user", content=context_message))
        state.scratchpad["search_completed"] = True
        return None

    # Stage 6 -----------------------------------------------------------------
    def finalize_response(self, state: AgentState, decision: AgentDecision) -> str:
        if decision.message:
            return decision.message.strip()

        if state.messages and state.messages[-1].role == "assistant":
            fallback = self._extract_answer(state.messages[-1].content)
            if fallback:
                return fallback
            return state.messages[-1].content.strip()

        return "Failed to generate an answer, please try again later.。"

    # Helpers -----------------------------------------------------------------
    def _messages_to_dicts(self, messages: List[Message]) -> List[Dict[str, str]]:
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def _extract_answer(self, content: str) -> Optional[str]:
        marker = "Final Answer:"
        lower_marker = marker.lower()
        lowered = content.lower()
        index = lowered.find(lower_marker)
        if index == -1:
            return None
        offset = index + len(marker)
        return content[offset:].strip() or None

    def _find_tool_name(self, keyword: str) -> Optional[str]:
        lowered = keyword.lower()
        for name in self.tools:
            if name.lower() == lowered:
                return name
        for name in self.tools:
            if lowered in name.lower():
                return name
        return None

    def _extract_visit_targets(self, text: str) -> List[str]:
        titles: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped[0].isdigit() and ". " in stripped:
                _, remainder = stripped.split(". ", 1)
                title = remainder.split("(score", 1)[0].strip()
                if title:
                    titles.append(title)
        return titles[:3]
