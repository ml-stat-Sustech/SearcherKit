from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .base import AgentDecision, AgentState, BaseAgent, Message
from ..llm.client import LLMClient
from ..agents.prompts import build_vanilla_system_prompt, build_vanilla_user_prompt


@dataclass
class VanillaRequest:
    """Payload passed to the Vanilla agent."""

    query: str


class VanillaAgent(BaseAgent):
    """Minimal agent that answers questions without using any tools."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        max_steps: int = 4,
    ) -> None:
        super().__init__(tools=None, max_steps=max_steps)
        self.llm = llm

    # Stage 1 -----------------------------------------------------------------
    def handle_user_message(self, user_input: VanillaRequest) -> AgentState:
        state = AgentState(user_input=user_input)
        state.messages.append(Message(role="system", content=build_vanilla_system_prompt()))
        state.messages.append(Message(role="user", content=build_vanilla_user_prompt(user_input.query)))
        state.scratchpad["query"] = user_input.query
        return state

    # Stage 2+ ----------------------------------------------------------------
    def generate_step_response(self, state: AgentState) -> str:
        return self.llm.complete(self._messages_to_dicts(state.messages))

    # Stage 3 -----------------------------------------------------------------
    def decide_next_action(self, state: AgentState) -> AgentDecision:
        if not state.messages:
            return AgentDecision(kind="continue")

        latest = state.messages[-1]
        if latest.role != "assistant":
            return AgentDecision(kind="continue")

        content = latest.content
        final = self._extract_answer(content)
        if final:
            return AgentDecision(kind="final", message=final)

        if "action:" in content.lower():
            return AgentDecision(kind="continue")

        if content.strip():
            return AgentDecision(kind="final", message=content.strip())

        return AgentDecision(kind="continue")

    # Stage 5 -----------------------------------------------------------------
    def process_tool_result(self, state: AgentState, result) -> Optional[List[object]]:  # type: ignore[override]
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

        return "Unable to provide an answer, please try again."

    # Helpers -----------------------------------------------------------------
    def _messages_to_dicts(self, messages: List[Message]) -> List[Dict[str, str]]:
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def _extract_answer(self, content: str) -> Optional[str]:
        marker = re.search(r"Final Answer:\s*", content, re.IGNORECASE)
        if not marker:
            return None
        answer = content[marker.end() :].strip()
        return answer or None
