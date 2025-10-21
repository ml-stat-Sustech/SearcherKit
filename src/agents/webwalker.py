from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .base import AgentDecision, AgentState, BaseAgent, Message
from ..tools.base import BaseTool, ToolCall, ToolResult
from ..llm.client import LLMClient
from ..common.memory_manager import MemoryManager
from ..prompts import SYSTEM_EXPLORER
from ..common.state import STATE
from ..tools import WEBWALKER_TOOLS
from ..common.utils import extract_links_with_text, get_info


@dataclass
class WebWalkerRequest:
    """CLI input payload for spinning up a WebWalker run."""

    website: str
    query: str


class WebWalkerAgent(BaseAgent):
    """Concrete WebWalker implementation built on top of `BaseAgent`."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        tools: Optional[Iterable[BaseTool]] = None,
        max_steps: int = 20,
    ) -> None:
        super().__init__(tools=list(tools or WEBWALKER_TOOLS.values()), max_steps=max_steps)
        self.llm = llm
        self.memory = MemoryManager(llm=llm, query="")

    # ---- Stage 1 ---------------------------------------------------------
    def handle_user_message(self, user_input: WebWalkerRequest) -> AgentState:
        STATE.reset(user_input.website)
        html, markdown, _ = asyncio.run(get_info(user_input.website, screenshot=False))
        buttons = extract_links_with_text(html)

        system_prompt = self._build_system_prompt(user_input.query)
        user_prompt = self._format_initial_prompt(markdown, buttons, user_input.query, user_input.website)

        state = AgentState(user_input=user_input)
        state.messages.append(Message(role="system", content=system_prompt))
        state.messages.append(Message(role="user", content=user_prompt))
        state.scratchpad.update(
            {
                "query": user_input.query,
                "website": user_input.website,
                "last_buttons": buttons,
                "last_markdown": markdown,
            }
        )
        self.memory.reset(user_input.query)
        return state

    # ---- Stage 2+ --------------------------------------------------------
    def generate_step_response(self, state: AgentState) -> str:
        return self.llm.complete(self._messages_to_dicts(state.messages))

    # ---- Stage 3 ---------------------------------------------------------
    def decide_next_action(self, state: AgentState) -> AgentDecision:
        if not state.messages or state.messages[-1].role != "assistant":
            return AgentDecision(kind="continue", message=None)

        latest = state.messages[-1].content
        final_answer = self._extract_final_answer(latest)
        # print(50*"*"+f'{latest}'+50*'*')
        # print(50*"="+f'{final_answer}'+50*'=')
        if final_answer:
            return AgentDecision(kind="final", message=final_answer)

        action = self._extract_action_block(latest)
        if action:
            tool_name, arguments = action
            if tool_name.lower() == "none" or tool_name not in self.tools:
                return AgentDecision(kind="continue", message=None)
            return AgentDecision(kind="tool", tool_call=ToolCall(name=tool_name, arguments=arguments))

        return AgentDecision(kind="continue", message=None)

    # ---- Stage 5 ---------------------------------------------------------
    def process_tool_result(self, state: AgentState, result: ToolResult) -> Optional[List[str]]:
        observation_text = result.output
        observation_message = f"Observation: {observation_text}"
        state.messages.append(Message(role="user", content=observation_message))

        follow_ups: List[str] = []
        try:
            extracted = self.memory.analyze_observation(observation_text)
        except Exception as exc:  # noqa: BLE001
            state.scratchpad.setdefault("memory_errors", []).append(str(exc))
            extracted = None

        if extracted:
            memory_lines = "\n".join(f"- {entry}" for entry in self.memory.entries)
            follow_ups.append(f"Memory:\n{memory_lines}")
            try:
                final = self.memory.judge_completion()
            except Exception as exc:  # noqa: BLE001
                state.scratchpad.setdefault("memory_errors", []).append(str(exc))
                final = None
            if final:
                final_text = f"Final Answer: {final}"
                state.scratchpad["final_answer"] = final_text

        return follow_ups or None

    # ---- Stage 6 ---------------------------------------------------------
    def finalize_response(self, state: AgentState, decision: AgentDecision) -> str:
        if decision.message:
            return decision.message.strip()
        if state.messages and state.messages[-1].role == "assistant":
            fallback = self._extract_final_answer(state.messages[-1].content)
            if fallback:
                return fallback
            return state.messages[-1].content.strip()
        return "Unable to produce a final answer."

    # ---- Helpers ---------------------------------------------------------
    def on_max_steps(self, state: AgentState) -> str:
        query = state.scratchpad.get("query", "")
        return (
            "Step budget exceeded before reaching a conclusion. "
            f"Consider rerunning with a larger limit. (Query: {query})"
        )

    def _build_system_prompt(self, query: str) -> str:
        tool_descs: List[str] = []
        for tool in self.tools.values():
            tool_descs.append(f"{tool.name}: {tool.description}")
        return SYSTEM_EXPLORER.format(
            tool_descs="\n\n".join(tool_descs),
            tool_names=", ".join(self.tools.keys()),
            query=f"Question: {query}",
        )

    def _format_initial_prompt(
        self,
        markdown: str,
        buttons: str,
        query: str,
        website: str,
    ) -> str:
        response = "🧭 website information:\n\n"
        if markdown:
            response += markdown + "\n\n"
        else:
            response += "🧭 The information of the current page is not accessible\n\n"

        if buttons:
            response += "🖱 clickable button:\n\n" + buttons + "\n\nEach button is wrapped in a <button> tag"
        else:
            response += (
                "🖱 clickable button:\n\nNo clickable buttons were detected.\n\nEach button is wrapped in a <button> tag"
            )

        start_prompt = f"query:\n{query} \nofficial website:\n{website}"
        start_prompt += "\nObservation:" + response + "\n\n"
        return start_prompt

    def _messages_to_dicts(self, messages: List[Message]) -> List[Dict[str, str]]:
        return [{"role": message.role, "content": message.content} for message in messages]

    def _extract_final_answer(self, content: str) -> Optional[str]:
        match = re.search(
            r"(?is)(?:^|\n)\s*(?:\*\*)?(?:Final\s+)?Answer(?:\s*\*\*)?\s*[:：](?:\*\*)?\s*(.+)",
            content,
        )
        if not match:
            return None
        return match.group(1).strip()

    def _extract_action_block(self, content: str) -> Optional[Tuple[str, Dict[str, object]]]:
        action_match = re.search(r"Action\s*:\s*([a-zA-Z0-9_]+)", content)
        if not action_match:
            return None
        name = action_match.group(1).strip()
        input_match = re.search(r"Action Input\s*:\s*(.*)", content, re.IGNORECASE | re.DOTALL)
        if not input_match:
            return name, {}
        raw_input = input_match.group(1).strip()
        return name, self._parse_arguments(raw_input)

    def _parse_arguments(self, raw: str) -> Dict[str, object]:
        cleaned = raw
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            newline_index = cleaned.find("\n")
            if newline_index != -1:
                cleaned = cleaned[newline_index + 1 :]
        brace_start = cleaned.find("{")
        brace_end = cleaned.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            cleaned = cleaned[brace_start : brace_end + 1]
        try:
            return json.loads(cleaned)
        except Exception:
            return {}
