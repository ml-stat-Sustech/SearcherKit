from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from openai import OpenAI

from .base import AgentDecision, AgentState, BaseAgent, Message
from ..tools.base import BaseTool, ToolCall, ToolResult
from ..llm.client import LLMClient
from ..prompts import build_webdancer_system_prompt, build_webdancer_user_prompt
from ..tools import build_webdancer_tools

SUMMARY_MODEL = os.getenv("WEBDANCER_SUMMARY_MODEL", "qwen/qwen-2.5-72b-instruct")
SUMMARY_API_KEY = (
    os.getenv("WEBDANCER_SUMMARY_API_KEY", "REMOVED_REVOKED_SECRET")
)
SUMMARY_BASE_URL = (
    os.getenv("WEBDANCER_SUMMARY_MODEL_SERVER", "https://openrouter.ai/api/v1")
)
SUMMARY_MAX_RETRIES = max(1, int(os.getenv("WEBDANCER_SUMMARY_MAX_RETRIES", "3")))

EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content** 
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning**: Locate the **specific sections/data** directly related to the user's goal within the webpage content.
2. **Key Extraction**: Identify and extract the **most relevant information** from the content, you never miss any important information
3. **Summary Output**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.


**Final Output Format using JSON format**:
{{
  "rational": "string",
  "evidence": "string",
  "summary": "string",
}}
"""

FORMAT_REMINDER = (
    "Please follow the required format.\n"
    "When calling a tool, respond with:\n"
    "<tool_call>\n"
    "{\"name\": \"search or visit\", \"arguments\": {...}}\n"
    "</tool_call>\n"
    "Wrap the final response for the user inside <answer></answer> once you are done."
)

FINAL_ROUND_REMINDER = (
    "You have reached the final reasoning step. Review all observations above and produce your definitive answer "
    "wrapped inside <answer></answer>. Do not call any additional tools."
)


@dataclass
class WebDancerRequest:
    query: str


class WebDancerAgent(BaseAgent):
    def __init__(
        self,
        llm: LLMClient,
        *,
        tools: Optional[Iterable[BaseTool]] = None,
        max_steps: int = 20,
    ) -> None:
        default_tools = list(tools) if tools is not None else list(build_webdancer_tools().values())
        super().__init__(tools=default_tools, max_steps=max_steps)
        self.llm = llm
        self._summary_client: Optional[OpenAI] = None

    # step 1
    def handle_user_message(self, user_input: WebDancerRequest) -> AgentState:
        state = AgentState(user_input=user_input)
        state.messages.append(Message(role="system", content=build_webdancer_system_prompt()))
        state.messages.append(Message(role="user", content=build_webdancer_user_prompt(user_input.query)))
        state.scratchpad["query"] = user_input.query
        return state

    # step 2
    def generate_step_response(self, state: AgentState) -> str:
        if (
            state.steps_taken == self.max_steps
            and not state.scratchpad.get("final_round_prompted")
        ):
            state.messages.append(Message(role="user", content=FINAL_ROUND_REMINDER))
            state.scratchpad["final_round_prompted"] = True

        raw_output = self.llm.complete(self._messages_to_dicts(state.messages))
        return self._normalise_action_blocks(raw_output)

    # step 3
    def decide_next_action(self, state: AgentState) -> AgentDecision:
        if state.steps_taken > 1:
            if not state.messages or state.messages[-1].role != "assistant":
                return AgentDecision(kind="continue", message=None)

        latest = state.messages[-1].content
        if state.steps_taken > 1:

            answer = self._extract_answer(latest)
            if answer:
                return AgentDecision(kind="final", message=answer)

        action = self._extract_tool_call(latest)
        if action:
            name, arguments, trimmed = action
            if trimmed and trimmed != latest:
                state.messages[-1].content = trimmed
            if name in self.tools:
                return AgentDecision(kind="tool", tool_call=ToolCall(name=name, arguments=arguments))
            return AgentDecision(kind="continue")

        structured = self._extract_structured_tool_call(latest)
        if structured:
            name, arguments, trimmed_content = structured
            if trimmed_content != latest:
                state.messages[-1].content = trimmed_content
            if name in self.tools:
                return AgentDecision(kind="tool", tool_call=ToolCall(name=name, arguments=arguments))
            return AgentDecision(kind="continue")

        if state.steps_taken > 1 and state.messages and state.messages[-1].role == "assistant":
            last_reminder_step = state.scratchpad.get("last_reminder_step")
            if last_reminder_step != state.steps_taken:
                state.messages.append(Message(role="user", content=FORMAT_REMINDER))
                state.scratchpad["last_reminder_step"] = state.steps_taken

        return AgentDecision(kind="continue")

    # step 5
    def process_tool_result(self, state: AgentState, result: ToolResult) -> Optional[List[object]]:
        output_text = result.output
        tool_name = (result.call.name or "").lower()
        if tool_name in {"visit", "visitforlocalwiki"}:
            try:
                processed = self._post_process_visit_result(state, result)
                if processed:
                    output_text = processed
            except Exception:
                # Fallback silently to avoid breaking the agent loop.
                pass
        response = f"<tool_response>\n{output_text}\n</tool_response>"
        state.messages.append(Message(role="user", content=response))
        return [{"text": response, "append_to_dialogue": False, "stage": "log"}]

    def finalize_response(self, state: AgentState, decision: AgentDecision) -> str:
        if decision.message:
            return decision.message.strip()
        if state.messages and state.messages[-1].role == "assistant":
            fallback = self._extract_answer(state.messages[-1].content)
            if fallback:
                return fallback
            return state.messages[-1].content.strip()
        return "Unable to provide an answer, please try again.。"

    def _messages_to_dicts(self, messages: List[Message]) -> List[Dict[str, str]]:
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def _extract_tool_call(self, content: str) -> Optional[Tuple[str, Dict[str, object], str]]:
        match = None
        pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
        for candidate in pattern.finditer(content):
            match = candidate
        if match is None:
            return None

        raw_payload = match.group(1).strip()
        cleaned = self._strip_code_fences(raw_payload)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return None

        name = data.get("name")
        arguments = data.get("arguments", {})
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return None

        trimmed_content = content[: match.end()].strip()
        return name, arguments, trimmed_content

    def _extract_structured_tool_call(self, content: str) -> Optional[Tuple[str, Dict[str, object], str]]:
        start = content.find('{"name"')
        if start == -1:
            return None

        brace_count = 0
        end = None
        in_string = False
        escape_next = False
        for idx in range(start, len(content)):
            char = content[idx]
            if escape_next:
                escape_next = False
                continue
            if char == "\\" and in_string:
                escape_next = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    end = idx + 1
                    break
        if end is None:
            return None

        segment = content[start:end]
        try:
            data = json.loads(segment)
        except json.JSONDecodeError:
            return None

        name = data.get("name")
        arguments = data.get("arguments", {})
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return None

        trimmed_content = content[:end]
        return name, arguments, trimmed_content

    @staticmethod
    def _strip_code_fences(payload: str) -> str:
        text = payload.strip()
        if text.startswith("```") and text.endswith("```"):
            lines = [line for line in text.splitlines()]
            if len(lines) >= 2:
                lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines).strip()
        return text

    def _extract_answer(self, content: str) -> Optional[str]:
        match = re.search(r"<answer>\s*(.*?)\s*</answer>", content, re.IGNORECASE | re.DOTALL)
        if not match:
            return None

        answer = match.group(1).strip()
        return answer or None


    def _normalise_action_blocks(self, content: str) -> str:
        return content.strip()

    def _post_process_visit_result(self, state: AgentState, result: ToolResult) -> Optional[str]:
        if not SUMMARY_MODEL:
            return None

        client = self._get_summary_client()
        if client is None:
            return None

        goal = str(result.call.arguments.get("goal") if result.call and result.call.arguments else "").strip()
        if not goal:
            goal = str(state.scratchpad.get("query") or "").strip()

        raw_output = result.output or ""
        if not raw_output.strip():
            return None

        segments = [segment.strip() for segment in raw_output.split("\n=======\n") if segment.strip()]
        if not segments:
            return None

        summaries: List[str] = []
        for segment in segments:
            summary = self._summarise_visit_chunk(segment, goal, client)
            if summary is None:
                return None
            summaries.append(summary)

        return "\n=======\n".join(summaries)

    def _summarise_visit_chunk(self, chunk: str, goal: str, client: OpenAI) -> Optional[str]:
        summary_data = self._call_summary_llm(chunk, goal, client)
        if summary_data is None:
            return None

        title = self._extract_title_from_chunk(chunk)
        actionable_links = self._extract_actionable_links(chunk)
        formatted = self._format_summary_block(title, goal, summary_data)
        if actionable_links:
            formatted = f"{formatted}\n\nActionable Links:\n{actionable_links}"
        return formatted

    def _call_summary_llm(self, chunk: str, goal: str, client: OpenAI) -> Optional[Dict[str, str]]:
        payload = EXTRACTOR_PROMPT.format(webpage_content=chunk, goal=goal or "N/A")
        for _ in range(SUMMARY_MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=SUMMARY_MODEL,
                    messages=[{"role": "user", "content": payload}],
                    response_format={"type": "json_object"},
                )
            except Exception:
                continue

            raw = response.choices[0].message.content if response.choices else None
            if not raw:
                continue
            cleaned = raw.replace("```json", "").replace("```", "").strip()
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                return None

            if isinstance(data, dict):
                return {
                    "rational": str(data.get("rational", "")).strip(),
                    "evidence": str(data.get("evidence", "")).strip(),
                    "summary": str(data.get("summary", "")).strip(),
                }
        return None

    @staticmethod
    def _extract_title_from_chunk(chunk: str) -> str:
        match = re.search(r"^Title:\s*(.+)", chunk, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
        match = re.search(r"Useful information in ([^\n]+)", chunk, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    @staticmethod
    def _extract_actionable_links(chunk: str) -> Optional[str]:
        match = re.search(r"Actionable Links:\s*(.*)", chunk, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        text = match.group(1).strip()
        return text or None

    @staticmethod
    def _format_summary_block(title: str, goal: str, data: Dict[str, str]) -> str:
        lines: List[str] = []
        source_label = title or "Unspecified Source"
        lines.append(f"Source: {source_label}")
        if goal:
            lines.append(f"Goal: {goal}")
        rational = data.get("rational") or ""
        evidence = data.get("evidence") or ""
        summary = data.get("summary") or ""

        if rational:
            lines.append("")
            lines.append("Reasoning:")
            lines.append(rational)

        lines.append("")
        lines.append("Evidence:")
        lines.append(evidence or "No evidence extracted.")

        lines.append("")
        lines.append("Summary:")
        lines.append(summary or "No summary available.")
        return "\n".join(lines).strip()

    def _get_summary_client(self) -> Optional[OpenAI]:
        if self._summary_client is not None:
            return self._summary_client

        api_key = SUMMARY_API_KEY
        if not api_key:
            return None

        client_kwargs = {"api_key": api_key}
        if SUMMARY_BASE_URL:
            client_kwargs["base_url"] = SUMMARY_BASE_URL

        try:
            client = OpenAI(**client_kwargs)
        except Exception:
            return None

        self._summary_client = client
        return client
