"""Prompt helpers for the WebDancer agent."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Tuple

from ..tools import build_webdancer_tools


def build_system_prompt() -> str:
    today = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    return (
        "You are a Web Information Seeking Master. Your task is to thoroughly seek the internet "
        "for information and provide accurate answers to questions with chinese language. "
        "You are also a Location-Based Services (LBS) assistant designed to help users find "
        "location-specific information. No matter how complex the query, you will not give up "
        "until you find the corresponding information.\n\n"
        "Principles:\n"
        "1. Persistent actions for answers — explore deeply until you have enough evidence.\n"
        "2. Repeated verification — cross-check facts before the final answer.\n"
        "3. Attention to detail — ensure sources are credible and up to date.\n\n"
        f"Current datetime: [{today}]. Consider the timestamp when responding."
    )


GUIDANCE_TEMPLATE = """The assistant communicates using tagged blocks:
<think> internal reasoning about the next action </think>
<tool_call>{{"name": "...", "arguments": {{...}}}}</tool_call>
<tool_response>tool output</tool_response>
Repeat as needed, then finish with:
<think>...</think>
<answer>final answer</answer>

Available tools (use exact names and valid JSON arguments):
{tool_descs}

Never invent new tool names. The `name` field must be one of: {tool_names}.
"""


def _render_tool_descriptions() -> Tuple[str, str]:
    tools = build_webdancer_tools()
    desc_blocks = []
    names = []
    for tool in tools.values():
        if tool.name in names:
            continue  # avoid duplicates from aliasing
        names.append(tool.name)
        schema = getattr(tool, "arguments_schema", None)
        schema_text = json.dumps(schema, ensure_ascii=False, indent=2) if schema else "{}"
        desc_blocks.append(f"{tool.name}: {tool.description}\nParameters:\n{schema_text}")
    return "\n\n".join(desc_blocks), ", ".join(names)


def build_user_prompt(query: str) -> str:
    tool_descs, tool_names = _render_tool_descriptions()
    guidance = GUIDANCE_TEMPLATE.format(tool_descs=tool_descs, tool_names=tool_names)
    return f"{guidance}\nUser: {query}"
