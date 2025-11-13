"""Prompt helpers for the WebDancer agent."""

from __future__ import annotations

from datetime import datetime
from typing import List

from ..tools import build_webdancer_tools


def build_system_prompt() -> str:
    today = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    tool_block = "\n".join(_render_tool_definitions())
    return (
        "You are a Web Information Seeking Master. Your task is to conduct thorough, multi-step research on the open web "
        "and provide accurate answers in English. Always gather evidence from reliable sources, verify information, "
        "and synthesise a final response only after you have sufficient support.\n\n"
        "You may invoke callable tools to assist you. The available functions are listed within <tools> tags:\n"
        "<tools>\n"
        f"{tool_block}\n"
        "</tools>\n\n"
        "When you call a function, you must output a single JSON object wrapped in <tool_call></tool_call> tags. Example:\n"
        "<tool_call>\n"
        '{"name": "search", "arguments": {"query": ["example query"]}}\n'
        "</tool_call>\n\n"
        "For every tool output you receive, integrate the information and decide on your next step. "
        "When you are ready to respond to the user, enclose the final answer within <answer></answer> tags.\n\n"
        f"Current datetime: [{today}]. Consider this when reasoning."
    )


def _render_tool_definitions() -> List[str]:
    tools = build_webdancer_tools()
    entries: List[str] = []
    seen = set()
    for tool in tools.values():
        if tool.name in seen:
            continue
        seen.add(tool.name)
        entry = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": _tool_parameters_schema(tool.name),
            },
        }
        entries.append(_json_dumps(entry))
    return entries


def build_user_prompt(query: str) -> str:
    return (
        "User request:\n"
        f"{query}\n\n"
        "Respond by thinking through the task, calling tools as needed via <tool_call> tags, "
        "and provide the final answer inside <answer></answer> when you are done."
    )


def _tool_parameters_schema(name: str):
    if name == "search":
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "Array of search queries to issue.",
                }
            },
            "required": ["query"],
        }
    if name in {"visit", "VisitForLocalWiki"}:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": ["string", "array"],
                    "items": {"type": "string"},
                    "description": "One or more webpage URLs or titles to open.",
                },
                "goal": {
                    "type": "string",
                    "description": "Optional description of what information you need from the page(s).",
                },
            },
            "required": ["url"],
        }
    return {"type": "object", "properties": {}}


def _json_dumps(obj: object) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)
