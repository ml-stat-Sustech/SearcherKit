"""Prompt helpers for the WebDancer agent."""

from __future__ import annotations

from typing import TYPE_CHECKING
from datetime import datetime
from typing import List

if TYPE_CHECKING:
    from ...tools import BaseTool
    from ...agents import WebDancerAgent


def build_system_prompt(agent: WebDancerAgent) -> str:
    today = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    tool_block = "\n".join(_render_tool_definitions(list(agent.tools.values())))
    return (
        "You are a Web Information Seeking Master. Your task is to thoroughly seek the internet for information and provide accurate answers to questions. with chinese language." \
        "And you are also a Location-Based Services (LBS) assistant designed to help users find location-specific information." \
        "No matter how complex the query, you will not give up until you find the corresponding information.\n\nAs you proceed, adhere to the following principles:\n\n" \
        "1. **Persistent Actions for Answers**: You will engage in many interactions, delving deeply into the topic to explore all possible aspects until a satisfactory answer is found.\n\n" \
        "2. **Repeated Verification**: Before presenting a Final Answer, you will **cross-check** and **validate the information** you've gathered to confirm its accuracy and reliability.\n\n" \
        "3. **Attention to Detail**: You will carefully analyze each information source to ensure that all data is current, relevant, and from credible origins.\n\n" \
        f" When responding, consider the time to provide contextually relevant information."
        "The available functions are listed within < tools > tags:\n"
        "<tools>\n"
        f"{tool_block}\n"
        "</tools>\n\n"
    )


def _render_tool_definitions(tools: List[BaseTool]) -> List[str]:
    entries: List[str] = []
    seen = set()
    for tool in tools:
        if tool.name in seen:
            continue
        seen.add(tool.name)
        entry = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.arguments_schema,
            },
        }
        entries.append(_json_dumps(entry))
    return entries


def build_user_prompt(query: str) -> str:
    # 注意：由于Qwen3系列的思维链十分冗长，为了提升效率， 在此关闭思维链模式，并使用<thought>来引导模型输出简短的分析和justification
    return (
        '''The assistant starts with one or more cycles of (thinking about which tool to use -> performing tool call -> waiting for tool response), and ends with (thinking about the answer -> answer of the question). The thinking processes, tool calls, tool responses, and answer are enclosed within their tags. There could be multiple thinking processes, tool calls, tool call parameters and tool response parameters.

Example response:
<thought> thinking process here </thought>
<tool_call>
{"name": "tool name here", "arguments": {"parameter name here": parameter value here, "another parameter name here": another parameter value here, ...}}
</tool_call>
<tool_response>
tool_response here
</tool_response>
<thought> thinking process here </thought>
<tool_call>
{"name": "another tool name here", "arguments": {...}}
</tool_call>
<tool_response>
tool_response here
</tool_response>
(more thinking processes, tool calls and tool responses here)
<thought> thinking process here </thought>
<answer> answer here </answer>

User: ''' + query
    )


# def _tool_parameters_schema(name: str):
#     if name == "search":
#         return {
#             "type": "object",
#             "properties": {
#                 "query": {
#                     "type": "array",
#                     "items": {"type": "string"},
#                     "minItems": 1,
#                     "description": "Array of search queries to issue.",
#                 }
#             },
#             "required": ["query"],
#         }
#     if name in {"visit", "VisitForLocalWiki"}:
#         return {
#             "type": "object",
#             "properties": {
#                 "url": {
#                     "type": ["string", "array"],
#                     "items": {"type": "string"},
#                     "description": "One or more webpage URLs or titles to open.",
#                 },
#                 "goal": {
#                     "type": "string",
#                     "description": "Optional description of what information you need from the page(s).",
#                 },
#             },
#             "required": ["docid"],
#         }
#     return {"type": "object", "properties": {}}


def _json_dumps(obj: object) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)
