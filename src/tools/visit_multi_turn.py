import asyncio
from typing import Dict

from ..agents.base import AgentState
from .base import BaseTool, ToolCall
from ..common.state import STATE
from ..common.utils import extract_links_with_text, get_info


def _normalize_button_label(raw_label: str) -> str:
    """Strip HTML-like wrappers the LLM might leave around a button label."""

    cleaned = raw_label.replace("<button>", "").replace("</button>", "")
    return cleaned.strip()


class VisitPage(BaseTool):
    """Navigate to a discovered sub-page and surface fresh content/buttons."""

    name = "visit_page"
    description = (
        "Follow a discovered button on the current website. The action input must be a JSON "
        'object with a single key "button" that matches one of the listed clickable buttons.'
    )

    def run(self, call: ToolCall, state: AgentState) -> str:
        button = call.arguments.get("button")
        if not isinstance(button, str):
            return 'Invalid input: expected {"button": "label"}.'

        label = _normalize_button_label(button)
        target_url = STATE.resolve_button(label)
        if not target_url:
            return f"The button '{label}' cannot be clicked from the current page."

        html, markdown, _ = asyncio.run(get_info(target_url, screenshot=False))
        STATE.visit_history.append(target_url)
        buttons = extract_links_with_text(html)

        if markdown:
            response = "\n🧭 website information:\n\n" + markdown + "\n\n"
        else:
            response = "\n🧭 The information of the current page is not accessible\n\n"

        if buttons:
            response += "\n🖱 clickable button:\n\n" + buttons + "\n\nEach button is wrapped in a <button> tag"
        else:
            response += (
                "\n🖱 clickable button:\n\nNo new clickable buttons found.\n\nEach button is wrapped in a <button> tag"
            )

        # Persist the latest observation details for downstream steps.
        state.scratchpad["last_url"] = target_url
        state.scratchpad["last_buttons"] = buttons
        state.scratchpad["last_markdown"] = markdown
        return response


TOOL_REGISTRY: Dict[str, BaseTool] = {VisitPage.name: VisitPage()}
