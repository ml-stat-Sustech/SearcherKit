from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class WebWalkerState:
    """Holds mutable runtime state for the CLI workflow."""

    root_url: str = ""
    button_url_map: Dict[str, str] = field(default_factory=dict)
    visit_history: List[str] = field(default_factory=list)

    def reset(self, root_url: str) -> None:
        """Initialize state for a new navigation session."""
        self.root_url = root_url
        self.button_url_map.clear()
        self.visit_history.clear()
        self.visit_history.append(root_url)

    def register_button(self, label: str, url: str) -> None:
        """Associate a button label with a fully-qualified URL."""
        self.button_url_map[label] = url

    def resolve_button(self, label: str) -> str:
        """Return the URL for a given button label, if known."""
        return self.button_url_map.get(label, "")


STATE = WebWalkerState()

