from __future__ import annotations

from dataclasses import dataclass

from searcherkit.interfaces.tui.ui.formatting import _display_width


@dataclass(frozen=True, slots=True)
class SplashNotice:
    """One-line transient feedback shown below the startup banner."""

    text: str
    style: str = "class:splash-notice"


def render_splash_notice(
    notice: SplashNotice, *, width: int
) -> list[tuple[str, str]]:
    """Render a notice as one width-bounded line."""

    text = _truncate_notice(notice.text, width=max(0, width))
    text_width = _display_width(text)
    left_pad = max(0, (width - text_width) // 2)
    parts: list[tuple[str, str]] = []
    if left_pad:
        parts.append(("", " " * left_pad))
    if text:
        parts.append((notice.style, text))
    parts.append(("", "\n"))
    return parts


def _truncate_notice(text: str, *, width: int) -> str:
    normalized = " ".join(text.splitlines()).strip()
    if width <= 0:
        return ""
    if _display_width(normalized) <= width:
        return normalized
    ellipsis = "…"
    if _display_width(ellipsis) > width:
        return ""
    prefix: list[str] = []
    current_width = 0
    target_width = width - _display_width(ellipsis)
    for char in normalized:
        char_width = _display_width(char)
        if current_width + char_width > target_width:
            break
        prefix.append(char)
        current_width += char_width
    return "".join(prefix) + ellipsis
