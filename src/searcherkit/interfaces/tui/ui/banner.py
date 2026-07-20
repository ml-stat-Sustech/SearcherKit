from __future__ import annotations

from searcherkit.interfaces.tui.ui.formatting import _display_width

BANNER_TEXT = "Searcher Kit"

# SEARCHER block (left-aligned) and KIT block (right-aligned to SEARCHER width).
_SEARCHER_ART = """\
███████╗███████╗ █████╗ ██████╗  ██████╗██╗  ██╗███████╗██████╗
██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝██║  ██║██╔════╝██╔══██╗
███████╗█████╗  ███████║██████╔╝██║     ███████║█████╗  ██████╔╝
╚════██║██╔══╝  ██╔══██║██╔══██╗██║     ██╔══██║██╔══╝  ██╔══██╗
███████║███████╗██║  ██║██║  ██║╚██████╗██║  ██║███████╗██║  ██║
╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝"""

# Slanted (italic) KIT: leading spaces encode the forward lean. Absolute
# horizontal placement is applied in ``_banner_art_lines``.
_KIT_ART = """\
                        ██╗  ██╗██╗████████╗
                       ██║ ██╔╝██║╚══██╔══╝
                      █████╔╝ ██║   ██║
                     ██╔═██╗ ██║   ██║
                    ██║  ██╗██║   ██║
                   ╚═╝  ╚═╝╚═╝   ╚═╝"""

BANNER_ART_SMALL = """\
 __
(_  _  _..__|_  _ ._ |/o_|_
__)(/_(_||(_| |(/_|  |\\| |_"""

# Aurora gradient: cyan -> indigo -> violet -> pink.
_GRADIENT_STOPS = ("#22d3ee", "#818cf8", "#c084fc", "#f472b6")
_GRADIENT_STOPS_RGB = tuple(
    tuple(int(stop[i : i + 2], 16) for i in (1, 3, 5)) for stop in _GRADIENT_STOPS
)
# Quantization step for gradient positions; larger steps merge more characters
# into a single styled run.
_GRADIENT_QUANTUM = 24
# Rows lean into the gradient by this many columns per line (diagonal sweep).
_GRADIENT_ROW_DRIFT = 3


def _banner_art_lines() -> list[str]:
    searcher = _SEARCHER_ART.splitlines()
    art_width = max(len(line) for line in searcher)
    kit = _right_align_slanted_kit(_KIT_ART.splitlines(), art_width=art_width)
    return [*searcher, "", *kit]


def _right_align_slanted_kit(kit_lines: list[str], *, art_width: int) -> list[str]:
    """Keep KIT's italic lean; shift so its rightmost edge matches ``art_width``."""
    max_end = max((len(line.rstrip()) for line in kit_lines), default=0)
    shift = art_width - max_end
    aligned: list[str] = []
    for line in kit_lines:
        body = line.rstrip()
        shifted = (" " * max(0, shift)) + body
        if len(shifted) > art_width:
            shifted = shifted[-art_width:]
        aligned.append(shifted.ljust(art_width))
    return aligned


BANNER_ART = "\n".join(_banner_art_lines())


def render_banner(*, chat_width: int) -> list[tuple[str, str]]:
    """Render the startup banner as prompt-toolkit style parts.

    Picks the largest art variant that fits ``chat_width``, centers it
    horizontally, and colors each non-space character with the aurora
    gradient swept diagonally across the art.
    """
    lines = _select_lines(chat_width)
    width = max(len(line) for line in lines)
    pad = " " * max(0, (chat_width - width) // 2)
    parts: list[tuple[str, str]] = []
    for row, line in enumerate(lines):
        parts.extend(_gradient_line_parts(line, row=row, width=width, pad=pad))
    return parts


def render_splash(
    *,
    chat_width: int,
    model_label: str = "",
    source_label: str = "",
    tool_label: str = "",
) -> list[tuple[str, str]]:
    """Render the startup splash: gradient banner with model/source/tool left of KIT.

    Vertical centering is the caller's job; this only produces the content
    lines, horizontally centered within ``chat_width``.
    """
    lines = _select_lines(chat_width)
    width = max(len(line) for line in lines)
    pad = " " * max(0, (chat_width - width) // 2)
    kit_start = _kit_start_row(lines)
    status_rows = _status_rows_for_kit(
        lines,
        kit_start=kit_start,
        art_width=width,
        model_label=model_label,
        source_label=source_label,
        tool_label=tool_label,
    )
    parts: list[tuple[str, str]] = []
    for row, line in enumerate(lines):
        status = status_rows.get(row)
        if status is None:
            parts.extend(_gradient_line_parts(line, row=row, width=width, pad=pad))
            continue
        parts.extend(
            _kit_status_line_parts(
                line,
                status=status,
                row=row,
                width=width,
                pad=pad,
            )
        )
    return parts


def _kit_start_row(lines: list[str]) -> int | None:
    """Return the first KIT row index, or None when the art has no KIT block."""
    if lines == _banner_art_lines():
        return len(_SEARCHER_ART.splitlines()) + 1
    return None


def _status_rows_for_kit(
    lines: list[str],
    *,
    kit_start: int | None,
    art_width: int,
    model_label: str,
    source_label: str,
    tool_label: str = "",
) -> dict[int, list[tuple[str, str]]]:
    """Map art row indices to left-gutter segments (bordered model/source/tool panel)."""
    if not model_label and not source_label and not tool_label:
        return {}
    if kit_start is None:
        # Compact art has no KIT gutter. Keep the fallback width-safe instead
        # of concatenating selection labels with the artwork.
        return {}

    kit_lines = lines[kit_start:]
    panel_height = len(kit_lines)
    # Stay clear of the leftmost slanted KIT column across all rows.
    panel_width = min(_leading_space_budget(line) for line in kit_lines)
    if panel_width <= 1:
        return {}
    panel_rows = _build_half_enclosed_status_panel(
        panel_width=panel_width,
        panel_height=panel_height,
        model_label=model_label,
        source_label=source_label,
        tool_label=tool_label,
    )
    return {kit_start + offset: row for offset, row in enumerate(panel_rows)}


def _build_half_enclosed_status_panel(
    *,
    panel_width: int,
    panel_height: int,
    model_label: str,
    source_label: str,
    tool_label: str = "",
) -> list[list[tuple[str, str]]]:
    """Build a left+bottom half-enclosure in the KIT gutter.

    The bottom edge is pinned to the last KIT row. Model/Source/Tool are
    left-aligned with each other and centered horizontally in the gutter;
    vertically they are centered in the rows above the bottom edge::

        │  Model:...
        │  Source:...
        │  Tool:...
        ╰──────────   ← aligns with KIT's bottom row
    """
    border = "class:splash-border"
    if panel_width <= 0 or panel_height <= 0:
        return [[("", " " * max(0, panel_width))] for _ in range(max(0, panel_height))]

    entries: list[list[tuple[str, str]]] = []
    if model_label:
        entries.append(
            [("class:splash-label", "Model:"), ("class:splash-model", model_label)]
        )
    if source_label:
        entries.append(
            [("class:splash-label", "Source:"), ("class:splash-source", source_label)]
        )
    if tool_label:
        entries.append(
            [("class:splash-label", "Tool:"), ("class:splash-tool", tool_label)]
        )
    if not entries:
        return [[("", " " * panel_width)] for _ in range(panel_height)]

    inner_width = max(0, panel_width - 1)
    fitted = [_fit_status(entry, budget=inner_width) for entry in entries]
    block_width = max(
        (sum(_display_width(text) for _, text in entry) for entry in fitted),
        default=0,
    )
    # Box spans the gutter; Model/Source/Tool stay left-aligned and centered as a block.
    box_inner = inner_width
    box_width = box_inner + 1
    box_left = max(0, (panel_width - box_width) // 2)
    box_right = max(0, panel_width - box_left - box_width)
    content_left = max(0, (box_inner - block_width) // 2)

    def _content_row(status: list[tuple[str, str]] | None) -> list[tuple[str, str]]:
        if status is None:
            return [
                ("", " " * box_left),
                (border, "│"),
                ("", " " * box_inner),
                ("", " " * box_right),
            ]
        status_width = sum(_display_width(text) for _, text in status)
        right_fill = max(0, box_inner - content_left - status_width)
        return [
            ("", " " * box_left),
            (border, "│"),
            ("", " " * content_left),
            *status,
            ("", " " * right_fill),
            ("", " " * box_right),
        ]

    # Last row is always the rounded bottom edge, aligned with KIT's bottom.
    content_height = max(0, panel_height - 1)
    top_pad = max(0, (content_height - len(fitted)) // 2)
    rows: list[list[tuple[str, str]]] = []
    for index in range(content_height):
        entry_index = index - top_pad
        if 0 <= entry_index < len(fitted):
            rows.append(_content_row(fitted[entry_index]))
        else:
            rows.append(_content_row(None))

    if box_width == 1:
        bottom = [("", " " * box_left), (border, "╰"), ("", " " * box_right)]
    else:
        bottom = [
            ("", " " * box_left),
            (border, "╰" + "─" * (box_width - 1)),
            ("", " " * box_right),
        ]
    rows.append(bottom)
    return rows[:panel_height]


def _leading_space_budget(line: str) -> int:
    stripped = line.lstrip(" ")
    if not stripped:
        return len(line)
    # Leave one blank column between the panel and KIT glyphs.
    return max(0, len(line) - len(stripped) - 1)


def _fit_status(
    status: list[tuple[str, str]], *, budget: int
) -> list[tuple[str, str]]:
    width = sum(_display_width(text) for _, text in status)
    if width <= budget:
        return status
    plain = "".join(text for _, text in status)
    if budget <= 0:
        return []
    if budget == 1:
        return [("class:splash-label", "…")]
    return [("class:splash-label", plain[: budget - 1] + "…")]


def _kit_status_line_parts(
    line: str,
    *,
    status: list[tuple[str, str]],
    row: int,
    width: int,
    pad: str,
) -> list[tuple[str, str]]:
    """Render one art line with a left-gutter panel segment beside KIT."""
    if not status:
        return _gradient_line_parts(line, row=row, width=width, pad=pad)
    status_width = sum(_display_width(text) for _, text in status)
    kit_index = len(line) - len(line.lstrip(" "))
    # Panel sits in the left gutter; KIT glyphs keep their absolute columns
    # so right-alignment against SEARCHER is preserved.
    fill = max(0, kit_index - status_width)
    kit = line[kit_index:] if kit_index < len(line) else ""
    parts: list[tuple[str, str]] = [("", pad)] if pad else []
    parts.extend(status)
    if fill:
        parts.append(("", " " * fill))
    if kit:
        parts.extend(
            _gradient_run_parts(kit, row=row, width=width, column_offset=kit_index)
        )
    trailing = max(0, width - status_width - fill - len(kit))
    if trailing:
        parts.append(("", " " * trailing))
    parts.append(("", "\n"))
    return parts


def _select_lines(chat_width: int) -> list[str]:
    for art in (BANNER_ART, BANNER_ART_SMALL):
        lines = art.splitlines()
        if max(len(line) for line in lines) <= chat_width:
            return lines
    return [BANNER_TEXT]


def _gradient_line_parts(
    line: str, *, row: int, width: int, pad: str
) -> list[tuple[str, str]]:
    if not line:
        return [("", "\n")]
    parts: list[tuple[str, str]] = [("", pad)] if pad else []
    parts.extend(_gradient_run_parts(line, row=row, width=width, column_offset=0))
    parts.append(("", "\n"))
    return parts


def _gradient_run_parts(
    text: str, *, row: int, width: int, column_offset: int
) -> list[tuple[str, str]]:
    parts: list[tuple[str, str]] = []
    style = None
    buffer: list[str] = []
    for column, char in enumerate(text):
        char_style = (
            ""
            if char == " "
            else _gradient_style(column_offset + column + row * _GRADIENT_ROW_DRIFT, width)
        )
        if char_style != style and buffer:
            parts.append((style or "", "".join(buffer)))
            buffer = []
        style = char_style
        buffer.append(char)
    if buffer:
        parts.append((style or "", "".join(buffer)))
    return parts


def _gradient_style(position: int, width: int) -> str:
    span = max(width - 1, 1)
    t = min(1.0, max(0.0, position / span))
    t = round(t * _GRADIENT_QUANTUM) / _GRADIENT_QUANTUM
    segment = t * (len(_GRADIENT_STOPS_RGB) - 1)
    index = min(int(segment), len(_GRADIENT_STOPS_RGB) - 2)
    fraction = segment - index
    start = _GRADIENT_STOPS_RGB[index]
    end = _GRADIENT_STOPS_RGB[index + 1]
    rgb = tuple(round(a + (b - a) * fraction) for a, b in zip(start, end, strict=True))
    return "#{:02x}{:02x}{:02x}".format(*rgb)
