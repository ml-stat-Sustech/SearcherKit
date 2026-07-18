from __future__ import annotations

import asyncio
from typing import Any

from searcherkit.agent.search_agent import SearchAgentConfig
from searcherkit.interfaces.tui.app import SearcherKitTui
from searcherkit.interfaces.tui.chat.conversation_entry import ConversationEntry
from searcherkit.runtime.interactive_selection import ModelOption, apply_active_model, discover_model_options, parse_model_command
from searcherkit.interfaces.tui.slash.slash_command import SlashCommandMenuState, TuiCommand
from searcherkit.interfaces.tui.ui.formatting import _display_width, _formatted_lines
from searcherkit.interfaces.tui.ui.selection_types import ChatPoint, ChatSelection
from searcherkit.llm.base import ClientConfig
from searcherkit.runtime.interactive import InteractiveQueryConfig
from searcherkit.common.live_events import LiveEvent
from searcherkit.sources import SourceConfig
from searcherkit.tools.base import ToolConfig


class _FakeBuffer:
    cursor_position = 0


class _FakeInput:
    def __init__(self, text: str = "") -> None:
        self.text = text
        self.buffer = _FakeBuffer()


def _render_text(app: SearcherKitTui) -> str:
    return "".join(text for _, text in app._render_chat_full())


def _set_test_input(app: SearcherKitTui, text: str = "") -> None:
    app.input_field.set_input_field(_FakeInput(text))


def _set_test_slash_menu(app: SearcherKitTui, commands: list[Any]) -> None:
    app.slash_menu.menu_state = SlashCommandMenuState(commands)


def _make_app(config: InteractiveQueryConfig | None = None, **view_overrides: Any) -> SearcherKitTui:
    if config is None:
        config = InteractiveQueryConfig(
            record_dir="/tmp/tui_test",
            agent=SearchAgentConfig(sources=[], tools=[]),
        )
    app = SearcherKitTui(config=config)
    for key, value in view_overrides.items():
        setattr(app.view_state, key, value)
    _set_test_input(app, "")
    return app


def _append_selection_entry(app: SearcherKitTui, prefix: str, label: str) -> None:
    app.chat_history.append_selection_entry(prefix, label)
    app.view_state.chat_scroll_top = None


def test_tui_hides_thinking_content_without_truncating_transcript() -> None:
    app = _make_app()
    app.chat_history._entries = [
        ConversationEntry(role="user", title="User", body="question", style="class:user"),
        ConversationEntry(
            role="assistant",
            title="Assistant",
            thinking="secret chain of thought",
            body="visible answer",
            style="class:assistant",
        ),
        ConversationEntry(role="tool-call", title="Tool call: search", body='{"query": "demo"}', style="class:tool"),
        ConversationEntry(role="tool-result", title="Tool result: search", body="result", style="class:tool-result"),
    ]
    app.view_state.show_thinking = False

    text = _render_text(app)

    assert "visible answer" in text
    assert "secret chain of thought" not in text
    assert "showing latest" not in text
    assert "Tool call: search" in text
    assert "Tool result: search" in text


def test_tui_shows_thinking_by_default() -> None:
    app = _make_app()
    app.chat_history._entries = [
        ConversationEntry(
            role="thinking",
            title="thinking completed",
            thinking="default visible thinking",
            style="class:thinking",
            status="completed",
        ),
    ]

    text = _render_text(app)

    assert "default visible thinking" in text


def test_tui_can_show_thinking_when_toggled() -> None:
    app = _make_app()
    app.chat_history._entries = [
        ConversationEntry(
            role="thinking",
            title="thinking completed",
            thinking="visible thinking",
            style="class:thinking",
            status="completed",
        ),
        ConversationEntry(
            role="assistant",
            title="ASSISTANT",
            body="visible answer",
            style="class:assistant",
        ),
    ]
    app.view_state.show_thinking = True

    text = _render_text(app)

    assert "visible thinking" in text
    assert "visible answer" in text
    assert "──" not in text


def test_tui_body_wrap_width_follows_terminal_width() -> None:
    app = _make_app()
    app.chat_history._entries = [
        ConversationEntry(
            role="assistant",
            title="Assistant",
            body="alpha beta gamma delta epsilon zeta",
            style="class:assistant",
        )
    ]
    app.view_state.show_thinking = True

    class FakeOutput:
        columns = 30

        def get_size(self):
            class Size:
                rows = 30
                columns = FakeOutput.columns

            return Size()

    class FakeApp:
        output = FakeOutput()

    app._pt_app = FakeApp()

    narrow_text = _render_text(app)
    FakeOutput.columns = 80
    wide_text = _render_text(app)

    assert "  alpha beta gamma delta\n  epsilon zeta" in narrow_text
    assert "  alpha beta gamma delta epsilon zeta" in wide_text


def test_tui_shaded_blocks_pad_cjk_content_by_display_width() -> None:
    app = _make_app()
    app.chat_history._entries = [
        ConversationEntry(role="user", title="USER", body="中文问题是什么？请回答。", style="class:user"),
        ConversationEntry(
            role="tool-interaction",
            title="search",
            meta="turn 1",
            style="class:tool-result",
            status="completed",
            arguments={"query": "中文检索"},
            result="中文结果第一行\n中文结果第二行",
        ),
        ConversationEntry(
            role="assistant",
            title="FINAL ANSWER",
            body="这是最终答案。",
            style="class:final-answer",
        ),
    ]
    app.view_state.show_thinking = False

    class FakeOutput:
        columns = 40

        def get_size(self):
            class Size:
                rows = 30
                columns = FakeOutput.columns

            return Size()

    class FakeApp:
        output = FakeOutput()

    app._pt_app = FakeApp()
    block_width = app.renderer._block_shadow_width(chat_width=app.layout_geometry.chat_view_width())
    shaded_styles = {
        "class:user-shadow",
        "class:user",
        "class:user-body",
        "class:tool-shadow",
        "class:tool-result",
        "class:tool-body",
        "class:tool-result-body",
        "class:final-answer-shadow",
        "class:final-answer",
        "class:final-answer-body",
    }

    widths = []
    for line in _formatted_lines(app._render_chat_full()):
        if any(style in shaded_styles or style == "class:tool-name" for style, _ in line):
            widths.append(_display_width("".join(text for _, text in line)))

    assert widths
    assert all(width == block_width for width in widths)


def test_tui_rule_title_width_follows_terminal_without_upper_cap() -> None:
    app = _make_app()

    class FakeOutput:
        columns = 140

        def get_size(self):
            class Size:
                rows = 30
                columns = FakeOutput.columns

            return Size()

    class FakeApp:
        output = FakeOutput()

    app._pt_app = FakeApp()

    assert len(app.renderer._rule_title("Assistant", chat_width=app.layout_geometry.chat_view_width())) == 137
    FakeOutput.columns = 80
    assert len(app.renderer._rule_title("Assistant", chat_width=app.layout_geometry.chat_view_width())) == 77


def test_tui_selection_entries_use_kind_titles_and_replace_consecutive_changes() -> None:
    app = _make_app()
    app.chat_history._entries = []
    app.view_state.chat_scroll_top = 0

    _append_selection_entry(app, "Model", "openai/qwen3:0.6b")
    _append_selection_entry(app, "Model", "openai/qwen3:1.7b")
    _append_selection_entry(app, "Source", "wiki")
    _append_selection_entry(app, "Source", "bcp")
    _append_selection_entry(app, "Model", "openai/qwen3:4b")

    assert [(entry.title, entry.body) for entry in app.chat_history.entries()] == [
        ("Active Model", "openai/qwen3:1.7b"),
        ("Active Source", "bcp"),
        ("Active Model", "openai/qwen3:4b"),
    ]
    assert app.view_state.chat_scroll_top is None


def test_tui_renders_interactive_events() -> None:
    app = _make_app()
    app.chat_history._entries = []
    app.view_state.show_thinking = False
    app.view_state.show_tool_detail = False
    app._pt_app = None

    app.chat_history.append_event(LiveEvent(kind="run_started", message="started"))
    app.chat_history.append_event(LiveEvent(kind="user_message", message="question"))
    app.chat_history.append_event(
        LiveEvent(
            kind="tool_call_started",
            message="search",
            data={"id": "call-1", "name": "search", "arguments": {"query": "demo"}, "turn": 1},
        )
    )

    assert [entry.role for entry in app.chat_history.entries()] == ["user", "tool-interaction"]
    assert app.chat_history.entries()[1].status == "running"
    assert "TOOL · turn 1" in _render_text(app)
    assert 'search "demo"' in _render_text(app)

    app.chat_history.append_event(
        LiveEvent(
            kind="tool_result",
            message="result",
            data={"id": "call-1", "name": "search", "result": "result", "status": "completed"},
        )
    )
    app.chat_history.append_event(LiveEvent(kind="run_completed", message="done"))

    assert [entry.role for entry in app.chat_history.entries()] == ["user", "tool-interaction"]
    assert app.chat_history.entries()[1].status == "completed"
    rendered = _render_text(app)
    assert app.chat_history.entries()[1].arguments == {"query": "demo"}
    assert "TOOL · turn 1" in rendered
    assert 'search "demo"' in rendered
    assert "result" in rendered


def test_tui_new_query_clears_existing_entries(tmp_path) -> None:
    async def run_case() -> None:
        app = SearcherKitTui(config=InteractiveQueryConfig(record_dir=str(tmp_path)))
        app.chat_history._entries = [ConversationEntry(role="assistant", title="Old", body="old", style="class:assistant")]
        app._pt_app = None
        app.query_controller._current_task = None

        class FakeRunner:
            def __init__(self, *, config):
                self.config = config

            async def run_query(self, query, *, live_event_sink=None):
                from searcherkit.runtime.interactive import InteractiveRunResult
                await live_event_sink(LiveEvent(kind="user_message", message="new"))
                return InteractiveRunResult(status="completed", record_path=tmp_path / "record.json", payload={})

        import searcherkit.interfaces.tui.runtime.query_controller as query_controller

        original = query_controller.InteractiveQueryRunner
        query_controller.InteractiveQueryRunner = FakeRunner
        try:
            await app.query_controller.run_query("new question")
        finally:
            query_controller.InteractiveQueryRunner = original

        assert "old" not in _render_text(app)
        assert "new" in _render_text(app)

    import asyncio

    asyncio.run(run_case())


def test_tui_keeps_thinking_separate_and_marks_it_completed() -> None:
    app = _make_app()
    app.chat_history._entries = []
    app.view_state.show_thinking = True
    app.view_state.show_tool_detail = False
    app._pt_app = None

    app.chat_history.append_event(LiveEvent(kind="assistant_turn_started", message="Assistant turn 1", data={"turn": 1}))
    running_text = _render_text(app)
    app.chat_history.append_event(
        LiveEvent(
            kind="assistant_message",
            message="assistant",
            data={"turn": 1, "thinking": "reasoning details", "content": "assistant text", "tool_calls": []},
        )
    )
    completed_text = _render_text(app)

    assert "thinking... · turn 1" not in running_text
    assert "THINKING" not in running_text
    assert [entry.role for entry in app.chat_history.entries()] == ["thinking", "assistant"]
    assert app.chat_history.entries()[0].status == "completed"
    assert "[✓] thinking completed · turn 1" not in completed_text
    assert "THINKING" not in completed_text
    assert "reasoning details" in completed_text
    assert "FINAL ANSWER · turn 1" in completed_text
    assert "assistant text" in completed_text


def test_tui_streams_assistant_delta_and_finalizes_existing_entry() -> None:
    app = _make_app()
    app.chat_history._entries = []
    app.view_state.show_thinking = False
    app.view_state.show_tool_detail = False
    app._pt_app = None

    app.chat_history.append_event(LiveEvent(kind="assistant_turn_started", message="Assistant turn 1", data={"turn": 1}))
    app.chat_history.append_event(
        LiveEvent(kind="assistant_delta", message="hello ", data={"turn": 1, "field": "content", "delta": "hello "})
    )
    app.chat_history.append_event(
        LiveEvent(kind="assistant_delta", message="world", data={"turn": 1, "field": "content", "delta": "world"})
    )

    streaming_text = _render_text(app)
    assert "hello world" in streaming_text
    assert [entry.role for entry in app.chat_history.entries()] == ["assistant"]
    assert app.chat_history.entries()[0].status == "streaming"

    app.chat_history.append_event(
        LiveEvent(
            kind="assistant_message",
            message="assistant",
            data={"turn": 1, "content": "hello world", "thinking": "", "tool_calls": []},
        )
    )

    assert [entry.role for entry in app.chat_history.entries()] == ["assistant"]
    assert app.chat_history.entries()[0].status == ""
    finalized_text = _render_text(app)
    assert "FINAL ANSWER · turn 1" in finalized_text
    assert "hello world" in finalized_text


def test_tui_streams_thinking_delta_and_respects_visibility_toggle() -> None:
    app = _make_app()
    app.chat_history._entries = []
    app.view_state.show_thinking = False
    app.view_state.show_tool_detail = False
    app._pt_app = None

    app.chat_history.append_event(
        LiveEvent(
            kind="assistant_delta",
            message="secret-reasoning",
            data={"turn": 1, "field": "thinking", "delta": "secret-reasoning"},
        )
    )

    hidden_text = _render_text(app)
    assert "secret-reasoning" not in hidden_text
    assert "reasoning block(s) hidden" in hidden_text

    app.view_state.show_thinking = True
    visible_text = _render_text(app)
    assert "secret-reasoning" in visible_text


def test_tui_marks_toolless_assistant_message_as_final_answer() -> None:
    app = _make_app()
    app.chat_history._entries = []
    app.view_state.show_thinking = False
    app.view_state.show_tool_detail = False
    app._pt_app = None

    app.chat_history.append_event(
        LiveEvent(
            kind="assistant_message",
            message="answer",
            data={"turn": 2, "content": "final answer", "tool_calls": []},
        )
    )

    text = _render_text(app)

    assert "FINAL ANSWER · turn 2" in text
    assert "final answer" in text


def test_tui_shows_only_answer_tag_content_for_final_answer() -> None:
    app = _make_app()
    app.chat_history._entries = []
    app.view_state.show_thinking = False
    app.view_state.show_tool_detail = False
    app._pt_app = None

    app.chat_history.append_event(
        LiveEvent(
            kind="assistant_message",
            message="answer",
            data={
                "turn": 2,
                "content": "reasoning summary <answer>Paris</answer> trailing note",
                "tool_calls": [],
            },
        )
    )

    text = _render_text(app)

    assert "FINAL ANSWER · turn 2" in text
    assert "Paris" in text
    assert "reasoning summary" not in text
    assert "trailing note" not in text
    assert "<answer>" not in text
    assert "</answer>" not in text


def test_tui_does_not_extract_answer_tag_content_for_tool_call_turn() -> None:
    app = _make_app()
    app.chat_history._entries = []
    app.view_state.show_thinking = False
    app.view_state.show_tool_detail = False
    app._pt_app = None

    app.chat_history.append_event(
        LiveEvent(
            kind="assistant_message",
            message="assistant",
            data={
                "turn": 1,
                "content": "Use this later: <answer>Paris</answer>",
                "tool_calls": [{"id": "call-1", "name": "search", "arguments": {}}],
            },
        )
    )

    text = _render_text(app)

    assert "ASSISTANT" not in text
    assert "──" not in text
    assert "Use this later: <answer>Paris</answer>" in text


def test_tui_shows_requested_tool_count_when_tool_turn_has_no_message() -> None:
    app = _make_app()
    app.chat_history._entries = []
    app.view_state.show_thinking = False
    app.view_state.show_tool_detail = False
    app._pt_app = None

    app.chat_history.append_event(
        LiveEvent(
            kind="assistant_message",
            message="assistant",
            data={
                "turn": 1,
                "content": None,
                "tool_calls": [{"id": "call-1", "name": "search", "arguments": {"query": "demo"}}],
            },
        )
    )

    entries = list(app.chat_history.entries())
    assert len(entries) == 1
    assert entries[0].body == "Requested 1 tool call(s)."
    assert entries[0].title == "ASSISTANT"
    text = _render_text(app)
    assert "Requested 1 tool call(s)." in text
    assert "ASSISTANT ·" not in text
    assert "──" not in text


def test_tui_shows_requested_tool_count_when_content_is_whitespace_only() -> None:
    app = _make_app()
    app.chat_history._entries = []
    app.view_state.show_thinking = False
    app.view_state.show_tool_detail = False
    app._pt_app = None

    app.chat_history.append_event(
        LiveEvent(
            kind="assistant_message",
            message="assistant",
            data={
                "turn": 1,
                "content": "\n  \n",
                "tool_calls": [
                    {"id": "call-1", "name": "search", "arguments": {}},
                    {"id": "call-2", "name": "visit", "arguments": {}},
                ],
            },
        )
    )

    entries = list(app.chat_history.entries())
    assert len(entries) == 1
    assert entries[0].body == "Requested 2 tool call(s)."
    text = _render_text(app)
    assert "Requested 2 tool call(s)." in text
    assert "ASSISTANT ·" not in text


def test_tui_replaces_streaming_assistant_with_requested_tool_count() -> None:
    app = _make_app()
    app.chat_history._entries = []
    app.view_state.show_thinking = False
    app.view_state.show_tool_detail = False
    app._pt_app = None

    app.chat_history.append_event(
        LiveEvent(
            kind="assistant_delta",
            message="x",
            data={"turn": 1, "field": "content", "delta": "partial"},
        )
    )
    assert [entry.role for entry in app.chat_history.entries()] == ["assistant"]

    app.chat_history.append_event(
        LiveEvent(
            kind="assistant_message",
            message="assistant",
            data={
                "turn": 1,
                "content": None,
                "tool_calls": [{"id": "call-1", "name": "search", "arguments": {}}],
            },
        )
    )

    entries = list(app.chat_history.entries())
    assert len(entries) == 1
    assert entries[0].body == "Requested 1 tool call(s)."
    assert entries[0].status == ""


def test_tui_drops_streaming_assistant_when_empty_final_answer() -> None:
    app = _make_app()
    app.chat_history._entries = []
    app.view_state.show_thinking = False
    app.view_state.show_tool_detail = False
    app._pt_app = None

    app.chat_history.append_event(
        LiveEvent(
            kind="assistant_delta",
            message="x",
            data={"turn": 1, "field": "content", "delta": "partial"},
        )
    )
    assert [entry.role for entry in app.chat_history.entries()] == ["assistant"]

    app.chat_history.append_event(
        LiveEvent(
            kind="assistant_message",
            message="assistant",
            data={"turn": 1, "content": None, "tool_calls": []},
        )
    )

    assert list(app.chat_history.entries()) == []


def test_tui_renders_thirty_turn_transcript_without_assistant_dividers() -> None:
    """Synthetic multi-turn Step-Level Live View stress check (no real LLM)."""
    app = _make_app()
    app.chat_history._entries = []
    app.view_state.show_thinking = False
    app.view_state.show_tool_detail = False
    app._pt_app = None

    total_turns = 30
    app.chat_history.append_event(
        LiveEvent(kind="user_message", message="multi-turn display stress query")
    )

    for turn in range(1, total_turns + 1):
        is_final = turn == total_turns
        call_id = f"call-{turn}"
        if is_final:
            app.chat_history.append_event(
                LiveEvent(
                    kind="assistant_message",
                    message="final",
                    data={
                        "turn": turn,
                        "content": f"<answer>done after {total_turns} turns</answer>",
                        "tool_calls": [],
                    },
                )
            )
            continue

        # Alternate empty-content tool turns and short-preamble tool turns.
        content = None if turn % 2 == 1 else f"Checking lead {turn}."
        app.chat_history.append_event(
            LiveEvent(
                kind="assistant_message",
                message="assistant",
                data={
                    "turn": turn,
                    "content": content,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "name": "search",
                            "arguments": {"query": [f"topic-{turn}"], "top_k": 3},
                        }
                    ],
                },
            )
        )
        app.chat_history.append_event(
            LiveEvent(
                kind="tool_call_started",
                message="search",
                data={
                    "id": call_id,
                    "name": "search",
                    "arguments": {"query": [f"topic-{turn}"], "top_k": 3},
                    "turn": turn,
                },
            )
        )
        app.chat_history.append_event(
            LiveEvent(
                kind="tool_result",
                message="result",
                data={
                    "id": call_id,
                    "name": "search",
                    "turn": turn,
                    "result": f"hit for topic-{turn}",
                    "status": "completed",
                    "extensions": {
                        "documents": [
                            {
                                "id": f"doc-{turn}",
                                "title": f"Doc {turn}",
                                "url": None,
                                "query": f"topic-{turn}",
                            }
                        ]
                    },
                },
            )
        )

    text = _render_text(app)

    assert "multi-turn display stress query" in text
    assert text.count("Requested 1 tool call(s).") == 15  # odd turns 1..29
    for turn in range(2, total_turns, 2):
        assert f"Checking lead {turn}." in text
    assert "ASSISTANT ·" not in text
    assert text.count("FINAL ANSWER ·") == 1
    assert f"FINAL ANSWER · turn {total_turns}" in text
    assert "done after 30 turns" in text
    assert "reasoning summary" not in text  # answer-tag extraction
    assert text.count("TOOL · turn") == total_turns - 1
    assert "Doc 1" in text
    assert "Doc 29" in text


def test_tui_tool_detail_toggles_preview_and_full_result() -> None:
    app = _make_app()
    app.chat_history._entries = []
    app.view_state.show_thinking = False
    app.view_state.show_tool_detail = False
    app._pt_app = None
    long_result = "\n".join(f"line-{index}" for index in range(8))

    app.chat_history.append_event(
        LiveEvent(
            kind="tool_call_started",
            message="search",
            data={"id": "call-1", "name": "search", "arguments": {"query": "demo", "top_k": 5}},
        )
    )
    app.chat_history.append_event(
        LiveEvent(
            kind="tool_result",
            message="result",
            data={
                "id": "call-1",
                "name": "search",
                "result": long_result,
                "status": "completed",
                "extensions": {
                    "searched_ids": ["doc-a", "doc-b"],
                    "documents": [
                        {"id": "doc-a", "title": "Doc A", "url": None, "query": "demo"},
                        {"id": "doc-b", "title": "Doc B", "url": None, "query": "demo"},
                    ],
                },
            },
        )
    )

    preview_text = _render_text(app)
    app.view_state.show_tool_detail = True
    full_text = _render_text(app)

    assert "TOOL" in preview_text
    assert "├─ Doc A" in preview_text
    assert "└─ Doc B" in preview_text
    assert "line-0" not in preview_text
    assert "use /tool-detail to expand details for 2 documents" in preview_text
    assert 'search "demo", 5' in preview_text
    assert "TOOL" in full_text
    assert "line-7" in full_text
    assert 'search "demo", 5' in full_text
    assert "├─ Doc A" not in full_text


def test_tui_collapses_search_results_to_document_summary() -> None:
    app = _make_app()
    app.chat_history._entries = []
    app.view_state.show_thinking = False
    app.view_state.show_tool_detail = False
    app._pt_app = None
    search_result = (
        "1. [02-rl-slime.qmd](None)\n"
        "guide/03-training/02-rl-slime.qmd:2:title: RL with slime\n\n"
        "2. [03-sft.qmd](None)\n"
        "guide/03-training/03-sft.qmd:2:title: SFT\n\n"
        "3. [None](https://example.com/doc)\n"
        "snippet text\n"
    )

    app.chat_history.append_event(
        LiveEvent(
            kind="tool_call_started",
            message="search",
            data={"id": "call-1", "name": "search", "arguments": {"query": "slime", "top_k": 5}},
        )
    )
    app.chat_history.append_event(
        LiveEvent(
            kind="tool_result",
            message="result",
            data={
                "id": "call-1",
                "name": "search",
                "result": search_result,
                "status": "completed",
                "extensions": {
                    "searched_ids": [
                        "guide/03-training/02-rl-slime.qmd",
                        "guide/03-training/03-sft.qmd",
                        "https://example.com/doc",
                    ],
                    "documents": [
                        {
                            "id": "guide/03-training/02-rl-slime.qmd",
                            "title": "02-rl-slime.qmd",
                            "url": None,
                            "query": "slime",
                        },
                        {
                            "id": "guide/03-training/03-sft.qmd",
                            "title": "03-sft.qmd",
                            "url": None,
                            "query": "slime",
                        },
                        {
                            "id": "https://example.com/doc",
                            "title": None,
                            "url": "https://example.com/doc",
                            "query": "slime",
                        },
                    ],
                },
            },
        )
    )

    preview_text = _render_text(app)
    app.view_state.show_tool_detail = True
    full_text = _render_text(app)

    assert "├─ 02-rl-slime.qmd" in preview_text
    assert "├─ 03-sft.qmd" in preview_text
    assert "└─ https://example.com/doc" in preview_text
    assert "use /tool-detail to expand details for 3 documents" in preview_text
    assert "guide/03-training/02-rl-slime.qmd:2:title" not in preview_text
    assert "guide/03-training/02-rl-slime.qmd:2:title" in full_text
    assert app.chat_history.entries()[0].extensions["documents"][0]["title"] == "02-rl-slime.qmd"


def test_tui_collapses_summarized_search_to_document_count() -> None:
    app = _make_app()
    app.chat_history._entries = []
    app.view_state.show_thinking = False
    app.view_state.show_tool_detail = False
    app._pt_app = None
    summary_result = (
        "The useful information for query slime as follows:\n\n"
        "Evidence in page:\n"
        "evidence\n\n"
        "Summary:\n"
        "summary text"
    )

    app.chat_history.append_event(
        LiveEvent(
            kind="tool_call_started",
            message="search",
            data={"id": "call-1", "name": "search", "arguments": {"query": "slime"}},
        )
    )
    app.chat_history.append_event(
        LiveEvent(
            kind="tool_result",
            message="result",
            data={
                "id": "call-1",
                "name": "search",
                "result": summary_result,
                "status": "completed",
                "extensions": {
                    "searched_ids": ["a.md", "b.md"],
                    "documents": [
                        {"id": "a.md", "title": "a.md", "url": None, "query": "slime"},
                        {"id": "b.md", "title": "b.md", "url": None, "query": "slime"},
                    ],
                },
            },
        )
    )

    preview_text = _render_text(app)
    assert "├─ a.md" in preview_text
    assert "└─ b.md" in preview_text
    assert "Evidence in page" not in preview_text
    assert "use /tool-detail to expand details for 2 documents" in preview_text


def test_tui_collapses_visit_results_to_document_tree() -> None:
    app = _make_app()
    app.chat_history._entries = []
    app.view_state.show_thinking = False
    app.view_state.show_tool_detail = False
    app._pt_app = None
    visit_result = "[source_files.md](None)\nfull document body\n"

    app.chat_history.append_event(
        LiveEvent(
            kind="tool_call_started",
            message="visit",
            data={
                "id": "call-1",
                "name": "visit",
                "arguments": {"document_id": "source_files.md", "goal": "read it"},
            },
        )
    )
    app.chat_history.append_event(
        LiveEvent(
            kind="tool_result",
            message="result",
            data={
                "id": "call-1",
                "name": "visit",
                "result": visit_result,
                "status": "completed",
                "extensions": {
                    "documents": [
                        {
                            "id": "source_files.md",
                            "title": "source_files.md",
                            "url": None,
                        }
                    ],
                },
            },
        )
    )

    preview_text = _render_text(app)
    app.view_state.show_tool_detail = True
    full_text = _render_text(app)

    assert "└─ source_files.md" in preview_text
    assert "use /tool-detail to expand details for 1 document" in preview_text
    assert "full document body" not in preview_text
    assert "full document body" in full_text


def test_tui_chat_viewport_does_not_pad_lines_so_terminal_selection_copies_clean_text() -> None:
    app = _make_app()
    app.chat_history._entries = [ConversationEntry(role="assistant", title="Assistant", body="short", style="class:assistant")]
    app.view_state.show_thinking = False
    app._pt_app = None
    app.view_state.chat_scroll_top = None
    app.view_state.chat_selection = None

    rendered = "".join(text for _, text in app.render_chat_viewport())

    assert "short" in rendered
    assert not any(line.endswith("  ") for line in rendered.splitlines())


def test_tui_internal_chat_selection_extracts_and_highlights_text() -> None:
    app = _make_app()
    app.chat_history._entries = [ConversationEntry(role="assistant", title="Assistant", body="alpha beta", style="class:assistant")]
    app.view_state.show_thinking = False
    app._pt_app = None
    app.view_state.chat_scroll_top = None
    app.view_state.chat_selection = None
    app.view_state.selecting = False

    lines = app._render_chat_plain_lines()
    row = next(index for index, line in enumerate(lines) if "alpha beta" in line)
    start = lines[row].index("alpha")
    end = lines[row].index(" beta")
    app.view_state.chat_selection = ChatSelection(
        anchor=ChatPoint(row=row, col=start),
        head=ChatPoint(row=row, col=end),
    )

    assert app.chat_selection.text() == "alpha"
    rendered = app.render_chat_viewport()
    selected = "".join(text for style, text in rendered if "class:selected" in style)
    assert selected == "alpha"


def test_tui_scrollbar_uses_visible_to_content_ratio() -> None:
    app = _make_app()
    app.chat_history._entries = [
        ConversationEntry(role="assistant", title=f"Assistant {index}", body="line", style="class:assistant")
        for index in range(30)
    ]
    app.view_state.show_thinking = False
    app._pt_app = None

    bottom_text = "".join(text for _, text in app.render_scrollbar())
    app._scroll_chat(-1000)
    top_text = "".join(text for _, text in app.render_scrollbar())

    assert "█" in bottom_text
    assert bottom_text.count("█") > 1
    assert bottom_text.count("█") < bottom_text.count("│")
    assert top_text != bottom_text
    assert top_text.startswith("█")



def test_tui_kicker_hints_lines_below_only_when_scrolled_up() -> None:
    app = _make_app()
    app.chat_history._entries = [
        ConversationEntry(role="assistant", title=f"Assistant {index}", body="line", style="class:assistant")
        for index in range(30)
    ]
    app.view_state.show_thinking = False
    app._pt_app = None
    app.view_state.chat_scroll_top = None

    assert "more" not in "".join(text for _, text in app.render_kicker())

    app._scroll_chat(-1000)
    kicker = "".join(text for _, text in app.render_kicker())
    assert "↓" in kicker
    assert "more (PgDn to bottom)" in kicker

    app._scroll_chat(1000)
    assert "more" not in "".join(text for _, text in app.render_kicker())


def test_tui_streaming_cursor_shows_while_streaming_and_stays_out_of_plain_lines() -> None:
    app = _make_app()
    app.chat_history.clear()

    app.chat_history.append_event(LiveEvent(
        kind="assistant_delta", message="think", data={"turn": 1, "field": "thinking", "delta": "think"}
    ))
    app.chat_history.append_event(LiveEvent(
        kind="assistant_delta", message="body", data={"turn": 1, "field": "content", "delta": "body"}
    ))

    rendered = _render_text(app)
    assert rendered.count("▌") == 2  # thinking + content both streaming
    assert "▌" not in "\n".join(app._render_chat_plain_lines())

    app.chat_history.append_event(LiveEvent(
        kind="assistant_message", message="assistant",
        data={"turn": 1, "thinking": "think", "content": "body", "tool_calls": []},
    ))
    assert "▌" not in _render_text(app)


def test_tui_detail_scroll_can_move_up_and_return_to_tail() -> None:
    app = _make_app()
    app.chat_history._entries = [
        ConversationEntry(role="assistant", title=f"Assistant {index}", body=f"line-{index}", style="class:assistant")
        for index in range(30)
    ]
    app.view_state.show_thinking = False
    app._pt_app = None
    app.view_state.chat_scroll_top = None

    bottom_text = "".join(text for _, text in app.render_chat_viewport())
    app._scroll_chat(-1000)
    upper_text = "".join(text for _, text in app.render_chat_viewport())
    app._scroll_chat(1000)
    bottom_again = "".join(text for _, text in app.render_chat_viewport())

    assert "line-29" in bottom_text
    assert "line-29" not in upper_text
    assert "line-29" in bottom_again
    assert app.view_state.chat_scroll_top is None


def test_tui_reuses_chat_layout_until_render_inputs_change(monkeypatch) -> None:
    app = _make_app()
    app.chat_history.clear()
    app.chat_history.append(
        role="tool-interaction",
        title="visit",
        body="result",
        status="completed",
    )
    render_calls = 0
    original_render_full = app.renderer.render_full

    def counted_render_full(*args, **kwargs):
        nonlocal render_calls
        render_calls += 1
        return original_render_full(*args, **kwargs)

    monkeypatch.setattr(app.renderer, "render_full", counted_render_full)

    app.render_chat_viewport()
    app.render_scrollbar()
    app._render_chat_plain_lines()
    app._scroll_chat(-1)

    assert render_calls == 1

    app.view_state.show_tool_detail = True
    app.render_scrollbar()
    assert render_calls == 2

    app.chat_history.append(role="assistant", title="Assistant", body="new content")
    app.render_chat_viewport()
    assert render_calls == 3


def test_tui_chat_layout_cache_invalidates_when_terminal_width_changes(monkeypatch) -> None:
    app = _make_app()
    render_calls = 0
    original_render_full = app.renderer.render_full

    class FakeOutput:
        columns = 40

        def get_size(self):
            class Size:
                rows = 30
                columns = FakeOutput.columns

            return Size()

    class FakeApp:
        output = FakeOutput()

    def counted_render_full(*args, **kwargs):
        nonlocal render_calls
        render_calls += 1
        return original_render_full(*args, **kwargs)

    app._pt_app = FakeApp()
    monkeypatch.setattr(app.renderer, "render_full", counted_render_full)

    app.render_chat_viewport()
    FakeOutput.columns = 100
    app.render_chat_viewport()

    assert render_calls == 2


def test_display_width_uses_prompt_toolkit_for_whole_string(monkeypatch) -> None:
    import searcherkit.interfaces.tui.ui.formatting as formatting

    calls: list[str] = []

    def fake_cwidth(text: str) -> int:
        calls.append(text)
        return len(text)

    monkeypatch.setattr(formatting, "_prompt_toolkit_cwidth", fake_cwidth)

    assert formatting._display_width("whole string") == 12
    assert calls == ["whole string"]


def test_slash_command_menu_filters_by_prefix_and_moves_selection() -> None:
    menu = SlashCommandMenuState([
        TuiCommand("models", "Choose the Active Model", kind="submenu"),
        TuiCommand("sources", "Choose the Active Source", kind="submenu"),
        TuiCommand("thinking", "Toggle assistant reasoning visibility"),
        TuiCommand("clear", "Clear current live view"),
        TuiCommand("quit", "Quit the TUI"),
    ])

    menu.update("/")

    assert [command.name for command in menu.candidates] == ["models", "sources", "thinking", "clear", "quit"]
    assert menu.selected_command is not None
    assert menu.selected_command.name == "models"

    menu.move_selection(1)
    assert menu.selected_command is not None
    assert menu.selected_command.name == "sources"

    menu.move_selection(-2)
    assert menu.selected_command is not None
    assert menu.selected_command.name == "quit"

    menu.move_selection(1)
    assert menu.selected_command is not None
    assert menu.selected_command.name == "models"

    menu.update("/q")
    assert [command.name for command in menu.candidates] == ["quit"]
    assert menu.selected_command is not None
    assert menu.selected_command.name == "quit"

    menu.update("/ui")
    assert menu.candidates == []


def test_slash_command_menu_can_be_dismissed_until_input_changes() -> None:
    menu = SlashCommandMenuState([
        TuiCommand("clear", "Clear current live view"),
        TuiCommand("quit", "Quit the TUI"),
    ])
    menu.update("/")

    assert menu.is_active

    menu.dismiss("/")

    assert not menu.is_active
    assert menu.candidates == []

    menu.update("/")

    assert not menu.is_active

    menu.update("/q")

    assert menu.is_active
    assert menu.selected_command is not None
    assert menu.selected_command.name == "quit"


def test_tui_can_dismiss_slash_menu_with_current_input() -> None:
    app = _make_app()
    app.slash_menu.menu_state = SlashCommandMenuState([
        TuiCommand("clear", "Clear current live view"),
        TuiCommand("quit", "Quit the TUI"),
    ])

    class FakeInput:
        text = "/"

    app.input_field.set_input_field(FakeInput())

    assert app.slash_candidates_visible() is True
    assert app.slash_menu.dismiss() is True
    assert app.slash_candidates_visible() is False
    assert app.input_field.text() == "/"

    app.input_field.set_text("/q")

    assert app.slash_candidates_visible() is True


def test_slash_submit_executes_selected_candidate_and_tab_completes() -> None:
    menu = SlashCommandMenuState([
        TuiCommand("clear", "Clear current live view"),
        TuiCommand("thinking", "Toggle assistant reasoning visibility"),
        TuiCommand("quit", "Quit the TUI"),
    ])
    menu.update("/q")

    decision = menu.decide_submit("/q")

    assert decision.action == "execute"
    assert decision.value == "quit"

    app = _make_app()
    app._pt_app = None
    app.slash_menu.menu_state = menu

    class FakeInput:
        text = "/q"

    app.input_field.set_input_field(FakeInput())

    assert app.slash_menu.complete_selected() is True
    assert app.input_field.text() == "/quit"


def test_slash_submit_rejects_unknown_and_arguments() -> None:
    menu = SlashCommandMenuState([TuiCommand("thinking", "Toggle assistant reasoning visibility")])
    menu.update("/wat")

    unknown = menu.decide_submit("/wat")

    assert unknown.action == "unknown"
    assert unknown.message == "Unknown command: /wat"

    menu.update("/thinking extra")
    argument_error = menu.decide_submit("/thinking extra")

    assert argument_error.action == "argument_error"
    assert argument_error.message == "/thinking does not accept arguments"


def test_tui_shows_slash_parse_errors() -> None:
    app = _make_app()
    app.slash_menu.menu_state = SlashCommandMenuState([
        TuiCommand("thinking", "Toggle assistant reasoning visibility")
    ])
    _set_test_input(app, "/thinking extra")

    app.handle_enter(object())

    error = app.chat_history.entries()[-1]
    assert error.title == "TUI Command Error"
    assert error.body == "/thinking does not accept arguments"
    assert app.input_field.text() == "/thinking extra"


def test_tui_thinking_command_is_allowed_while_running() -> None:
    app = _make_app()
    app.chat_history._entries = []
    app.query_controller.running = True
    app.view_state.show_thinking = False
    app.input_field.set_input_field(None)
    app._pt_app = None
    app.slash_menu.menu_state = SlashCommandMenuState([TuiCommand("thinking", "Toggle assistant reasoning visibility")])

    assert app.execute_tui_command("thinking") is True

    assert app.view_state.show_thinking is True


def test_tui_clear_command_rejected_while_running_and_preserves_input() -> None:
    app = _make_app()
    app.chat_history._entries = [ConversationEntry(role="assistant", title="Assistant", body="active", style="class:assistant")]
    app.query_controller.running = True
    app.view_state.show_thinking = False
    app._pt_app = None
    app.slash_menu.menu_state = SlashCommandMenuState([TuiCommand("clear", "Clear current live view")])

    class FakeInput:
        text = "/clear"

    app.input_field.set_input_field(FakeInput())

    assert app.execute_tui_command("clear") is False

    error = app.chat_history.entries()[-1]
    assert error.title == "TUI Command Error"
    assert "Cannot clear while a query is running" in error.body
    assert app.input_field.text() == "/clear"


def test_tui_clear_command_clears_entries_when_idle(tmp_path) -> None:
    app = SearcherKitTui(config=InteractiveQueryConfig(record_dir=str(tmp_path)))
    app.chat_history._entries = [ConversationEntry(role="assistant", title="Assistant", body="old", style="class:assistant")]
    app.query_controller.running = False
    app.view_state.show_thinking = False
    app.view_state.chat_scroll_top = 3
    app._pt_app = None
    app.slash_menu.menu_state = SlashCommandMenuState([TuiCommand("clear", "Clear current live view")])

    class FakeInput:
        text = "/clear"

    app.input_field.set_input_field(FakeInput())

    assert app.execute_tui_command("clear") is True

    assert [entry.title for entry in app.chat_history.entries()] == ["SearcherKit Interactive Query", "Clear"]
    assert app.view_state.chat_scroll_top is None
    assert app.input_field.text() == ""
    assert app.chat_history.entries()[-1].body == "Conversation cleared. Ready for new queries."


def test_tui_intro_and_clear_entry_widths_follow_terminal(tmp_path) -> None:
    app = SearcherKitTui(config=InteractiveQueryConfig(record_dir=str(tmp_path)))
    app.query_controller.running = False
    app.view_state.show_thinking = False
    app.slash_menu.menu_state = SlashCommandMenuState([TuiCommand("clear", "Clear current live view")])

    class FakeInput:
        text = "/clear"

    class FakeOutput:
        columns = 40

        def get_size(self):
            class Size:
                rows = 30
                columns = FakeOutput.columns

            return Size()

    class FakeApp:
        output = FakeOutput()

        def invalidate(self) -> None:
            return None

    app.input_field.set_input_field(FakeInput())
    app._pt_app = FakeApp()
    app._set_intro()

    narrow_text = _render_text(app)
    assert len(narrow_text.splitlines()[0]) == 37
    assert "Records:" not in narrow_text
    assert "  Type a query and press Enter. Each\n  query starts" in narrow_text

    FakeOutput.columns = 100
    wide_text = _render_text(app)
    assert len(wide_text.splitlines()[0]) == 97
    assert "  Type a query and press Enter. Each query starts" in wide_text

    assert app.execute_tui_command("clear") is True
    clear_text = _render_text(app)
    assert len(clear_text.splitlines()[0]) == 97
    assert "  Conversation cleared. Ready for new queries." in clear_text


def test_input_field_set_text_can_move_cursor_to_end() -> None:
    app = _make_app()
    app.slash_menu.menu_state = SlashCommandMenuState([])

    class FakeBuffer:
        cursor_position = 1

    class FakeInput:
        text = "old"
        buffer = FakeBuffer()

    app.input_field.set_input_field(FakeInput())

    app.input_field.set_text("history item", cursor_end=True)

    assert app.input_field.text() == "history item"
    assert app.input_field._input_field.buffer.cursor_position == len("history item")


def test_tui_dynamic_input_height_and_fixed_slash_height_share_terminal_rows() -> None:
    app = _make_app()
    app.slash_menu.menu_state = SlashCommandMenuState([TuiCommand(f"cmd{i}", f"Command {i}") for i in range(10)])

    class FakeInput:
        text = "/"

    class FakeOutput:
        def get_size(self):
            class Size:
                rows = 30
                columns = 10

            return Size()

    class FakeApp:
        output = FakeOutput()

    app.input_field.set_input_field(FakeInput())
    app._pt_app = FakeApp()

    assert app.input_view_height() == 1
    assert app.slash_candidates_height() == 5
    assert app.chat_view_height() == 19

    app.input_field.set_text("123456789\nsecond\nthird")

    assert app.input_view_height() == 4
    assert app.chat_view_height() == 21


def test_tui_renders_slash_candidates_below_input() -> None:
    app = _make_app()
    app._pt_app = None
    app.slash_menu.menu_state = SlashCommandMenuState([
        TuiCommand("models", "Choose the Active Model", kind="submenu"),
        TuiCommand("sources", "Choose the Active Source", kind="submenu"),
        TuiCommand("thinking", "Toggle assistant reasoning visibility"),
        TuiCommand("clear", "Clear current live view"),
        TuiCommand("quit", "Quit the TUI"),
    ])

    class FakeInput:
        text = "/"

    app.input_field.set_input_field(FakeInput())
    app.slash_menu.is_active()  # sync menu state with input

    rendered = "".join(text for _, text in app.slash_menu_renderer.render(app.slash_menu.menu_state))

    assert "/models" in rendered
    assert "/sources" in rendered
    assert "/thinking" in rendered
    assert rendered.index("/models") < rendered.index("/sources") < rendered.index("/thinking")


def _multi_source_interactive_config(tmp_path) -> InteractiveQueryConfig:
    return InteractiveQueryConfig(
        record_dir=str(tmp_path),
        agent=SearchAgentConfig(
            sources=[
                SourceConfig(
                    type="file",
                    name="wiki",
                    root_path="docs",
                ),
                SourceConfig(
                    type="file",
                    name="bcp",
                    root_path="tests/fixtures/files",
                ),
            ],
            tools=[
                ToolConfig(type="search", name="search", source=["wiki"]),
                ToolConfig(type="visit", name="visit", source=["wiki"]),
                ToolConfig(type="mcp", name="external", source=["wiki"]),
            ],
        ),
    )


def test_tui_source_commands_are_generated_and_rendered(tmp_path) -> None:
    app = SearcherKitTui(config=_multi_source_interactive_config(tmp_path))

    class FakeInput:
        text = "/sources:"

    app.input_field.set_input_field(FakeInput())
    app.slash_menu.is_active()  # sync menu state with input
    app._pt_app = None

    rendered = "".join(text for _, text in app.slash_menu_renderer.render(app.slash_menu.menu_state))

    assert "/sources:wiki(file)" in rendered
    assert "/sources:bcp(file)" in rendered


def test_tui_source_command_completes_and_executes(tmp_path) -> None:
    app = SearcherKitTui(config=_multi_source_interactive_config(tmp_path))
    app._pt_app = None
    app.query_controller.running = False

    class FakeInput:
        text = "/sources:b"

    app.input_field.set_input_field(FakeInput())

    assert app.slash_menu.complete_selected() is True
    assert app.input_field.text() == "/sources:bcp(file)"

    assert app.execute_tui_command("sources:bcp(file)") is True

    assert app.session_state.active_source.name == "bcp"
    assert app.chat_history.entries()[-1].role == "meta"
    assert app.chat_history.entries()[-1].title == "Active Source"
    assert app.chat_history.entries()[-1].body == "bcp(file)"
    assert app.input_field.text() == ""


def test_tui_active_source_scopes_next_query_config(tmp_path) -> None:
    config = _multi_source_interactive_config(tmp_path)
    app = SearcherKitTui(config=config)
    app._pt_app = None
    app.query_controller.running = False

    assert app.execute_tui_command("sources:bcp(file)") is True

    run_config = app.query_controller.build_run_config()

    source_by_tool = {tool.name: tool.source for tool in run_config.agent.tools}
    assert source_by_tool["search"] == ["bcp"]
    assert source_by_tool["visit"] == ["bcp"]
    assert source_by_tool["external"] == ["wiki"]
    assert {tool.name: tool.source for tool in config.agent.tools}["search"] == ["wiki"]


def test_model_command_parsing_and_application_updates_provider_config() -> None:
    option = ModelOption(provider="openai", model="llama3.2:1b", base_url="http://127.0.0.1:11434/v1")
    config = ClientConfig(type="openai", model="qwen2.5:0.5b")

    assert parse_model_command("models:openai/llama3.2:1b", [option]) == option

    apply_active_model(config, option)

    assert config.type == "openai"
    assert config.model == "llama3.2:1b"
    assert config.base_url == "http://127.0.0.1:11434/v1"


def test_tui_model_commands_are_generated_and_rendered(tmp_path) -> None:
    option = ModelOption(provider="openai", model="qwen2.5:0.5b", base_url="http://127.0.0.1:11434/v1")
    app = SearcherKitTui(config=InteractiveQueryConfig(record_dir=str(tmp_path)), model_options=[option])

    class FakeInput:
        text = "/models:"

    app.input_field.set_input_field(FakeInput())
    app.slash_menu.is_active()  # sync menu state with input
    app._pt_app = None

    rendered = "".join(text for _, text in app.slash_menu_renderer.render(app.slash_menu.menu_state))

    assert "/models:openai/qwen2.5:0.5b" in rendered
    assert "Use model openai/qwen2.5:0.5b" in rendered


def test_tui_status_renders_model_and_source_on_highlighted_first_line(tmp_path) -> None:
    app = SearcherKitTui(config=_multi_source_interactive_config(tmp_path))

    rendered_parts = app.render_status()
    rendered = "".join(text for _, text in rendered_parts)
    styles = [style for style, _ in rendered_parts]

    assert rendered.startswith("Model:openai/model · Source:wiki(file) · Records:")
    assert str(tmp_path) in rendered
    assert "class:status-model" in styles
    assert "class:status-source" in styles
    assert "class:status-records" in styles


def test_tui_run_query_renders_provider_not_found_as_friendly_error(tmp_path, monkeypatch) -> None:
    import httpx
    from openai import NotFoundError

    class FakeRunner:
        def __init__(self, *, config):
            self.config = config

        async def run_query(self, query, *, live_event_sink):
            request = httpx.Request("POST", "http://127.0.0.1:11434/v1/chat/completions")
            response = httpx.Response(
                404,
                request=request,
                json={"error": {"message": "model 'qwen2.5:0.5b' not found"}},
            )
            raise NotFoundError(
                "Error code: 404 - {'error': {'message': \"model 'qwen2.5:0.5b' not found\"}}",
                response=response,
                body={"error": {"message": "model 'qwen2.5:0.5b' not found"}},
            )

    monkeypatch.setattr("searcherkit.interfaces.tui.runtime.query_controller.InteractiveQueryRunner", FakeRunner)
    app = SearcherKitTui(config=InteractiveQueryConfig(record_dir=str(tmp_path)))
    app._pt_app = None

    asyncio.run(app.query_controller.run_query("hello"))

    assert app.query_controller.running is False
    assert app.chat_history.entries()[-1].title == "Provider Error"
    assert app.chat_history.entries()[-1].style == "class:error"
    assert app.chat_history.entries()[-1].body == (
        "Model not found: qwen2.5:0.5b. Use /models to choose an available model, "
        "or install/start this model in the provider before retrying."
    )
    assert "Traceback" not in app.chat_history.entries()[-1].body
    assert "Error code: 404" not in app.chat_history.entries()[-1].body


def test_tui_run_query_renders_bad_request_as_friendly_error(tmp_path, monkeypatch) -> None:
    import httpx
    from openai import BadRequestError

    class FakeRunner:
        def __init__(self, *, config):
            self.config = config

        async def run_query(self, query, *, live_event_sink):
            request = httpx.Request("POST", "http://127.0.0.1:11434/v1/chat/completions")
            response = httpx.Response(
                400,
                request=request,
                json={"error": {"message": "model 'nomic-embed-text' does not support chat completions"}},
            )
            raise BadRequestError(
                "Error code: 400 - {'error': {'message': \"model 'nomic-embed-text' does not support chat completions\"}}",
                response=response,
                body={"error": {"message": "model 'nomic-embed-text' does not support chat completions"}},
            )

    monkeypatch.setattr("searcherkit.interfaces.tui.runtime.query_controller.InteractiveQueryRunner", FakeRunner)
    app = SearcherKitTui(config=InteractiveQueryConfig(record_dir=str(tmp_path)))
    app._pt_app = None
    app.session_state.active_model = ModelOption(provider="openai", model="nomic-embed-text", base_url="http://127.0.0.1:11434/v1")

    asyncio.run(app.query_controller.run_query("hello"))

    assert app.chat_history.entries()[-1].title == "Provider Error"
    assert app.chat_history.entries()[-1].body == (
        "Model cannot be used for chat: openai/nomic-embed-text. Use /models to choose a chat-capable model."
    )
    assert "Traceback" not in app.chat_history.entries()[-1].body
    assert "Error code: 400" not in app.chat_history.entries()[-1].body


def test_tui_model_command_appends_selection_entry(tmp_path) -> None:
    option = ModelOption(provider="openai", model="llama3.2:1b", base_url="http://127.0.0.1:11434/v1")
    config = InteractiveQueryConfig(record_dir=str(tmp_path))
    app = SearcherKitTui(config=config, model_options=[option])
    app._pt_app = None
    app.query_controller.running = False

    assert app.execute_tui_command("models:openai/llama3.2:1b") is True

    assert app.chat_history.entries()[-1].role == "meta"
    assert app.chat_history.entries()[-1].title == "Active Model"
    assert app.chat_history.entries()[-1].body == "openai/llama3.2:1b"


def test_tui_model_command_scopes_next_query_config(tmp_path) -> None:
    option = ModelOption(provider="openai", model="llama3.2:1b", base_url="http://127.0.0.1:11434/v1")
    config = InteractiveQueryConfig(
        record_dir=str(tmp_path),
        agent=SearchAgentConfig(
            llm_client=ClientConfig(
                type="openai",
                model="qwen2.5:0.5b",
                base_url="http://127.0.0.1:11434/v1",
            )
        ),
    )
    app = SearcherKitTui(config=config, model_options=[option])
    app._pt_app = None
    app.query_controller.running = False

    assert app.execute_tui_command("models:openai/llama3.2:1b") is True

    run_config = app.query_controller.build_run_config()

    assert run_config.agent.llm_client.model == "llama3.2:1b"
    assert run_config.agent.llm_client.base_url == "http://127.0.0.1:11434/v1"
    assert config.agent.llm_client.model == "qwen2.5:0.5b"


def test_tui_model_command_rejected_while_running(tmp_path) -> None:
    option = ModelOption(provider="openai", model="llama3.2:1b", base_url="http://127.0.0.1:11434/v1")
    app = SearcherKitTui(config=InteractiveQueryConfig(record_dir=str(tmp_path)), model_options=[option])
    app._pt_app = None
    app.query_controller.running = True

    class FakeInput:
        text = "/models:openai/llama3.2:1b"

    app.input_field.set_input_field(FakeInput())

    assert app.execute_tui_command("models:openai/llama3.2:1b") is False

    assert app.session_state.active_model is None
    assert app.input_field.text() == "/models:openai/llama3.2:1b"


def test_active_model_application_keeps_openai_parser_and_prompts_unchanged(tmp_path) -> None:
    option = ModelOption(provider="openai", model="model-b", base_url="http://127.0.0.1:8000/v1")
    config = InteractiveQueryConfig(
        record_dir=str(tmp_path),
        agent=SearchAgentConfig(
            llm_client=ClientConfig(
                type="openai",
                model="model-a",
                base_url="http://127.0.0.1:8001/v1",
            ),
            system_prompt="system",
            query_prompt="Q: {query}",
        ),
    )
    app = SearcherKitTui(config=config, model_options=[option])
    app._pt_app = None
    app.query_controller.running = False

    assert app.execute_tui_command("models:openai/model-b") is True

    run_config = app.query_controller.build_run_config()

    assert run_config.agent.llm_client.model == "model-b"
    assert run_config.agent.llm_client.base_url == "http://127.0.0.1:8000/v1"
    assert run_config.agent.parser == config.agent.parser
    assert run_config.agent.system_prompt == "system"
    assert run_config.agent.query_prompt == "Q: {query}"


def test_discover_model_options_uses_openai_compatible_models_list(monkeypatch) -> None:
    import asyncio
    import openai
    from types import SimpleNamespace

    class FakeModels:
        async def list(self):
            return SimpleNamespace(
                data=[
                    SimpleNamespace(id="qwen2.5:0.5b"),
                    SimpleNamespace(id="llama3.2:1b"),
                ]
            )

    closed_clients = []

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.models = FakeModels()

        async def close(self):
            closed_clients.append(self)

    monkeypatch.setattr(openai, "AsyncOpenAI", FakeAsyncOpenAI)

    result = asyncio.run(
        discover_model_options(
            ClientConfig(
                type="openai",
                model="qwen2.5:0.5b",
                base_url="http://127.0.0.1:11434/v1",
            )
        )
    )

    assert result.failed is False
    assert [option.command_name for option in result.options] == [
        "models:openai/qwen2.5:0.5b",
        "models:openai/llama3.2:1b",
    ]
    assert {option.base_url for option in result.options} == {"http://127.0.0.1:11434/v1"}
    assert len(closed_clients) == 1


def test_discover_model_options_deduplicates_by_command(monkeypatch) -> None:
    import asyncio
    import openai
    from types import SimpleNamespace

    class FakeModels:
        async def list(self):
            return SimpleNamespace(data=[SimpleNamespace(id="same-model")])

    closed_clients = []

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.models = FakeModels()

        async def close(self):
            closed_clients.append(self)

    monkeypatch.setattr(openai, "AsyncOpenAI", FakeAsyncOpenAI)

    result = asyncio.run(
        discover_model_options(
            ClientConfig(
                type="vllm",
                model="same-model",
                base_url=["http://a/v1", "http://b/v1"],
            )
        )
    )

    assert [option.command_name for option in result.options] == ["models:vllm/same-model"]
    assert result.options[0].base_url == "http://a/v1"
    assert len(closed_clients) == 2


def test_discover_model_options_failure_returns_no_commands(monkeypatch) -> None:
    import asyncio
    import openai

    closed_clients = []

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.models = self

        async def list(self):
            raise ValueError("no models")

        async def close(self):
            closed_clients.append(self)

    monkeypatch.setattr(openai, "AsyncOpenAI", FakeAsyncOpenAI)

    result = asyncio.run(
        discover_model_options(
            ClientConfig(
                type="openai",
                model="qwen2.5:0.5b",
                base_url="http://127.0.0.1:11434/v1",
            )
        )
    )

    assert result.options == []
    assert result.failed is True
    assert "Model discovery failed for openai" in result.message
    assert len(closed_clients) == 1


def test_discover_model_options_unsupported_provider_returns_no_commands() -> None:
    import asyncio

    result = asyncio.run(discover_model_options(ClientConfig(type="transformers", model="local-model")))

    assert result.options == []
    assert result.failed is False
    assert result.message == ""
