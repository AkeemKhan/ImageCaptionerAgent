from __future__ import annotations

from typing import Any

import pytest

from captioner_agent.agent import CaptionAgent
from captioner_agent.llm import LLMError, TurnResult
from captioner_agent.prompts import ASK_USER_TOOL, DESCRIBE_TOOL
from captioner_agent.settings import Settings
from captioner_agent.store import Session, SessionStore
from captioner_agent.toolbox import ToolResult

from .conftest import FakeProvider, FakeToolbox, text_block, tool_use

pytestmark = pytest.mark.anyio


async def run_turn(
    agent: CaptionAgent, session: Session, text: str, uploaded: list[str] | None = None
) -> list[dict[str, Any]]:
    return [event async for event in agent.run(session, text, uploaded or [])]


def build(
    settings: Settings, store: SessionStore, turns: list[TurnResult | Exception]
) -> tuple[CaptionAgent, FakeProvider, FakeToolbox, Session]:
    provider = FakeProvider(turns)
    toolbox = FakeToolbox()
    return CaptionAgent(provider, toolbox, store, settings), provider, toolbox, store.create()


async def test_tool_call_then_answer(settings: Settings, store: SessionStore) -> None:
    path = "sid/uploads/a.png"
    agent, provider, toolbox, session = build(
        settings,
        store,
        [
            TurnResult(
                [text_block("Checking."), tool_use("t1", "read_png_metadata", {"path": path})], "tool_use"
            ),
            TurnResult([text_block("Done.")], "end_turn"),
        ],
    )

    events = await run_turn(agent, session, "caption this")

    assert [e["type"] for e in events] == [
        "text",
        "text_end",
        "tool_start",
        "tool_end",
        "text",
        "text_end",
        "done",
    ]
    assert toolbox.calls == [("read_png_metadata", {"path": path})]
    # The second request carries the tool result back to the model.
    assert provider.requests[1]["messages"][-1] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "t1",
                "content": '{"has_prompt": false}',
                "is_error": False,
            }
        ],
    }

    saved = store.get(session.id)
    assert saved is not None
    assert saved.title == "caption this"
    assert [m["role"] for m in saved.messages] == ["user", "assistant", "user", "assistant"]
    assert [t["type"] for t in saved.transcript] == ["user", "assistant", "tool", "assistant"]


async def test_first_message_lists_session_paths_and_criteria(
    settings: Settings, store: SessionStore
) -> None:
    agent, provider, _, session = build(settings, store, [TurnResult([text_block("ok")], "end_turn")])

    await run_turn(agent, session, "hello")

    request = provider.requests[0]
    user_text = request["messages"][0]["content"]
    assert f"upload_dir: {session.id}/uploads" in user_text
    assert "<caption_results>" not in user_text
    assert user_text.endswith("hello")
    assert "Keep captions under 20 words." in request["system"]
    assert {ASK_USER_TOOL, DESCRIBE_TOOL} <= {tool["name"] for tool in request["tools"]}


async def test_question_pauses_turn_and_answer_resumes_it(settings: Settings, store: SessionStore) -> None:
    ask = tool_use("q1", ASK_USER_TOOL, {"question": "Flip too?", "options": ["Yes", "No"]})
    read = tool_use("t1", "read_png_metadata", {"path": "x.png"})
    agent, provider, _, session = build(
        settings,
        store,
        [TurnResult([read, ask], "tool_use"), TurnResult([text_block("Skipping flips.")], "end_turn")],
    )

    first = await run_turn(agent, session, "caption")

    assert first[-2] == {"type": "choice", "id": "q1", "question": "Flip too?", "options": ["Yes", "No"]}
    assert len(provider.requests) == 1
    paused = store.get(session.id)
    assert paused is not None and paused.pending is not None
    # History stops at the assistant's tool calls until the user answers.
    assert paused.messages[-1]["role"] == "assistant"

    await run_turn(agent, paused, "No")

    resumed = provider.requests[1]["messages"][-1]
    assert resumed["role"] == "user"
    assert [(b["tool_use_id"], b["content"]) for b in resumed["content"]] == [
        ("t1", '{"has_prompt": false}'),
        ("q1", "No"),
    ]
    finished = store.get(session.id)
    assert finished is not None and finished.pending is None


async def test_invalid_tool_input_is_rejected_without_calling_the_tool(
    settings: Settings, store: SessionStore
) -> None:
    agent, provider, toolbox, session = build(
        settings,
        store,
        [
            TurnResult([tool_use("t1", "read_png_metadata", {"path": 5})], "tool_use"),
            TurnResult([tool_use("t2", "no_such_tool", {})], "tool_use"),
            TurnResult([text_block("Sorry.")], "end_turn"),
        ],
    )

    events = await run_turn(agent, session, "go")

    assert toolbox.calls == []
    assert [e["ok"] for e in events if e["type"] == "tool_end"] == [False, False]
    assert provider.requests[1]["messages"][-1]["content"][0]["is_error"] is True
    assert "Unknown tool" in provider.requests[2]["messages"][-1]["content"][0]["content"]


async def test_failed_tool_is_reported_to_the_model(settings: Settings, store: SessionStore) -> None:
    agent, provider, toolbox, session = build(
        settings,
        store,
        [
            TurnResult([tool_use("t1", "read_png_metadata", {"path": "a.png"})], "tool_use"),
            TurnResult([text_block("One failed.")], "end_turn"),
        ],
    )
    toolbox.results["read_png_metadata"] = ToolResult("image not found: a.png", is_error=True)

    await run_turn(agent, session, "go")

    result = provider.requests[1]["messages"][-1]["content"][0]
    assert (result["content"], result["is_error"]) == ("image not found: a.png", True)


async def test_model_error_leaves_session_unchanged(settings: Settings, store: SessionStore) -> None:
    agent, _, _, session = build(settings, store, [LLMError("Rate limited.")])

    events = await run_turn(agent, session, "go")

    assert events == [{"type": "error", "message": "Rate limited."}]
    saved = store.get(session.id)
    assert saved is not None and saved.messages == [] and saved.transcript == []


async def test_refusal_is_shown_as_a_notice(settings: Settings, store: SessionStore) -> None:
    agent, _, _, session = build(
        settings, store, [TurnResult([], "refusal", refusal_explanation="Declined.")]
    )

    events = await run_turn(agent, session, "go")

    assert {"type": "notice", "text": "Declined."} in events
    assert events[-1]["type"] == "done"


async def test_truncated_tool_call_is_not_executed(settings: Settings, store: SessionStore) -> None:
    agent, _, toolbox, session = build(
        settings, store, [TurnResult([tool_use("t1", "read_png_metadata", {"path": "a.png"})], "max_tokens")]
    )

    events = await run_turn(agent, session, "go")

    assert toolbox.calls == []
    assert events[-1]["type"] == "error"


async def test_malformed_stream_is_retried(settings: Settings, store: SessionStore) -> None:
    agent, provider, _, session = build(
        settings, store, [ValueError("bad json"), TurnResult([text_block("ok")], "end_turn")]
    )

    events = await run_turn(agent, session, "go")

    assert len(provider.requests) == 2
    assert events[-1]["type"] == "done"
