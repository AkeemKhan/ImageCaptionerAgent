"""Manual captioning: detecting it at the start, and handing the images to MANUAL_CAPTION."""

from __future__ import annotations

import json
from typing import Any

import pytest

from captioner_agent.agent import CaptionAgent
from captioner_agent.files import SessionFiles
from captioner_agent.llm import LLMError, TurnResult
from captioner_agent.prompts import MANUAL_TOOL
from captioner_agent.settings import Settings
from captioner_agent.store import Session, SessionStore
from captioner_agent.toolbox import McpToolbox, ToolResult

from .conftest import FakeProvider, FakeToolbox, make_png, text_block, tool_use

pytestmark = pytest.mark.anyio


def build(
    settings: Settings,
    store: SessionStore,
    images: list[str],
    turns: list[TurnResult | Exception] | None = None,
    manual: bool | Exception = True,
) -> tuple[CaptionAgent, FakeProvider, FakeToolbox, Session]:
    provider = FakeProvider(turns if turns is not None else [TurnResult([text_block("Saved.")], "end_turn")])
    provider.manual_intent = manual
    toolbox = FakeToolbox()
    session = store.create()
    files = SessionFiles(settings.sessions_dir, session.id)
    for name in images:
        make_png(files.uploads / name)
    return CaptionAgent(provider, toolbox, store, settings), provider, toolbox, session


async def turn(
    agent: CaptionAgent, store: SessionStore, session: Session, text: str, uploaded: list[str] | None = None
) -> list[dict[str, Any]]:
    fresh = store.get(session.id)
    assert fresh is not None
    return [event async for event in agent.run(fresh, text, uploaded or [])]


async def test_manual_request_skips_every_automatic_step(settings: Settings, store: SessionStore) -> None:
    settings.vision_backend = "ask"
    agent, provider, toolbox, session = build(settings, store, ["a.png", "b.png"])
    text = "a.png is a red car, b.png is a blue boat"

    events = await turn(agent, store, session, text, ["a.png", "b.png"])

    # No PNG info, no BLIP, no Anthropic vision, and no question about which to use.
    assert toolbox.calls == [] and provider.vision_calls == []
    assert not [e for e in events if e["type"] in ("choice", "tool_start", "tool_end")]
    assert {
        "type": "notice",
        "text": "Manual captioning: the automatic captioning steps were skipped.",
    } in events
    assert provider.router_calls == [{"text": text, "instructions": provider.router_calls[0]["instructions"]}]

    sent: str = provider.requests[0]["messages"][0]["content"]
    assert "<manual_captioning>" in sent and "<caption_results>" not in sent
    assert f"{session.id}/uploads/a.png" in sent and f"{session.id}/uploads/b.png" in sent
    assert sent.endswith(text)
    assert MANUAL_TOOL in provider.requests[0]["system"]


async def test_the_decision_is_made_before_png_info(settings: Settings, store: SessionStore) -> None:
    agent, provider, toolbox, session = build(settings, store, ["a.png"], manual=False)

    await turn(agent, store, session, "please caption", ["a.png"])

    # An automatic request runs the normal pipeline, with the check made first.
    assert len(provider.router_calls) == 1
    assert [n for n, _ in toolbox.calls][:1] == ["caption_from_png_info"]
    assert "<manual_captioning>" not in provider.requests[0]["messages"][0]["content"]


@pytest.mark.parametrize(
    ("text", "uploaded", "setting_on", "resuming"),
    [
        ("", ["a.png"], True, False),  # images alone: nothing to classify
        ("   ", ["a.png"], True, False),
        ("a.png is a car", [], True, False),  # no images: Claude handles it with the tool
        ("a.png is a car", ["a.png"], False, False),  # detection switched off
        ("BLIP", ["a.png"], True, True),  # an answer to the BLIP-or-Anthropic question
    ],
)
async def test_the_check_is_skipped_when_it_cannot_apply(
    settings: Settings,
    store: SessionStore,
    text: str,
    uploaded: list[str],
    setting_on: bool,
    resuming: bool,
) -> None:
    settings.vision_backend = "ask"
    settings.detect_manual_captioning = setting_on
    agent, provider, _, session = build(settings, store, ["a.png"], manual=True)
    if resuming:
        await turn(agent, store, session, "", ["a.png"])  # leaves the vision question pending
        provider.router_calls.clear()

    await turn(agent, store, session, text, uploaded)

    assert provider.router_calls == []


async def test_a_failed_check_falls_back_to_automatic_captioning(
    settings: Settings, store: SessionStore
) -> None:
    agent, _, toolbox, session = build(settings, store, ["a.png"], manual=LLMError("Rate limited."))

    events = await turn(agent, store, session, "a.png is a car", ["a.png"])

    assert [n for n, _ in toolbox.calls][:1] == ["caption_from_png_info"]
    assert not [e for e in events if e["type"] == "notice"]
    assert events[-1]["type"] == "done"


async def test_claude_saves_the_users_captions_with_manual_caption(
    settings: Settings, store: SessionStore
) -> None:
    entries = [
        {"path": "s/uploads/a.png", "caption": "a red car"},
        {"path": "s/uploads/b.png", "caption": "a boat"},
    ]
    turns = [
        TurnResult(
            [tool_use("m1", MANUAL_TOOL, {"captions": entries, "output_dir": "s/outputs"})], "tool_use"
        ),
        TurnResult([text_block("Saved 2 captions.")], "end_turn"),
    ]
    agent, provider, toolbox, session = build(settings, store, ["a.png", "b.png"], turns)
    toolbox.definitions = lambda: [  # type: ignore[method-assign]
        {"name": MANUAL_TOOL, "description": "", "input_schema": {"type": "object"}}
    ]
    toolbox.results[MANUAL_TOOL] = ToolResult('{"written": 2}')

    events = await turn(agent, store, session, "a.png is a red car, b.png is a boat", ["a.png", "b.png"])

    assert toolbox.calls == [(MANUAL_TOOL, {"captions": entries, "output_dir": "s/outputs"})]
    assert [(e["name"], e["ok"]) for e in events if e["type"] == "tool_end"] == [(MANUAL_TOOL, True)]
    assert MANUAL_TOOL in {t["name"] for t in provider.requests[0]["tools"]}


async def test_a_later_message_without_images_can_still_use_the_tool(
    settings: Settings, store: SessionStore
) -> None:
    agent, provider, _, session = build(settings, store, ["a.png"])
    await turn(agent, store, session, "I will caption these myself", ["a.png"])
    provider.turns = [TurnResult([text_block("Saved.")], "end_turn")]
    provider.router_calls.clear()

    await turn(agent, store, session, "a.png is a red car")

    # No images with the second message, so there is nothing to route.
    assert provider.router_calls == []
    history = provider.requests[1]["messages"]
    assert "<manual_captioning>" in history[0]["content"]
    assert history[-1]["content"].endswith("a.png is a red car")


async def test_manual_mode_with_a_pending_claude_question_still_answers_it(
    settings: Settings, store: SessionStore
) -> None:
    ask = tool_use("q1", "ask_user", {"question": "Flip?", "options": ["Yes", "No"]})
    agent, provider, _, session = build(
        settings,
        store,
        ["a.png"],
        [TurnResult([ask], "tool_use"), TurnResult([text_block("Ok.")], "end_turn")],
        manual=False,
    )
    await turn(agent, store, session, "hello")
    provider.manual_intent = True

    await turn(agent, store, session, "a.png is a car", ["a.png"])

    last = provider.requests[1]["messages"][-1]["content"]
    assert last[0]["type"] == "tool_result" and last[0]["tool_use_id"] == "q1"
    assert "<manual_captioning>" in last[1]["text"]


async def test_the_real_server_saves_manual_captions_verbatim(
    settings: Settings, store: SessionStore
) -> None:
    from image_captioner_mcp.captioner import StubCaptioner
    from image_captioner_mcp.config import ServerConfig
    from image_captioner_mcp.server import create_server
    from image_captioner_mcp.service import CaptionService

    session = store.create()
    files = SessionFiles(settings.sessions_dir, session.id)
    make_png(files.uploads / "a.png")
    make_png(files.uploads / "b.png")
    out = files.tool_path("outputs")
    entries = [
        {"path": files.tool_path("uploads", "a.png"), "caption": "sks dog, (running:1.2)"},
        {"path": files.tool_path("uploads", "b.png"), "caption": "sks dog, sleeping"},
        {"path": "../outside.png", "caption": "nope"},
    ]
    provider = FakeProvider(
        [
            TurnResult([tool_use("m1", MANUAL_TOOL, {"captions": entries, "output_dir": out})], "tool_use"),
            TurnResult([text_block("Saved 2, 1 failed.")], "end_turn"),
        ]
    )
    provider.manual_intent = True
    server = create_server(
        CaptionService(
            ServerConfig(allowed_root=settings.sessions_dir, backend="stub", preload=False), StubCaptioner()
        )
    )
    toolbox = McpToolbox(settings, target=server)
    await toolbox.connect()
    try:
        agent = CaptionAgent(provider, toolbox, store, settings)
        fresh = store.get(session.id)
        assert fresh is not None
        events = [
            e
            async for e in agent.run(fresh, "a is sks dog running, b is sks dog sleeping", ["a.png", "b.png"])
        ]
    finally:
        await toolbox.close()

    assert (files.outputs / "a.txt").read_text(encoding="utf-8") == "sks dog, (running:1.2)"
    assert (files.outputs / "b.txt").read_text(encoding="utf-8") == "sks dog, sleeping"
    result = json.loads(provider.requests[1]["messages"][-1]["content"][0]["content"])
    assert (result["written"], result["failed"]) == (2, 1)
    assert "outside the allowed root" in result["results"][2]["error"]
    assert [e["name"] for e in events if e["type"] == "tool_end"] == [MANUAL_TOOL]
