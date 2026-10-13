"""The ENHANCE_CAPTION tool: an optional rewrite of a caption that already exists."""

from __future__ import annotations

import json
from typing import Any

import pytest

from captioner_agent.agent import ENHANCE_DEFINITION, CaptionAgent
from captioner_agent.files import SessionFiles
from captioner_agent.llm import LLMError, TurnResult
from captioner_agent.prompts import ENHANCE_TOOL
from captioner_agent.settings import Settings
from captioner_agent.store import Session, SessionStore

from .conftest import FakeProvider, FakeToolbox, make_png, text_block, tool_use

pytestmark = pytest.mark.anyio


def build(
    settings: Settings, store: SessionStore, calls: list[dict[str, Any]]
) -> tuple[CaptionAgent, FakeProvider, FakeToolbox, Session, SessionFiles]:
    provider = FakeProvider([TurnResult(calls, "tool_use"), TurnResult([text_block("Done.")], "end_turn")])
    toolbox = FakeToolbox()
    session = store.create()
    files = SessionFiles(settings.sessions_dir, session.id)
    files.ensure()
    return CaptionAgent(provider, toolbox, store, settings), provider, toolbox, session, files


async def run_turn(agent: CaptionAgent, session: Session) -> list[dict[str, Any]]:
    return [event async for event in agent.run(session, "enhance them", [])]


def tool_results(provider: FakeProvider) -> dict[str, dict[str, Any]]:
    return {b["tool_use_id"]: b for b in provider.requests[1]["messages"][-1]["content"]}


def test_tool_is_named_as_requested() -> None:
    assert ENHANCE_TOOL == "ENHANCE_CAPTION"
    assert ENHANCE_DEFINITION["name"] == "ENHANCE_CAPTION"
    assert ENHANCE_DEFINITION["input_schema"]["properties"]["mode"]["enum"] == ["improved", "creative"]


async def test_enhances_existing_caption_and_saves_it(settings: Settings, store: SessionStore) -> None:
    agent, provider, toolbox, session, files = build(
        settings,
        store,
        [tool_use("e1", ENHANCE_TOOL, {"path": "s/uploads/a.png", "mode": "creative", "guidance": "moody"})],
    )
    make_png(files.uploads / "a.png")
    (files.outputs / "a.txt").write_text("masterpiece, a knight, stone bridge, sunset", encoding="utf-8")
    provider.enhanced_caption = "  A lone knight stands\n on a stone bridge at sunset.  "

    events = await run_turn(agent, session)

    # Claude receives the caption text and the mode's instructions, never the image.
    assert provider.vision_calls == []
    assert len(provider.enhance_calls) == 1
    call = provider.enhance_calls[0]
    assert call["caption"] == "masterpiece, a knight, stone bridge, sunset"
    assert "more vivid" in call["instructions"] and "moody" in call["instructions"]
    assert "Keep captions under 20 words." in call["instructions"]

    enhanced = "A lone knight stands on a stone bridge at sunset."
    assert toolbox.calls == [
        (
            "write_caption",
            {
                "image_path": f"{session.id}/uploads/a.png",
                "text": enhanced,
                "output_dir": f"{session.id}/outputs",
            },
        )
    ]
    assert json.loads(tool_results(provider)["e1"]["content"]) == {
        "image": "a.png",
        "mode": "creative",
        "original": "masterpiece, a knight, stone bridge, sunset",
        "enhanced": enhanced,
    }
    assert [(e["name"], e["ok"]) for e in events if e["type"] == "tool_end"] == [(ENHANCE_TOOL, True)]


async def test_improved_mode_uses_its_own_instructions(settings: Settings, store: SessionStore) -> None:
    agent, provider, _, session, files = build(
        settings, store, [tool_use("e1", ENHANCE_TOOL, {"path": "s/uploads/a.png", "mode": "improved"})]
    )
    make_png(files.uploads / "a.png")
    (files.outputs / "a.txt").write_text("a cat, sofa", encoding="utf-8")

    await run_turn(agent, session)

    instructions = provider.enhance_calls[0]["instructions"]
    assert "clear, natural English" in instructions and "more vivid" not in instructions


async def test_refuses_an_image_that_is_not_captioned_yet(settings: Settings, store: SessionStore) -> None:
    agent, provider, toolbox, session, files = build(
        settings,
        store,
        [
            tool_use("e1", ENHANCE_TOOL, {"path": "s/uploads/a.png", "mode": "improved"}),
            tool_use("e2", ENHANCE_TOOL, {"path": "s/uploads/missing.png", "mode": "improved"}),
            tool_use("e3", ENHANCE_TOOL, {"path": "s/uploads/a.png", "mode": "poetic"}),
            tool_use("e4", ENHANCE_TOOL, {"path": "../../etc/a.png", "mode": "improved"}),
        ],
    )
    make_png(files.uploads / "a.png")

    await run_turn(agent, session)

    results = tool_results(provider)
    assert all(results[key]["is_error"] for key in ("e1", "e2", "e3", "e4"))
    assert "no caption yet" in results["e1"]["content"]
    assert results["e2"]["content"] == "image not found: missing.png"
    assert "INVALID_JSON" in results["e3"]["content"]
    assert "inside upload_dir or output_dir" in results["e4"]["content"]
    assert provider.enhance_calls == [] and toolbox.calls == []


async def test_model_failure_leaves_the_caption_unchanged(settings: Settings, store: SessionStore) -> None:
    agent, provider, toolbox, session, files = build(
        settings, store, [tool_use("e1", ENHANCE_TOOL, {"path": "s/uploads/a.png", "mode": "improved"})]
    )
    make_png(files.uploads / "a.png")
    (files.outputs / "a.txt").write_text("a cat, sofa", encoding="utf-8")
    provider.enhanced_caption = LLMError("Rate limited.")

    await run_turn(agent, session)

    result = tool_results(provider)["e1"]
    assert (result["content"], result["is_error"]) == ("Rate limited.", True)
    assert toolbox.calls == []
    assert (files.outputs / "a.txt").read_text(encoding="utf-8") == "a cat, sofa"


async def test_system_prompt_offers_enhancement_before_flipping(
    settings: Settings, store: SessionStore
) -> None:
    agent, provider, _, session, _ = build(settings, store, [])
    provider.turns = [TurnResult([text_block("ok")], "end_turn")]

    await run_turn(agent, session)

    system: str = provider.requests[0]["system"]
    assert system.index("# Enhancing captions") < system.index("# Flipped copies")
    assert ENHANCE_TOOL in {tool["name"] for tool in provider.requests[0]["tools"]}
