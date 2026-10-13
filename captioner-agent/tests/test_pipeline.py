"""The fixed captioning order: PNG info first, a vision model only when there is none."""

from __future__ import annotations

import json
from typing import Any

import pytest

from captioner_agent.agent import CaptionAgent
from captioner_agent.files import SessionFiles
from captioner_agent.llm import LLMError, TurnResult
from captioner_agent.prompts import DESCRIBE_TOOL
from captioner_agent.settings import Settings
from captioner_agent.store import Session, SessionStore
from captioner_agent.toolbox import ToolResult

from .conftest import FakeProvider, FakeToolbox, make_png, text_block, tool_use

pytestmark = pytest.mark.anyio


def build(
    settings: Settings,
    store: SessionStore,
    images: list[str],
    turns: list[TurnResult | Exception] | None = None,
) -> tuple[CaptionAgent, FakeProvider, FakeToolbox, Session]:
    provider = FakeProvider(
        turns if turns is not None else [TurnResult([text_block("Summary.")], "end_turn")]
    )
    toolbox = FakeToolbox()
    session = store.create()
    files = SessionFiles(settings.sessions_dir, session.id)
    for name in images:
        make_png(files.uploads / name)
    return CaptionAgent(provider, toolbox, store, settings), provider, toolbox, session


async def run_turn(agent: CaptionAgent, session: Session, uploaded: list[str]) -> list[dict[str, Any]]:
    return [event async for event in agent.run(session, "", uploaded)]


def caption_results(provider: FakeProvider) -> list[dict[str, str]]:
    text: str = provider.requests[0]["messages"][0]["content"]
    block = text.split("<caption_results>\n")[1].split("\n</caption_results>")[0]
    results: list[dict[str, str]] = json.loads(block)
    return results


async def test_png_info_is_used_and_claude_never_sees_the_image(
    settings: Settings, store: SessionStore
) -> None:
    agent, provider, toolbox, session = build(settings, store, ["a.png"])
    toolbox.png_info["a.png"] = "a cat on a sofa"

    events = await run_turn(agent, session, ["a.png"])

    assert provider.vision_calls == []
    assert [name for name, _ in toolbox.calls] == ["caption_from_png_info"]
    assert toolbox.calls[0][1] == {
        "path": f"{session.id}/uploads/a.png",
        "output_dir": f"{session.id}/outputs",
    }
    assert [(e["type"], e["name"]) for e in events[:2]] == [
        ("tool_start", "caption_from_png_info"),
        ("tool_end", "caption_from_png_info"),
    ]
    assert caption_results(provider) == [
        {"image": f"{session.id}/uploads/a.png", "source": "png_info", "caption": "a cat on a sofa"}
    ]


async def test_vision_is_used_only_after_png_info_is_missing(settings: Settings, store: SessionStore) -> None:
    agent, provider, toolbox, session = build(settings, store, ["a.png"])
    provider.vision_caption = "  A red\n square.  "

    events = await run_turn(agent, session, ["a.png"])

    # PNG info is tried first; the vision caption is saved afterwards.
    assert [name for name, _ in toolbox.calls] == ["caption_from_png_info", "write_caption"]
    assert toolbox.calls[1][1] == {
        "image_path": f"{session.id}/uploads/a.png",
        "text": "A red square.",
        "output_dir": f"{session.id}/outputs",
    }
    assert len(provider.vision_calls) == 1
    call = provider.vision_calls[0]
    assert call["media_type"] == "image/jpeg" and call["image"].startswith(b"\xff\xd8")
    assert "Keep captions under 20 words." in call["instructions"]
    # Two separate steps in the UI: PNG info finds nothing, then the vision step captions it.
    ends = [(e["name"], e["ok"], e["summary"]) for e in events if e["type"] == "tool_end"]
    assert ends == [
        ("caption_from_png_info", True, "No PNG info"),
        ("caption_with_vision", True, "A red square."),
    ]
    assert caption_results(provider)[0]["source"] == "vision"


async def test_mixed_batch_keeps_upload_order_and_sends_only_missing_images_to_vision(
    settings: Settings, store: SessionStore
) -> None:
    names = ["a.png", "b.png", "c.png"]
    agent, provider, toolbox, session = build(settings, store, names)
    toolbox.png_info = {"a.png": "prompt a", "c.png": "prompt c"}

    await run_turn(agent, session, names)

    assert len(provider.vision_calls) == 1
    assert [(r["image"].rsplit("/", 1)[-1], r["source"]) for r in caption_results(provider)] == [
        ("a.png", "png_info"),
        ("b.png", "vision"),
        ("c.png", "png_info"),
    ]


async def test_vision_failure_is_reported_and_does_not_stop_the_batch(
    settings: Settings, store: SessionStore
) -> None:
    agent, provider, toolbox, session = build(settings, store, ["a.png", "b.png"])
    toolbox.png_info["b.png"] = "prompt b"
    provider.vision_caption = LLMError("Rate limited.")

    events = await run_turn(agent, session, ["a.png", "b.png"])

    results = caption_results(provider)
    assert results[0] == {
        "image": f"{session.id}/uploads/a.png",
        "source": "failed",
        "via": "claude",
        "error": "Rate limited.",
    }
    assert results[1]["source"] == "png_info"
    # Both PNG info steps succeed; only the vision step for a.png fails.
    assert sorted(e["ok"] for e in events if e["type"] == "tool_end") == [False, True, True]
    assert events[-1]["type"] == "done"


async def test_png_info_tool_error_does_not_fall_through_to_vision(
    settings: Settings, store: SessionStore
) -> None:
    agent, provider, toolbox, session = build(settings, store, ["a.png"])
    toolbox.results["caption_from_png_info"] = ToolResult("the file is not a readable image", is_error=True)

    await run_turn(agent, session, ["a.png"])

    assert provider.vision_calls == []
    assert caption_results(provider)[0]["source"] == "failed"


async def test_local_backend_uses_the_server_model_instead_of_claude(
    settings: Settings, store: SessionStore
) -> None:
    settings.vision_backend = "local"
    agent, provider, toolbox, session = build(settings, store, ["a.png"])
    toolbox.results["caption_image"] = ToolResult('{"caption": "a local caption"}')

    await run_turn(agent, session, ["a.png"])

    assert provider.vision_calls == []
    assert [name for name, _ in toolbox.calls] == ["caption_from_png_info", "caption_image", "write_caption"]
    assert caption_results(provider)[0] == {
        "image": f"{session.id}/uploads/a.png",
        "source": "vision",
        "via": "local",
        "caption": "a local caption",
    }


async def test_blip_backend_calls_blip_captioner_only_when_there_is_no_png_info(
    settings: Settings, store: SessionStore
) -> None:
    settings.vision_backend = "blip"
    settings.caption_style = "short"
    agent, provider, toolbox, session = build(settings, store, ["a.png", "b.png"])
    toolbox.png_info["a.png"] = "prompt a"
    toolbox.results["BLIP_CAPTIONER"] = ToolResult('{"caption": "a house on a field"}')

    events = await run_turn(agent, session, ["a.png", "b.png"])

    blip_calls = [args for name, args in toolbox.calls if name == "BLIP_CAPTIONER"]
    assert blip_calls == [{"path": f"{session.id}/uploads/b.png", "style": "short"}]
    assert provider.vision_calls == []
    # Images run concurrently, so the order that matters is per image: PNG info
    # is tried first for b.png, and only then is BLIP called for it.
    b = f"{session.id}/uploads/b.png"
    b_calls = [name for name, args in toolbox.calls if b in args.values()]
    assert b_calls == ["caption_from_png_info", "BLIP_CAPTIONER", "write_caption"]
    assert [name for name, args in toolbox.calls if f"{session.id}/uploads/a.png" in args.values()] == [
        "caption_from_png_info"
    ]
    assert [(r["source"], r["caption"]) for r in caption_results(provider)] == [
        ("png_info", "prompt a"),
        ("vision", "a house on a field"),
    ]
    assert [e["name"] for e in events if e["type"] == "tool_end"].count("BLIP_CAPTIONER") == 1


async def test_blip_failure_is_reported_for_that_image_only(settings: Settings, store: SessionStore) -> None:
    settings.vision_backend = "blip"
    agent, provider, toolbox, session = build(settings, store, ["a.png"])
    toolbox.results["BLIP_CAPTIONER"] = ToolResult("the file is not a readable image", is_error=True)

    await run_turn(agent, session, ["a.png"])

    assert caption_results(provider)[0]["source"] == "failed"
    assert "write_caption" not in [name for name, _ in toolbox.calls]


def test_asking_the_user_is_the_shipped_default_and_each_backend_can_be_pinned() -> None:
    assert Settings().vision_backend == "ask"
    pinned = {Settings(vision_backend=b).vision_backend for b in ("ask", "blip", "claude", "local")}
    assert pinned == {"ask", "blip", "claude", "local"}


async def test_claude_gets_blip_captioner_but_not_the_other_local_model_tools(
    settings: Settings, store: SessionStore
) -> None:
    agent, provider, toolbox, session = build(settings, store, [])
    toolbox.definitions = lambda: [  # type: ignore[method-assign]
        {"name": name, "description": "", "input_schema": {"type": "object"}}
        for name in (
            "caption_from_png_info",
            "caption_image",
            "caption_folder",
            "BLIP_CAPTIONER",
            "write_caption",
        )
    ]

    await run_turn(agent, session, [])

    names = [tool["name"] for tool in provider.requests[0]["tools"]]
    assert names == [
        "caption_from_png_info",
        "BLIP_CAPTIONER",
        "write_caption",
        DESCRIBE_TOOL,
        "ENHANCE_CAPTION",
        "ask_user",
    ]


async def test_system_prompt_offers_anthropic_and_blip_enhancement(
    settings: Settings, store: SessionStore
) -> None:
    agent, provider, _, session = build(settings, store, [])

    await run_turn(agent, session, [])

    system: str = provider.requests[0]["system"]
    enhancing = system.split("# Enhancing captions")[1].split("# Flipped copies")[0]
    for option in (
        "Keep them as they are",
        "Improve with Anthropic",
        "Make creative with Anthropic",
        "Re-caption with BLIP",
    ):
        assert option in enhancing
    assert "BLIP_CAPTIONER" in enhancing and "ENHANCE_CAPTION" in enhancing


async def test_describe_image_tool_is_confined_to_the_session(
    settings: Settings, store: SessionStore
) -> None:
    agent, provider, _, session = build(
        settings,
        store,
        ["a.png"],
        [
            TurnResult(
                [
                    # Whatever folder prefix Claude sends, the path resolves in this session.
                    tool_use("d1", DESCRIBE_TOOL, {"path": "another-session/uploads/a.png"}),
                    tool_use("d2", DESCRIBE_TOOL, {"path": "../../secrets/key.png"}),
                    tool_use("d3", DESCRIBE_TOOL, {"path": "x/uploads/missing.png"}),
                ],
                "tool_use",
            ),
            TurnResult([text_block("Done.")], "end_turn"),
        ],
    )

    events = [event async for event in agent.run(session, "describe a.png again", [])]

    assert {e["id"]: e["ok"] for e in events if e["type"] == "tool_end"} == {
        "d1": True,
        "d2": False,
        "d3": False,
    }
    results = {b["tool_use_id"]: b["content"] for b in provider.requests[1]["messages"][-1]["content"]}
    assert results["d1"] == "a vision caption"
    assert "inside upload_dir or output_dir" in results["d2"]
    assert results["d3"] == "image not found: missing.png"
