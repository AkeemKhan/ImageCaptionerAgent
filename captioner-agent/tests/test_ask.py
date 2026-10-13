"""Images without PNG info: the user chooses BLIP or Anthropic."""

from __future__ import annotations

import json
from typing import Any

import pytest

from captioner_agent.agent import CaptionAgent
from captioner_agent.files import SessionFiles
from captioner_agent.llm import TurnResult
from captioner_agent.pipeline import ANTHROPIC_OPTION, BLIP_OPTION, parse_backend_choice
from captioner_agent.settings import Settings
from captioner_agent.store import Session, SessionStore
from captioner_agent.toolbox import ToolResult

from .conftest import FakeProvider, FakeToolbox, make_png, text_block, tool_use

pytestmark = pytest.mark.anyio


@pytest.fixture
def ask_settings(settings: Settings) -> Settings:
    settings.vision_backend = "ask"
    return settings


def build(
    settings: Settings, store: SessionStore, images: list[str]
) -> tuple[CaptionAgent, FakeProvider, FakeToolbox, Session]:
    provider = FakeProvider([TurnResult([text_block("Summary.")], "end_turn")])
    toolbox = FakeToolbox()
    toolbox.results["BLIP_CAPTIONER"] = ToolResult('{"caption": "a blip caption"}')
    session = store.create()
    files = SessionFiles(settings.sessions_dir, session.id)
    for name in images:
        make_png(files.uploads / name)
    return CaptionAgent(provider, toolbox, store, settings), provider, toolbox, session


async def turn(
    agent: CaptionAgent,
    store: SessionStore,
    session: Session,
    text: str = "",
    uploaded: list[str] | None = None,
) -> list[dict[str, Any]]:
    fresh = store.get(session.id)
    assert fresh is not None
    return [event async for event in agent.run(fresh, text, uploaded or [])]


def results_sent_to_claude(provider: FakeProvider) -> list[dict[str, str]]:
    text: str = provider.requests[0]["messages"][0]["content"]
    block = text.split("<caption_results>\n")[1].split("\n</caption_results>")[0]
    parsed: list[dict[str, str]] = json.loads(block)
    return parsed


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (BLIP_OPTION, "blip"),
        (ANTHROPIC_OPTION, "claude"),
        ("blip", "blip"),
        ("Use BLIP please", "blip"),
        ("Anthropic", "claude"),
        ("claude", "claude"),
        ("CLAUDE, thanks", "claude"),
        ("both", None),
        ("blip or anthropic?", None),
        ("whatever", None),
        ("", None),
    ],
)
def test_parse_backend_choice(answer: str, expected: str | None) -> None:
    assert parse_backend_choice(answer) == expected


async def test_png_info_runs_first_then_the_user_is_asked_and_claude_is_not_called(
    ask_settings: Settings, store: SessionStore
) -> None:
    agent, provider, toolbox, session = build(ask_settings, store, ["a.png", "b.png", "c.png"])
    toolbox.png_info = {"a.png": "prompt a"}

    events = await turn(agent, store, session, "caption these", ["a.png", "b.png", "c.png"])

    # PNG info was tried for every image; nothing was captioned by a model yet.
    assert [name for name, _ in toolbox.calls] == ["caption_from_png_info"] * 3
    assert provider.requests == [] and provider.vision_calls == []
    choice = events[-2]
    assert choice["type"] == "choice"
    assert choice["options"] == [BLIP_OPTION, ANTHROPIC_OPTION]
    assert choice["question"].startswith("2 images have no PNG info (b.png, c.png)")
    assert events[-1]["type"] == "done"

    saved = store.get(session.id)
    assert saved is not None and saved.pending is not None
    assert saved.pending["kind"] == "vision" and saved.pending["missing"] == ["b.png", "c.png"]
    assert saved.messages == []  # Claude has not been told anything yet
    assert [t["type"] for t in saved.transcript] == ["user", "tool", "tool", "tool", "choice"]
    assert saved.title == "caption these"


@pytest.mark.parametrize(
    ("answer", "backend", "blip_calls", "anthropic_calls"),
    [(BLIP_OPTION, "blip", 1, 0), (ANTHROPIC_OPTION, "claude", 0, 1), ("Use Claude", "claude", 0, 1)],
)
async def test_the_answer_picks_the_backend_and_the_pipeline_resumes(
    ask_settings: Settings,
    store: SessionStore,
    answer: str,
    backend: str,
    blip_calls: int,
    anthropic_calls: int,
) -> None:
    agent, provider, toolbox, session = build(ask_settings, store, ["a.png", "b.png"])
    toolbox.png_info = {"a.png": "prompt a"}
    await turn(agent, store, session, "caption these", ["a.png", "b.png"])
    calls_before = len(toolbox.calls)

    events = await turn(agent, store, session, answer)

    assert [n for n, _ in toolbox.calls[calls_before:]].count("BLIP_CAPTIONER") == blip_calls
    assert len(provider.vision_calls) == anthropic_calls
    # PNG info is not run again for images already handled.
    assert [n for n, _ in toolbox.calls].count("caption_from_png_info") == 2
    assert events[-1]["type"] == "done"

    # Claude's first message has the original text and every image's result in
    # upload order, and does not contain the button the user clicked.
    sent: str = provider.requests[0]["messages"][0]["content"]
    assert sent.endswith("caption these") and answer not in sent
    assert [(r["image"].rsplit("/", 1)[-1], r["source"]) for r in results_sent_to_claude(provider)] == [
        ("a.png", "png_info"),
        ("b.png", "vision"),
    ]
    # Claude is told which backend the user picked, so its summary can say so.
    assert [r.get("via") for r in results_sent_to_claude(provider)] == [None, backend]
    finished = store.get(session.id)
    assert finished is not None and finished.pending is None
    assert [m["role"] for m in finished.messages] == ["user", "assistant"]


async def test_an_unclear_answer_asks_again_without_captioning_anything(
    ask_settings: Settings, store: SessionStore
) -> None:
    agent, provider, toolbox, session = build(ask_settings, store, ["b.png"])
    first = await turn(agent, store, session, "go", ["b.png"])
    calls_before = len(toolbox.calls)

    second = await turn(agent, store, session, "hmm, not sure")

    assert len(toolbox.calls) == calls_before and provider.requests == [] and provider.vision_calls == []
    assert second[-2]["type"] == "choice" and second[-2]["id"] != first[-2]["id"]
    pending = store.get(session.id)
    assert pending is not None and pending.pending is not None and pending.pending["text"] == "go"

    await turn(agent, store, session, "BLIP")
    assert [n for n, _ in toolbox.calls].count("BLIP_CAPTIONER") == 1


async def test_images_added_with_the_answer_join_the_same_choice(
    ask_settings: Settings, store: SessionStore
) -> None:
    agent, provider, toolbox, session = build(ask_settings, store, ["a.png", "b.png", "c.png"])
    toolbox.png_info = {"c.png": "prompt c"}
    await turn(agent, store, session, "first", ["a.png"])

    await turn(agent, store, session, BLIP_OPTION, ["b.png", "c.png"])

    blipped = [args["path"].rsplit("/", 1)[-1] for n, args in toolbox.calls if n == "BLIP_CAPTIONER"]
    assert sorted(blipped) == ["a.png", "b.png"]
    assert [(r["image"].rsplit("/", 1)[-1], r["source"]) for r in results_sent_to_claude(provider)] == [
        ("a.png", "vision"),
        ("b.png", "vision"),
        ("c.png", "png_info"),
    ]


async def test_no_question_when_every_image_has_png_info(ask_settings: Settings, store: SessionStore) -> None:
    agent, provider, toolbox, session = build(ask_settings, store, ["a.png", "b.png"])
    toolbox.png_info = {"a.png": "prompt a", "b.png": "prompt b"}

    events = await turn(agent, store, session, "", ["a.png", "b.png"])

    assert not [e for e in events if e["type"] == "choice"]
    assert len(provider.requests) == 1
    assert [n for n, _ in toolbox.calls] == ["caption_from_png_info"] * 2


@pytest.mark.parametrize("backend", ["blip", "claude", "local"])
async def test_a_pinned_backend_never_asks(ask_settings: Settings, store: SessionStore, backend: str) -> None:
    ask_settings.vision_backend = backend  # type: ignore[assignment]
    agent, provider, toolbox, session = build(ask_settings, store, ["a.png"])
    toolbox.results["caption_image"] = ToolResult('{"caption": "local caption"}')

    events = await turn(agent, store, session, "", ["a.png"])

    assert not [e for e in events if e["type"] == "choice"]
    assert results_sent_to_claude(provider)[0]["source"] == "vision"


async def test_a_pending_claude_question_is_not_mistaken_for_the_vision_question(
    ask_settings: Settings, store: SessionStore
) -> None:
    ask = tool_use("q1", "ask_user", {"question": "Flip?", "options": ["Yes", "No"]})
    agent, provider, _, session = build(ask_settings, store, [])
    provider.turns = [TurnResult([ask], "tool_use"), TurnResult([text_block("Ok.")], "end_turn")]
    await turn(agent, store, session, "hello")

    await turn(agent, store, session, "No")

    answered = provider.requests[1]["messages"][-1]["content"]
    assert [(b["tool_use_id"], b["content"]) for b in answered] == [("q1", "No")]


async def test_the_pending_choice_is_exposed_to_the_ui_and_survives_a_restart(
    ask_settings: Settings, store: SessionStore
) -> None:
    agent, _, _, session = build(ask_settings, store, ["b.png"])
    events = await turn(agent, store, session, "", ["b.png"])

    reloaded = store.get(session.id)  # as the server does after a restart

    assert reloaded is not None and reloaded.pending is not None
    assert reloaded.pending["choice"]["id"] == events[-2]["id"]
    assert reloaded.pending["choice"]["options"] == [BLIP_OPTION, ANTHROPIC_OPTION]
