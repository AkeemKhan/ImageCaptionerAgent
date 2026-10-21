"""Gemini for cleanup and upscaling: the user chooses first, and the app enforces it."""

from __future__ import annotations

from typing import Any

import pytest

from captioner_agent.agent import CaptionAgent
from captioner_agent.cleanup_choice import (
    CLEANUP_OPTIONS,
    GEMINI_OPTION,
    LOCAL_OPTION,
    gate_message,
    parse_cleanup_choice,
)
from captioner_agent.llm import TurnResult
from captioner_agent.prompts import ASK_USER_TOOL, CLEANUP_TOOL, build_system_prompt
from captioner_agent.settings import Settings
from captioner_agent.store import Session, SessionStore
from captioner_agent.toolbox import ToolResult

from .conftest import FakeProvider, FakeToolbox, text_block, tool_use

pytestmark = pytest.mark.anyio

IMAGE = "sid/uploads/a.png"


class CleanupToolbox(FakeToolbox):
    """Adds the cleanup tool, with the schema the MCP server publishes."""

    def definitions(self) -> list[dict[str, Any]]:
        return [
            *super().definitions(),
            {
                "name": CLEANUP_TOOL,
                "description": "Clean up",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "backend": {"type": "string", "enum": ["local", "gemini"]},
                    },
                    "required": ["path"],
                },
            },
        ]


@pytest.fixture
def gemini_settings(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    settings.gemini_cleanup = True
    return settings


def build(
    settings: Settings, store: SessionStore, turns: list[TurnResult]
) -> tuple[CaptionAgent, FakeProvider, CleanupToolbox, Session]:
    provider = FakeProvider(turns)
    toolbox = CleanupToolbox()
    toolbox.results[CLEANUP_TOOL] = ToolResult('{"backend": "x"}')
    return CaptionAgent(provider, toolbox, store, settings), provider, toolbox, store.create()


async def say(agent: CaptionAgent, store: SessionStore, session: Session, text: str) -> list[dict[str, Any]]:
    fresh = store.get(session.id)
    assert fresh is not None
    return [event async for event in agent.run(fresh, text, [])]


def cleanup_call(call_id: str = "c1", **extra: Any) -> TurnResult:
    return TurnResult([tool_use(call_id, CLEANUP_TOOL, {"path": IMAGE, **extra})], "tool_use")


def ask(options: list[str], call_id: str = "q1") -> TurnResult:
    question = tool_use(call_id, ASK_USER_TOOL, {"question": "Which backend?", "options": options})
    return TurnResult([question], "tool_use")


def end(text: str = "Done.") -> TurnResult:
    return TurnResult([text_block(text)], "end_turn")


def tool_results(provider: FakeProvider, request: int) -> list[dict[str, Any]]:
    content = provider.requests[request]["messages"][-1]["content"]
    return [block for block in content if block["type"] == "tool_result"]


def cleanup_section(prompt: str) -> str:
    return prompt.split("# Cleaning up")[1].split("# Other requests")[0]


# --- reading the answer -----------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (LOCAL_OPTION, "local"),
        (GEMINI_OPTION, "gemini"),
        ("local", "local"),
        ("Use Gemini", "gemini"),
        ("GEMINI please", "gemini"),
        ("google", "gemini"),
        ("keep it on this machine", "local"),
        ("local or gemini?", None),
        ("both", None),
        ("whatever", None),
        ("yes", None),
        ("", None),
    ],
)
def test_parse_cleanup_choice(answer: str, expected: str | None) -> None:
    assert parse_cleanup_choice(answer) == expected


def test_the_two_options_name_different_backends() -> None:
    assert [parse_cleanup_choice(o) for o in CLEANUP_OPTIONS] == ["local", "gemini"]
    assert "Google" in GEMINI_OPTION and "this machine" in LOCAL_OPTION


def test_the_gate_message_names_the_tool_the_question_tool_and_the_options() -> None:
    message = gate_message(CLEANUP_TOOL, ASK_USER_TOOL)

    assert message.startswith("NOT RUN")
    assert CLEANUP_TOOL in message and ASK_USER_TOOL in message
    assert all(option in message for option in CLEANUP_OPTIONS)


# --- the prompt -------------------------------------------------------------


def test_the_prompt_mentions_gemini_only_when_it_is_available() -> None:
    off = build_system_prompt("rules")
    on = build_system_prompt("rules", gemini_cleanup=True)

    assert "Gemini" not in cleanup_section(off)
    assert GEMINI_OPTION in on and LOCAL_OPTION in on
    assert "Never pass its backend" in on


# --- the gate ---------------------------------------------------------------


async def test_without_gemini_cleanup_runs_locally_whatever_claude_passes(
    settings: Settings, store: SessionStore
) -> None:
    agent, provider, toolbox, session = build(settings, store, [cleanup_call(backend="gemini"), end()])

    await say(agent, store, session, "remove the watermark")

    assert toolbox.calls == [(CLEANUP_TOOL, {"path": IMAGE, "backend": "local"})]
    assert "Gemini" not in cleanup_section(provider.requests[0]["system"])


async def test_gemini_needs_the_switch_and_a_key(
    settings: Settings, store: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings.gemini_cleanup = True  # switched on, but no key
    agent, _, toolbox, session = build(settings, store, [cleanup_call(), end()])

    await say(agent, store, session, "clean it")

    assert toolbox.calls == [(CLEANUP_TOOL, {"path": IMAGE, "backend": "local"})]


async def test_a_cleanup_before_the_user_has_chosen_is_refused(
    gemini_settings: Settings, store: SessionStore
) -> None:
    agent, provider, toolbox, session = build(gemini_settings, store, [cleanup_call(), end("Asking.")])

    events = await say(agent, store, session, "remove the watermark")

    assert toolbox.calls == []
    (result,) = tool_results(provider, 1)
    assert result["is_error"] is True and result["content"].startswith("NOT RUN")
    assert [e["ok"] for e in events if e["type"] == "tool_end"] == [False]
    assert GEMINI_OPTION in provider.requests[0]["system"]


async def test_claude_cannot_pick_gemini_for_itself(gemini_settings: Settings, store: SessionStore) -> None:
    agent, _, toolbox, session = build(gemini_settings, store, [cleanup_call(backend="gemini"), end()])

    await say(agent, store, session, "remove the watermark")

    assert toolbox.calls == []


@pytest.mark.parametrize(
    ("answer", "backend"),
    [(GEMINI_OPTION, "gemini"), (LOCAL_OPTION, "local"), ("gemini", "gemini"), ("Local", "local")],
)
async def test_the_users_answer_decides_the_backend(
    gemini_settings: Settings, store: SessionStore, answer: str, backend: str
) -> None:
    # Claude asks, then (after the answer) calls the tool and tries to pass the other backend.
    other = "local" if backend == "gemini" else "gemini"
    agent, _, toolbox, session = build(
        gemini_settings, store, [ask(CLEANUP_OPTIONS), cleanup_call(backend=other), end()]
    )

    asked = await say(agent, store, session, "remove the watermark from 3 images")
    assert asked[-2]["type"] == "choice" and asked[-2]["options"] == CLEANUP_OPTIONS
    assert toolbox.calls == []

    answered = await say(agent, store, session, answer)

    assert toolbox.calls == [(CLEANUP_TOOL, {"path": IMAGE, "backend": backend})]
    # The step shown to the user carries the backend that was really used.
    (start,) = [e for e in answered if e["type"] == "tool_start"]
    assert start["input"]["backend"] == backend


async def test_an_unclear_answer_does_not_count_as_consent(
    gemini_settings: Settings, store: SessionStore
) -> None:
    agent, provider, toolbox, session = build(
        gemini_settings, store, [ask(CLEANUP_OPTIONS), cleanup_call(), end("Which one?")]
    )
    await say(agent, store, session, "clean these")

    await say(agent, store, session, "local or gemini?")

    assert toolbox.calls == []
    (result,) = tool_results(provider, 2)
    assert result["content"].startswith("NOT RUN")


async def test_the_choice_does_not_carry_over_to_the_next_message(
    gemini_settings: Settings, store: SessionStore
) -> None:
    agent, provider, toolbox, session = build(
        gemini_settings,
        store,
        [ask(CLEANUP_OPTIONS), cleanup_call("c1"), end("Cleaned."), cleanup_call("c2"), end("Asking.")],
    )
    await say(agent, store, session, "clean these")
    await say(agent, store, session, GEMINI_OPTION)
    assert toolbox.calls == [(CLEANUP_TOOL, {"path": IMAGE, "backend": "gemini"})]

    await say(agent, store, session, "now clean the other one too")

    assert len(toolbox.calls) == 1  # refused: the user is asked again
    assert tool_results(provider, 4)[0]["content"].startswith("NOT RUN")


async def test_an_answer_to_some_other_question_is_not_consent(
    gemini_settings: Settings, store: SessionStore
) -> None:
    agent, _, toolbox, session = build(gemini_settings, store, [ask(["Yes", "No"]), cleanup_call(), end()])
    await say(agent, store, session, "flip them?")

    await say(agent, store, session, "gemini")  # names Gemini, but nobody asked about cleanup

    assert toolbox.calls == []


async def test_a_batch_is_decided_once_for_every_image(
    gemini_settings: Settings, store: SessionStore
) -> None:
    batch = TurnResult(
        [tool_use(f"c{i}", CLEANUP_TOOL, {"path": f"sid/uploads/{i}.png"}) for i in range(3)], "tool_use"
    )
    agent, _, toolbox, session = build(gemini_settings, store, [ask(CLEANUP_OPTIONS), batch, end()])
    await say(agent, store, session, "clean all three")

    await say(agent, store, session, GEMINI_OPTION)

    assert sorted(args["path"] for _, args in toolbox.calls) == [f"sid/uploads/{i}.png" for i in range(3)]
    assert {args["backend"] for _, args in toolbox.calls} == {"gemini"}


# --- the key reaches the server only when Gemini is on ---------------------


def server_env(settings: Settings) -> dict[str, str]:
    from captioner_agent.toolbox import McpToolbox

    target = McpToolbox(settings)._target()
    assert not isinstance(target, str)
    return dict(target.env or {})


def test_the_server_gets_no_gemini_key_unless_gemini_cleanup_is_on(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "other-secret")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-secret")

    env = server_env(settings)  # gemini_cleanup is off by default

    assert not any("secret" in value for value in env.values())
    assert "CAPTIONER_GEMINI_ENABLED" not in env


def test_the_server_gets_the_key_and_settings_when_gemini_cleanup_is_on(
    gemini_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "other-secret")
    gemini_settings.gemini_model = "some-image-model"
    gemini_settings.gemini_upscale_fallback = "resize"

    env = server_env(gemini_settings)

    assert env["GEMINI_API_KEY"] == "test-key"
    assert env["CAPTIONER_GEMINI_ENABLED"] == "true"
    assert env["CAPTIONER_GEMINI_MODEL"] == "some-image-model"
    assert env["CAPTIONER_GEMINI_UPSCALE_FALLBACK"] == "resize"
    assert env["CAPTIONER_GEMINI_IMAGE_SIZE"] == "2K"
    assert env["CAPTIONER_GEMINI_PROMPT_FILE"].endswith("gemini_cleanup_prompt.md")
    assert "CAPTIONER_GEMINI_MAX_DRIFT" not in env
    assert "ANTHROPIC_API_KEY" not in env


def test_switched_on_without_a_key_sends_nothing(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings.gemini_cleanup = True

    assert "CAPTIONER_GEMINI_ENABLED" not in server_env(settings)
