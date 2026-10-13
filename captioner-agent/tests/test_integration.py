"""The agent driving the real MCP server (in-process, stub captioning model).

Only Claude is faked: its tool calls are scripted, and everything they touch is real.
"""

from __future__ import annotations

import pytest

from captioner_agent.agent import CaptionAgent
from captioner_agent.files import SessionFiles
from captioner_agent.llm import TurnResult
from captioner_agent.settings import Settings
from captioner_agent.store import SessionStore
from captioner_agent.toolbox import McpToolbox
from image_captioner_mcp.captioner import StubCaptioner
from image_captioner_mcp.config import ServerConfig
from image_captioner_mcp.server import create_server
from image_captioner_mcp.service import CaptionService

from .conftest import FakeProvider, make_png, text_block, tool_use

pytestmark = pytest.mark.anyio


async def test_caption_and_flip_through_the_mcp_server(settings: Settings, store: SessionStore) -> None:
    session = store.create()
    files = SessionFiles(settings.sessions_dir, session.id)
    files.ensure()
    make_png(
        files.uploads / "Image1.png",
        {"parameters": "(a cat:1.2) on the left, <lora:fur:0.7>\nNegative prompt: dog\nSteps: 20"},
    )
    make_png(files.uploads / "Image2.png")
    image1 = files.tool_path("uploads", "Image1.png")
    out = files.tool_path("outputs")

    # Claude's scripted part comes after the pipeline: it flips Image1 and
    # tries one path outside the allowed root.
    provider = FakeProvider(
        [
            TurnResult(
                [
                    tool_use("f1", "flip_image", {"path": image1, "output_dir": out}),
                    tool_use("r1", "mirror_caption", {"text": "a cat on the left"}),
                    tool_use("x1", "read_png_metadata", {"path": "../outside.png"}),
                ],
                "tool_use",
            ),
            TurnResult(
                [
                    tool_use(
                        "w2",
                        "write_caption",
                        {
                            "image_path": f"{out}/Image1_flipped.png",
                            "text": "a cat on the right",
                            "output_dir": out,
                        },
                    )
                ],
                "tool_use",
            ),
            TurnResult([text_block("Finished.")], "end_turn"),
        ]
    )
    server = create_server(
        CaptionService(
            ServerConfig(allowed_root=settings.sessions_dir, backend="stub", preload=False), StubCaptioner()
        )
    )
    toolbox = McpToolbox(settings, target=server)
    await toolbox.connect()
    try:
        agent = CaptionAgent(provider, toolbox, store, settings)
        events = [event async for event in agent.run(session, "caption these", ["Image1.png", "Image2.png"])]
    finally:
        await toolbox.close()

    # Pipeline: Image1 from PNG info by the server script, Image2 by vision.
    pipeline_steps = sorted(
        (e["input"]["path"].rsplit("/", 1)[-1], e["name"], e["ok"])
        for e in events
        if e["type"] == "tool_end" and e["id"].startswith("auto_")
    )
    assert pipeline_steps == [
        ("Image1.png", "caption_from_png_info", True),
        ("Image2.png", "caption_with_vision", True),
    ]
    assert len(provider.vision_calls) == 1

    # Claude was told the outcome instead of being asked to caption.
    first_message: str = provider.requests[0]["messages"][0]["content"]
    assert '"source": "png_info"' in first_message and '"caption": "a cat on the left"' in first_message
    assert '"source": "vision"' in first_message

    claude_steps = {e["id"]: e["ok"] for e in events if e["type"] == "tool_end" and e["id"][0] != "a"}
    assert claude_steps == {"f1": True, "r1": True, "x1": False, "w2": True}
    results = {b["tool_use_id"]: b for b in provider.requests[1]["messages"][-1]["content"]}
    assert '"text": "a cat on the right"' in results["r1"]["content"]
    assert results["x1"]["is_error"] and "outside the allowed root" in results["x1"]["content"]

    assert [(i["name"], i["caption"]) for i in files.list_items()] == [
        ("Image1.png", "a cat on the left"),
        ("Image2.png", "a vision caption"),
        ("Image1_flipped.png", "a cat on the right"),
    ]
