from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from captioner_agent.llm import LLMError, TurnResult
from captioner_agent.settings import Settings
from captioner_agent.store import SessionStore
from captioner_agent.toolbox import ToolResult


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    criteria = tmp_path / "criteria.md"
    criteria.write_text("- Keep captions under 20 words.", encoding="utf-8")
    # Claude vision unless a test picks another backend; the shipped default is BLIP.
    return Settings(data_dir=tmp_path / "data", criteria_file=criteria, vision_backend="claude")


@pytest.fixture
def store(settings: Settings) -> SessionStore:
    return SessionStore(settings.db_path)


def make_png(path: Path, text: dict[str, str] | None = None) -> Path:
    info = PngInfo()
    for key, value in (text or {}).items():
        info.add_text(key, value)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 2), "red").save(path, pnginfo=info)
    return path


def png_bytes(text: dict[str, str] | None = None) -> bytes:
    import io

    info = PngInfo()
    for key, value in (text or {}).items():
        info.add_text(key, value)
    buffer = io.BytesIO()
    Image.new("RGB", (4, 2), "red").save(buffer, "PNG", pnginfo=info)
    return buffer.getvalue()


def text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def tool_use(tool_id: str, name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}


@dataclass
class FakeProvider:
    """Replays scripted model responses and records every request it receives."""

    turns: list[TurnResult | Exception]
    requests: list[dict[str, Any]] = field(default_factory=list)
    vision_caption: str | Exception = "a vision caption"
    vision_calls: list[dict[str, Any]] = field(default_factory=list)
    enhanced_caption: str | Exception = "An enhanced caption."
    enhance_calls: list[dict[str, Any]] = field(default_factory=list)

    async def enhance_caption(self, *, caption: str, instructions: str) -> str:
        self.enhance_calls.append({"caption": caption, "instructions": instructions})
        if isinstance(self.enhanced_caption, Exception):
            raise self.enhanced_caption
        return self.enhanced_caption

    async def caption_image(self, *, image: bytes, media_type: str, instructions: str) -> str:
        self.vision_calls.append({"image": image, "media_type": media_type, "instructions": instructions})
        if isinstance(self.vision_caption, Exception):
            raise self.vision_caption
        return self.vision_caption

    async def stream_turn(
        self, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[str | TurnResult]:
        import copy

        self.requests.append({"system": system, "messages": copy.deepcopy(messages), "tools": tools})
        if not self.turns:
            raise LLMError("no scripted response left")
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        for block in turn.content:
            if block["type"] == "text":
                yield block["text"]
        yield turn


@dataclass
class FakeToolbox:
    """A toolbox that records calls. By default no image has PNG info.

    ``png_info`` maps an image file name to the caption its PNG info yields.
    """

    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    results: dict[str, ToolResult] = field(default_factory=dict)
    png_info: dict[str, str] = field(default_factory=dict)

    def definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "read_png_metadata",
                "description": "Read metadata",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        self.calls.append((name, arguments))
        if name in self.results:
            return self.results[name]
        if name == "caption_from_png_info":
            caption = self.png_info.get(arguments["path"].rsplit("/", 1)[-1])
            if caption is None:
                return ToolResult('{"found": false}')
            return ToolResult(json.dumps({"found": True, "caption": caption}))
        return ToolResult('{"has_prompt": false}')
