"""End-to-end tests through the MCP protocol, using an in-process client."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from mcp import Client

from image_captioner_mcp.server import create_server
from image_captioner_mcp.service import CaptionService

from .conftest import make_png

pytestmark = pytest.mark.anyio


async def call(client: Client, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result = await client.call_tool(name, arguments)
    assert not result.is_error, result.content
    payload: dict[str, Any] = json.loads(result.content[0].text)  # type: ignore[union-attr]
    return payload


async def test_lists_all_tools(service: CaptionService) -> None:
    async with Client(create_server(service)) as client:
        names = {tool.name for tool in (await client.list_tools()).tools}

    assert names == {
        "list_images",
        "read_png_metadata",
        "caption_from_png_info",
        "caption_image",
        "write_caption",
        "flip_image",
        "mirror_caption",
        "caption_folder",
    }


async def test_metadata_then_caption_then_write(service: CaptionService, root: Path) -> None:
    make_png(root / "s1" / "uploads" / "Image1.png", {"parameters": "a cat\nSteps: 20"})
    make_png(root / "s1" / "uploads" / "Image2.png")

    async with Client(create_server(service)) as client:
        listed = await call(client, "list_images", {"directory": "s1/uploads"})
        with_prompt = await call(client, "read_png_metadata", {"path": "s1/uploads/Image1.png"})
        without = await call(client, "read_png_metadata", {"path": "s1/uploads/Image2.png"})
        caption = await call(client, "caption_image", {"path": "s1/uploads/Image2.png", "style": "short"})
        written = await call(
            client,
            "write_caption",
            {"image_path": "s1/uploads/Image2.png", "text": caption["caption"], "output_dir": "s1/outputs"},
        )

    assert listed["images"] == ["s1/uploads/Image1.png", "s1/uploads/Image2.png"]
    assert with_prompt["has_prompt"] and with_prompt["prompt"] == "a cat"
    assert not without["has_prompt"]
    assert written["caption_path"] == "s1/outputs/Image2.txt"
    assert (root / "s1" / "outputs" / "Image2.txt").read_text(encoding="utf-8") == caption["caption"]


async def test_flip_and_mirror(service: CaptionService, root: Path) -> None:
    make_png(root / "a.png")

    async with Client(create_server(service)) as client:
        flipped = await call(client, "flip_image", {"path": "a.png", "output_dir": "out"})
        mirrored = await call(client, "mirror_caption", {"text": "arm on the left"})

    assert flipped["flipped_path"] == "out/a_flipped.png"
    assert (root / "out" / "a_flipped.png").is_file()
    assert mirrored["text"] == "arm on the right"


async def test_path_escape_is_a_tool_error_with_a_message(service: CaptionService) -> None:
    async with Client(create_server(service)) as client:
        result = await client.call_tool("read_png_metadata", {"path": "../secret.png"})

    assert result.is_error
    assert "outside the allowed root" in result.content[0].text  # type: ignore[union-attr]


async def test_non_image_file_is_a_tool_error(service: CaptionService, root: Path) -> None:
    (root / "fake.png").write_text("not an image", encoding="utf-8")

    async with Client(create_server(service)) as client:
        result = await client.call_tool("caption_image", {"path": "fake.png"})

    assert result.is_error
    assert "not a readable image" in result.content[0].text  # type: ignore[union-attr]


async def test_caption_folder_reports_progress_and_skips_existing(
    service: CaptionService, root: Path
) -> None:
    make_png(root / "in" / "a.png")
    make_png(root / "in" / "b.png")
    (root / "out").mkdir()
    (root / "out" / "a.txt").write_text("kept", encoding="utf-8")
    progress: list[tuple[float, float | None]] = []

    async def on_progress(done: float, total: float | None, message: str | None) -> None:
        progress.append((done, total))

    async with Client(create_server(service)) as client:
        result = await client.call_tool(
            "caption_folder", {"directory": "in", "output_dir": "out"}, progress_callback=on_progress
        )

    summary = json.loads(result.content[0].text)  # type: ignore[union-attr]
    assert (summary["written"], summary["skipped"], summary["error"]) == (1, 1, 0)
    assert (root / "out" / "a.txt").read_text(encoding="utf-8") == "kept"
    assert (root / "out" / "b.txt").is_file()
    assert progress == [(1, 2), (2, 2)]


async def test_captions_resource_and_prompt(service: CaptionService, root: Path) -> None:
    (root / "out").mkdir()
    (root / "out" / "a.txt").write_text("a caption", encoding="utf-8")

    async with Client(create_server(service)) as client:
        resource = await client.read_resource("captions://out")
        prompt = await client.get_prompt("simplify_caption", {"raw_prompt": "(1girl:1.2)"})

    assert json.loads(resource.contents[0].text) == {"a.txt": "a caption"}  # type: ignore[union-attr]
    assert "(1girl:1.2)" in prompt.messages[0].content.text  # type: ignore[union-attr]
