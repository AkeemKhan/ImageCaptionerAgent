"""The MANUAL_CAPTION tool: captions the user wrote, saved as written."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mcp import Client

from image_captioner_mcp.server import create_server
from image_captioner_mcp.service import MAX_MANUAL_CAPTIONS, CaptionService

from .conftest import make_png


def test_saves_each_caption_next_to_its_image_by_default(service: CaptionService, root: Path) -> None:
    make_png(root / "Image1.png")
    make_png(root / "Image2.png")

    result = service.manual_caption(
        [{"path": "Image1.png", "caption": "a red car"}, {"path": "Image2.png", "caption": "a blue boat"}]
    )

    assert (result["total"], result["written"], result["failed"]) == (2, 2, 0)
    assert (root / "Image1.txt").read_text(encoding="utf-8") == "a red car"
    assert (root / "Image2.txt").read_text(encoding="utf-8") == "a blue boat"
    assert result["results"][0] == {
        "path": "Image1.png",
        "status": "written",
        "caption_path": "Image1.txt",
        "caption": "a red car",
    }


def test_keeps_the_users_wording_including_prompt_syntax(service: CaptionService, root: Path) -> None:
    make_png(root / "a.png")
    text = "sks dog, (running:1.2) <lora:style:0.8>\n  on   grass: sunny"

    service.manual_caption([{"path": "a.png", "caption": text}], output_dir="out")

    # Only whitespace is normalised; (), <>, : and weights survive untouched.
    saved = (root / "out" / "a.txt").read_text(encoding="utf-8")
    assert saved == "sks dog, (running:1.2) <lora:style:0.8> on grass: sunny"


def test_one_bad_entry_does_not_stop_the_rest(service: CaptionService, root: Path) -> None:
    make_png(root / "ok.png")
    (root / "notes.txt").write_text("x", encoding="utf-8")

    result = service.manual_caption(
        [
            {"path": "missing.png", "caption": "x"},
            {"path": "../outside.png", "caption": "x"},
            {"path": "notes.txt", "caption": "x"},
            {"path": "ok.png", "caption": "   "},
            {"path": "ok.png", "caption": "fine"},
        ]
    )

    assert (result["written"], result["failed"]) == (1, 4)
    errors = [r["error"] for r in result["results"] if r["status"] == "error"]
    assert "image not found" in errors[0]
    assert "outside the allowed root" in errors[1]
    assert "unsupported image type" in errors[2]
    assert "caption text is empty" in errors[3]
    assert (root / "ok.txt").read_text(encoding="utf-8") == "fine"


def test_overwrite_false_keeps_an_existing_caption(service: CaptionService, root: Path) -> None:
    make_png(root / "a.png")
    (root / "a.txt").write_text("old", encoding="utf-8")

    kept = service.manual_caption([{"path": "a.png", "caption": "new"}], overwrite=False)
    replaced = service.manual_caption([{"path": "a.png", "caption": "new"}])

    assert kept["results"][0]["status"] == "error" and "already exists" in kept["results"][0]["error"]
    assert replaced["written"] == 1 and (root / "a.txt").read_text(encoding="utf-8") == "new"


def test_rejects_an_empty_or_oversized_batch_and_an_output_dir_outside_the_root(
    service: CaptionService, root: Path
) -> None:
    make_png(root / "a.png")

    with pytest.raises(ValueError, match="between 1 and"):
        service.manual_caption([])
    with pytest.raises(ValueError, match="between 1 and"):
        service.manual_caption([{"path": "a.png", "caption": "x"}] * (MAX_MANUAL_CAPTIONS + 1))
    with pytest.raises(ValueError, match="outside the allowed root"):
        service.manual_caption([{"path": "a.png", "caption": "x"}], output_dir="../elsewhere")


@pytest.mark.anyio
async def test_manual_caption_tool_over_mcp(service: CaptionService, root: Path) -> None:
    make_png(root / "a.png")

    async with Client(create_server(service)) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        ok = await client.call_tool(
            "MANUAL_CAPTION", {"captions": [{"path": "a.png", "caption": "a cat"}], "output_dir": "out"}
        )
        bad_dir = await client.call_tool(
            "MANUAL_CAPTION", {"captions": [{"path": "a.png", "caption": "a cat"}], "output_dir": "../x"}
        )

    schema = tools["MANUAL_CAPTION"].input_schema
    assert schema["required"] == ["captions"]
    assert not ok.is_error and json.loads(ok.content[0].text)["written"] == 1  # type: ignore[union-attr]
    assert (root / "out" / "a.txt").read_text(encoding="utf-8") == "a cat"
    assert bad_dir.is_error and "outside the allowed root" in bad_dir.content[0].text  # type: ignore[union-attr]
