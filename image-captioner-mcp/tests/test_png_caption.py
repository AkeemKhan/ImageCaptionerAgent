from __future__ import annotations

from pathlib import Path

import pytest

from image_captioner_mcp import png_caption
from image_captioner_mcp.service import CaptionService

from .conftest import make_png


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a cat on a sofa", "a cat on a sofa"),
        ("(masterpiece:1.2), ((1girl)), red dress", "masterpiece, 1girl, red dress"),
        ("a dog <lora:style_v2:0.8>, park", "a dog, park"),
        ("[soft light:0.5], {vivid}, a \\(real\\) lake", "soft light, vivid, a real lake"),
        ("style:anime, a boat", "style anime, a boat"),
        ("a cat\nNegative prompt: blurry, (bad:1.4)\nSteps: 20", "a cat"),
        ("a cat,\n\nsitting, \n", "a cat, sitting"),
        ("<lora:only:1>", ""),
        ("negative prompt: only this", ""),
    ],
)
def test_clean_prompt(raw: str, expected: str) -> None:
    assert png_caption.clean_prompt(raw) == expected


def test_caption_from_png_info_reads_a1111(root: Path) -> None:
    text = "(a red fox:1.3), <lora:fur:0.6> snow\nNegative prompt: lowres\nSteps: 20, Sampler: Euler"
    image = make_png(root / "fox.png", {"parameters": text})

    assert png_caption.caption_from_png_info(image) == "a red fox, snow"


def test_caption_from_png_info_without_metadata(root: Path) -> None:
    assert png_caption.caption_from_png_info(make_png(root / "plain.png")) is None
    # Metadata that cleans to nothing counts as none.
    assert (
        png_caption.caption_from_png_info(make_png(root / "tags.png", {"parameters": "<lora:x:1>"})) is None
    )


def test_service_writes_caption_only_when_found(service: CaptionService, root: Path) -> None:
    make_png(root / "in" / "a.png", {"parameters": "(a cat:1.1)\nNegative prompt: dog"})
    make_png(root / "in" / "b.png")

    found = service.caption_from_png_info("in/a.png", "out")
    missing = service.caption_from_png_info("in/b.png", "out")

    assert found == {"path": "in/a.png", "found": True, "caption": "a cat", "caption_path": "out/a.txt"}
    assert missing == {"path": "in/b.png", "found": False}
    assert (root / "out" / "a.txt").read_text(encoding="utf-8") == "a cat"
    assert not (root / "out" / "b.txt").exists()


def test_cli_captions_a_folder(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_png(root / "a.png", {"parameters": "a (tall:1.2) tree\nNegative prompt: x"})
    make_png(root / "b.png")

    exit_code = png_caption.main([str(root), "--output-dir", str(root / "captions")])

    assert exit_code == 0
    assert (root / "captions" / "a.txt").read_text(encoding="utf-8") == "a tall tree"
    assert not (root / "captions" / "b.txt").exists()
    assert "1 caption(s) written, 1 image(s) without PNG info" in capsys.readouterr().out
