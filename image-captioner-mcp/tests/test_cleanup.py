"""CLEANUP_AND_ENHANCE: text removal and upscaling, tested without the models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from mcp import Client
from PIL import Image, ImageDraw

from image_captioner_mcp import cleanup
from image_captioner_mcp.captioner import StubCaptioner, TextBox
from image_captioner_mcp.cleanup import StubEnhancer, clean_image, find_caption_bar, restore_bar
from image_captioner_mcp.config import ServerConfig
from image_captioner_mcp.server import create_server
from image_captioner_mcp.service import CaptionService

from .conftest import make_png


class FakeFinder:
    """Reports fixed text boxes and records the size of every image it is shown."""

    def __init__(self, found: list[TextBox] | None = None) -> None:
        self.found = found or []
        self.seen: list[tuple[int, int]] = []

    def find_text(self, image: Image.Image) -> list[TextBox]:
        self.seen.append(image.size)
        return list(self.found) if len(self.seen) == 1 else []


def gradient(width: int = 320, height: int = 240) -> Image.Image:
    yy, xx = np.mgrid[0:height, 0:width]
    pixels = np.stack([xx / width * 180 + 40, yy / height * 160 + 50, np.full_like(xx, 120.0)], -1)
    return Image.fromarray(pixels.astype(np.uint8), "RGB")


def with_caption_bar(image: Image.Image, top: int, bottom: int) -> Image.Image:
    """Darken rows [top, bottom) to half brightness and write white text on them, Snapchat style."""
    pixels = np.asarray(image, np.float32).copy()
    pixels[top:bottom] *= 0.5
    barred = Image.fromarray(pixels.round().astype(np.uint8), "RGB")
    ImageDraw.Draw(barred).rectangle((130, top + 6, 190, bottom - 7), fill="white")
    return barred


@pytest.fixture
def cleaning_service(root: Path) -> CaptionService:
    config = ServerConfig(allowed_root=root, backend="stub", preload=False, max_output_side=4096)
    finder = FakeFinder([TextBox("@watermark", (200.0, 100.0, 300.0, 116.0))])
    return CaptionService(config, StubCaptioner(), enhancer=StubEnhancer(), text_finder=finder)


def lum(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), np.float32)


# --- caption bars -----------------------------------------------------------


def test_finds_a_dark_caption_bar_behind_text() -> None:
    image = with_caption_bar(gradient(), 100, 130)

    assert find_caption_bar(lum(image), (130, 106, 190, 123)) == (100, 130)


def test_no_bar_behind_text_on_a_plain_picture() -> None:
    image = gradient()
    ImageDraw.Draw(image).rectangle((130, 106, 190, 123), fill="white")

    assert find_caption_bar(lum(image), (130, 106, 190, 123)) is None


def test_a_dark_object_that_does_not_span_the_width_is_not_a_bar() -> None:
    image = gradient()
    ImageDraw.Draw(image).rectangle((100, 100, 220, 130), fill=(20, 20, 20))

    assert find_caption_bar(lum(image), (130, 106, 190, 123)) is None


def test_restoring_a_bar_brings_back_the_picture_under_it() -> None:
    original = gradient()
    barred = np.asarray(with_caption_bar(original, 100, 130), np.float32).copy()
    text_columns = np.zeros(original.width, bool)
    text_columns[120:200] = True

    restore_bar(barred, 100, 130, text_columns)

    truth = np.asarray(original, np.float32)
    error = np.abs(barred[103:127, :120] - truth[103:127, :120])
    assert error.mean() < 2


# --- clean_image ------------------------------------------------------------


def test_removes_found_text_and_upscales() -> None:
    image = gradient()
    ImageDraw.Draw(image).text((200, 200), "@watermark", fill="white")
    finder = FakeFinder([TextBox("@watermark", (200.0, 200.0, 260.0, 212.0))])

    result = clean_image(image, finder, StubEnhancer(), scale=2)

    assert [t.text for t in result.removed] == ["@watermark"]
    assert result.image.size == (640, 480)
    assert result.upscaled_by == 2.0
    # The text area changed; a far corner did not (beyond the upscale).
    before = np.asarray(image.resize((640, 480), Image.Resampling.LANCZOS), np.float32)
    after = np.asarray(result.image, np.float32)
    assert np.abs(after[400:424, 400:520] - before[400:424, 400:520]).mean() > 5
    assert np.abs(after[:40, :40] - before[:40, :40]).mean() < 2


def test_keep_text_spares_matching_text() -> None:
    image = gradient()
    finder = FakeFinder(
        [TextBox("STOP", (10.0, 10.0, 40.0, 20.0)), TextBox("@someone", (200.0, 200.0, 260.0, 212.0))]
    )

    result = clean_image(image, finder, StubEnhancer(), keep_text=["stop"], scale=1)

    assert [t.text for t in result.kept] == ["STOP"]
    assert [t.text for t in result.removed] == ["@someone"]
    assert np.array_equal(np.asarray(result.image)[5:25, 5:45], np.asarray(image)[5:25, 5:45])


def test_remove_text_false_only_upscales() -> None:
    image = gradient()
    finder = FakeFinder([TextBox("@someone", (200.0, 200.0, 260.0, 212.0))])

    result = clean_image(image, finder, StubEnhancer(), remove_text=False, scale=4)

    assert finder.seen == []
    assert result.removed == []
    assert result.image.size == (1280, 960)


def test_caption_bar_is_restored_and_counted() -> None:
    image = with_caption_bar(gradient(), 100, 130)
    finder = FakeFinder([TextBox("omg", (130.0, 106.0, 190.0, 123.0))])

    result = clean_image(image, finder, StubEnhancer(), scale=1)

    assert result.caption_bars == 1
    # Away from the text and the bar's edges, the bar is back to full brightness.
    truth = np.asarray(gradient(), np.float32)[110:120, :80]
    assert np.abs(np.asarray(result.image, np.float32)[110:120, :80] - truth).mean() < 3


def test_upscaling_is_capped_by_max_side() -> None:
    result = clean_image(gradient(), FakeFinder(), StubEnhancer(), scale=4, max_side=800)

    assert result.image.size == (800, 600)
    assert result.upscaled_by == 2.5


def test_an_image_already_at_max_side_is_not_upscaled() -> None:
    image = gradient()

    result = clean_image(image, FakeFinder(), StubEnhancer(), scale=2, max_side=320)

    assert result.image.size == image.size
    assert result.upscaled_by == 1.0


def test_regions_are_removed_without_ocr() -> None:
    image = gradient()
    ImageDraw.Draw(image).rectangle((295, 5, 304, 14), fill="black")

    result = clean_image(image, FakeFinder(), StubEnhancer(), regions=[(0.85, 0.0, 1.0, 0.15)], scale=1)

    after, before = np.asarray(result.image, np.float32), np.asarray(image, np.float32)
    assert result.regions == 1
    assert after[5:15, 295:305].mean() > before[5:15, 295:305].mean() + 20
    assert np.array_equal(after[40:, :], before[40:, :])


@pytest.mark.parametrize("region", [(0.5, 0.0, 0.4, 1.0), (0.0, 0.0, 1.5, 1.0)])
def test_rejects_bad_regions(region: tuple[float, float, float, float]) -> None:
    with pytest.raises(ValueError, match="region"):
        clean_image(gradient(), FakeFinder(), StubEnhancer(), regions=[region], scale=1)


def test_rejects_unsupported_scale() -> None:
    with pytest.raises(ValueError, match="scale"):
        clean_image(gradient(), FakeFinder(), StubEnhancer(), scale=3)


def test_large_images_are_also_read_in_tiles() -> None:
    image = Image.new("RGB", (3000, 2000))
    finder = FakeFinder([TextBox("@big", (2800.0, 1900.0, 2950.0, 1950.0))])

    found = cleanup.find_all_text(finder, image)

    assert len(finder.seen) > 1
    assert all(max(size) < 3000 for size in finder.seen[1:])
    assert [t.text for t in found] == ["@big"]


def test_duplicate_boxes_from_tiles_are_reported_once() -> None:
    boxes = [TextBox("@me", (10.0, 10.0, 50.0, 20.0)), TextBox("@me", (11.0, 10.0, 50.0, 20.0))]

    assert len(cleanup._dedupe(boxes)) == 1


def test_slivers_are_not_text() -> None:
    boxes = [TextBox("@me", (10.0, 10.0, 50.0, 20.0)), TextBox("٠١", (80.0, 10.0, 81.0, 18.0))]

    assert [t.text for t in cleanup._dedupe(boxes)] == ["@me"]


# --- service and MCP tool ---------------------------------------------------


def test_service_saves_a_clean_copy_and_leaves_the_original(
    cleaning_service: CaptionService, root: Path
) -> None:
    source = make_png(root / "in" / "Image1.png", {"parameters": "a cat"}, size=(320, 240))
    original_bytes = source.read_bytes()

    result = cleaning_service.cleanup_and_enhance("in/Image1.png", "out", scale=2)

    assert result["cleaned_path"] == "out/Image1_clean.png"
    assert result["removed_text"] == [{"text": "@watermark", "box": [200, 100, 300, 116]}]
    assert (result["original_size"], result["size"], result["upscaled_by"]) == ([320, 240], [640, 480], 2.0)
    assert source.read_bytes() == original_bytes
    with Image.open(root / "out" / "Image1_clean.png") as cleaned:
        assert cleaned.size == (640, 480)
        assert cleaned.text["parameters"] == "a cat"  # type: ignore[attr-defined]


def test_service_keeps_jpeg_format_and_alpha(cleaning_service: CaptionService, root: Path) -> None:
    gradient().save(root / "photo.jpg")
    Image.new("RGBA", (40, 30), (255, 0, 0, 128)).save(root / "sticker.png")

    jpg = cleaning_service.cleanup_and_enhance("photo.jpg", scale=1)
    cleaning_service.cleanup_and_enhance("sticker.png", scale=2)

    assert jpg["cleaned_path"] == "photo_clean.jpg"
    with Image.open(root / "sticker_clean.png") as sticker:
        assert sticker.mode == "RGBA"
        assert sticker.size == (80, 60)
        assert sticker.getpixel((40, 30))[3] == 128  # type: ignore[index]


def test_service_respects_overwrite_false(cleaning_service: CaptionService, root: Path) -> None:
    make_png(root / "a.png", size=(32, 32))
    cleaning_service.cleanup_and_enhance("a.png", scale=1)

    with pytest.raises(FileExistsError):
        cleaning_service.cleanup_and_enhance("a.png", scale=1, overwrite=False)


def test_stub_backend_finds_no_text(service: CaptionService, root: Path) -> None:
    make_png(root / "a.png", size=(32, 32))

    result = service.cleanup_and_enhance("a.png", scale=2)

    assert result["removed_text"] == []
    assert result["size"] == [64, 64]


@pytest.mark.anyio
async def test_tool_through_mcp(cleaning_service: CaptionService, root: Path) -> None:
    make_png(root / "s" / "Image1.png", size=(320, 240))
    async with Client(create_server(cleaning_service)) as client:
        result = await client.call_tool(
            "CLEANUP_AND_ENHANCE",
            {
                "path": "s/Image1.png",
                "output_dir": "s/out",
                "keep_text": ["WATER"],
                "regions": [{"left": 0.0, "top": 0.0, "right": 0.1, "bottom": 0.1}],
                "scale": 4,
            },
        )
        bad = await client.call_tool("CLEANUP_AND_ENHANCE", {"path": "../x.png"})

    assert not result.is_error, result.content
    payload = json.loads(result.content[0].text)  # type: ignore[union-attr]
    assert payload["cleaned_path"] == "s/out/Image1_clean.png"
    assert [t["text"] for t in payload["kept_text"]] == ["@watermark"]
    assert payload["regions_removed"] == 1
    assert payload["size"] == [1280, 960]
    assert bad.is_error
    assert "outside the allowed root" in bad.content[0].text  # type: ignore[union-attr]
