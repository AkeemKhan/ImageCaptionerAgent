from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from image_captioner_mcp import imaging

from .conftest import make_png


@pytest.mark.parametrize(
    "bad",
    ["../outside.png", "sub/../../outside.png", "", "a\x00b.png"],
)
def test_resolve_inside_rejects_escapes(root: Path, bad: str) -> None:
    with pytest.raises(imaging.PathNotAllowedError):
        imaging.resolve_inside(root, bad)


def test_resolve_inside_rejects_absolute_path_outside_root(root: Path, tmp_path: Path) -> None:
    with pytest.raises(imaging.PathNotAllowedError):
        imaging.resolve_inside(root, str(tmp_path / "elsewhere.png"))


def test_resolve_inside_accepts_relative_and_absolute(root: Path) -> None:
    expected = (root / "sub" / "a.png").resolve()
    assert imaging.resolve_inside(root, "sub/a.png") == expected
    assert imaging.resolve_inside(root, str(expected)) == expected
    assert imaging.resolve_inside(root, ".") == root.resolve()


def test_flip_horizontal_mirrors_pixels_and_names_file(root: Path) -> None:
    source = make_png(root / "Image1.png", text={"parameters": "a cat"})
    flipped = imaging.flip_image(source, root / "out")

    assert flipped == root / "out" / "Image1_flipped.png"
    with Image.open(flipped) as img:
        assert img.getpixel((0, 0)) == (0, 0, 255)
        assert img.getpixel((3, 0)) == (255, 0, 0)
        assert "parameters" not in img.info


def test_flip_rejects_unknown_direction(root: Path) -> None:
    source = make_png(root / "a.png")
    with pytest.raises(ValueError):
        imaging.flip_image(source, root, "diagonal")  # type: ignore[arg-type]


def test_write_caption_uses_image_stem(root: Path) -> None:
    target = imaging.write_caption(root / "Image1.png", "  a red square \n", root / "out")

    assert target == root / "out" / "Image1.txt"
    assert target.read_text(encoding="utf-8") == "a red square"


def test_write_caption_respects_overwrite_flag(root: Path) -> None:
    imaging.write_caption(root / "a.png", "first", root)
    with pytest.raises(FileExistsError):
        imaging.write_caption(root / "a.png", "second", root, overwrite=False)
    imaging.write_caption(root / "a.png", "second", root)
    assert (root / "a.txt").read_text(encoding="utf-8") == "second"


def test_write_caption_rejects_empty_text(root: Path) -> None:
    with pytest.raises(ValueError):
        imaging.write_caption(root / "a.png", "   ", root)


def test_mirror_caption_swaps_words_and_keeps_case() -> None:
    text = "Left hand raised, looking to the right; RIGHT side lit, leftover copyright"
    assert imaging.mirror_caption(text) == (
        "Right hand raised, looking to the left; LEFT side lit, leftover copyright"
    )


def test_list_images_filters_and_sorts(root: Path) -> None:
    make_png(root / "b.png")
    make_png(root / "A.png")
    (root / "notes.txt").write_text("x", encoding="utf-8")
    assert [p.name for p in imaging.list_images(root)] == ["A.png", "b.png"]


def test_load_rgb_downscales(root: Path) -> None:
    source = make_png(root / "big.png", size=(400, 200))
    assert imaging.load_rgb(source, max_side=100).size == (100, 50)
