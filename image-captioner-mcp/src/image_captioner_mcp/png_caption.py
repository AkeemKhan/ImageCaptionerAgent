"""Caption an image from its embedded PNG info, with no model involved.

The caption is the generation prompt up to ``Negative prompt:``, with prompt
syntax stripped. Run as a script to caption files or folders::

    python -m image_captioner_mcp.png_caption Image1.png
    python -m image_captioner_mcp.png_caption D:\\dataset --output-dir D:\\dataset\\captions
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from . import imaging, png_meta

# Characters that only carry prompt syntax. A colon becomes a space so that
# "style:anime" does not fuse into one word.
_STRIP_CHARS = str.maketrans(
    {"(": "", ")": "", "<": "", ">": "", "[": "", "]": "", "{": "", "}": "", "\\": "", ":": " "}
)

_NEGATIVE_PROMPT = re.compile(r"Negative prompt:", re.IGNORECASE)
# <lora:name:0.8>, <hypernet:name:1>: these name files, not things in the image.
_ANGLE_TAG = re.compile(r"<[^<>]*>")
# The ":1.2" in "(word:1.2)" or "[word:0.5]".
_WEIGHT = re.compile(r":\s*-?\d*\.?\d+\s*(?=[)\]])")
_SPACES = re.compile(r"[ \t]+")
_EMPTY_ITEMS = re.compile(r"\s*,(\s*,)+\s*")
_SPACE_BEFORE_COMMA = re.compile(r"\s+,")


def clean_prompt(text: str) -> str:
    """Turn a raw generation prompt into a plain caption.

    Keeps everything before ``Negative prompt:``, drops ``<...>`` tags and
    attention weights, and removes ``( ) [ ] { } < > :`` and backslashes.
    """
    text = _NEGATIVE_PROMPT.split(text, maxsplit=1)[0]
    text = _ANGLE_TAG.sub("", text)
    text = _WEIGHT.sub("", text)
    text = text.translate(_STRIP_CHARS)
    # Prompts are often split over lines; a caption is a single line.
    text = ", ".join(line.strip(" ,") for line in text.splitlines() if line.strip(" ,"))
    text = _SPACES.sub(" ", text)
    text = _SPACE_BEFORE_COMMA.sub(",", text)
    text = _EMPTY_ITEMS.sub(", ", text)
    return text.strip(" ,")


def caption_from_png_info(path: Path) -> str | None:
    """The cleaned caption for ``path``, or ``None`` if it has no usable PNG info."""
    prompt = png_meta.read_metadata(path).prompt
    if not prompt:
        return None
    return clean_prompt(prompt) or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write <image>.txt captions from embedded PNG info (prompt before 'Negative prompt:')."
    )
    parser.add_argument("paths", nargs="+", type=Path, help="image files or folders")
    parser.add_argument(
        "--output-dir", type=Path, help="where to write .txt files (default: next to each image)"
    )
    parser.add_argument("--no-overwrite", action="store_true", help="skip images that already have a .txt")
    args = parser.parse_args(argv)

    images: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            images.extend(imaging.list_images(path))
        elif imaging.is_image(path):
            images.append(path)
        else:
            print(f"skipped (not an image): {path}", file=sys.stderr)

    written = missing = 0
    for image in images:
        output_dir: Path = args.output_dir or image.parent
        if args.no_overwrite and imaging.caption_path_for(image, output_dir).exists():
            continue
        caption = caption_from_png_info(image)
        if caption is None:
            missing += 1
            print(f"no PNG info: {image}")
            continue
        target = imaging.write_caption(image, caption, output_dir)
        written += 1
        print(f"{image.name} -> {target}")
    print(f"{written} caption(s) written, {missing} image(s) without PNG info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
