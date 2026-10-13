"""Path confinement, image loading, flipping and caption-file helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from PIL import Image

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
FLIP_SUFFIX = "_flipped"

FlipDirection = Literal["horizontal", "vertical"]

_LEFT_RIGHT = re.compile(r"\b(left|right)\b", re.IGNORECASE)


class PathNotAllowedError(ValueError):
    """Raised when a path resolves outside the allowed root."""


def resolve_inside(root: Path, user_path: str) -> Path:
    """Resolve ``user_path`` (absolute, or relative to ``root``) and confine it to ``root``.

    Resolution follows ``..`` and symlinks before the check, so neither can escape.
    """
    if not user_path or "\x00" in user_path:
        raise PathNotAllowedError("path is empty or invalid")
    root = root.resolve()
    candidate = Path(user_path)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if resolved != root and not resolved.is_relative_to(root):
        raise PathNotAllowedError(f"path is outside the allowed root: {user_path}")
    return resolved


def display_path(root: Path, path: Path) -> str:
    """Return ``path`` relative to ``root`` with forward slashes, as tools report it."""
    return path.resolve().relative_to(root.resolve()).as_posix()


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def list_images(directory: Path) -> list[Path]:
    """Images directly inside ``directory``, sorted by name."""
    if not directory.is_dir():
        raise FileNotFoundError(f"not a directory: {directory.name}")
    return sorted((p for p in directory.iterdir() if is_image(p)), key=lambda p: p.name.lower())


def require_image(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path.name}")
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"unsupported image type: {path.suffix or '(none)'}")
    return path


def load_rgb(path: Path, max_side: int) -> Image.Image:
    """Open an image as RGB, downscaled so its longest side is at most ``max_side``."""
    with Image.open(require_image(path)) as img:
        img.load()
        rgb = img.convert("RGB")
    if max(rgb.size) > max_side:
        rgb.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return rgb


def flip_image(path: Path, output_dir: Path, direction: FlipDirection = "horizontal") -> Path:
    """Save a mirrored copy as ``<stem>_flipped<ext>`` in ``output_dir``.

    Embedded generation metadata is deliberately not copied: it describes the
    unflipped image, so left/right wording in it would be wrong for the copy.
    """
    if direction not in ("horizontal", "vertical"):
        raise ValueError("direction must be 'horizontal' or 'vertical'")
    method = Image.Transpose.FLIP_LEFT_RIGHT if direction == "horizontal" else Image.Transpose.FLIP_TOP_BOTTOM
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{path.stem}{FLIP_SUFFIX}{path.suffix.lower()}"
    with Image.open(require_image(path)) as img:
        img.load()
        flipped = img.transpose(method)
    if target.suffix in {".jpg", ".jpeg"}:
        flipped.convert("RGB").save(target, quality=95)
    else:
        flipped.save(target)
    return target


def caption_path_for(image_path: Path, output_dir: Path) -> Path:
    """``Image1.png`` -> ``<output_dir>/Image1.txt``."""
    return output_dir / f"{image_path.stem}.txt"


def write_caption(image_path: Path, text: str, output_dir: Path, overwrite: bool = True) -> Path:
    """Write the caption for ``image_path`` as UTF-8 text and return the file path."""
    caption = text.strip()
    if not caption:
        raise ValueError("caption text is empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = caption_path_for(image_path, output_dir)
    if target.exists() and not overwrite:
        raise FileExistsError(f"caption already exists: {target.name}")
    target.write_text(caption, encoding="utf-8", newline="\n")
    return target


def mirror_caption(text: str) -> str:
    """Swap the words "left" and "right", keeping each word's capitalisation."""

    def swap(match: re.Match[str]) -> str:
        word = match.group(0)
        other = "right" if word.lower() == "left" else "left"
        if word.isupper():
            return other.upper()
        if word[0].isupper():
            return other.capitalize()
        return other

    return _LEFT_RIGHT.sub(swap, text)
