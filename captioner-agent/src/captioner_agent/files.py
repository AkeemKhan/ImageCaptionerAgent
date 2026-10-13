"""Session folders: uploads, caption outputs, thumbnails and ZIP export."""

from __future__ import annotations

import io
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any, Literal

from PIL import Image

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
FLIP_SUFFIX = "_flipped"
THUMBNAIL_SIDE = 256
VISION_MAX_SIDE = 1568
MAX_STEM_CHARS = 100

FileKind = Literal["uploads", "outputs"]

_UNSAFE_CHARS = re.compile(r"[^\w.\- ()]+")
_WINDOWS_RESERVED = re.compile(r"^(con|prn|aux|nul|com\d|lpt\d)$", re.IGNORECASE)


class InvalidFileError(ValueError):
    """An uploaded file or a requested file name is not acceptable."""


class SessionFiles:
    """The ``uploads`` and ``outputs`` folders of one session."""

    def __init__(self, sessions_dir: Path, session_id: str) -> None:
        self.session_id = session_id
        self.root = sessions_dir / session_id
        self.uploads = self.root / "uploads"
        self.outputs = self.root / "outputs"

    def ensure(self) -> None:
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.outputs.mkdir(parents=True, exist_ok=True)

    def remove(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def tool_path(self, kind: FileKind, name: str = "") -> str:
        """Path as the MCP server sees it: relative to the sessions folder."""
        return "/".join(part for part in (self.session_id, kind, name) if part)

    def new_upload_path(self, original_name: str) -> Path:
        """Pick a safe, unused name in ``uploads`` for an incoming file.

        Names are unique by stem, not just by full name, because ``a.png`` and
        ``a.jpg`` would otherwise both map to the caption file ``a.txt``.
        """
        stem, suffix = safe_name_parts(original_name)
        taken = {p.stem.lower() for p in self.uploads.iterdir()} if self.uploads.is_dir() else set()
        candidate, counter = stem, 2
        while candidate.lower() in taken:
            candidate = f"{stem}_{counter}"
            counter += 1
        return self.uploads / f"{candidate}{suffix}"

    def resolve(self, kind: FileKind, name: str) -> Path:
        """Map a client-supplied file name to an existing file in one of the two folders."""
        folder = self.uploads if kind == "uploads" else self.outputs
        if not name or name != Path(name).name or name in {".", ".."}:
            raise InvalidFileError("invalid file name")
        path = folder / name
        if not path.is_file():
            raise FileNotFoundError(name)
        return path

    def caption_file(self, image_name: str) -> Path:
        return self.outputs / f"{Path(image_name).stem}.txt"

    def write_caption(self, image_name: str, text: str) -> None:
        """Save a caption edited in the UI for an image that exists in this session."""
        if not any(item["name"] == image_name for item in self.list_items()):
            raise FileNotFoundError(image_name)
        self.outputs.mkdir(parents=True, exist_ok=True)
        self.caption_file(image_name).write_text(text.strip(), encoding="utf-8", newline="\n")

    def list_items(self) -> list[dict[str, Any]]:
        """Every image in the session with its caption, originals before flipped copies."""
        items: list[dict[str, Any]] = []
        for kind, folder in (("uploads", self.uploads), ("outputs", self.outputs)):
            if not folder.is_dir():
                continue
            for path in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                caption_file = self.caption_file(path.name)
                items.append(
                    {
                        "name": path.name,
                        "kind": kind,
                        "flipped": kind == "outputs",
                        "caption_file": caption_file.name,
                        "caption": caption_file.read_text(encoding="utf-8")
                        if caption_file.is_file()
                        else None,
                    }
                )
        return items

    def build_zip(self, include_originals: bool = True) -> bytes:
        """ZIP of the outputs folder, optionally with the original images alongside."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            if self.outputs.is_dir():
                for path in sorted(self.outputs.iterdir()):
                    if path.is_file():
                        archive.write(path, path.name)
            if include_originals and self.uploads.is_dir():
                for path in sorted(self.uploads.iterdir()):
                    if path.is_file():
                        archive.write(path, path.name)
        return buffer.getvalue()


def safe_name_parts(original_name: str) -> tuple[str, str]:
    """Reduce a client file name (possibly a relative path) to a safe ``(stem, suffix)``."""
    name = original_name.replace("\\", "/").rsplit("/", 1)[-1]
    suffix = Path(name).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise InvalidFileError(f"unsupported file type: {suffix or '(none)'}")
    stem = _UNSAFE_CHARS.sub("_", Path(name).stem).strip(" .")[:MAX_STEM_CHARS]
    if not stem or _WINDOWS_RESERVED.match(stem):
        stem = f"image_{stem}".rstrip("_")
    return stem, suffix


def verify_image(path: Path) -> None:
    """Raise ``InvalidFileError`` unless ``path`` decodes as an image."""
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as exc:
        raise InvalidFileError("the file is not a readable image") from exc


def encode_for_vision(path: Path) -> tuple[bytes, str]:
    """The image as JPEG bytes sized for a vision model, with its media type.

    Downscaling keeps the request small and cheap; captioning does not need
    more than ``VISION_MAX_SIDE`` pixels on the long side.
    """
    try:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
    except Exception as exc:
        raise InvalidFileError("the file is not a readable image") from exc
    rgb.thumbnail((VISION_MAX_SIDE, VISION_MAX_SIDE))
    buffer = io.BytesIO()
    rgb.save(buffer, "JPEG", quality=90)
    return buffer.getvalue(), "image/jpeg"


def make_thumbnail(path: Path) -> bytes:
    with Image.open(path) as img:
        img.thumbnail((THUMBNAIL_SIDE, THUMBNAIL_SIDE))
        buffer = io.BytesIO()
        img.convert("RGB").save(buffer, "JPEG", quality=80)
    return buffer.getvalue()
