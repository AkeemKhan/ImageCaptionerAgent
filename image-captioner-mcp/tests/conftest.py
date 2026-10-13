from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from image_captioner_mcp.captioner import StubCaptioner
from image_captioner_mcp.config import ServerConfig
from image_captioner_mcp.service import CaptionService


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    allowed = tmp_path / "root"
    allowed.mkdir()
    return allowed


@pytest.fixture
def service(root: Path) -> CaptionService:
    config = ServerConfig(allowed_root=root, backend="stub", preload=False)
    return CaptionService(config, StubCaptioner())


def make_png(path: Path, text: dict[str, str] | None = None, size: tuple[int, int] = (4, 2)) -> Path:
    """Write a small PNG whose left half is red and right half is blue."""
    image = Image.new("RGB", size, "blue")
    image.paste("red", (0, 0, size[0] // 2, size[1]))
    info = PngInfo()
    for key, value in (text or {}).items():
        info.add_text(key, value)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, pnginfo=info)
    return path
