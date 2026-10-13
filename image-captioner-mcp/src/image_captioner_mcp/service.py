"""Tool logic, independent of the MCP transport so it can be tested directly."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import imaging, png_caption, png_meta
from .captioner import CAPTION_STYLES, Captioner, CaptionStyle, create_blip_captioner
from .config import ServerConfig


class CaptionService:
    """Implements each tool against paths confined to ``config.allowed_root``."""

    def __init__(
        self, config: ServerConfig, captioner: Captioner, blip_captioner: Captioner | None = None
    ) -> None:
        self.config = config
        self.captioner = captioner
        # Creating a BLIP captioner loads nothing; its model loads on first use.
        self.blip_captioner = blip_captioner or create_blip_captioner(config)

    def resolve(self, path: str) -> Path:
        return imaging.resolve_inside(self.config.allowed_root, path)

    def _display(self, path: Path) -> str:
        return imaging.display_path(self.config.allowed_root, path)

    def _output_dir(self, output_dir: str | None, default: Path) -> Path:
        return self.resolve(output_dir) if output_dir else default

    def list_images(self, directory: str) -> dict[str, Any]:
        images = imaging.list_images(self.resolve(directory))
        return {"count": len(images), "images": [self._display(p) for p in images]}

    def read_png_metadata(self, path: str) -> dict[str, Any]:
        image = imaging.require_image(self.resolve(path))
        return {"path": self._display(image), **png_meta.read_metadata(image).to_dict()}

    def caption_from_png_info(
        self, path: str, output_dir: str | None = None, overwrite: bool = True
    ) -> dict[str, Any]:
        image = imaging.require_image(self.resolve(path))
        caption = png_caption.caption_from_png_info(image)
        if caption is None:
            return {"path": self._display(image), "found": False}
        target = imaging.write_caption(image, caption, self._output_dir(output_dir, image.parent), overwrite)
        return {
            "path": self._display(image),
            "found": True,
            "caption": caption,
            "caption_path": self._display(target),
        }

    def caption_image(self, path: str, style: CaptionStyle = "detailed") -> dict[str, Any]:
        return self._caption_with(self.captioner, path, style)

    def blip_caption(self, path: str, style: CaptionStyle = "short") -> dict[str, Any]:
        return self._caption_with(self.blip_captioner, path, style)

    def _caption_with(self, captioner: Captioner, path: str, style: CaptionStyle) -> dict[str, Any]:
        if style not in CAPTION_STYLES:
            raise ValueError(f"style must be one of {', '.join(CAPTION_STYLES)}")
        image_path = self.resolve(path)
        image = imaging.load_rgb(image_path, self.config.max_image_side)
        caption = captioner.caption(image, style)
        return {
            "path": self._display(image_path),
            "caption": caption,
            "style": style,
            "backend": captioner.name,
            "device": captioner.device,
        }

    def write_caption(
        self, image_path: str, text: str, output_dir: str | None = None, overwrite: bool = True
    ) -> dict[str, Any]:
        image = self.resolve(image_path)
        target = imaging.write_caption(image, text, self._output_dir(output_dir, image.parent), overwrite)
        return {"caption_path": self._display(target), "characters": len(text.strip())}

    def flip_image(
        self,
        path: str,
        output_dir: str | None = None,
        direction: imaging.FlipDirection = "horizontal",
    ) -> dict[str, Any]:
        image = self.resolve(path)
        target = imaging.flip_image(image, self._output_dir(output_dir, image.parent), direction)
        return {"flipped_path": self._display(target), "direction": direction}

    def caption_to_file(
        self, image: Path, output_dir: Path, style: CaptionStyle, overwrite: bool
    ) -> dict[str, Any]:
        """Caption one image with the model and write its ``.txt``; used by batch runs."""
        target = imaging.caption_path_for(image, output_dir)
        if target.exists() and not overwrite:
            return {"image": image.name, "status": "skipped", "caption_path": self._display(target)}
        caption = self.captioner.caption(imaging.load_rgb(image, self.config.max_image_side), style)
        imaging.write_caption(image, caption, output_dir, overwrite=True)
        return {"image": image.name, "status": "written", "caption_path": self._display(target)}

    def read_captions(self, folder: str) -> dict[str, str]:
        directory = self.resolve(folder)
        if not directory.is_dir():
            raise FileNotFoundError(f"not a directory: {folder}")
        return {
            p.name: p.read_text(encoding="utf-8")
            for p in sorted(directory.glob("*.txt"), key=lambda p: p.name.lower())
        }
