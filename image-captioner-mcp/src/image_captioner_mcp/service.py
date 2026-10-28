"""Tool logic, independent of the MCP transport so it can be tested directly."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from . import cleanup, gemini_cleanup, imaging, png_caption, png_meta
from .captioner import CAPTION_STYLES, Captioner, CaptionStyle, TextBox, create_blip_captioner
from .cleanup import Enhancer, NoTextFinder, TextFinder
from .config import ServerConfig
from .gemini import GeminiEditor, HttpGeminiEditor

MAX_MANUAL_CAPTIONS = 500


def _create_gemini(config: ServerConfig) -> GeminiEditor | None:
    """The Gemini editor, only if the operator turned it on and gave a key."""
    if not (config.gemini_enabled and config.gemini_api_key):
        return None
    return HttpGeminiEditor(config.gemini_api_key, config.gemini_model, config.gemini_timeout_seconds)


class CaptionService:
    """Implements each tool against paths confined to ``config.allowed_root``."""

    def __init__(
        self,
        config: ServerConfig,
        captioner: Captioner,
        blip_captioner: Captioner | None = None,
        enhancer: Enhancer | None = None,
        text_finder: TextFinder | None = None,
        gemini: GeminiEditor | None = None,
    ) -> None:
        self.config = config
        self.captioner = captioner
        # Creating a BLIP captioner loads nothing; its model loads on first use.
        self.blip_captioner = blip_captioner or create_blip_captioner(config)
        # Likewise the cleanup models. Text is found with the captioning model
        # (Florence-2) when it can read text, so no second copy is loaded.
        self.enhancer = enhancer or cleanup.create_enhancer(config)
        if text_finder is None:
            text_finder = captioner if isinstance(captioner, TextFinder) else NoTextFinder()
        self.text_finder = text_finder
        # Only present when the operator enabled Gemini and gave a key; None means
        # no image can be sent to Google by this server.
        self.gemini = gemini or _create_gemini(config)

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

    def manual_caption(
        self, captions: list[dict[str, str]], output_dir: str | None = None, overwrite: bool = True
    ) -> dict[str, Any]:
        """Save captions the user wrote, each as "<image stem>.txt".

        The text is kept as written, including prompt syntax such as ``(word:1.2)``;
        only line breaks and runs of spaces become single spaces, because a caption
        file holds one line. One bad entry is reported and does not stop the rest.
        """
        if not 0 < len(captions) <= MAX_MANUAL_CAPTIONS:
            raise ValueError(f"send between 1 and {MAX_MANUAL_CAPTIONS} captions per call")
        target_dir = self.resolve(output_dir) if output_dir else None
        results: list[dict[str, str]] = []
        for entry in captions:
            path = entry.get("path", "")
            try:
                image = imaging.require_image(self.resolve(path))
                text = " ".join(entry.get("caption", "").split())
                target = imaging.write_caption(image, text, target_dir or image.parent, overwrite)
            except (ValueError, FileNotFoundError, FileExistsError) as exc:
                results.append({"path": path, "status": "error", "error": str(exc)})
                continue
            results.append(
                {
                    "path": self._display(image),
                    "status": "written",
                    "caption_path": self._display(target),
                    "caption": text,
                }
            )
        written = sum(r["status"] == "written" for r in results)
        return {
            "total": len(results),
            "written": written,
            "failed": len(results) - written,
            "results": results,
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

    def cleanup_and_enhance(
        self,
        path: str,
        output_dir: str | None = None,
        remove_text: bool = True,
        keep_text: Sequence[str] = (),
        regions: Sequence[cleanup.Region] = (),
        scale: int = 2,
        overwrite: bool = True,
        backend: str = "local",
        instructions: str = "",
    ) -> dict[str, Any]:
        """Remove overlaid text and upscale; save ``<stem>_clean<ext>``, leaving the original alone.

        ``backend`` is ``local`` (models on this machine) or ``gemini`` (the image is sent to
        Google). Gemini works from a prompt and falls back to local when it cannot be used;
        ``instructions`` are the user's own wishes, added to its prompt.
        """
        if backend not in ("local", "gemini"):
            raise ValueError("backend must be local or gemini")
        if backend == "gemini" and self.gemini is None:
            raise ValueError(
                "Gemini cleanup is not enabled on this server: set CAPTIONER_GEMINI_ENABLED=true "
                "and provide GEMINI_API_KEY"
            )
        source = self.resolve(path)
        target_dir = self._output_dir(output_dir, source.parent)
        image, text = imaging.load_upright(source)
        alpha = image.getchannel("A") if "A" in image.getbands() else None
        options = {
            "remove_text": remove_text,
            "keep_text": keep_text,
            "regions": regions,
            "scale": scale,
            "max_side": self.config.max_output_side,
        }
        rgb = image.convert("RGB")
        if backend == "gemini" and self.gemini is not None:
            result = gemini_cleanup.clean_image_with_gemini(
                rgb,
                self.text_finder,
                self.enhancer,
                self.gemini,
                upscale_fallback=self.config.gemini_upscale_fallback,
                image_size=self.config.gemini_image_size,
                instructions=instructions,
                # Read on every call, so edits to the prompt file apply without a restart.
                base_prompt=gemini_cleanup.load_prompt(self.config.gemini_prompt_file),
                **options,  # type: ignore[arg-type]
            )
        else:
            result = cleanup.clean_image(rgb, self.text_finder, self.enhancer, **options)  # type: ignore[arg-type]
        cleaned = result.image
        if alpha is not None:
            cleaned.putalpha(alpha.resize(cleaned.size, Image.Resampling.LANCZOS))
        target = imaging.save_cleaned(cleaned, source, target_dir, text, overwrite)

        def listed(items: list[TextBox]) -> list[dict[str, Any]]:
            shown = items[: cleanup.MAX_REPORTED_TEXT]
            return [{"text": t.text, "box": [round(v) for v in t.box]} for t in shown]

        sent_to_google = backend == "gemini" and result.backend == "gemini"
        report: dict[str, Any] = {
            "path": self._display(source),
            "cleaned_path": self._display(target),
            "removed_text": listed(result.removed),
            "kept_text": listed(result.kept),
            "caption_bars": result.caption_bars,
            "regions_removed": result.regions,
            "original_size": list(image.size),
            "size": list(cleaned.size),
            "upscaled_by": result.upscaled_by,
            "backend": result.backend,
            "device": self.enhancer.device if result.backend == "local" else "gemini",
        }
        if backend == "gemini":
            report["requested_backend"] = "gemini"
            report["image_sent_to_google"] = result.backend == "gemini" or result.fallback_reason is not None
        if result.fallback_reason:
            report["fallback_reason"] = result.fallback_reason
        if sent_to_google:
            report["gemini_model"] = self.gemini.model if self.gemini else self.config.gemini_model
            report["gemini_size"] = list(result.gemini_size or ())
        return report

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
