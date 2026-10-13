"""The fixed captioning order for uploaded images.

For every image, in this order:

1. PNG info: the server's ``caption_from_png_info`` script writes the caption
   from the embedded prompt. No model is involved.
2. Only if the image has no PNG info: a vision model looks at the image and
   writes the caption (Claude by default, the local model if configured).

The order is enforced here in code, not left to the chat model to follow.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

from .files import FileKind, InvalidFileError, SessionFiles, encode_for_vision
from .llm import LLMError, LLMProvider
from .prompts import build_enhance_prompt, build_vision_prompt, load_criteria
from .settings import Settings
from .toolbox import Toolbox

logger = logging.getLogger(__name__)

PNG_INFO_TOOL = "caption_from_png_info"
# Shown in the UI for step 2; not a server tool.
VISION_STEP = "caption_with_vision"

Event = dict[str, Any]


@dataclass(frozen=True)
class CaptionOutcome:
    """How one uploaded image was captioned."""

    image: str
    source: Literal["png_info", "vision", "failed"]
    caption: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, str]:
        data = {"image": self.image, "source": self.source}
        if self.caption is not None:
            data["caption"] = self.caption
        if self.error is not None:
            data["error"] = self.error
        return data


class CaptionPipeline:
    def __init__(self, provider: LLMProvider, toolbox: Toolbox, settings: Settings) -> None:
        self._provider = provider
        self._toolbox = toolbox
        self._settings = settings

    async def run(
        self, files: SessionFiles, uploaded: list[str], outcomes: list[CaptionOutcome]
    ) -> AsyncIterator[Event]:
        """Caption ``uploaded`` images, yielding UI events and filling ``outcomes`` in upload order."""
        limit = asyncio.Semaphore(self._settings.vision_concurrency)
        step_ids = {name: f"auto_{uuid.uuid4().hex[:12]}" for name in uploaded}
        for name in uploaded:
            yield _event("tool_start", step_ids[name], PNG_INFO_TOOL, files.tool_path("uploads", name))

        async def caption_one(name: str) -> tuple[str, str, CaptionOutcome]:
            async with limit:
                step, outcome = await self._caption_one(files, name)
            return name, step, outcome

        finished: dict[str, CaptionOutcome] = {}
        for task in asyncio.as_completed([caption_one(name) for name in uploaded]):
            name, step, outcome = await task
            finished[name] = outcome
            yield {
                **_event("tool_end", step_ids[name], step, outcome.image),
                "ok": outcome.source != "failed",
                "summary": outcome.caption or outcome.error or "",
            }
        outcomes.extend(finished[name] for name in uploaded)

    async def _caption_one(self, files: SessionFiles, name: str) -> tuple[str, CaptionOutcome]:
        """Returns the step that produced the result and the outcome."""
        image = files.tool_path("uploads", name)
        output_dir = files.tool_path("outputs")

        result = await self._toolbox.call(PNG_INFO_TOOL, {"path": image, "output_dir": output_dir})
        if result.is_error:
            return PNG_INFO_TOOL, CaptionOutcome(image, "failed", error=result.text)
        info = _parse_json(result.text)
        if info.get("found"):
            return PNG_INFO_TOOL, CaptionOutcome(image, "png_info", caption=str(info.get("caption", "")))

        try:
            caption = await self.describe(files, "uploads", name)
        except (LLMError, InvalidFileError, OSError) as exc:
            return VISION_STEP, CaptionOutcome(image, "failed", error=str(exc))
        saved = await self._toolbox.call(
            "write_caption", {"image_path": image, "text": caption, "output_dir": output_dir}
        )
        if saved.is_error:
            return VISION_STEP, CaptionOutcome(image, "failed", error=saved.text)
        return VISION_STEP, CaptionOutcome(image, "vision", caption=caption)

    async def describe(self, files: SessionFiles, kind: FileKind, name: str) -> str:
        """Caption one image by looking at it, with the configured vision backend."""
        path = files.resolve(kind, name)
        if self._settings.vision_backend == "local":
            result = await self._toolbox.call(
                "caption_image", {"path": files.tool_path(kind, name), "style": self._settings.caption_style}
            )
            if result.is_error:
                raise LLMError(result.text)
            return str(_parse_json(result.text).get("caption", "")).strip() or _no_caption()
        data, media_type = await asyncio.to_thread(encode_for_vision, path)
        instructions = build_vision_prompt(load_criteria(self._settings.criteria_file))
        caption = await self._provider.caption_image(
            image=data, media_type=media_type, instructions=instructions
        )
        # A caption file holds a single line.
        return " ".join(caption.split()) or _no_caption()

    async def enhance(
        self, files: SessionFiles, kind: FileKind, name: str, mode: str, guidance: str = ""
    ) -> dict[str, str]:
        """Rewrite an image's existing caption with Claude and save it over the old one.

        Only the caption text is sent; the image is not. Raises ``LLMError`` if
        the image has no caption yet, because enhancement comes after captioning.
        """
        files.resolve(kind, name)
        caption_file = files.caption_file(name)
        if not caption_file.is_file() or not (original := caption_file.read_text(encoding="utf-8").strip()):
            raise LLMError(f"{name} has no caption yet; caption it before enhancing")
        instructions = build_enhance_prompt(load_criteria(self._settings.criteria_file), mode, guidance)
        reply = await self._provider.enhance_caption(caption=original, instructions=instructions)
        enhanced = " ".join(reply.split())
        if not enhanced:
            raise LLMError("The model returned no caption.")
        saved = await self._toolbox.call(
            "write_caption",
            {
                "image_path": files.tool_path(kind, name),
                "text": enhanced,
                "output_dir": files.tool_path("outputs"),
            },
        )
        if saved.is_error:
            raise LLMError(saved.text)
        return {"image": name, "mode": mode, "original": original, "enhanced": enhanced}


def _event(kind: str, step_id: str, name: str, path: str) -> Event:
    return {"type": kind, "id": step_id, "name": name, "input": {"path": path}}


def _parse_json(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _no_caption() -> str:
    raise LLMError("The vision model returned no caption.")
