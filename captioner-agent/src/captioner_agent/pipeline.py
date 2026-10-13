"""The fixed captioning order for uploaded images.

For every image, in this order:

1. PNG info: the server's ``caption_from_png_info`` script writes the caption
   from the embedded prompt. No model is involved.
2. Only for images with no PNG info: a vision model looks at the image and
   writes the caption. Which one is the user's choice, BLIP (local) or
   Anthropic (Claude), unless the settings pin one.

The order is enforced here in code, not left to the chat model to follow. The
two stages are separate methods so the app can ask the user between them.
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
from .prompts import BLIP_TOOL, build_enhance_prompt, build_vision_prompt, load_criteria
from .settings import Settings
from .toolbox import Toolbox

logger = logging.getLogger(__name__)

PNG_INFO_TOOL = "caption_from_png_info"

Backend = Literal["blip", "claude", "local"]
# Vision backends that run as tools on the MCP server. "claude" is not one:
# it sends the image to the Anthropic API from the agent itself.
SERVER_VISION_TOOLS = {"blip": BLIP_TOOL, "local": "caption_image"}
# The step name shown in the UI for each backend.
VISION_STEPS = {"blip": BLIP_TOOL, "claude": "caption_with_vision", "local": "caption_image"}

# The two buttons the user picks between. parse_backend_choice reads them back.
BLIP_OPTION = "BLIP (local, images stay on this machine)"
ANTHROPIC_OPTION = "Anthropic (Claude, images are sent to Anthropic)"

Event = dict[str, Any]
_SOURCES = ("png_info", "vision", "failed")


@dataclass(frozen=True)
class CaptionOutcome:
    """How one uploaded image was captioned."""

    image: str
    source: Literal["png_info", "vision", "failed"]
    caption: str | None = None
    error: str | None = None
    # Which vision backend was used (blip, claude or local); None for PNG info.
    via: str | None = None

    def to_dict(self) -> dict[str, str]:
        data = {"image": self.image, "source": self.source}
        if self.via is not None:
            data["via"] = self.via
        if self.caption is not None:
            data["caption"] = self.caption
        if self.error is not None:
            data["error"] = self.error
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaptionOutcome:
        source = data.get("source")
        return cls(
            image=str(data["image"]),
            source=source if source in _SOURCES else "failed",
            caption=data.get("caption"),
            error=data.get("error"),
            via=data.get("via"),
        )


def parse_backend_choice(answer: str) -> Literal["blip", "claude"] | None:
    """Read the user's reply to the BLIP-or-Anthropic question.

    Accepts a button label or free text. Returns ``None`` if the reply names
    neither backend, or names both, so the question is asked again.
    """
    text = answer.lower()
    wants_blip = "blip" in text
    wants_anthropic = "anthropic" in text or "claude" in text
    if wants_blip == wants_anthropic:
        return None
    return "blip" if wants_blip else "claude"


def vision_question(names: list[str]) -> dict[str, Any]:
    """The choice shown to the user when images have no PNG info."""
    count = len(names)
    shown = ", ".join(names[:3]) + (f" and {count - 3} more" if count > 3 else "")
    noun = "image has" if count == 1 else "images have"
    return {
        "id": f"vision_{uuid.uuid4().hex[:12]}",
        "question": f"{count} {noun} no PNG info ({shown}). Which should caption them?",
        "options": [BLIP_OPTION, ANTHROPIC_OPTION],
    }


class CaptionPipeline:
    def __init__(self, provider: LLMProvider, toolbox: Toolbox, settings: Settings) -> None:
        self._provider = provider
        self._toolbox = toolbox
        self._settings = settings

    async def png_stage(
        self, files: SessionFiles, names: list[str], results: dict[str, CaptionOutcome]
    ) -> AsyncIterator[Event]:
        """Step 1: caption each image from its PNG info, if it has any.

        Fills ``results`` for images that were captioned or failed. An image
        with no PNG info is left out of ``results``: it is for step 2.
        """
        step_ids = {name: f"auto_{uuid.uuid4().hex[:12]}" for name in names}
        for name in names:
            yield _event("tool_start", step_ids[name], PNG_INFO_TOOL, files.tool_path("uploads", name))

        async def one(name: str) -> tuple[str, CaptionOutcome | None]:
            return name, await self._png_info(files, name)

        for task in asyncio.as_completed([one(name) for name in names]):
            name, outcome = await task
            if outcome is not None:
                results[name] = outcome
            yield {
                **_event("tool_end", step_ids[name], PNG_INFO_TOOL, files.tool_path("uploads", name)),
                "ok": outcome is None or outcome.source != "failed",
                "summary": (outcome.caption or outcome.error or "") if outcome else "No PNG info",
            }

    async def vision_stage(
        self,
        files: SessionFiles,
        names: list[str],
        backend: Backend,
        results: dict[str, CaptionOutcome],
    ) -> AsyncIterator[Event]:
        """Step 2: caption images that have no PNG info with ``backend``."""
        limit = asyncio.Semaphore(self._settings.vision_concurrency)
        step = VISION_STEPS[backend]
        step_ids = {name: f"auto_{uuid.uuid4().hex[:12]}" for name in names}
        for name in names:
            yield _event("tool_start", step_ids[name], step, files.tool_path("uploads", name))

        async def one(name: str) -> tuple[str, CaptionOutcome]:
            async with limit:
                return name, await self._vision_one(files, name, backend)

        for task in asyncio.as_completed([one(name) for name in names]):
            name, outcome = await task
            results[name] = outcome
            yield {
                **_event("tool_end", step_ids[name], step, files.tool_path("uploads", name)),
                "ok": outcome.source != "failed",
                "summary": outcome.caption or outcome.error or "",
            }

    async def _png_info(self, files: SessionFiles, name: str) -> CaptionOutcome | None:
        image = files.tool_path("uploads", name)
        result = await self._toolbox.call(
            PNG_INFO_TOOL, {"path": image, "output_dir": files.tool_path("outputs")}
        )
        if result.is_error:
            return CaptionOutcome(image, "failed", error=result.text)
        info = _parse_json(result.text)
        if info.get("found"):
            return CaptionOutcome(image, "png_info", caption=str(info.get("caption", "")))
        return None

    async def _vision_one(self, files: SessionFiles, name: str, backend: Backend) -> CaptionOutcome:
        image = files.tool_path("uploads", name)
        try:
            caption = await self.describe(files, "uploads", name, backend)
        except (LLMError, InvalidFileError, OSError) as exc:
            return CaptionOutcome(image, "failed", error=str(exc), via=backend)
        saved = await self._toolbox.call(
            "write_caption",
            {"image_path": image, "text": caption, "output_dir": files.tool_path("outputs")},
        )
        if saved.is_error:
            return CaptionOutcome(image, "failed", error=saved.text, via=backend)
        return CaptionOutcome(image, "vision", caption=caption, via=backend)

    async def describe(self, files: SessionFiles, kind: FileKind, name: str, backend: Backend) -> str:
        """Caption one image by looking at it with ``backend``. The caption is not saved."""
        path = files.resolve(kind, name)
        server_tool = SERVER_VISION_TOOLS.get(backend)
        if server_tool is not None:
            result = await self._toolbox.call(
                server_tool, {"path": files.tool_path(kind, name), "style": self._settings.caption_style}
            )
            if result.is_error:
                raise LLMError(result.text)
            caption = str(_parse_json(result.text).get("caption", "")).strip()
            return " ".join(caption.split()) or _no_caption()
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
