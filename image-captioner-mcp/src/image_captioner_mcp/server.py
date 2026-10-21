"""MCP server: tools, a resource and a prompt for image captioning."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Literal
from urllib.parse import unquote

import anyio
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from PIL import UnidentifiedImageError
from pydantic import BaseModel, Field

from . import imaging
from .captioner import Captioner, CaptionStyle, create_captioner
from .config import ServerConfig
from .service import CaptionService

logger = logging.getLogger(__name__)

INSTRUCTIONS = """\
Image captioning tools. Every path is relative to the server's allowed root \
(absolute paths inside it also work); anything outside is rejected.

Typical flow for one image: read_png_metadata to see whether the generator \
embedded a prompt (caption_from_png_info captions from it directly); if not, \
describe the image with a local model: caption_image (Florence-2) or \
BLIP_CAPTIONER (BLIP); then write_caption to save "<image stem>.txt". If the \
user wrote the captions, MANUAL_CAPTION saves them as written. \
flip_image creates a mirrored copy, and mirror_caption swaps left/right \
wording so a caption fits that copy. CLEANUP_AND_ENHANCE saves a copy with \
watermarks, labels and app captions removed, upscaled.
"""

DEFAULT_CRITERIA = """\
- Write one natural-language caption on a single line.
- Describe the subject first, then setting, then notable details.
- Remove prompt syntax: attention weights such as (word:1.2), brackets, LoRA \
and embedding tags, BREAK keywords.
- Remove quality and style boilerplate (masterpiece, best quality, 8k, ...).
- Do not invent details that are not in the source text.
"""


class ManualCaptionEntry(BaseModel):
    """One image and the caption the user wrote for it."""

    path: str = Field(description="Image path, relative to the allowed root.")
    caption: str = Field(description="The caption text, exactly as the user wrote it.")


class ImageRegion(BaseModel):
    """An area to remove, as fractions of the image width and height (0 to 1)."""

    left: float = Field(ge=0, le=1)
    top: float = Field(ge=0, le=1)
    right: float = Field(ge=0, le=1)
    bottom: float = Field(ge=0, le=1)


@contextmanager
def _tool_errors() -> Iterator[None]:
    """Turn expected failures into tool errors whose message reaches the client.

    Any other exception is reported by the SDK as a generic failure, so
    internal details are not sent to the model.
    """
    try:
        yield
    except UnidentifiedImageError as exc:
        raise ToolError("the file is not a readable image") from exc
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        raise ToolError(str(exc)) from exc


def create_server(service: CaptionService) -> MCPServer[None]:
    """Build the MCP server around ``service``."""
    config = service.config

    @asynccontextmanager
    async def lifespan(_server: MCPServer[None]) -> AsyncIterator[None]:
        # Load in the background so the client handshake is not held up by a
        # multi-second model load; the first caption call waits on the
        # captioner's lock if loading is still in progress.
        if config.preload:
            threading.Thread(target=_preload, args=(service.captioner,), daemon=True).start()
        if config.preload_blip:
            threading.Thread(target=_preload, args=(service.blip_captioner,), daemon=True).start()
        try:
            yield
        finally:
            service.captioner.unload()
            service.blip_captioner.unload()
            service.enhancer.unload()

    server: MCPServer[None] = MCPServer("image-captioner", instructions=INSTRUCTIONS, lifespan=lifespan)

    @server.tool()
    def list_images(directory: str) -> dict[str, Any]:
        """List the image files (png, jpg, jpeg, webp) directly inside a directory."""
        with _tool_errors():
            return service.list_images(directory)

    @server.tool()
    def read_png_metadata(path: str) -> dict[str, Any]:
        """Read the generation prompt embedded in an image by AI image generators.

        Understands AUTOMATIC1111/Forge, ComfyUI, NovelAI, InvokeAI and Fooocus
        metadata. When has_prompt is true, `prompt` holds the positive prompt.
        When it is false, `other_text` holds any other embedded text, untouched.
        """
        with _tool_errors():
            return service.read_png_metadata(path)

    @server.tool()
    def caption_from_png_info(
        path: str, output_dir: str | None = None, overwrite: bool = True
    ) -> dict[str, Any]:
        """Caption an image from its embedded PNG info and save "<image stem>.txt".

        The caption is the generation prompt before "Negative prompt:", with
        prompt syntax such as ( ) : < > and LoRA tags removed. No model is used.
        When the image has no PNG info, `found` is false and nothing is written.
        """
        with _tool_errors():
            return service.caption_from_png_info(path, output_dir, overwrite)

    @server.tool()
    async def caption_image(path: str, style: CaptionStyle = "detailed") -> dict[str, Any]:
        """Describe an image with the locally hosted captioning model.

        Use this when the image carries no embedded prompt. `style` controls
        length: short (one phrase), detailed (a sentence or two), more_detailed
        (a paragraph).
        """
        with _tool_errors():
            return await anyio.to_thread.run_sync(service.caption_image, path, style)

    @server.tool(name="BLIP_CAPTIONER")
    async def blip_captioner(path: str, style: CaptionStyle = "short") -> dict[str, Any]:
        """Caption an image with the BLIP model running on this machine.

        BLIP looks at the image itself and writes a new, plain caption, so it
        replaces an existing caption rather than building on it. `style`
        controls length: short (one sentence), detailed, more_detailed. The
        caption is returned and not saved; save it with write_caption.
        """
        with _tool_errors():
            return await anyio.to_thread.run_sync(service.blip_caption, path, style)

    @server.tool(name="MANUAL_CAPTION")
    def manual_caption(
        captions: list[ManualCaptionEntry], output_dir: str | None = None, overwrite: bool = True
    ) -> dict[str, Any]:
        """Save captions that the user wrote themselves, as "<image stem>.txt".

        Use this when the user supplies the caption text. No model is involved
        and the wording is kept exactly as written, including syntax such as
        (word:1.2); only line breaks become spaces. Send one entry per image,
        with the same text repeated for images that share a caption. Each
        image must exist. Entries that fail are reported and the rest are saved.
        """
        with _tool_errors():
            return service.manual_caption([c.model_dump() for c in captions], output_dir, overwrite)

    @server.tool()
    def write_caption(
        image_path: str, text: str, output_dir: str | None = None, overwrite: bool = True
    ) -> dict[str, Any]:
        """Save a caption as "<image stem>.txt", e.g. Image1.png -> Image1.txt.

        The file goes in `output_dir`, or next to the image when that is omitted.
        With overwrite=false an existing caption file is an error.
        """
        with _tool_errors():
            return service.write_caption(image_path, text, output_dir, overwrite)

    @server.tool()
    def flip_image(
        path: str, output_dir: str | None = None, direction: imaging.FlipDirection = "horizontal"
    ) -> dict[str, Any]:
        """Save a mirrored copy of an image as "<stem>_flipped<ext>".

        The copy carries no embedded metadata, because the original prompt's
        left/right wording would no longer match it.
        """
        with _tool_errors():
            return service.flip_image(path, output_dir, direction)

    @server.tool(name="CLEANUP_AND_ENHANCE")
    async def cleanup_and_enhance(
        path: str,
        output_dir: str | None = None,
        remove_text: bool = True,
        keep_text: list[str] | None = None,
        regions: list[ImageRegion] | None = None,
        scale: Literal[1, 2, 4] = 2,
        overwrite: bool = True,
    ) -> dict[str, Any]:
        """Remove text laid over an image and upscale it, saving "<stem>_clean<ext>".

        Removes watermarks, labels, usernames and app captions such as a
        Snapchat caption bar: an OCR model finds the text, a semi-transparent
        bar behind it is undone, and an inpainting model paints the text out.
        Then the image is upscaled by `scale` (1 means no upscaling; the
        longest side is capped). The original is not changed, and the copy
        needs its own caption file.

        OCR cannot tell an overlay from text that belongs in the picture (a
        sign, a shirt), so every removed text is listed in `removed_text`. Run
        again with `keep_text` (case-insensitive substrings) to keep some, or
        remove_text=false to only upscale. `regions` removes extra areas the
        OCR cannot read, such as a logo, given as fractions of the image size.
        """
        boxes = [(r.left, r.top, r.right, r.bottom) for r in regions or []]
        with _tool_errors():
            return await anyio.to_thread.run_sync(
                lambda: service.cleanup_and_enhance(
                    path, output_dir, remove_text, keep_text or [], boxes, scale, overwrite
                )
            )

    @server.tool()
    def mirror_caption(text: str) -> dict[str, str]:
        """Swap the words "left" and "right" so a caption describes the horizontally flipped image."""
        return {"text": imaging.mirror_caption(text)}

    @server.tool()
    async def caption_folder(
        directory: str,
        ctx: Context,
        output_dir: str | None = None,
        style: CaptionStyle = "detailed",
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Caption every image in a directory with the local model and write each .txt.

        Reports progress per image. With overwrite=false, images that already
        have a caption file are skipped. One failing image does not stop the batch.
        """
        with _tool_errors():
            source = service.resolve(directory)
            target = service.resolve(output_dir) if output_dir else source
            images = imaging.list_images(source)
        results: list[dict[str, Any]] = []
        for index, image in enumerate(images, start=1):
            try:
                result = await anyio.to_thread.run_sync(
                    service.caption_to_file, image, target, style, overwrite
                )
            except Exception:  # report per image and keep going
                logger.exception("Captioning failed for %s", image.name)
                result = {"image": image.name, "status": "error"}
            results.append(result)
            await ctx.report_progress(index, len(images), image.name)
        counts = {
            status: sum(r["status"] == status for r in results) for status in ("written", "skipped", "error")
        }
        return {"total": len(images), **counts, "results": results}

    @server.resource("captions://{folder}", mime_type="application/json")
    def captions(folder: str) -> str:
        """The caption .txt files in a folder, as a JSON object of file name to text.

        URL-encode slashes for nested folders, e.g. captions://session1%2Foutputs.
        """
        return json.dumps(service.read_captions(unquote(folder)), ensure_ascii=False, indent=2)

    @server.prompt()
    def simplify_caption(raw_prompt: str, criteria: str = "") -> str:
        """Rewrite a raw image-generation prompt as a clean caption that follows the criteria."""
        rules = criteria.strip() or _load_criteria(config)
        return (
            "Rewrite the image-generation prompt below as a caption that follows "
            "these criteria. Reply with the caption only.\n\n"
            f"<criteria>\n{rules}\n</criteria>\n\n"
            f"<raw_prompt>\n{raw_prompt}\n</raw_prompt>"
        )

    return server


def _load_criteria(config: ServerConfig) -> str:
    if config.criteria_file and config.criteria_file.is_file():
        return config.criteria_file.read_text(encoding="utf-8").strip()
    return DEFAULT_CRITERIA.strip()


def _preload(captioner: Captioner) -> None:
    try:
        captioner.load()
    except Exception:
        logger.exception("Preloading the %s model failed; its tool will report the error", captioner.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Image captioning MCP server")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1", help="bind address for streamable-http")
    parser.add_argument("--port", type=int, default=8765, help="port for streamable-http")
    args = parser.parse_args()

    # stdout carries the MCP protocol on stdio, so all logging goes to stderr.
    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = ServerConfig.from_env()
    config.allowed_root.mkdir(parents=True, exist_ok=True)
    logger.info("Allowed root: %s (backend: %s)", config.allowed_root, config.backend)
    server = create_server(CaptionService(config, create_captioner(config)))
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
