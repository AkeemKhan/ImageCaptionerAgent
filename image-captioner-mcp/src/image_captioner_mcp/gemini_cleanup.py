"""``CLEANUP_AND_ENHANCE`` with Gemini doing the whole job from a prompt.

Gemini is told, in one request, to remove overlaid text together with whatever
holds it (bubbles, caption bars, boxes, borders), to fill those areas with what
belongs behind them, and to return a sharper, high-resolution image. No local
model decides what to remove: the prompt does. The base prompt lives in a file
the user can edit (``CAPTIONER_GEMINI_PROMPT_FILE``); the details for each image
(size, text to keep, extra areas, the user's own instructions) are added here.

If Gemini declines or fails, the image is cleaned by the local pipeline instead
and the result says why. Nothing is sent to Google a second time.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from .cleanup import (
    UPSCALE_FACTORS,
    CleanupResult,
    Enhancer,
    Region,
    TextFinder,
    clean_image,
    region_box,
)
from .gemini import (
    ASPECT_RATIOS,
    IMAGE_SIZES,
    GeminiEditor,
    GeminiError,
    closest_aspect_ratio,
    image_size_for,
)

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = """\
[Task: Photo Restoration, Overlay Removal and Inpainting]

1. Remove:
- Completely remove all text and every box, bubble, bar, banner, sticker and app UI \
element added on top of the photo, including their borders and shadows.

2. Inpaint:
- Fill every removed area with what belongs behind it, continuing the clothing, body \
and background seamlessly with matching lighting, shadows and texture. No smudges, \
patches or ghost outlines.

3. Restore:
- Restore the photo to a clean, natural, high-quality image: remove film grain, noise, \
filters, JPEG compression artefacts and blur.
- Make it sharp and in focus, never soft or blurry: crisp facial features, clear eyes, \
realistic skin texture, individual hair strands and detailed fabric.

4. Preserve:
- Preserve the person's identity, expression, posture and outfit and the overall \
composition, showing the whole scene without cropping.
- Printed text on real objects in the scene, such as a sign or a shirt, stays.
"""

# The picture sent to Gemini is no larger than this on its longest side.
GEMINI_INPUT_SIDE = 2048
# How far Gemini's answer may stray from the original's proportions.
ASPECT_TOLERANCE = 0.04
# Local upscaling is only worth its time when Gemini's image is this much too small.
LOCAL_UPSCALE_MIN_RATIO = 1.5
MAX_KEEP_TEXT_CHARS = 60
MAX_INSTRUCTION_CHARS = 500


def load_prompt(path: Path | None) -> str:
    """The base prompt: the file's text if it exists and is not empty, else the built-in one."""
    if path is not None:
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            text = ""
        if text:
            return text
    return DEFAULT_PROMPT.strip()


def clean_image_with_gemini(
    image: Image.Image,
    finder: TextFinder,
    enhancer: Enhancer,
    gemini: GeminiEditor,
    *,
    remove_text: bool = True,
    keep_text: Sequence[str] = (),
    regions: Sequence[Region] = (),
    scale: int = 2,
    max_side: int = 4096,
    image_size: str = "2K",
    instructions: str = "",
    base_prompt: str = DEFAULT_PROMPT,
    upscale_fallback: str = "none",
) -> CleanupResult:
    """Clean and enhance ``image`` with Gemini, falling back to the local pipeline.

    ``scale`` 1 keeps the original size; 2 or 4 ask for an ``image_size`` result
    (``1K``, ``2K`` or ``4K``), never smaller than the original and capped at
    ``max_side``. ``finder`` and ``enhancer`` are used only by the fallback and,
    when Gemini's image is too small, to upscale it.
    """
    if scale not in UPSCALE_FACTORS:
        raise ValueError(f"scale must be one of {', '.join(map(str, UPSCALE_FACTORS))}")
    if image_size not in IMAGE_SIZES:
        raise ValueError(f"image_size must be one of {', '.join(IMAGE_SIZES)}")
    width, height = image.size
    for region in regions:
        region_box(region, width, height)  # validates before anything is sent
    target = _target_size(width, height, scale, image_size, max_side)
    if not remove_text and not regions and not instructions.strip() and target == image.size:
        return CleanupResult(image)

    ratio = closest_aspect_ratio(width, height)
    framed, inner = _frame(image, ratio)
    prompt = build_prompt(
        base_prompt,
        remove_text=remove_text,
        keep_text=keep_text,
        regions=regions,
        letterboxed=inner is not None,
        instructions=instructions,
    )
    sent = framed.copy()
    sent.thumbnail((GEMINI_INPUT_SIDE, GEMINI_INPUT_SIDE), Image.Resampling.LANCZOS)

    def local(reason: str) -> CleanupResult:
        logger.warning("Gemini cleanup not used: %s", reason)
        result = clean_image(
            image,
            finder,
            enhancer,
            remove_text=remove_text,
            keep_text=keep_text,
            regions=regions,
            scale=scale,
            max_side=max_side,
        )
        result.fallback_reason = reason
        return result

    try:
        edited = gemini.edit(sent, prompt, ratio, image_size_for(max(target)))
    except GeminiError as exc:
        return local(str(exc))
    returned = edited.size
    if inner is not None:
        edited = _unframe(edited, inner, sent.size)
    if abs((edited.width / edited.height) / (width / height) - 1) > ASPECT_TOLERANCE:
        return local("Gemini changed the proportions of the picture")

    final = _fit(edited, target, enhancer, upscale_fallback)
    return CleanupResult(
        final,
        regions=len(regions),
        upscaled_by=round(max(final.size) / max(width, height), 3),
        backend="gemini",
        gemini_size=returned,
    )


def build_prompt(
    base_prompt: str,
    *,
    remove_text: bool = True,
    keep_text: Sequence[str] = (),
    regions: Sequence[Region] = (),
    letterboxed: bool = False,
    instructions: str = "",
) -> str:
    """The base prompt plus the details for this image.

    ``keep_text`` and ``instructions`` come from the user through the chat; they
    are trimmed and stripped of quotes and line breaks before they are added.
    """
    details: list[str] = []
    if not remove_text:
        details.append("Leave the text and overlays in place; only restore and sharpen the photo.")
    keep = [t for t in (_clean(k, MAX_KEEP_TEXT_CHARS) for k in keep_text) if t]
    if keep and remove_text:
        quoted = ", ".join(f'"{t}"' for t in keep)
        details.append(f"Leave this text in place, it belongs in the scene: {quoted}.")
    if regions:
        areas = "; ".join(f"{r[0]:.0%}-{r[2]:.0%} across, {r[1]:.0%}-{r[3]:.0%} down" for r in regions)
        details.append(
            "Also remove whatever is overlaid in these areas, measured from the top-left, "
            f"and rebuild what belongs behind them: {areas}."
        )
    if letterboxed:
        details.append(
            "The picture has been placed on a black background to fit the frame. Leave those "
            "black margins plain black and do not extend the scene into them."
        )
    wish = _clean(instructions, MAX_INSTRUCTION_CHARS)
    if wish:
        details.append(f"The user also asked: {wish}")
    lines = [base_prompt.strip()]
    if details:
        lines += ["", "For this image:", *(f"- {d}" for d in details)]
    return "\n".join([*lines, "", "Reply with the edited image only."])


def _clean(text: str, limit: int) -> str:
    return " ".join(text.replace('"', "").replace("\\", "").split())[:limit].strip()


def _target_size(width: int, height: int, scale: int, image_size: str, max_side: int) -> tuple[int, int]:
    """The finished image's size: the original for scale 1, otherwise the ``image_size`` bucket.

    The bucket never shrinks an image that is already larger, and is capped at ``max_side``.
    """
    long_side = max(width, height)
    if scale == 1:
        return width, height
    wanted = max(long_side, min(IMAGE_SIZES[image_size], max_side))
    factor = wanted / long_side
    return round(width * factor), round(height * factor)


def _frame(image: Image.Image, ratio: str) -> tuple[Image.Image, tuple[float, float, float, float] | None]:
    """Letterbox ``image`` to an aspect ratio Gemini accepts, if it is not close to one already.

    Returns the image to send and, when letterboxed, where the picture sits in it
    as ``(left, top, right, bottom)`` fractions, to crop Gemini's answer back.
    """
    width, height = image.size
    wanted = ASPECT_RATIOS[ratio]
    if abs(wanted / (width / height) - 1) <= ASPECT_TOLERANCE:
        return image, None
    if wanted > width / height:
        canvas = Image.new("RGB", (round(height * wanted), height), "black")
        x = (canvas.width - width) // 2
        canvas.paste(image, (x, 0))
        return canvas, (x / canvas.width, 0.0, (x + width) / canvas.width, 1.0)
    canvas = Image.new("RGB", (width, round(width / wanted)), "black")
    y = (canvas.height - height) // 2
    canvas.paste(image, (0, y))
    return canvas, (0.0, y / canvas.height, 1.0, (y + height) / canvas.height)


def _unframe(
    edited: Image.Image, inner: tuple[float, float, float, float], sent: tuple[int, int]
) -> Image.Image:
    """Crop the letterbox margins off Gemini's answer.

    Half a pixel of rounding in what was sent becomes several pixels at 4K, so
    the crop goes one sent pixel (plus one) inside the picture on the padded
    sides, leaving no sliver of black along the edge.
    """
    left, top, right, bottom = inner
    inset_x = math.ceil(edited.width / sent[0]) + 1 if left > 0 else 0
    inset_y = math.ceil(edited.height / sent[1]) + 1 if top > 0 else 0
    return edited.crop(
        (
            round(left * edited.width) + inset_x,
            round(top * edited.height) + inset_y,
            round(right * edited.width) - inset_x,
            round(bottom * edited.height) - inset_y,
        )
    )


def _fit(
    edited: Image.Image,
    target: tuple[int, int],
    enhancer: Enhancer,
    fallback: str,
) -> Image.Image:
    """Bring Gemini's image to ``target`` size.

    Too large: scaled down. Too small: kept at Gemini's own size when ``fallback``
    is ``none`` (the default: stretching only blurs it, even back up to the
    original's size); otherwise upscaled with the local model (``local``, when
    far too small) or resized.
    """
    if edited.size == target:
        return edited
    if max(edited.size) < max(target) and fallback == "none":
        return edited
    needed = max(target) / max(edited.size)
    if needed >= LOCAL_UPSCALE_MIN_RATIO and fallback == "local":
        edited = enhancer.upscale_x4(edited)
    return edited.resize(target, Image.Resampling.LANCZOS)
