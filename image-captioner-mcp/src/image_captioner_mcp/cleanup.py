"""Removing overlaid text from images, and upscaling them.

Three steps, each behind a small interface so the logic can be tested without
the models:

1. find text: Florence-2 OCR, the same model ``caption_image`` uses;
2. paint it out with LaMa, after first undoing any semi-transparent caption
   bar behind it (a Snapchat caption), so the picture under the bar is kept
   rather than repainted;
3. upscale with Swin2SR (x4), then resize to the requested scale.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageFilter

from .captioner import TextBox, pick_device
from .config import ServerConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")

UPSCALE_FACTORS = (1, 2, 4)
LAMA_FILE = "big-lama.pt"
# LaMa is trained at 512 pixels and leaves streaks well above it, so larger
# crops are scaled down for it.
LAMA_MAX_SIDE = 512
# Swin2SR runs on tiles of this size, overlapping so tile seams do not show.
SR_TILE = 256
SR_OVERLAP = 16
# Text in a large image is too small for OCR on the whole frame, so it is also
# read in tiles, at most this many per side.
OCR_FULL_FRAME_SIDE = 1536
OCR_MAX_TILES_PER_SIDE = 3
OCR_TILE_OVERLAP = 128
MAX_REPORTED_TEXT = 50
MIN_TEXT_SIDE = 4

Array = npt.NDArray[Any]
Box = tuple[int, int, int, int]
Region = tuple[float, float, float, float]


@runtime_checkable
class TextFinder(Protocol):
    def find_text(self, image: Image.Image) -> list[TextBox]: ...


class NoTextFinder:
    """Finds nothing; used when no OCR model is configured."""

    def find_text(self, image: Image.Image) -> list[TextBox]:
        return []


class Enhancer(Protocol):
    """The two image-to-image models."""

    @property
    def device(self) -> str: ...

    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Repaint the white pixels of ``mask`` (mode ``L``, same size) in RGB ``image``."""
        ...

    def upscale_x4(self, image: Image.Image) -> Image.Image: ...

    def unload(self) -> None: ...


class StubEnhancer:
    """No models: blur fills the mask and Lanczos upscales. For tests and wiring checks."""

    @property
    def device(self) -> str:
        return "none"

    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        blurred = image.filter(ImageFilter.GaussianBlur(8))
        return Image.composite(blurred, image, mask)

    def upscale_x4(self, image: Image.Image) -> Image.Image:
        return image.resize((image.width * 4, image.height * 4), Image.Resampling.LANCZOS)

    def unload(self) -> None:
        return None


class TorchEnhancer:
    """LaMa and Swin2SR on the GPU when there is room, each loaded on first use.

    Like the captioners, calls are serialised by a lock and a CUDA
    out-of-memory error moves the models to the CPU instead of failing.
    """

    def __init__(self, config: ServerConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._torch: Any = None
        self._lama: Any = None
        self._sr: Any = None
        self._device = "unloaded"

    @property
    def device(self) -> str:
        return self._device

    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        return self._run(self._load_lama, self._inpaint, image, mask)

    def upscale_x4(self, image: Image.Image) -> Image.Image:
        return self._run(self._load_sr, self._upscale_x4, image)

    def unload(self) -> None:
        with self._lock:
            self._lama = None
            self._sr = None
            if self._torch is not None and self._device == "cuda":
                self._torch.cuda.empty_cache()
            self._device = "unloaded"

    def _run(self, load: Callable[[], None], fn: Callable[..., T], *args: Any) -> T:
        with self._lock:
            self._import_torch()
            load()
            try:
                return fn(*args)
            except self._torch.cuda.OutOfMemoryError:
                if self._device != "cuda":
                    raise
                logger.warning("CUDA out of memory; moving the cleanup models to CPU")
                self._device = "cpu"
                if self._lama is not None:
                    self._lama = self._lama.to("cpu")
                if self._sr is not None:
                    self._sr = self._sr.to("cpu").float()
                self._torch.cuda.empty_cache()
                return fn(*args)

    def _import_torch(self) -> None:
        if self._torch is not None:
            return
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Cleanup needs torch and transformers. "
                "Install them with: pip install 'image-captioner-mcp[florence]'"
            ) from exc
        self._torch = torch

    def _pick(self) -> str:
        # The first model to load picks the device; the second follows it.
        if self._device == "unloaded":
            self._device = pick_device(self._torch, self._config)
        return self._device

    def _load_lama(self) -> None:
        if self._lama is not None:
            return
        from huggingface_hub import hf_hub_download

        # A TorchScript archive: loading it runs no Python from the repository,
        # and CAPTIONER_LAMA_MODEL_REVISION pins the exact file.
        path = hf_hub_download(
            self._config.lama_model_id, LAMA_FILE, revision=self._config.lama_model_revision
        )
        device = self._pick()
        logger.info("Loading LaMa on %s", device)
        self._lama = self._torch.jit.load(path, map_location=device).eval()

    def _load_sr(self) -> None:
        if self._sr is not None:
            return
        try:
            from transformers import Swin2SRForImageSuperResolution
        except ImportError as exc:
            raise RuntimeError(
                "Upscaling needs transformers. Install it with: pip install 'image-captioner-mcp[florence]'"
            ) from exc
        device = self._pick()
        dtype = self._torch.float16 if device == "cuda" else self._torch.float32
        logger.info("Loading Swin2SR on %s", device)
        kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if self._config.upscale_model_revision:
            kwargs["revision"] = self._config.upscale_model_revision
        model = Swin2SRForImageSuperResolution.from_pretrained(self._config.upscale_model_id, **kwargs)
        self._sr = model.to(device).eval()

    def _tensor(self, array: Array, multiple: int) -> tuple[Any, int, int]:
        """HxWxC float array -> 1xCxHxW tensor, edge-padded to a multiple of ``multiple``."""
        torch = self._torch
        height, width = array.shape[:2]
        tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))[None]
        pad_h, pad_w = -height % multiple, -width % multiple
        if pad_h or pad_w:
            tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
        return tensor.to(self._device), height, width

    def _inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        torch = self._torch
        pixels, height, width = self._tensor(np.asarray(image, np.float32) / 255, 8)
        holes, _, _ = self._tensor((np.asarray(mask)[..., None] > 127).astype(np.float32), 8)
        with torch.inference_mode():
            result = self._lama(pixels, holes)[0, :, :height, :width]
        return _to_image(result.permute(1, 2, 0).float().cpu().numpy())

    def _upscale_x4(self, image: Image.Image) -> Image.Image:
        source = np.asarray(image, np.float32) / 255
        height, width = source.shape[:2]
        # uint8 keeps a 4x image of a large photo to a quarter of the memory of floats.
        output = np.zeros((height * 4, width * 4, 3), np.uint8)
        step = SR_TILE - 2 * SR_OVERLAP
        for y in range(0, height, step):
            for x in range(0, width, step):
                y0, x0 = max(0, y - SR_OVERLAP), max(0, x - SR_OVERLAP)
                y1, x1 = min(height, y + step + SR_OVERLAP), min(width, x + step + SR_OVERLAP)
                tile = self._sr_tile(source[y0:y1, x0:x1])
                # Keep only the tile's centre; the overlap is context for the model.
                keep_h, keep_w = min(step, height - y), min(step, width - x)
                oy, ox = (y - y0) * 4, (x - x0) * 4
                output[y * 4 : (y + keep_h) * 4, x * 4 : (x + keep_w) * 4] = tile[
                    oy : oy + keep_h * 4, ox : ox + keep_w * 4
                ]
        return Image.fromarray(output, "RGB")

    def _sr_tile(self, tile: Array) -> Array:
        torch = self._torch
        pixels, height, width = self._tensor(tile, 8)
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        with torch.inference_mode():
            result = self._sr(pixel_values=pixels.to(dtype)).reconstruction
        result = result[0, :, : height * 4, : width * 4]
        pixels = result.permute(1, 2, 0).float().clamp(0, 1).mul(255).round().byte()
        return np.asarray(pixels.cpu().numpy())


def create_enhancer(config: ServerConfig) -> Enhancer:
    """The models behind ``CLEANUP_AND_ENHANCE``. Nothing loads until first use."""
    return StubEnhancer() if config.backend == "stub" else TorchEnhancer(config)


@dataclass
class CleanupResult:
    image: Image.Image
    removed: list[TextBox] = field(default_factory=list)
    kept: list[TextBox] = field(default_factory=list)
    caption_bars: int = 0
    regions: int = 0
    upscaled_by: float = 1.0


def clean_image(
    image: Image.Image,
    finder: TextFinder,
    enhancer: Enhancer,
    *,
    remove_text: bool = True,
    keep_text: Sequence[str] = (),
    regions: Sequence[Region] = (),
    scale: int = 2,
    max_side: int = 4096,
) -> CleanupResult:
    """Remove text and the given regions from an RGB ``image``, then upscale it.

    Text containing any of ``keep_text`` (case-insensitive) is left alone.
    ``regions`` are extra areas to remove, such as a logo without text, as
    ``(left, top, right, bottom)`` fractions of the width and height. The
    upscaled image's longest side is capped at ``max_side``.
    """
    if scale not in UPSCALE_FACTORS:
        raise ValueError(f"scale must be one of {', '.join(map(str, UPSCALE_FACTORS))}")
    result = CleanupResult(image)
    width, height = image.size
    holes: list[Box] = []

    if remove_text:
        wanted = [k.strip().lower() for k in keep_text if k.strip()]
        for found in find_all_text(finder, image):
            if any(k in found.text.lower() for k in wanted):
                result.kept.append(found)
            else:
                result.removed.append(found)

    if result.removed:
        pixels = np.asarray(image, np.float32).copy()
        lum = pixels @ np.array([0.299, 0.587, 0.114], np.float32)
        bars: list[tuple[int, int]] = []
        for found in result.removed:
            box = _clip(found.box, width, height)
            holes.append(_pad(box, width, height))
            bar = find_caption_bar(lum, box)
            if bar and bar not in bars:
                bars.append(bar)
        for top, bottom in bars:
            restore_bar(pixels, top, bottom, _text_columns(result.removed, width))
            # The bar's edges are soft, so they are painted over too.
            seam = max(3, (bottom - top) // 16)
            holes.append((0, max(0, top - seam), width, min(height, top + seam)))
            holes.append((0, max(0, bottom - seam), width, min(height, bottom + seam)))
        result.caption_bars = len(bars)
        image = _to_image(pixels / 255)

    for region in regions:
        holes.append(_region_box(region, width, height))
    result.regions = len(regions)

    if holes:
        image = _inpaint_holes(image, holes, enhancer)

    factor = min(float(scale), max_side / max(width, height))
    if factor > 1:
        upscaled = enhancer.upscale_x4(image)
        target = (round(width * factor), round(height * factor))
        image = upscaled if upscaled.size == target else upscaled.resize(target, Image.Resampling.LANCZOS)
        result.upscaled_by = round(factor, 3)
    result.image = image
    return result


def find_all_text(finder: TextFinder, image: Image.Image) -> list[TextBox]:
    """OCR the whole image and, if it is large, overlapping tiles of it too.

    The OCR model sees a downscaled frame, which loses small text such as a
    corner watermark in a big photo; tiles catch it. Boxes found twice are
    reported once.
    """
    found = list(finder.find_text(image))
    width, height = image.size
    if max(width, height) > OCR_FULL_FRAME_SIDE:
        for left, top, right, bottom in _tiles(width, height):
            for item in finder.find_text(image.crop((left, top, right, bottom))):
                x0, y0, x1, y1 = item.box
                found.append(TextBox(item.text, (x0 + left, y0 + top, x1 + left, y1 + top)))
    return _dedupe(found)


def _tiles(width: int, height: int) -> Iterable[Box]:
    cols = min(OCR_MAX_TILES_PER_SIDE, math.ceil(width / (OCR_FULL_FRAME_SIDE / 2)))
    rows = min(OCR_MAX_TILES_PER_SIDE, math.ceil(height / (OCR_FULL_FRAME_SIDE / 2)))
    tile_w, tile_h = math.ceil(width / cols), math.ceil(height / rows)
    for row in range(rows):
        for col in range(cols):
            yield (
                max(0, col * tile_w - OCR_TILE_OVERLAP),
                max(0, row * tile_h - OCR_TILE_OVERLAP),
                min(width, (col + 1) * tile_w + OCR_TILE_OVERLAP),
                min(height, (row + 1) * tile_h + OCR_TILE_OVERLAP),
            )


def _dedupe(found: list[TextBox]) -> list[TextBox]:
    """Drop boxes mostly covered by a larger box already kept, and slivers.

    The OCR occasionally "reads" a few pixels of texture as a character; a box
    thinner than ``MIN_TEXT_SIDE`` is never real overlaid text.
    """
    kept: list[TextBox] = []
    for item in sorted(found, key=lambda t: -_area(t.box)):
        x0, y0, x1, y1 = item.box
        if min(x1 - x0, y1 - y0) < MIN_TEXT_SIDE:
            continue
        if all(_overlap(item.box, other.box) < 0.6 * max(_area(item.box), 1e-6) for other in kept):
            kept.append(item)
    return sorted(kept, key=lambda t: (t.box[1], t.box[0]))


def find_caption_bar(lum: Array, box: Box) -> tuple[int, int] | None:
    """Rows ``(top, bottom)`` of a semi-transparent dark bar behind the text in ``box``.

    Snapchat and similar apps draw a caption on a full-width band that darkens
    the picture. Along the band's top and bottom edges nearly every column away
    from the text gets abruptly darker or lighter, which ordinary picture
    content does not do. Returns None when there is no such band.
    """
    height, width = lum.shape
    x0, y0, x1, y1 = box
    text_h = max(1, y1 - y0)
    columns = np.ones(width, bool)
    columns[max(0, x0 - text_h) : min(width, x1 + text_h)] = False
    if columns.sum() < max(16, width // 10):
        return None
    reach = 2 * text_h
    top = _strongest_edge(lum, columns, range(max(2, y0 - reach), max(2, y0) + 1), entering=True)
    bottom = _strongest_edge(lum, columns, range(y1, min(height - 2, y1 + reach) + 1), entering=False)
    if top is None or bottom is None or bottom - top > max(4 * text_h, height // 4):
        return None
    return top, bottom


def _strongest_edge(lum: Array, columns: Array, rows: range, entering: bool) -> int | None:
    """The row in ``rows`` where most columns step into (or out of) a darker band.

    ``entering`` finds the first row inside the band; otherwise the first row
    after it. A soft edge darkens over two or three rows, all of which pass
    the test, so among them the sharpest step wins.
    """
    best, best_key = None, (0.0, 0.0)
    for row in rows:
        if row < 1 or row >= lum.shape[0]:
            continue
        before, after = lum[row - 1, columns], lum[row, columns]
        outside, inside = (before, after) if entering else (after, before)
        darker = (inside < outside * 0.85) & (outside - inside > 6)
        score = float(darker.mean())
        if score < 0.8:
            continue
        step = float(np.median(1 - inside / np.maximum(outside, 1)))
        if (step, score) > best_key:
            best, best_key = row, (step, score)
    return best


def restore_bar(pixels: Array, top: int, bottom: int, text_columns: Array) -> None:
    """Undo a dark overlay on rows ``[top, bottom)`` of ``pixels`` (HxWx3, 0-255) in place.

    The overlay multiplies the picture by a constant, which is measured per
    colour channel across the bar's edges, away from the text. Rows near an
    edge are skipped, since a soft edge blends the bar into the picture over a
    distance that grows with the image size.
    """
    height = pixels.shape[0]
    columns = ~text_columns
    gap = max(2, (bottom - top) // 8)
    ratios = []
    for outside, inside in (
        (pixels[max(0, top - gap - 3) : max(0, top - gap)], pixels[top + gap : top + gap + 3]),
        (
            pixels[min(height, bottom + gap) : min(height, bottom + gap + 3)],
            pixels[bottom - gap - 3 : bottom - gap],
        ),
    ):
        if len(outside) == 0 or len(inside) == 0 or bottom - top < 2 * gap + 6:
            continue
        out, ins = outside[:, columns].mean(axis=0), inside[:, columns].mean(axis=0)
        usable = out.min(axis=1) > 10
        if usable.any():
            ratios.append(np.median(ins[usable] / out[usable], axis=0))
    if not ratios:
        return
    factor = np.clip(np.mean(ratios, axis=0), 0.2, 1.0)
    pixels[top:bottom] = np.clip(pixels[top:bottom] / factor, 0, 255)


def _text_columns(found: list[TextBox], width: int) -> Array:
    columns = np.zeros(width, bool)
    for item in found:
        x0, _, x1, _ = _clip(item.box, width, 1)
        pad = max(4, (x1 - x0) // 10)
        columns[max(0, x0 - pad) : min(width, x1 + pad)] = True
    return columns


def _inpaint_holes(image: Image.Image, holes: list[Box], enhancer: Enhancer) -> Image.Image:
    """Inpaint each cluster of holes in a crop with some surrounding context.

    Crops keep VRAM use flat however large the photo is. Only the masked
    pixels are taken from the model's output; the rest stays untouched.
    """
    width, height = image.size
    mask = Image.new("L", image.size, 0)
    for box in holes:
        mask.paste(255, box)
    result = image.copy()
    for crop in _merge([_context(box, width, height) for box in holes]):
        part, part_mask = image.crop(crop), mask.crop(crop)
        shrink = max(part.size) / LAMA_MAX_SIDE
        if shrink > 1:
            small = (max(8, round(part.width / shrink)), max(8, round(part.height / shrink)))
            # MaxFilter first so thin strips of mask survive the downscale.
            grown = part_mask.filter(ImageFilter.MaxFilter(_odd(math.ceil(shrink) * 2 + 1)))
            filled = enhancer.inpaint(
                part.resize(small, Image.Resampling.LANCZOS),
                grown.resize(small, Image.Resampling.BILINEAR).point(lambda v: 255 if v else 0),
            ).resize(part.size, Image.Resampling.LANCZOS)
        else:
            filled = enhancer.inpaint(part, part_mask)
        result.paste(filled, crop[:2], part_mask)
    return result


def _context(box: Box, width: int, height: int) -> Box:
    x0, y0, x1, y1 = box
    margin = max(32, 2 * min(x1 - x0, y1 - y0))
    return max(0, x0 - margin), max(0, y0 - margin), min(width, x1 + margin), min(height, y1 + margin)


def _merge(boxes: list[Box]) -> list[Box]:
    """Merge overlapping boxes until none overlap."""
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        out: list[Box] = []
        for box in merged:
            for i, other in enumerate(out):
                if _overlap(box, other) > 0:
                    out[i] = (
                        min(box[0], other[0]),
                        min(box[1], other[1]),
                        max(box[2], other[2]),
                        max(box[3], other[3]),
                    )
                    changed = True
                    break
            else:
                out.append(box)
        merged = out
    return merged


def _pad(box: Box, width: int, height: int) -> Box:
    """Grow a text box to cover outlines, shadows and anti-aliasing."""
    x0, y0, x1, y1 = box
    pad = max(3, round(0.25 * (y1 - y0)))
    return max(0, x0 - pad), max(0, y0 - pad), min(width, x1 + pad), min(height, y1 + pad)


def _clip(box: Sequence[float], width: int, height: int) -> Box:
    x0, y0, x1, y1 = box
    return (
        max(0, min(width, math.floor(x0))),
        max(0, min(height, math.floor(y0))),
        max(0, min(width, math.ceil(x1))),
        max(0, min(height, math.ceil(y1))),
    )


def _region_box(region: Region, width: int, height: int) -> Box:
    left, top, right, bottom = region
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError("each region needs 0 <= left < right <= 1 and 0 <= top < bottom <= 1")
    return _clip((left * width, top * height, right * width, bottom * height), width, height)


def _area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _overlap(a: Sequence[float], b: Sequence[float]) -> float:
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def _odd(n: int) -> int:
    return n if n % 2 else n + 1


def _to_image(array: Array) -> Image.Image:
    """HxWx3 float array in 0-1 -> RGB image."""
    return Image.fromarray((np.clip(array, 0, 1) * 255).round().astype(np.uint8), "RGB")
