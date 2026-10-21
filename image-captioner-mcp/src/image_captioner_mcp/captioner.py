"""Captioning backends behind one small interface."""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable
from typing import Any, Literal, NamedTuple, Protocol, TypeVar, get_args

from PIL import Image

from .config import ServerConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")

CaptionStyle = Literal["short", "detailed", "more_detailed"]
CAPTION_STYLES: tuple[str, ...] = get_args(CaptionStyle)

_FLORENCE_OCR_TASK = "<OCR_WITH_REGION>"

_FLORENCE_TASKS: dict[str, str] = {
    "short": "<CAPTION>",
    "detailed": "<DETAILED_CAPTION>",
    "more_detailed": "<MORE_DETAILED_CAPTION>",
}

# BLIP writes one short sentence by default, so the longer styles ask for a
# minimum length. BLIP repeats itself at length, so those styles also forbid
# repeating any three-word run.
_BLIP_GENERATION: dict[str, dict[str, Any]] = {
    "short": {"max_new_tokens": 30, "num_beams": 3},
    "detailed": {
        "min_new_tokens": 20,
        "max_new_tokens": 60,
        "num_beams": 5,
        "repetition_penalty": 1.4,
        "no_repeat_ngram_size": 3,
    },
    "more_detailed": {
        "min_new_tokens": 35,
        "max_new_tokens": 90,
        "num_beams": 5,
        "repetition_penalty": 1.4,
        "no_repeat_ngram_size": 3,
    },
}

# BLIP-large often opens a caption with the nonsense token "arafed".
_BLIP_ARTIFACT = re.compile(r"\barafed\b\s*", re.IGNORECASE)
# ...and with "there is a" / "there are two", a habit of its training captions.
_THERE_IS = re.compile(r"^there (?:is|are)\s+", re.IGNORECASE)


def clean_blip_text(text: str) -> str:
    """Remove BLIP's known artefacts and tidy whitespace.

    Drops "arafed" and a leading "there is/are", so "there is a red truck on a
    road" becomes "a red truck on a road". A "there is" later in the caption
    is real wording and is left alone.
    """
    text = " ".join(_BLIP_ARTIFACT.sub("", text).split())
    return _THERE_IS.sub("", text)


def pick_device(torch: Any, config: ServerConfig) -> str:
    """Choose ``cuda`` or ``cpu`` for a model about to load.

    CUDA is used only when enough VRAM is free; ``CAPTIONER_DEVICE=cuda`` turns
    a shortfall into an error instead of a quiet fall back to the CPU.
    """
    wanted = config.device
    if wanted == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        if wanted == "cuda":
            raise RuntimeError("CAPTIONER_DEVICE=cuda but CUDA is not available")
        return "cpu"
    free_bytes, _total = torch.cuda.mem_get_info()
    free_mb = free_bytes // (1024 * 1024)
    if free_mb < config.min_free_vram_mb:
        message = f"only {free_mb} MB of VRAM free, below the {config.min_free_vram_mb} MB minimum"
        if wanted == "cuda":
            raise RuntimeError(f"CAPTIONER_DEVICE=cuda but {message}")
        logger.warning("Using CPU: %s", message)
        return "cpu"
    return "cuda"


class TextBox(NamedTuple):
    """Text found in an image and its bounding box ``(x0, y0, x1, y1)`` in pixels."""

    text: str
    box: tuple[float, float, float, float]


class Captioner(Protocol):
    """A model that turns an RGB image into a caption."""

    name: str

    @property
    def device(self) -> str: ...

    def load(self) -> None: ...

    def caption(self, image: Image.Image, style: CaptionStyle = "detailed") -> str: ...

    def unload(self) -> None: ...


class StubCaptioner:
    """Deterministic backend with no model, for tests and wiring checks."""

    name = "stub"

    @property
    def device(self) -> str:
        return "none"

    def load(self) -> None:
        return None

    def caption(self, image: Image.Image, style: CaptionStyle = "detailed") -> str:
        return f"a {style} caption of a {image.width}x{image.height} image"

    def unload(self) -> None:
        return None


class TorchCaptioner:
    """Loading and inference shared by the ``transformers`` based captioners.

    GPU safety:

    * the model loads once and inference is serialised by a lock, so VRAM use
      is bounded by a single forward pass;
    * CUDA is only used when enough VRAM is free at load time;
    * a CUDA out-of-memory error frees the cache and moves the model to the CPU
      for the rest of the process instead of failing the request.

    Subclasses provide ``_build`` and ``_generate``.
    """

    name = "torch"

    def __init__(self, config: ServerConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._device = "unloaded"

    @property
    def device(self) -> str:
        return self._device

    def load(self) -> None:
        with self._lock:
            self._load_locked()

    def _build(self, dtype: Any) -> tuple[Any, Any]:
        """Return ``(processor, model)``, with weights loaded in ``dtype``."""
        raise NotImplementedError

    def _generate(self, image: Image.Image, style: str) -> str:
        raise NotImplementedError

    def _load_locked(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                f"The {self.name} backend needs torch and transformers. "
                "Install them with: pip install 'image-captioner-mcp[florence]'"
            ) from exc

        self._torch = torch
        device = pick_device(torch, self._config)
        dtype = torch.float16 if device == "cuda" else torch.float32
        logger.info("Loading %s on %s", self.name, device)
        processor, model = self._build(dtype)
        self._processor = processor
        self._model = model.to(device).eval()
        self._device = device
        logger.info("%s ready on %s", self.name, device)

    @property
    def _dtype(self) -> Any:
        return self._torch.float16 if self._device == "cuda" else self._torch.float32

    def caption(self, image: Image.Image, style: CaptionStyle = "detailed") -> str:
        if style not in CAPTION_STYLES:
            raise ValueError(f"style must be one of {', '.join(CAPTION_STYLES)}")
        return self._run(self._generate, image, style)

    def _run(self, fn: Callable[..., T], *args: Any) -> T:
        """Call ``fn`` with the model loaded, holding the lock, retrying on the CPU after a CUDA OOM."""
        with self._lock:
            self._load_locked()
            torch = self._torch
            try:
                return fn(*args)
            except torch.cuda.OutOfMemoryError:
                if self._device != "cuda":
                    raise
                logger.warning("CUDA out of memory; moving the model to CPU")
                self._model = self._model.to("cpu").float()
                self._device = "cpu"
                torch.cuda.empty_cache()
                return fn(*args)

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._processor = None
            if self._torch is not None and self._device == "cuda":
                self._torch.cuda.empty_cache()
            self._device = "unloaded"


class Florence2Captioner(TorchCaptioner):
    """Microsoft Florence-2 running locally through ``transformers``."""

    name = "florence2"

    def _build(self, dtype: Any) -> tuple[Any, Any]:
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "The Florence-2 backend needs transformers. "
                "Install it with: pip install 'image-captioner-mcp[florence]'"
            ) from exc
        model_id = self._config.model_id
        # trust_remote_code runs Python shipped in the model repository; pin
        # CAPTIONER_MODEL_REVISION to a reviewed commit to keep that code fixed.
        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self._config.model_revision:
            kwargs["revision"] = self._config.model_revision
        processor = AutoProcessor.from_pretrained(model_id, **kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, **kwargs)
        return processor, model

    def find_text(self, image: Image.Image) -> list[TextBox]:
        """Every piece of text Florence-2 reads in ``image``, with its box in pixels."""
        parsed = self._run(self._task, image, _FLORENCE_OCR_TASK)
        found: list[TextBox] = []
        for quad, label in zip(parsed.get("quad_boxes", []), parsed.get("labels", []), strict=False):
            xs, ys = quad[0::2], quad[1::2]
            text = str(label).replace("</s>", "").replace("<s>", "").strip()
            found.append(TextBox(text, (min(xs), min(ys), max(xs), max(ys))))
        return found

    def _generate(self, image: Image.Image, style: str) -> str:
        task = _FLORENCE_TASKS[style]
        return str(self._task(image, task)[task]).strip()

    def _task(self, image: Image.Image, task: str) -> Any:
        torch = self._torch
        inputs = self._processor(text=task, images=image, return_tensors="pt")
        with torch.inference_mode():
            generated = self._model.generate(
                input_ids=inputs["input_ids"].to(self._device),
                pixel_values=inputs["pixel_values"].to(self._device, self._dtype),
                max_new_tokens=self._config.max_new_tokens,
                num_beams=3,
                do_sample=False,
            )
        decoded = self._processor.batch_decode(generated, skip_special_tokens=False)[0]
        parsed = self._processor.post_process_generation(
            decoded, task=task, image_size=(image.width, image.height)
        )
        return parsed[task] if task == _FLORENCE_OCR_TASK else parsed


class BlipCaptioner(TorchCaptioner):
    """Salesforce BLIP image captioning through ``transformers``.

    Unlike Florence-2, BLIP needs no ``trust_remote_code``: the model classes
    ship with ``transformers`` and the weights are plain tensors.
    """

    name = "blip"

    def _build(self, dtype: Any) -> tuple[Any, Any]:
        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError as exc:
            raise RuntimeError(
                "The BLIP backend needs transformers. "
                "Install it with: pip install 'image-captioner-mcp[florence]'"
            ) from exc
        model_id = self._config.blip_model_id
        kwargs: dict[str, Any] = {}
        if self._config.blip_model_revision:
            kwargs["revision"] = self._config.blip_model_revision
        processor = BlipProcessor.from_pretrained(model_id, **kwargs)
        model = BlipForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype, **kwargs)
        return processor, model

    def _generate(self, image: Image.Image, style: str) -> str:
        torch = self._torch
        inputs = self._processor(images=image, return_tensors="pt")
        with torch.inference_mode():
            generated = self._model.generate(
                pixel_values=inputs["pixel_values"].to(self._device, self._dtype),
                do_sample=False,
                **_BLIP_GENERATION[style],
            )
        return clean_blip_text(self._processor.decode(generated[0], skip_special_tokens=True))


def create_captioner(config: ServerConfig) -> Captioner:
    """The captioner behind the ``caption_image`` tool."""
    if config.backend == "florence2":
        return Florence2Captioner(config)
    if config.backend == "stub":
        return StubCaptioner()
    raise ValueError(f"unknown CAPTIONER_BACKEND: {config.backend!r} (use florence2 or stub)")


def create_blip_captioner(config: ServerConfig) -> Captioner:
    """The captioner behind the ``BLIP_CAPTIONER`` tool. Its model loads on first use."""
    return StubCaptioner() if config.backend == "stub" else BlipCaptioner(config)
