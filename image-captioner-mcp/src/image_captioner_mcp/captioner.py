"""Captioning backends behind one small interface."""

from __future__ import annotations

import logging
import threading
from typing import Any, Literal, Protocol, get_args

from PIL import Image

from .config import ServerConfig

logger = logging.getLogger(__name__)

CaptionStyle = Literal["short", "detailed", "more_detailed"]
CAPTION_STYLES: tuple[str, ...] = get_args(CaptionStyle)

_FLORENCE_TASKS: dict[str, str] = {
    "short": "<CAPTION>",
    "detailed": "<DETAILED_CAPTION>",
    "more_detailed": "<MORE_DETAILED_CAPTION>",
}


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


class Florence2Captioner:
    """Microsoft Florence-2 running locally through ``transformers``.

    GPU safety:

    * the model loads once and inference is serialised by a lock, so VRAM use
      is bounded by a single forward pass;
    * CUDA is only used when enough VRAM is free at load time;
    * a CUDA out-of-memory error frees the cache and moves the model to the CPU
      for the rest of the process instead of failing the request.
    """

    name = "florence2"

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

    def _load_locked(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "The Florence-2 backend needs torch and transformers. "
                "Install them with: pip install 'image-captioner-mcp[florence]'"
            ) from exc

        self._torch = torch
        device = self._pick_device()
        dtype = torch.float16 if device == "cuda" else torch.float32
        logger.info("Loading %s on %s", self._config.model_id, device)
        # trust_remote_code runs Python shipped in the model repository; pin
        # CAPTIONER_MODEL_REVISION to a reviewed commit to keep that code fixed.
        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self._config.model_revision:
            kwargs["revision"] = self._config.model_revision
        self._processor = AutoProcessor.from_pretrained(self._config.model_id, **kwargs)
        model = AutoModelForCausalLM.from_pretrained(self._config.model_id, torch_dtype=dtype, **kwargs)
        self._model = model.to(device).eval()
        self._device = device
        logger.info("Model ready on %s", device)

    def _pick_device(self) -> str:
        torch = self._torch
        wanted = self._config.device
        if wanted == "cpu":
            return "cpu"
        if not torch.cuda.is_available():
            if wanted == "cuda":
                raise RuntimeError("CAPTIONER_DEVICE=cuda but CUDA is not available")
            return "cpu"
        free_bytes, _total = torch.cuda.mem_get_info()
        free_mb = free_bytes // (1024 * 1024)
        if free_mb < self._config.min_free_vram_mb:
            message = f"only {free_mb} MB of VRAM free, below the {self._config.min_free_vram_mb} MB minimum"
            if wanted == "cuda":
                raise RuntimeError(f"CAPTIONER_DEVICE=cuda but {message}")
            logger.warning("Using CPU: %s", message)
            return "cpu"
        return "cuda"

    def caption(self, image: Image.Image, style: CaptionStyle = "detailed") -> str:
        if style not in _FLORENCE_TASKS:
            raise ValueError(f"style must be one of {', '.join(CAPTION_STYLES)}")
        task = _FLORENCE_TASKS[style]
        with self._lock:
            self._load_locked()
            torch = self._torch
            try:
                return self._generate(image, task)
            except torch.cuda.OutOfMemoryError:
                if self._device != "cuda":
                    raise
                logger.warning("CUDA out of memory; moving the model to CPU")
                self._model = self._model.to("cpu").float()
                self._device = "cpu"
                torch.cuda.empty_cache()
                return self._generate(image, task)

    def _generate(self, image: Image.Image, task: str) -> str:
        torch = self._torch
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        inputs = self._processor(text=task, images=image, return_tensors="pt")
        with torch.inference_mode():
            generated = self._model.generate(
                input_ids=inputs["input_ids"].to(self._device),
                pixel_values=inputs["pixel_values"].to(self._device, dtype),
                max_new_tokens=self._config.max_new_tokens,
                num_beams=3,
                do_sample=False,
            )
        decoded = self._processor.batch_decode(generated, skip_special_tokens=False)[0]
        parsed = self._processor.post_process_generation(
            decoded, task=task, image_size=(image.width, image.height)
        )
        return str(parsed[task]).strip()

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._processor = None
            if self._torch is not None and self._device == "cuda":
                self._torch.cuda.empty_cache()
            self._device = "unloaded"


def create_captioner(config: ServerConfig) -> Captioner:
    if config.backend == "florence2":
        return Florence2Captioner(config)
    if config.backend == "stub":
        return StubCaptioner()
    raise ValueError(f"unknown CAPTIONER_BACKEND: {config.backend!r} (use florence2 or stub)")
