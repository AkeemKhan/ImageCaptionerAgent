"""Server configuration, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ENV_PREFIX = "CAPTIONER_"

DEFAULT_LAMA = "fashn-ai/LaMa"
# LaMa is a TorchScript file from a community repository, so the default is
# pinned to the commit it was tested with.
DEFAULT_LAMA_REVISION = "da55aaf971ce271b42eddca79831e0f646c7711e"
DEFAULT_UPSCALER = "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"
# The Gemini model that did text removal and a 2K upscale in testing; 2.5 Flash Image refused
# to remove watermarks. Image-output models change often, so it is a setting.
DEFAULT_GEMINI_MODEL = "gemini-3-pro-image-preview"


def _env(name: str, default: str) -> str:
    return os.environ.get(ENV_PREFIX + name, default)


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _gemini_fallback(value: str) -> str:
    value = value.strip().lower()
    if value not in {"none", "local", "resize"}:
        raise ValueError(f"{ENV_PREFIX}GEMINI_UPSCALE_FALLBACK must be none, local or resize, got {value!r}")
    return value


def _gemini_size(value: str) -> str:
    value = value.strip().upper()
    if value not in {"1K", "2K", "4K"}:
        raise ValueError(f"{ENV_PREFIX}GEMINI_IMAGE_SIZE must be 1K, 2K or 4K, got {value!r}")
    return value


@dataclass(frozen=True)
class ServerConfig:
    """Settings for the captioning server.

    Every file path a tool receives must resolve inside ``allowed_root``.
    """

    allowed_root: Path
    backend: str = "florence2"
    model_id: str = "microsoft/Florence-2-large"
    model_revision: str | None = None
    device: str = "auto"
    min_free_vram_mb: int = 3000
    max_image_side: int = 1024
    max_new_tokens: int = 512
    preload: bool = True
    # BLIP is a second, smaller model behind the BLIP_CAPTIONER tool.
    blip_model_id: str = "Salesforce/blip-image-captioning-large"
    blip_model_revision: str | None = None
    preload_blip: bool = False
    # Text removal (LaMa) and upscaling (Swin2SR) for CLEANUP_AND_ENHANCE.
    lama_model_id: str = DEFAULT_LAMA
    lama_model_revision: str | None = DEFAULT_LAMA_REVISION
    upscale_model_id: str = DEFAULT_UPSCALER
    upscale_model_revision: str | None = None
    max_output_side: int = 4096
    # Gemini as an optional cleanup backend. It sends the image to Google, so it
    # is off unless the operator turns it on and supplies a key.
    gemini_enabled: bool = False
    gemini_api_key: str | None = field(default=None, repr=False)
    gemini_model: str = DEFAULT_GEMINI_MODEL
    # A 4K edit can take several minutes.
    gemini_timeout_seconds: float = 600.0
    # When Gemini returns a smaller image than asked for: keep its size ("none", sharpest),
    # upscale it with the local model ("local") or just resize it ("resize").
    gemini_upscale_fallback: str = "none"
    # The size Gemini is asked for when upscaling: 1K, 2K or 4K. 2K: quality comes from the
    # prompt, and 4K is slower and costs more.
    gemini_image_size: str = "2K"
    # The base prompt, editable without touching code; the built-in one if unset or empty.
    gemini_prompt_file: Path | None = None
    criteria_file: Path | None = None

    @classmethod
    def from_env(cls) -> ServerConfig:
        root = Path(_env("ALLOWED_ROOT", str(Path.cwd()))).expanduser().resolve()
        device = _env("DEVICE", "auto").lower()
        if device not in {"auto", "cuda", "cpu"}:
            raise ValueError(f"{ENV_PREFIX}DEVICE must be auto, cuda or cpu, got {device!r}")
        return cls(
            allowed_root=root,
            backend=_env("BACKEND", "florence2").lower(),
            model_id=_env("MODEL_ID", "microsoft/Florence-2-large"),
            model_revision=_env("MODEL_REVISION", "") or None,
            device=device,
            min_free_vram_mb=int(_env("MIN_FREE_VRAM_MB", "3000")),
            max_image_side=int(_env("MAX_IMAGE_SIDE", "1024")),
            max_new_tokens=int(_env("MAX_NEW_TOKENS", "512")),
            preload=_env_bool("PRELOAD", True),
            blip_model_id=_env("BLIP_MODEL_ID", "Salesforce/blip-image-captioning-large"),
            blip_model_revision=_env("BLIP_MODEL_REVISION", "") or None,
            preload_blip=_env_bool("PRELOAD_BLIP", False),
            lama_model_id=_env("LAMA_MODEL_ID", DEFAULT_LAMA),
            lama_model_revision=_env("LAMA_MODEL_REVISION", DEFAULT_LAMA_REVISION) or None,
            upscale_model_id=_env("UPSCALE_MODEL_ID", DEFAULT_UPSCALER),
            upscale_model_revision=_env("UPSCALE_MODEL_REVISION", "") or None,
            max_output_side=int(_env("MAX_OUTPUT_SIDE", "4096")),
            gemini_enabled=_env_bool("GEMINI_ENABLED", False),
            # The key keeps Google's own variable name, so one .env serves every tool.
            gemini_api_key=os.environ.get("GEMINI_API_KEY", "").strip() or None,
            gemini_model=_env("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            gemini_timeout_seconds=float(_env("GEMINI_TIMEOUT_SECONDS", "600")),
            gemini_upscale_fallback=_gemini_fallback(_env("GEMINI_UPSCALE_FALLBACK", "none")),
            gemini_image_size=_gemini_size(_env("GEMINI_IMAGE_SIZE", "2K")),
            gemini_prompt_file=Path(prompt).expanduser()
            if (prompt := _env("GEMINI_PROMPT_FILE", ""))
            else None,
            criteria_file=Path(criteria).expanduser() if (criteria := _env("CRITERIA_FILE", "")) else None,
        )
