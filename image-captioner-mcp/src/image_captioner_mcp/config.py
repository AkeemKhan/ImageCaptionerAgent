"""Server configuration, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_PREFIX = "CAPTIONER_"


def _env(name: str, default: str) -> str:
    return os.environ.get(ENV_PREFIX + name, default)


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


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
            criteria_file=Path(criteria).expanduser() if (criteria := _env("CRITERIA_FILE", "")) else None,
        )
