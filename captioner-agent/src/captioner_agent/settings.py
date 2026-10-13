"""Application settings, loaded from ``config/settings.yaml``."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

SETTINGS_ENV = "CAPTIONER_AGENT_SETTINGS"
DEFAULT_SETTINGS_PATH = Path("config") / "settings.yaml"


class McpSettings(BaseModel):
    transport: Literal["stdio", "http"] = "stdio"
    command: str = ""
    args: list[str] = Field(default_factory=list)
    url: str = "http://127.0.0.1:8765/mcp"
    env: dict[str, str] = Field(default_factory=dict)


class Settings(BaseModel):
    model: str = "claude-opus-5-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    max_tokens: int = 32000
    refusal_fallbacks: bool = True

    host: str = "127.0.0.1"
    port: int = 8000
    max_upload_mb: int = 50

    data_dir: Path = Path("data")
    criteria_file: Path = Path("config") / "caption_criteria.md"
    # Who captions an image that has no PNG info: Claude, or the local model.
    vision_backend: Literal["claude", "local"] = "claude"
    vision_concurrency: int = Field(default=4, ge=1, le=16)
    caption_style: Literal["short", "detailed", "more_detailed"] = "detailed"

    mcp: McpSettings = Field(default_factory=McpSettings)

    @property
    def sessions_dir(self) -> Path:
        """Parent of every session folder; also the MCP server's allowed root."""
        return self.data_dir / "sessions"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "agent.db"


def load_settings(path: Path | None = None) -> Settings:
    """Load settings, resolving relative paths against the settings file's project folder.

    With no ``path``, uses ``$CAPTIONER_AGENT_SETTINGS`` or ``config/settings.yaml``
    in the working directory, and falls back to defaults if that file is absent.
    """
    explicit = path or (Path(env) if (env := os.environ.get(SETTINGS_ENV)) else None)
    settings_path = (explicit or DEFAULT_SETTINGS_PATH).resolve()
    if settings_path.is_file():
        raw = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
        base_dir = settings_path.parent.parent
    elif explicit is not None:
        raise FileNotFoundError(f"settings file not found: {settings_path}")
    else:
        raw = {}
        base_dir = Path.cwd()

    settings = Settings.model_validate(raw)
    settings.data_dir = (base_dir / settings.data_dir).resolve()
    settings.criteria_file = (base_dir / settings.criteria_file).resolve()
    return settings
