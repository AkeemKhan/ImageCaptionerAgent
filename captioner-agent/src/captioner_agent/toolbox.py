"""The agent's tools: a connection to the image-captioner MCP server."""

from __future__ import annotations

import logging
import os
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Protocol

from mcp import Client, StdioServerParameters

from .settings import GEMINI_KEY_ENV, Settings

logger = logging.getLogger(__name__)

# The first caption call can wait on a model download and load.
TOOL_TIMEOUT_SECONDS = 900.0
# Credentials the captioning server has no use for and must not inherit. The Gemini key
# is the exception, and is handed over explicitly below when Gemini cleanup is on.
_WITHHELD_ENV_PREFIXES = ("ANTHROPIC_", "GEMINI_", "GOOGLE_API")


@dataclass(frozen=True)
class ToolResult:
    text: str
    is_error: bool = False


class Toolbox(Protocol):
    """A set of tools Claude can call."""

    def definitions(self) -> list[dict[str, Any]]:
        """Tool definitions in Claude API format."""
        ...

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult: ...


class McpToolbox:
    """Tools served by an MCP server, over stdio (spawned here) or streamable HTTP."""

    def __init__(self, settings: Settings, target: Any = None) -> None:
        """``target`` overrides the configured server with anything ``mcp.Client``
        accepts, such as an in-process server instance in tests."""
        self._settings = settings
        self._target_override = target
        self._stack = AsyncExitStack()
        self._client: Client | None = None
        self._definitions: list[dict[str, Any]] = []

    async def connect(self) -> None:
        target = self._target_override if self._target_override is not None else self._target()
        self._client = await self._stack.enter_async_context(Client(target))
        listed = await self._client.list_tools()
        self._definitions = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.input_schema,
                # Inputs stream as generated; the agent validates them before use.
                "eager_input_streaming": True,
            }
            for tool in listed.tools
        ]
        logger.info("Connected to MCP server with tools: %s", [d["name"] for d in self._definitions])

    async def close(self) -> None:
        await self._stack.aclose()
        self._client = None

    def _target(self) -> StdioServerParameters | str:
        mcp = self._settings.mcp
        if mcp.transport == "http":
            return mcp.url
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.upper().startswith(_WITHHELD_ENV_PREFIXES)
        }
        # Load into VRAM at startup only the model that captions images without
        # PNG info. The other loads on first use, if the user ever asks for it.
        backend = self._settings.vision_backend
        env["CAPTIONER_PRELOAD"] = str(backend == "local").lower()
        env["CAPTIONER_PRELOAD_BLIP"] = str(backend in ("ask", "blip")).lower()
        if self._settings.gemini_cleanup_ready:
            env.update(
                {
                    GEMINI_KEY_ENV: os.environ[GEMINI_KEY_ENV].strip(),
                    "CAPTIONER_GEMINI_ENABLED": "true",
                    "CAPTIONER_GEMINI_MODEL": self._settings.gemini_model,
                    "CAPTIONER_GEMINI_UPSCALE_FALLBACK": self._settings.gemini_upscale_fallback,
                    "CAPTIONER_GEMINI_IMAGE_SIZE": self._settings.gemini_image_size,
                    "CAPTIONER_GEMINI_PROMPT_FILE": str(self._settings.gemini_prompt_file),
                }
            )
        env.update(mcp.env)
        env["CAPTIONER_ALLOWED_ROOT"] = str(self._settings.sessions_dir)
        env["CAPTIONER_CRITERIA_FILE"] = str(self._settings.criteria_file)
        if mcp.command:
            return StdioServerParameters(command=mcp.command, args=mcp.args, env=env)
        return StdioServerParameters(
            command=sys.executable, args=["-m", "image_captioner_mcp.server", *mcp.args], env=env
        )

    def definitions(self) -> list[dict[str, Any]]:
        return self._definitions

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if self._client is None:
            return ToolResult("The captioning server is not connected.", is_error=True)
        try:
            result = await self._client.call_tool(name, arguments, read_timeout_seconds=TOOL_TIMEOUT_SECONDS)
        except Exception:
            logger.exception("MCP call %s failed", name)
            return ToolResult("The captioning server did not respond. It may have stopped.", is_error=True)
        text = "\n".join(block.text for block in result.content if block.type == "text")
        return ToolResult(text or "(no output)", is_error=bool(result.is_error))
