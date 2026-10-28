"""Which backend cleans up and upscales images: this machine, or Gemini.

Gemini sends the image to Google, so the user must choose it. The choice is
enforced in code (see ``CaptionAgent``): a cleanup call is not run until the
user has answered the question these options belong to.
"""

from __future__ import annotations

import json
from typing import Literal

CleanupBackend = Literal["local", "gemini"]

LOCAL_OPTION = "Local (images stay on this machine)"
GEMINI_OPTION = (
    "Gemini (removes overlays, restores and upscales; images are sent to Google, billed per image)"
)
CLEANUP_OPTIONS = [LOCAL_OPTION, GEMINI_OPTION]


def parse_cleanup_choice(answer: str) -> CleanupBackend | None:
    """Read the user's reply to the local-or-Gemini question.

    Accepts a button label or free text. Returns ``None`` if the reply names
    neither backend, or both, so the question is asked again.
    """
    text = answer.lower()
    wants_gemini = "gemini" in text or "google" in text
    wants_local = "local" in text or "this machine" in text
    if wants_gemini == wants_local:
        return None
    return "gemini" if wants_gemini else "local"


def gate_message(tool: str, ask_tool: str) -> str:
    """The tool result Claude gets when it calls a cleanup before the user has chosen."""
    return (
        f"NOT RUN. {tool} can use Gemini, which sends images to Google, so the user must choose "
        f"the backend first. Ask them with {ask_tool} using exactly these options: "
        f"{json.dumps(CLEANUP_OPTIONS)}. Then call {tool} again; the app applies their choice. "
        "Do not choose for them."
    )
