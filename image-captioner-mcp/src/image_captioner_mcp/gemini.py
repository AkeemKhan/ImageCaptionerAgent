"""A small client for editing an image with Gemini.

This is the only code that sends an image to Google. It is used by
``CLEANUP_AND_ENHANCE`` when the caller picks the ``gemini`` backend, which the
operator must first enable with ``CAPTIONER_GEMINI_ENABLED`` and a
``GEMINI_API_KEY``.
"""

from __future__ import annotations

import base64
import contextlib
import io
import logging
from typing import Any, Protocol

import httpx
from PIL import Image

logger = logging.getLogger(__name__)

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
# The aspect ratios the image models accept. Output comes back in one of these.
ASPECT_RATIOS: dict[str, float] = {
    "1:1": 1.0,
    "2:3": 2 / 3,
    "3:2": 3 / 2,
    "3:4": 3 / 4,
    "4:3": 4 / 3,
    "4:5": 4 / 5,
    "5:4": 5 / 4,
    "9:16": 9 / 16,
    "16:9": 16 / 9,
    "21:9": 21 / 9,
}
# The output sizes the image models offer, by the length of the longest side.
IMAGE_SIZES: dict[str, int] = {"1K": 1024, "2K": 2048, "4K": 4096}
# A request larger than this is sent as JPEG instead of PNG.
MAX_PNG_BYTES = 12 * 1024 * 1024
MAX_ERROR_CHARS = 300


class GeminiError(RuntimeError):
    """The request failed. The message is safe to show: it never contains the API key."""


class GeminiRefusal(GeminiError):
    """Gemini answered but returned no image, usually because it declined the edit."""


class GeminiEditor(Protocol):
    model: str

    def edit(self, image: Image.Image, prompt: str, aspect_ratio: str, image_size: str) -> Image.Image:
        """Return ``image`` edited as ``prompt`` says, at about ``image_size`` (``1K``, ``2K`` or ``4K``)."""
        ...


def closest_aspect_ratio(width: int, height: int) -> str:
    """The supported aspect ratio nearest to ``width`` x ``height``."""
    ratio = width / height
    return min(ASPECT_RATIOS, key=lambda name: abs(ASPECT_RATIOS[name] / ratio - 1))


def image_size_for(long_side: int) -> str:
    """The smallest size bucket that covers an output whose longest side is ``long_side``."""
    for name, side in IMAGE_SIZES.items():
        if long_side <= side:
            return name
    return "4K"


class HttpGeminiEditor:
    """Gemini's ``generateContent`` REST endpoint, authenticated with an API key."""

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: float = 300.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("a Gemini API key is required")
        self._api_key = api_key
        self.model = model
        self._client = client or httpx.Client(timeout=httpx.Timeout(timeout_seconds, connect=15.0))

    def edit(self, image: Image.Image, prompt: str, aspect_ratio: str, image_size: str) -> Image.Image:
        image_config: dict[str, str] = {"aspectRatio": aspect_ratio}
        # 2.5 Flash Image has a fixed output size and does not take imageSize.
        if "2.5" not in self.model:
            image_config["imageSize"] = image_size
        body = {
            "contents": [{"parts": [{"text": prompt}, {"inline_data": _encode(image)}]}],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"], "imageConfig": image_config},
        }
        payload = self._post(body)
        return _extract_image(payload)

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{API_ROOT}/models/{self.model.removeprefix('models/')}:generateContent"
        try:
            response = self._client.post(url, headers={"x-goog-api-key": self._api_key}, json=body)
        except httpx.TimeoutException as exc:
            raise GeminiError("Gemini did not answer in time") from exc
        except httpx.HTTPError as exc:
            raise GeminiError(f"could not reach Gemini ({type(exc).__name__})") from exc
        if response.status_code != 200:
            raise GeminiError(self._http_error(response))
        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            raise GeminiError("Gemini returned something that is not JSON") from exc
        return payload

    def _http_error(self, response: httpx.Response) -> str:
        status = response.status_code
        detail = ""
        with contextlib.suppress(ValueError, AttributeError):
            detail = str(response.json().get("error", {}).get("message", ""))
        detail = detail.replace(self._api_key, "<key>")[:MAX_ERROR_CHARS]
        if status in (401, 403):
            return f"Gemini rejected the API key or its permissions ({status}). {detail}".strip()
        if status == 402:
            return f"Gemini reports no credits left on the project. {detail}".strip()
        if status == 429:
            return f"Gemini quota or rate limit reached. {detail}".strip()
        if status >= 500:
            return f"Gemini had a server error ({status})."
        return f"Gemini rejected the request ({status}). {detail}".strip()


def _encode(image: Image.Image) -> dict[str, str]:
    """The image as an inline-data part: PNG, or JPEG when the PNG is too large to send."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "PNG")
    mime = "image/png"
    if buffer.tell() > MAX_PNG_BYTES:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, "JPEG", quality=95)
        mime = "image/jpeg"
    return {"mime_type": mime, "data": base64.b64encode(buffer.getvalue()).decode("ascii")}


def _extract_image(payload: dict[str, Any]) -> Image.Image:
    """The final image in a reply.

    Some models send intermediate "thought" images before the answer; those
    are skipped, and when several images remain the last one is the result.
    """
    candidates = payload.get("candidates") or []
    if not candidates:
        reason = (payload.get("promptFeedback") or {}).get("blockReason")
        raise GeminiRefusal(f"Gemini blocked the request{f' ({reason})' if reason else ''}")
    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    images: list[Image.Image] = []
    texts: list[str] = []
    for part in parts:
        if part.get("thought"):
            continue
        data = part.get("inlineData") or part.get("inline_data")
        if data and data.get("data"):
            try:
                with Image.open(io.BytesIO(base64.b64decode(data["data"]))) as decoded:
                    decoded.load()
                    images.append(decoded.convert("RGB"))
            except (ValueError, OSError) as exc:
                raise GeminiError("Gemini returned an image that could not be read") from exc
        elif part.get("text"):
            texts.append(str(part["text"]).strip())
    if images:
        return images[-1]
    said = " ".join(texts)[:MAX_ERROR_CHARS]
    reason = candidate.get("finishReason")
    detail = said or (f"finish reason {reason}" if reason and reason != "STOP" else "no image in the reply")
    raise GeminiRefusal(f"Gemini did not return an image: {detail}")
