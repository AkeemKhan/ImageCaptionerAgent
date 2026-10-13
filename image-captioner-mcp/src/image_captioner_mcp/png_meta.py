"""Extract the generation prompt that image generators embed in their output.

Supported sources:

* AUTOMATIC1111 / Forge / SD.Next: ``parameters`` text chunk (and EXIF UserComment
  for JPEG/WebP).
* ComfyUI: ``prompt`` text chunk holding the executed node graph as JSON.
* NovelAI: ``Description`` and ``Comment`` (JSON) text chunks.
* InvokeAI: ``invokeai_metadata`` JSON text chunk.
* Fooocus: ``parameters`` text chunk holding JSON.

Anything else is returned untouched in ``other_text`` so a caller can inspect it.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

MAX_OTHER_TEXT_CHARS = 2000
_EXIF_IFD = 0x8769
_EXIF_USER_COMMENT = 0x9286
_COMFY_MAX_DEPTH = 12
_COMFY_TEXT_KEYS = ("text", "text_g", "text_l", "string", "prompt", "value")

_A1111_SETTINGS = re.compile(r"(?:^|\n)(Steps: \d+.*)\Z", re.DOTALL)
_A1111_NEGATIVE = re.compile(r"(?:^|\n)Negative prompt: ?")


@dataclass
class ImageMetadata:
    """Generation metadata found in an image file."""

    has_prompt: bool = False
    generator: str | None = None
    prompt: str | None = None
    negative_prompt: str | None = None
    settings: str | None = None
    other_text: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_metadata(path: Path) -> ImageMetadata:
    """Read embedded generation metadata from ``path``."""
    with Image.open(path) as img:
        chunks = {str(k): v for k, v in getattr(img, "text", {}).items() if isinstance(v, str)}
        user_comment = _exif_user_comment(img)

    parsed = _parse_chunks(chunks)
    if parsed is None and user_comment:
        parsed = parse_a1111(user_comment)
    if parsed is not None and parsed.prompt:
        parsed.has_prompt = True
        return parsed

    other = {k: _truncate(v) for k, v in chunks.items()}
    if user_comment:
        other["exif:UserComment"] = _truncate(user_comment)
    return ImageMetadata(other_text=other)


def _parse_chunks(chunks: dict[str, str]) -> ImageMetadata | None:
    parameters = chunks.get("parameters")
    if parameters:
        return _parse_fooocus(parameters) or parse_a1111(parameters)
    if "prompt" in chunks:
        comfy = parse_comfyui(chunks["prompt"])
        if comfy is not None:
            return comfy
    if "invokeai_metadata" in chunks:
        invoke = _parse_invokeai(chunks["invokeai_metadata"])
        if invoke is not None:
            return invoke
    if chunks.get("Software", "").startswith("NovelAI") or "Comment" in chunks:
        return _parse_novelai(chunks)
    return None


def parse_a1111(text: str) -> ImageMetadata | None:
    """Parse the ``parameters`` block: prompt, ``Negative prompt:`` line, settings line."""
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return None
    settings: str | None = None
    match = _A1111_SETTINGS.search(text)
    if match:
        settings = match.group(1).strip()
        text = text[: match.start()]
    parts = _A1111_NEGATIVE.split(text, maxsplit=1)
    prompt = parts[0].strip()
    negative = parts[1].strip() if len(parts) > 1 else None
    return ImageMetadata(
        generator="a1111",
        prompt=prompt or None,
        negative_prompt=negative or None,
        settings=settings,
    )


def parse_comfyui(prompt_json: str) -> ImageMetadata | None:
    """Follow each sampler's positive/negative links back to its prompt text."""
    try:
        graph = json.loads(prompt_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(graph, dict):
        return None

    for node in graph.values():
        inputs = node.get("inputs") if isinstance(node, dict) else None
        if not isinstance(inputs, dict) or "positive" not in inputs or "negative" not in inputs:
            continue
        positive = _comfy_text(graph, inputs["positive"], "positive")
        if positive:
            negative = _comfy_text(graph, inputs["negative"], "negative")
            return ImageMetadata(generator="comfyui", prompt=positive, negative_prompt=negative or None)

    # No sampler with a traceable prompt: fall back to the first text-encode node.
    for node in graph.values():
        if isinstance(node, dict) and "TextEncode" in str(node.get("class_type", "")):
            text = _comfy_node_text(graph, node, "positive", 0)
            if text:
                return ImageMetadata(generator="comfyui", prompt=text)
    return None


def _comfy_text(graph: dict[str, Any], ref: Any, role: str, depth: int = 0) -> str | None:
    """Resolve a node input (a literal or a ``[node_id, output_index]`` link) to text."""
    if isinstance(ref, str):
        return ref.strip() or None
    if depth > _COMFY_MAX_DEPTH or not (isinstance(ref, list) and len(ref) == 2):
        return None
    node = graph.get(str(ref[0]))
    if not isinstance(node, dict):
        return None
    return _comfy_node_text(graph, node, role, depth + 1)


def _comfy_node_text(graph: dict[str, Any], node: dict[str, Any], role: str, depth: int) -> str | None:
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        return None
    texts = [
        text
        for key in _COMFY_TEXT_KEYS
        if key in inputs and (text := _comfy_text(graph, inputs[key], role, depth))
    ]
    if texts:
        # SDXL encoders carry the same prompt in text_g and text_l.
        return ", ".join(dict.fromkeys(texts))
    # Pass-through nodes (ControlNet, conditioning combine/concat, ...).
    for key in (role, "conditioning", "conditioning_1", "conditioning_to"):
        if key in inputs:
            text = _comfy_text(graph, inputs[key], role, depth)
            if text:
                return text
    return None


def _parse_novelai(chunks: dict[str, str]) -> ImageMetadata | None:
    comment: dict[str, Any] = {}
    try:
        loaded = json.loads(chunks.get("Comment", ""))
        if isinstance(loaded, dict):
            comment = loaded
    except json.JSONDecodeError:
        pass
    prompt = comment.get("prompt") or chunks.get("Description")
    if not isinstance(prompt, str) or not prompt.strip():
        return None
    negative = comment.get("uc")
    return ImageMetadata(
        generator="novelai",
        prompt=prompt.strip(),
        negative_prompt=negative.strip() if isinstance(negative, str) and negative.strip() else None,
    )


def _parse_invokeai(text: str) -> ImageMetadata | None:
    return _from_json_keys(text, "invokeai", "positive_prompt", "negative_prompt")


def _parse_fooocus(text: str) -> ImageMetadata | None:
    if not text.lstrip().startswith("{"):
        return None
    return _from_json_keys(text, "fooocus", "prompt", "negative_prompt")


def _from_json_keys(text: str, generator: str, prompt_key: str, negative_key: str) -> ImageMetadata | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    prompt = data.get(prompt_key)
    if not isinstance(prompt, str) or not prompt.strip():
        return None
    negative = data.get(negative_key)
    return ImageMetadata(
        generator=generator,
        prompt=prompt.strip(),
        negative_prompt=negative.strip() if isinstance(negative, str) and negative.strip() else None,
    )


def _exif_user_comment(img: Image.Image) -> str | None:
    try:
        value = img.getexif().get_ifd(_EXIF_IFD).get(_EXIF_USER_COMMENT)
    except Exception:  # malformed EXIF is common and never worth failing a read for
        return None
    if isinstance(value, str):
        return value.strip() or None
    if not isinstance(value, bytes) or len(value) <= 8:
        return None
    code, body = value[:8], value[8:]
    encoding = "utf-16-be" if code.startswith(b"UNICODE") else "utf-8"
    return body.decode(encoding, errors="replace").strip("\x00 \n") or None


def _truncate(value: str) -> str:
    if len(value) <= MAX_OTHER_TEXT_CHARS:
        return value
    return value[:MAX_OTHER_TEXT_CHARS] + f"... [truncated, {len(value)} chars total]"
