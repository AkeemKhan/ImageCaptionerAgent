"""The Gemini client, against a mocked HTTP layer. Nothing here reaches Google."""

from __future__ import annotations

import base64
import io
import json
from typing import Any

import httpx
import pytest
from PIL import Image

from image_captioner_mcp.gemini import (
    GeminiError,
    GeminiRefusal,
    HttpGeminiEditor,
    closest_aspect_ratio,
    image_size_for,
)

KEY = "test-gemini-key-0123456789"


def encoded(size: tuple[int, int] = (8, 6), color: str = "red") -> str:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, "PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def reply(*parts: dict[str, Any], finish: str = "STOP") -> dict[str, Any]:
    return {"candidates": [{"content": {"parts": list(parts)}, "finishReason": finish}]}


def image_part(size: tuple[int, int] = (8, 6), color: str = "red", **extra: Any) -> dict[str, Any]:
    return {"inlineData": {"mimeType": "image/png", "data": encoded(size, color)}, **extra}


def editor(
    handler: Any, model: str = "gemini-3-pro-image-preview"
) -> tuple[HttpGeminiEditor, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = httpx.Client(transport=httpx.MockTransport(record))
    return HttpGeminiEditor(KEY, model, client=client), seen


def ok(payload: dict[str, Any]) -> Any:
    return lambda request: httpx.Response(200, json=payload)


SOURCE = Image.new("RGB", (40, 30), "blue")


def test_request_has_the_key_in_a_header_not_the_url_and_asks_for_an_image() -> None:
    gemini, seen = editor(ok(reply(image_part())))

    gemini.edit(SOURCE, "remove the text", "4:3", "2K")

    request = seen[0]
    assert request.url.path == "/v1beta/models/gemini-3-pro-image-preview:generateContent"
    assert request.headers["x-goog-api-key"] == KEY
    assert KEY not in str(request.url)
    body = json.loads(request.content)
    assert body["generationConfig"] == {
        "responseModalities": ["TEXT", "IMAGE"],
        "imageConfig": {"aspectRatio": "4:3", "imageSize": "2K"},
    }
    text_part, image_part_ = body["contents"][0]["parts"]
    assert text_part == {"text": "remove the text"}
    assert image_part_["inline_data"]["mime_type"] == "image/png"
    sent = Image.open(io.BytesIO(base64.b64decode(image_part_["inline_data"]["data"])))
    assert sent.size == (40, 30)


def test_models_prefix_is_accepted_and_2_5_flash_gets_no_image_size() -> None:
    gemini, seen = editor(ok(reply(image_part())), model="models/gemini-2.5-flash-image")

    gemini.edit(SOURCE, "x", "1:1", "2K")

    assert seen[0].url.path == "/v1beta/models/gemini-2.5-flash-image:generateContent"
    assert json.loads(seen[0].content)["generationConfig"]["imageConfig"] == {"aspectRatio": "1:1"}


def test_returns_the_last_image_and_skips_thought_images() -> None:
    payload = reply(
        {"text": "thinking", "thought": True},
        image_part(color="green", thought=True),
        {"text": "Here you go."},
        image_part((16, 12), "red"),
    )
    gemini, _ = editor(ok(payload))

    result = gemini.edit(SOURCE, "x", "4:3", "1K")

    assert result.size == (16, 12)
    assert result.getpixel((0, 0)) == (255, 0, 0)


def test_snake_case_inline_data_is_read_too() -> None:
    part = {"inline_data": {"mime_type": "image/png", "data": encoded()}}
    gemini, _ = editor(ok(reply(part)))

    assert gemini.edit(SOURCE, "x", "4:3", "1K").size == (8, 6)


def test_text_only_reply_is_a_refusal_carrying_what_gemini_said() -> None:
    gemini, _ = editor(ok(reply({"text": "I'm unable to remove watermarks from images."})))

    with pytest.raises(GeminiRefusal, match="unable to remove watermarks"):
        gemini.edit(SOURCE, "x", "4:3", "1K")


def test_a_blocked_prompt_is_a_refusal() -> None:
    gemini, _ = editor(ok({"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}}))

    with pytest.raises(GeminiRefusal, match="PROHIBITED_CONTENT"):
        gemini.edit(SOURCE, "x", "4:3", "1K")


def test_an_image_safety_stop_is_a_refusal() -> None:
    gemini, _ = editor(ok({"candidates": [{"finishReason": "IMAGE_SAFETY"}]}))

    with pytest.raises(GeminiRefusal, match="IMAGE_SAFETY"):
        gemini.edit(SOURCE, "x", "4:3", "1K")


@pytest.mark.parametrize(
    ("status", "words"),
    [
        (400, "rejected the request"),
        (401, "API key"),
        (403, "API key"),
        (402, "no credits"),
        (429, "quota or rate limit"),
        (503, "server error"),
    ],
)
def test_http_errors_become_clear_messages_that_never_hold_the_key(status: int, words: str) -> None:
    body = {"error": {"message": f"details mention {KEY} by mistake"}}
    gemini, _ = editor(lambda request: httpx.Response(status, json=body))

    with pytest.raises(GeminiError) as caught:
        gemini.edit(SOURCE, "x", "4:3", "1K")

    assert words in str(caught.value)
    assert KEY not in str(caught.value)
    assert not isinstance(caught.value, GeminiRefusal)


def test_timeouts_and_connection_failures_are_gemini_errors() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    with pytest.raises(GeminiError, match="in time"):
        editor(timeout)[0].edit(SOURCE, "x", "4:3", "1K")
    with pytest.raises(GeminiError, match="could not reach Gemini"):
        editor(refused)[0].edit(SOURCE, "x", "4:3", "1K")


def test_unreadable_image_data_and_non_json_replies_are_errors() -> None:
    bad = {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(b"not an image").decode()}}

    with pytest.raises(GeminiError, match="could not be read"):
        editor(ok(reply(bad)))[0].edit(SOURCE, "x", "4:3", "1K")
    with pytest.raises(GeminiError, match="not JSON"):
        editor(lambda request: httpx.Response(200, text="<html>"))[0].edit(SOURCE, "x", "4:3", "1K")


def test_a_key_is_required() -> None:
    with pytest.raises(ValueError, match="API key"):
        HttpGeminiEditor("", "gemini-3-pro-image-preview")


@pytest.mark.parametrize(
    ("size", "ratio"),
    [
        ((640, 480), "4:3"),
        ((480, 640), "3:4"),
        ((1000, 1000), "1:1"),
        ((1920, 1080), "16:9"),
        ((2560, 1920), "4:3"),
    ],
)
def test_closest_aspect_ratio(size: tuple[int, int], ratio: str) -> None:
    assert closest_aspect_ratio(*size) == ratio


@pytest.mark.parametrize(
    ("side", "bucket"), [(512, "1K"), (1024, "1K"), (1025, "2K"), (2048, "2K"), (4096, "4K")]
)
def test_image_size_buckets(side: int, bucket: str) -> None:
    assert image_size_for(side) == bucket
