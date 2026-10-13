from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from captioner_agent.app import create_app
from captioner_agent.llm import TurnResult
from captioner_agent.settings import Settings

from .conftest import FakeProvider, FakeToolbox, png_bytes, text_block, tool_use


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider([])


@pytest.fixture
def client(settings: Settings, provider: FakeProvider) -> Iterator[TestClient]:
    settings.max_upload_mb = 1
    with TestClient(create_app(settings, provider=provider, toolbox=FakeToolbox())) as test_client:
        yield test_client


def new_session(client: TestClient) -> str:
    session_id: str = client.post("/api/sessions").json()["id"]
    return session_id


def events_of(response: Any) -> list[dict[str, Any]]:
    return [json.loads(line[6:]) for line in response.text.split("\n\n") if line.startswith("data: ")]


def test_serves_the_ui(client: TestClient) -> None:
    assert "Image Captioner" in client.get("/").text
    assert client.get("/app.js").status_code == 200


def test_upload_accepts_images_and_rejects_everything_else(client: TestClient) -> None:
    sid = new_session(client)

    response = client.post(
        f"/api/sessions/{sid}/files",
        files=[
            ("files", ("photos/Image1.png", png_bytes(), "image/png")),
            ("files", ("Image1.png", png_bytes(), "image/png")),
            ("files", ("notes.txt", b"hello", "text/plain")),
            ("files", ("fake.png", b"not an image", "image/png")),
            ("files", ("huge.png", b"\x00" * (1024 * 1024 + 1), "image/png")),
        ],
    )

    body = response.json()
    assert body["saved"] == ["Image1.png", "Image1_2.png"]
    assert [r["name"] for r in body["rejected"]] == ["notes.txt", "fake.png", "huge.png"]
    listed = client.get(f"/api/sessions/{sid}/outputs").json()
    assert [item["name"] for item in listed] == ["Image1.png", "Image1_2.png"]


def test_message_streams_events_and_persists_transcript(client: TestClient, provider: FakeProvider) -> None:
    sid = new_session(client)
    client.post(f"/api/sessions/{sid}/files", files=[("files", ("a.png", png_bytes(), "image/png"))])
    provider.turns = [
        TurnResult([tool_use("t1", "read_png_metadata", {"path": f"{sid}/uploads/a.png"})], "tool_use"),
        TurnResult([text_block("All done.")], "end_turn"),
    ]

    response = client.post(f"/api/sessions/{sid}/messages", json={"text": "caption it", "files": ["a.png"]})

    assert response.headers["content-type"].startswith("text/event-stream")
    events = events_of(response)
    # The pipeline captions the upload first (no PNG info, so vision), then Claude's turn runs.
    assert [(e["type"], e.get("name")) for e in events] == [
        ("tool_start", "caption_from_png_info"),
        ("tool_end", "caption_from_png_info"),
        ("tool_start", "caption_with_vision"),
        ("tool_end", "caption_with_vision"),
        ("tool_start", "read_png_metadata"),
        ("tool_end", "read_png_metadata"),
        ("text", None),
        ("text_end", None),
        ("done", None),
    ]
    session = client.get(f"/api/sessions/{sid}").json()
    assert session["title"] == "caption it"
    assert [t["type"] for t in session["transcript"]] == ["user", "tool", "tool", "tool", "assistant"]
    assert client.get("/api/sessions").json()[0]["id"] == sid


def test_pending_choice_is_exposed(client: TestClient, provider: FakeProvider) -> None:
    sid = new_session(client)
    provider.turns = [
        TurnResult([tool_use("q1", "ask_user", {"question": "Flip?", "options": ["Yes", "No"]})], "tool_use")
    ]

    client.post(f"/api/sessions/{sid}/messages", json={"text": "hi"})

    assert client.get(f"/api/sessions/{sid}").json()["pending_choice"] == {
        "id": "q1",
        "question": "Flip?",
        "options": ["Yes", "No"],
    }


def test_message_validation(client: TestClient) -> None:
    sid = new_session(client)

    assert client.post(f"/api/sessions/{sid}/messages", json={"text": "  "}).status_code == 400
    assert (
        client.post(f"/api/sessions/{sid}/messages", json={"text": "x", "files": ["missing.png"]}).status_code
        == 400
    )
    assert client.post(f"/api/sessions/{'0' * 32}/messages", json={"text": "x"}).status_code == 404
    assert client.post("/api/sessions/not-an-id/messages", json={"text": "x"}).status_code == 404


def test_caption_edit_thumbnail_and_zip(client: TestClient, settings: Settings) -> None:
    sid = new_session(client)
    client.post(f"/api/sessions/{sid}/files", files=[("files", ("a.png", png_bytes(), "image/png"))])

    assert client.put(f"/api/sessions/{sid}/captions/a.png", json={"text": "a red tile"}).status_code == 204
    assert client.put(f"/api/sessions/{sid}/captions/zzz.png", json={"text": "x"}).status_code == 404
    assert client.put(f"/api/sessions/{sid}/captions/a.png", json={"text": " "}).status_code == 400

    assert (settings.sessions_dir / sid / "outputs" / "a.txt").read_text(encoding="utf-8") == "a red tile"
    assert client.get(f"/api/sessions/{sid}/outputs").json()[0]["caption"] == "a red tile"

    thumb = client.get(f"/api/sessions/{sid}/files/uploads/a.png", params={"thumb": 1})
    assert thumb.headers["content-type"] == "image/jpeg"
    assert client.get(f"/api/sessions/{sid}/files/uploads/a.png").status_code == 200
    assert client.get(f"/api/sessions/{sid}/files/uploads/nope.png").status_code == 404
    assert client.get(f"/api/sessions/{sid}/files/secrets/a.png").status_code == 422

    archive = zipfile.ZipFile(io.BytesIO(client.get(f"/api/sessions/{sid}/download").content))
    assert sorted(archive.namelist()) == ["a.png", "a.txt"]


def test_delete_removes_files_and_history(client: TestClient, settings: Settings) -> None:
    sid = new_session(client)
    client.post(f"/api/sessions/{sid}/files", files=[("files", ("a.png", png_bytes(), "image/png"))])

    assert client.delete(f"/api/sessions/{sid}").status_code == 204

    assert client.get(f"/api/sessions/{sid}").status_code == 404
    assert not (settings.sessions_dir / sid).exists()
