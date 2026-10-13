from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from captioner_agent.files import InvalidFileError, SessionFiles, safe_name_parts, verify_image
from captioner_agent.store import SessionStore, is_session_id, make_title

from .conftest import make_png

SESSION_ID = "0123456789abcdef0123456789abcdef"


@pytest.fixture
def files(tmp_path: Path) -> SessionFiles:
    session_files = SessionFiles(tmp_path, SESSION_ID)
    session_files.ensure()
    return session_files


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Image1.PNG", ("Image1", ".png")),
        ("folder/sub/photo.jpeg", ("photo", ".jpeg")),
        ("..\\..\\evil.png", ("evil", ".png")),
        ("my <weird>:name?.webp", ("my _weird_name_", ".webp")),
        ("con.png", ("image_con", ".png")),
        ("....png", ("image", ".png")),
    ],
)
def test_safe_name_parts(name: str, expected: tuple[str, str]) -> None:
    assert safe_name_parts(name) == expected


@pytest.mark.parametrize("name", ["notes.txt", "archive.zip", "noextension", "image.png.exe"])
def test_safe_name_parts_rejects_non_images(name: str) -> None:
    with pytest.raises(InvalidFileError):
        safe_name_parts(name)


def test_upload_names_are_unique_by_stem(files: SessionFiles) -> None:
    make_png(files.uploads / "cat.png")

    assert files.new_upload_path("cat.png").name == "cat_2.png"
    # A different extension would still collide on cat.txt.
    assert files.new_upload_path("CAT.jpg").name == "CAT_2.jpg"


def test_resolve_rejects_paths_and_missing_files(files: SessionFiles) -> None:
    make_png(files.uploads / "a.png")

    assert files.resolve("uploads", "a.png") == files.uploads / "a.png"
    for bad in ["../a.png", "sub/a.png", "", ".."]:
        with pytest.raises(InvalidFileError):
            files.resolve("uploads", bad)
    with pytest.raises(FileNotFoundError):
        files.resolve("outputs", "a.png")


def test_list_items_pairs_images_with_captions(files: SessionFiles) -> None:
    make_png(files.uploads / "a.png")
    make_png(files.uploads / "b.png")
    make_png(files.outputs / "a_flipped.png")
    (files.outputs / "a.txt").write_text("a caption", encoding="utf-8")

    items = files.list_items()

    assert [(i["name"], i["flipped"], i["caption"]) for i in items] == [
        ("a.png", False, "a caption"),
        ("b.png", False, None),
        ("a_flipped.png", True, None),
    ]


def test_write_caption_only_for_known_images(files: SessionFiles) -> None:
    make_png(files.uploads / "a.png")

    files.write_caption("a.png", "  edited  ")
    assert (files.outputs / "a.txt").read_text(encoding="utf-8") == "edited"
    with pytest.raises(FileNotFoundError):
        files.write_caption("../../outside.png", "x")


def test_zip_contains_captions_and_optionally_originals(files: SessionFiles) -> None:
    make_png(files.uploads / "a.png")
    (files.outputs / "a.txt").write_text("a caption", encoding="utf-8")

    with_originals = zipfile.ZipFile(io.BytesIO(files.build_zip())).namelist()
    captions_only = zipfile.ZipFile(io.BytesIO(files.build_zip(include_originals=False))).namelist()

    assert sorted(with_originals) == ["a.png", "a.txt"]
    assert captions_only == ["a.txt"]


def test_verify_image(tmp_path: Path) -> None:
    verify_image(make_png(tmp_path / "ok.png"))
    fake = tmp_path / "fake.png"
    fake.write_text("not an image", encoding="utf-8")
    with pytest.raises(InvalidFileError):
        verify_image(fake)


def test_store_round_trip(store: SessionStore) -> None:
    session = store.create()
    session.title = "First"
    session.messages = [{"role": "user", "content": "hi"}]
    session.transcript = [{"type": "user", "text": "hi", "files": []}]
    session.pending = {"tool_use_id": "t1", "results": [], "choice": {"id": "t1"}}
    store.save(session)

    loaded = store.get(session.id)

    assert loaded is not None
    assert (loaded.title, loaded.messages, loaded.pending) == ("First", session.messages, session.pending)
    assert [s["id"] for s in store.list()] == [session.id]

    store.delete(session.id)
    assert store.get(session.id) is None


def test_store_rejects_malformed_ids(store: SessionStore) -> None:
    assert store.get("../etc") is None
    assert not is_session_id("0123")
    assert is_session_id(SESSION_ID)


def test_make_title() -> None:
    assert make_title("  caption   these\nplease ", 2) == "caption these please"
    assert make_title("", 3) == "3 images"
    assert make_title("", 1) == "1 image"
    assert len(make_title("x" * 200, 0)) == 60
