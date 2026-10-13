"""BLIP captioning: the BLIP_CAPTIONER tool, model-free helpers and the download plan."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mcp import Client
from PIL import Image

from image_captioner_mcp.captioner import BlipCaptioner, StubCaptioner, clean_blip_text, create_blip_captioner
from image_captioner_mcp.config import ServerConfig
from image_captioner_mcp.download import RepoFile, select_files
from image_captioner_mcp.server import create_server
from image_captioner_mcp.service import CaptionService

from .conftest import make_png


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("arafed a red house on a green field", "a red house on a green field"),
        ("a red house arafed  on a field", "a red house on a field"),
        ("Arafed man   standing", "man standing"),
        ("a gearafed thing", "a gearafed thing"),
        ("  plain caption  ", "plain caption"),
        ("there is a red truck on a road", "a red truck on a road"),
        ("There are two dogs in a park", "two dogs in a park"),
        ("arafed there is a man", "a man"),
        ("a house where there is a lake", "a house where there is a lake"),
        ("theres a dog", "theres a dog"),
    ],
)
def test_clean_blip_text(raw: str, expected: str) -> None:
    assert clean_blip_text(raw) == expected


def test_blip_captioner_validates_style_before_loading_anything(root: Path) -> None:
    captioner = BlipCaptioner(ServerConfig(allowed_root=root))

    assert captioner.device == "unloaded"
    with pytest.raises(ValueError, match="style must be one of"):
        captioner.caption(Image.new("RGB", (4, 4)), "poetic")  # type: ignore[arg-type]
    assert captioner.device == "unloaded"


def test_blip_is_a_separate_captioner_with_its_own_model(root: Path) -> None:
    config = ServerConfig(allowed_root=root)

    captioner = create_blip_captioner(config)

    assert isinstance(captioner, BlipCaptioner)
    assert config.blip_model_id == "Salesforce/blip-image-captioning-large"
    assert not config.preload_blip
    assert isinstance(create_blip_captioner(ServerConfig(allowed_root=root, backend="stub")), StubCaptioner)


def test_blip_settings_come_from_the_environment(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setenv("CAPTIONER_ALLOWED_ROOT", str(root))
    monkeypatch.setenv("CAPTIONER_BLIP_MODEL_ID", "Salesforce/blip-image-captioning-base")
    monkeypatch.setenv("CAPTIONER_BLIP_MODEL_REVISION", "abc123")
    monkeypatch.setenv("CAPTIONER_PRELOAD_BLIP", "true")

    config = ServerConfig.from_env()

    assert (config.blip_model_id, config.blip_model_revision, config.preload_blip) == (
        "Salesforce/blip-image-captioning-base",
        "abc123",
        True,
    )


def test_service_blip_caption_defaults_to_short_and_checks_the_path(
    service: CaptionService, root: Path
) -> None:
    make_png(root / "a.png")

    result = service.blip_caption("a.png")

    assert result == {
        "path": "a.png",
        "caption": "a short caption of a 4x2 image",
        "style": "short",
        "backend": "stub",
        "device": "none",
    }
    with pytest.raises(ValueError, match="style must be one of"):
        service.blip_caption("a.png", "poetic")  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_blip_captioner_tool_over_mcp(service: CaptionService, root: Path) -> None:
    make_png(root / "a.png")

    async with Client(create_server(service)) as client:
        ok = await client.call_tool("BLIP_CAPTIONER", {"path": "a.png", "style": "detailed"})
        escaped = await client.call_tool("BLIP_CAPTIONER", {"path": "../a.png"})
        not_image = await client.call_tool("BLIP_CAPTIONER", {"path": "missing.png"})

    assert not ok.is_error
    assert json.loads(ok.content[0].text)["style"] == "detailed"  # type: ignore[union-attr]
    assert escaped.is_error and "outside the allowed root" in escaped.content[0].text  # type: ignore[union-attr]
    assert not_image.is_error


def test_download_keeps_one_copy_of_the_weights() -> None:
    repo = [
        RepoFile("config.json", 1),
        RepoFile("vocab.txt", 1),
        RepoFile("model.safetensors", 1879),
        RepoFile("pytorch_model.bin", 1879),
        RepoFile("tf_model.h5", 1879),
    ]

    assert [f.name for f in select_files(repo)] == ["config.json", "vocab.txt", "model.safetensors"]


def test_download_falls_back_to_pytorch_bin_then_to_everything() -> None:
    base = [RepoFile("config.json", 1), RepoFile("pytorch_model.bin", 990), RepoFile("tf_model.h5", 990)]
    odd = [RepoFile("config.json", 1), RepoFile("weights.ckpt", 5), RepoFile("weights.pt", 5)]

    assert [f.name for f in select_files(base)] == ["config.json", "pytorch_model.bin"]
    assert [f.name for f in select_files(odd)] == ["config.json", "weights.ckpt", "weights.pt"]


def test_download_keeps_all_shards_of_the_chosen_format() -> None:
    repo = [
        RepoFile("model.safetensors.index.json", 1),
        RepoFile("model-00001-of-00002.safetensors", 5),
        RepoFile("model-00002-of-00002.safetensors", 5),
        RepoFile("pytorch_model-00001-of-00002.bin", 5),
    ]

    assert {f.name for f in select_files(repo)} == {
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
