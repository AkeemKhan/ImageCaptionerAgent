from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from image_captioner_mcp import png_meta

from .conftest import make_png

A1111 = (
    "masterpiece, (1girl:1.2), red dress, <lora:style:0.8>\n"
    "standing in a field\n"
    "Negative prompt: lowres, bad hands\n"
    "Steps: 28, Sampler: Euler a, CFG scale: 7, Seed: 1, Size: 512x768"
)


def test_a1111_parameters(root: Path) -> None:
    meta = png_meta.read_metadata(make_png(root / "a.png", {"parameters": A1111}))

    assert meta.has_prompt
    assert meta.generator == "a1111"
    assert meta.prompt == "masterpiece, (1girl:1.2), red dress, <lora:style:0.8>\nstanding in a field"
    assert meta.negative_prompt == "lowres, bad hands"
    assert meta.settings is not None and meta.settings.startswith("Steps: 28")


def test_a1111_without_negative_prompt() -> None:
    meta = png_meta.parse_a1111("a cat\nSteps: 20, Sampler: DDIM")

    assert meta is not None
    assert meta.prompt == "a cat"
    assert meta.negative_prompt is None


def test_comfyui_follows_sampler_links(root: Path) -> None:
    graph = {
        "3": {
            "class_type": "KSampler",
            "inputs": {"positive": ["10", 0], "negative": ["7", 0], "seed": 5},
        },
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a castle on a hill", "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry", "clip": ["4", 1]}},
        # A ControlNet node sits between the positive encoder and the sampler.
        "10": {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {"positive": ["6", 0], "negative": ["7", 0], "strength": 1.0},
        },
    }
    meta = png_meta.read_metadata(make_png(root / "c.png", {"prompt": json.dumps(graph)}))

    assert meta.generator == "comfyui"
    assert meta.prompt == "a castle on a hill"
    assert meta.negative_prompt == "blurry"


def test_comfyui_text_supplied_by_another_node() -> None:
    graph = {
        "1": {"class_type": "PrimitiveString", "inputs": {"value": "a red fox"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ["1", 0]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
        "4": {"class_type": "KSampler", "inputs": {"positive": ["2", 0], "negative": ["3", 0]}},
    }
    meta = png_meta.parse_comfyui(json.dumps(graph))

    assert meta is not None
    assert meta.prompt == "a red fox"
    assert meta.negative_prompt is None


def test_comfyui_link_cycle_terminates() -> None:
    graph = {
        "1": {"class_type": "Reroute", "inputs": {"conditioning": ["2", 0]}},
        "2": {"class_type": "Reroute", "inputs": {"conditioning": ["1", 0]}},
        "3": {"class_type": "KSampler", "inputs": {"positive": ["1", 0], "negative": ["2", 0]}},
    }
    assert png_meta.parse_comfyui(json.dumps(graph)) is None


def test_novelai(root: Path) -> None:
    chunks = {
        "Software": "NovelAI",
        "Description": "1girl, solo",
        "Comment": json.dumps({"prompt": "1girl, solo, smile", "uc": "lowres"}),
    }
    meta = png_meta.read_metadata(make_png(root / "n.png", chunks))

    assert meta.generator == "novelai"
    assert meta.prompt == "1girl, solo, smile"
    assert meta.negative_prompt == "lowres"


def test_invokeai_and_fooocus(root: Path) -> None:
    invoke = png_meta.read_metadata(
        make_png(root / "i.png", {"invokeai_metadata": json.dumps({"positive_prompt": "a boat"})})
    )
    fooocus = png_meta.read_metadata(
        make_png(root / "f.png", {"parameters": json.dumps({"prompt": "a tree", "negative_prompt": "x"})})
    )

    assert (invoke.generator, invoke.prompt) == ("invokeai", "a boat")
    assert (fooocus.generator, fooocus.prompt, fooocus.negative_prompt) == ("fooocus", "a tree", "x")


def test_no_metadata(root: Path) -> None:
    meta = png_meta.read_metadata(make_png(root / "plain.png"))

    assert not meta.has_prompt
    assert meta.prompt is None
    assert meta.other_text == {}


def test_unrecognised_text_is_returned_truncated(root: Path) -> None:
    long_value = "x" * (png_meta.MAX_OTHER_TEXT_CHARS + 50)
    meta = png_meta.read_metadata(make_png(root / "o.png", {"Title": "holiday", "Blob": long_value}))

    assert not meta.has_prompt
    assert meta.other_text["Title"] == "holiday"
    assert meta.other_text["Blob"].endswith("chars total]")


def test_jpeg_exif_user_comment(root: Path) -> None:
    path = root / "photo.jpg"
    exif = Image.Exif()
    exif[0x8769] = {0x9286: b"UNICODE\x00" + A1111.encode("utf-16-be")}
    Image.new("RGB", (4, 4), "green").save(path, exif=exif)

    meta = png_meta.read_metadata(path)

    assert meta.generator == "a1111"
    assert meta.negative_prompt == "lowres, bad hands"
