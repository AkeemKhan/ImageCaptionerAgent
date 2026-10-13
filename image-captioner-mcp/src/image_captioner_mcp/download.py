"""Download the captioning models ahead of time.

Without this, each model downloads the first time it is used, which can take
minutes and looks like a hang. Run it once::

    captioner-download-models              # BLIP only (about 1.9 GB)
    captioner-download-models --all        # BLIP and Florence-2
    captioner-download-models --dry-run    # list what would be fetched

Model repositories often hold the same weights in several formats (PyTorch,
safetensors, TensorFlow). Only one copy is fetched here, preferring safetensors.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .config import ServerConfig

DEFAULT_BLIP = "Salesforce/blip-image-captioning-large"
DEFAULT_FLORENCE = "microsoft/Florence-2-large"

_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".h5", ".msgpack", ".onnx", ".ot", ".pt", ".pth", ".ckpt")


@dataclass(frozen=True)
class RepoFile:
    name: str
    size: int


def is_weight(name: str) -> bool:
    return name.lower().endswith(_WEIGHT_SUFFIXES)


def select_files(files: Iterable[RepoFile]) -> list[RepoFile]:
    """Choose the files to download: everything except duplicate weight formats.

    Of the weight files, keep the safetensors set if there is one (safe to load,
    no pickle), otherwise the PyTorch ``.bin`` set, otherwise everything.
    """
    everything = list(files)
    weights = [f for f in everything if is_weight(f.name)]
    for suffix in (".safetensors", ".bin"):
        preferred = [f for f in weights if f.name.lower().endswith(suffix)]
        if preferred:
            keep = {f.name for f in preferred}
            break
    else:
        keep = {f.name for f in weights}
    return [f for f in everything if not is_weight(f.name) or f.name in keep]


def plan(repo_id: str, revision: str | None) -> list[RepoFile]:
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    files = [RepoFile(s.rfilename, s.size or 0) for s in info.siblings or []]
    return select_files(files)


def download(repo_id: str, revision: str | None = None, dry_run: bool = False) -> None:
    chosen = plan(repo_id, revision)
    total_mb = sum(f.size for f in chosen) / 1e6
    print(f"{repo_id}: {len(chosen)} files, {total_mb:,.0f} MB")
    for f in sorted(chosen, key=lambda f: -f.size)[:5]:
        print(f"    {f.name:<36} {f.size / 1e6:>8,.1f} MB")
    if dry_run:
        return
    from huggingface_hub import snapshot_download

    # One file at a time: on Windows without symlink rights, parallel downloads
    # race while the library probes for symlink support and fail with
    # "[WinError 1314] A required privilege is not held by the client". The
    # weights are a single large file, so nothing is lost.
    path = snapshot_download(
        repo_id, revision=revision, allow_patterns=[f.name for f in chosen], max_workers=1
    )
    print(f"    saved to {path}")


def main(argv: Sequence[str] | None = None) -> int:
    config = ServerConfig.from_env()
    parser = argparse.ArgumentParser(description="Download the captioning models.")
    parser.add_argument("--blip", action="store_true", help="download BLIP (the default)")
    parser.add_argument("--florence", action="store_true", help="download Florence-2")
    parser.add_argument("--all", action="store_true", help="download both models")
    parser.add_argument("--dry-run", action="store_true", help="list the files without downloading")
    args = parser.parse_args(argv)

    targets: list[tuple[str, str | None]] = []
    if args.blip or args.all or not args.florence:
        targets.append((config.blip_model_id, config.blip_model_revision))
    if args.florence or args.all:
        targets.append((config.model_id, config.model_revision))

    for repo_id, revision in targets:
        try:
            download(repo_id, revision, args.dry_run)
        except Exception as exc:
            print(f"Could not download {repo_id}: {exc}", file=sys.stderr)
            return 1
    if not args.dry_run:
        print("Done. The models are cached and load without a network connection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
