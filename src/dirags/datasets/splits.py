"""Portable seed42 split validation; sample membership and order are immutable."""

from __future__ import annotations
import json
from pathlib import Path
from ..config import canonical_hash
from .kvasir_seg import build_pairs


def validate_split_manifest(manifest):
    expected = {
        "version",
        "dataset",
        "seed",
        "counts",
        "splits",
        "source_manifest_sha256",
        "sha256",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected:
        raise ValueError("portable split manifest schema mismatch")
    payload = {key: value for key, value in manifest.items() if key != "sha256"}
    if canonical_hash(payload) != manifest["sha256"]:
        raise ValueError("split manifest hash mismatch")
    if (
        type(manifest["version"]) is not int
        or manifest["version"] != 2
        or manifest["dataset"] != "Kvasir-SEG"
        or type(manifest["seed"]) is not int
        or manifest["seed"] != 42
    ):
        raise ValueError("expected the portable Kvasir-SEG seed42 split")
    origin = manifest["source_manifest_sha256"]
    if (
        not isinstance(origin, str)
        or len(origin) != 64
        or any(c not in "0123456789abcdef" for c in origin)
    ):
        raise ValueError("invalid original split hash")
    groups = manifest["splits"]
    if not isinstance(groups, dict) or set(groups) != {"train", "val", "test"}:
        raise ValueError("expected train, val and test split groups")
    seen = set()
    for name in ["train", "val", "test"]:
        group = groups[name]
        if not isinstance(group, list) or any(
            not isinstance(v, str)
            or not v
            or "/" in v
            or "\\" in v
            or ":" in v
            or v in {".", ".."}
            for v in group
        ):
            raise ValueError("split entries must be image filenames")
        if len(group) != len(set(group)) or seen & set(group):
            raise ValueError("duplicate or overlapping split membership")
        seen.update(group)
    counts = {name: len(groups[name]) for name in ["train", "val", "test"]}
    if manifest["counts"] != counts or counts != {
        "train": 800,
        "val": 100,
        "test": 100,
    }:
        raise ValueError("paper split must contain 800/100/100 images")


def load_split_manifest(path, data_root):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_split_manifest(manifest)
    expected = {name for group in manifest["splits"].values() for name in group}
    actual = {pair.name for pair in build_pairs(Path(data_root).expanduser().resolve())}
    if actual != expected:
        raise ValueError("dataset image/mask pairs do not match the frozen split")
    return manifest
