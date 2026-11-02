"""Frozen portable OOD4 membership and metadata checks."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path, PurePosixPath

DATASETS = ("ClinicDB", "ColonDB", "ETIS", "PolypGen")
SEQUENCES = tuple(f"seq{i}" for i in range(18, 23))
COUNTS = {
    "ClinicDB": {"total": 612, "positive": 612, "negative": 0},
    "ColonDB": {"total": 380, "positive": 380, "negative": 0},
    "ETIS": {"total": 196, "positive": 196, "negative": 0},
    "PolypGen": {"total": 273, "positive": 211, "negative": 62},
}
MANIFEST_SHA256 = "0da5ddd1aa6be87d888cc33e03748e250978cb51c928aa8411914145ad717265"
DIMENSIONS = (
    "original_image_width",
    "original_image_height",
    "original_mask_width",
    "original_mask_height",
)
RECORD_FIELDS = {
    "dataset",
    "sequence",
    "name",
    "image_relative_path",
    "mask_relative_path",
    "gt_empty",
    *DIMENSIONS,
}


def record_key(record):
    return record["dataset"], record["sequence"] or "", record["name"]


def validate_records(records):
    if not isinstance(records, list):
        raise ValueError("manifest records must be a list")
    seen = set()
    counts = {d: dict(total=0, positive=0, negative=0) for d in DATASETS}
    for record in records:
        if not isinstance(record, dict) or set(record) != RECORD_FIELDS:
            raise ValueError("OOD4 manifest record schema mismatch")
        d = record["dataset"]
        if d not in DATASETS:
            raise ValueError("unknown target domain")
        key = record_key(record)
        if key in seen:
            raise ValueError("duplicate target image identity")
        seen.add(key)
        if (
            not isinstance(record["name"], str)
            or not record["name"]
            or "/" in record["name"]
            or "\\" in record["name"]
        ):
            raise ValueError("invalid target image name")
        if (d == "PolypGen" and record["sequence"] not in SEQUENCES) or (
            d != "PolypGen" and record["sequence"] != ""
        ):
            raise ValueError("target sequence identity changed")
        for field in ["image_relative_path", "mask_relative_path"]:
            value = record[field]
            if (
                not isinstance(value, str)
                or not value
                or "\\" in value
                or ":" in value
                or PurePosixPath(value).is_absolute()
                or ".." in PurePosixPath(value).parts
            ):
                raise ValueError("target paths must be portable relative paths")
        if type(record["gt_empty"]) is not int or record["gt_empty"] not in (0, 1):
            raise ValueError("gt_empty must be 0 or 1")
        if any(type(record[k]) is not int or record[k] <= 0 for k in DIMENSIONS):
            raise ValueError("invalid original image/mask dimensions")
        counts[d]["total"] += 1
        counts[d]["negative" if record["gt_empty"] else "positive"] += 1
    if counts != COUNTS:
        raise ValueError("frozen OOD4 domain/positive/negative counts changed")
    if [r["dataset"] for r in records] != [
        d for d in DATASETS for _ in range(COUNTS[d]["total"])
    ]:
        raise ValueError("frozen target domain order changed")
    return counts


def load_manifest(path):
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != MANIFEST_SHA256:
        raise ValueError("frozen target manifest file hash mismatch")
    manifest = json.loads(content)
    if not isinstance(manifest, dict) or set(manifest) != {"records"}:
        raise ValueError("target manifest schema mismatch")
    validate_records(manifest["records"])
    return manifest["records"]


def verify_target_pairs(records, roots):
    """Check the complete discovered target layouts against the frozen list."""
    from .ood import build_target_pairs

    validate_records(records)
    pairs = {}
    for d in DATASETS:
        root = Path(roots[d]).expanduser().resolve()
        discovered = build_target_pairs(d, root)
        actual = [
            (
                p.dataset,
                p.sequence or "",
                p.name,
                p.image.resolve().relative_to(root).as_posix(),
                p.mask.resolve().relative_to(root).as_posix(),
            )
            for p in discovered
        ]
        expected = [
            (
                r["dataset"],
                r["sequence"],
                r["name"],
                r["image_relative_path"],
                r["mask_relative_path"],
            )
            for r in records
            if r["dataset"] == d
        ]
        if actual != expected:
            raise ValueError(
                f"{d} target membership/order/path differs from frozen manifest"
            )
        pairs[d] = discovered
    return pairs


def validate_rows(rows, records):
    if len(rows) != len(records):
        raise ValueError("evaluated target count changed")
    for row, record in zip(rows, records):
        if record_key(row) != record_key(record):
            raise ValueError("evaluated target identity/order changed")
        if row["gt_empty"] != record["gt_empty"] or any(
            row[k] != record[k] for k in DIMENSIONS
        ):
            raise ValueError(
                "target ground-truth status or original dimensions changed"
            )
