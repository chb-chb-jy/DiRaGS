"""Frozen OOD4 evaluation: PIL-resized 256px masks and 256x256 logits.

Positive-frame metrics and domain weighting preserve the original evaluator.
Negative frames are summarized separately and never enter positive means.
"""

from __future__ import annotations
import hashlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from ..datasets.kvasir_seg import IMAGENET_MEAN, IMAGENET_STD
from .metrics import _boundary_metrics, _overlap_metrics

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


SUMMARY_METRICS = (
    "dice",
    "iou",
    "precision",
    "recall",
    "mae",
    "boundary_f1",
    "boundary_iou",
)


SERIOUS_FP_PIXELS = 5000


@dataclass(frozen=True)
class CrossDomainPair:
    dataset: str
    name: str
    image: Path
    mask: Path
    sequence: str | None = None


def _files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"directory not found: {directory}")
    return sorted(
        (
            p
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        ),
        key=lambda p: p.name,
    )


def _pair_directories(
    dataset: str,
    image_dir: Path,
    mask_dir: Path,
    *,
    mask_key=lambda stem: stem,
    sequence: str | None = None,
) -> list[CrossDomainPair]:
    images = {p.stem.lower(): p for p in _files(image_dir)}
    masks: dict[str, Path] = {}
    for path in _files(mask_dir):
        key = str(mask_key(path.stem.lower()))
        if key in masks:
            raise ValueError(f"duplicate normalized mask stem {key!r} in {mask_dir}")
        masks[key] = path
    missing = sorted(images.keys() - masks.keys())
    orphan = sorted(masks.keys() - images.keys())
    if missing or orphan:
        raise ValueError(
            f"{dataset} pairing failed: missing={missing[:10]}, orphan={orphan[:10]}"
        )
    return [
        CrossDomainPair(dataset, images[key].name, images[key], masks[key], sequence)
        for key in sorted(images)
    ]


def build_target_pairs(dataset: str, root: Path) -> list[CrossDomainPair]:
    """Build exact pairs for the four frozen target-domain layouts."""
    root = Path(root)
    key = dataset.lower().replace("_", "-")
    if key in {"clinicdb", "cvc-clinicdb"}:
        return _pair_directories(
            "ClinicDB", root / "PNG" / "Original", root / "PNG" / "Ground Truth"
        )
    if key in {"colondb", "cvc-colondb"}:
        return _pair_directories("ColonDB", root / "images", root / "masks")
    if key in {"etis", "etis-laribpolypdb"}:
        return _pair_directories(
            "ETIS",
            root / "original",
            root / "ground_truth",
            mask_key=lambda stem: stem[1:] if stem.startswith("p") else stem,
        )
    if key == "polypgen":
        base = root / "PolypGen2021_MultiCenterData_v3" / "sequenceData" / "positive"
        pairs: list[CrossDomainPair] = []
        for index in range(18, 23):
            sequence = f"seq{index}"
            pairs.extend(
                _pair_directories(
                    "PolypGen",
                    base / sequence / f"images_{sequence}",
                    base / sequence / f"masks_{sequence}",
                    mask_key=lambda stem: stem.removesuffix("_mask"),
                    sequence=sequence,
                )
            )
        return pairs
    raise ValueError(f"unsupported dataset: {dataset}")


class CrossDomainDataset(Dataset):
    def __init__(self, pairs: Iterable[CrossDomainPair], image_size: int = 256):
        self.pairs = list(pairs)
        if not self.pairs:
            raise ValueError("cross-domain dataset is empty")
        self.image_size = int(image_size)
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        pair = self.pairs[index]
        with Image.open(pair.image) as source:
            image = source.convert("RGB")
        with Image.open(pair.mask) as source:
            mask_image = source.convert("L")
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
        )
        mask_image = TF.resize(
            mask_image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )
        mask = np.asarray(mask_image) >= 128
        image_tensor = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        return image_tensor, torch.from_numpy(mask.copy()).bool(), pair


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _largest_component(mask: np.ndarray) -> int:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    largest = 0
    for row, col in zip(*np.nonzero(mask)):
        if seen[row, col]:
            continue
        stack = [(int(row), int(col))]
        seen[row, col] = True
        size = 0
        while stack:
            r, c = stack.pop()
            size += 1
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (
                    0 <= rr < height
                    and 0 <= cc < width
                    and mask[rr, cc]
                    and not seen[rr, cc]
                ):
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        largest = max(largest, size)
    return largest


def fp_component_statistics(
    prediction: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
) -> dict[str, int | float]:
    """Return the frozen Gate 2C per-image FP statistics using true 8CC."""
    prediction_array = np.asarray(
        (
            prediction.detach().cpu()
            if isinstance(prediction, torch.Tensor)
            else prediction
        ),
        dtype=bool,
    )
    target_array = np.asarray(
        target.detach().cpu() if isinstance(target, torch.Tensor) else target,
        dtype=bool,
    )
    if prediction_array.ndim != 2 or prediction_array.shape != target_array.shape:
        raise ValueError("prediction and target must be equal two-dimensional masks")

    fp_mask = prediction_array & ~target_array
    visited = np.zeros_like(fp_mask, dtype=bool)
    component_areas: list[int] = []
    height, width = fp_mask.shape
    for row, col in np.argwhere(fp_mask):
        row, col = int(row), int(col)
        if visited[row, col]:
            continue
        visited[row, col] = True
        queue: deque[tuple[int, int]] = deque([(row, col)])
        area = 0
        while queue:
            y, x = queue.popleft()
            area += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    yy, xx = y + dy, x + dx
                    if (
                        0 <= yy < height
                        and 0 <= xx < width
                        and fp_mask[yy, xx]
                        and not visited[yy, xx]
                    ):
                        visited[yy, xx] = True
                        queue.append((yy, xx))
        component_areas.append(area)

    total = int(np.count_nonzero(fp_mask))
    largest = max(component_areas, default=0)
    count = len(component_areas)
    pixels = int(fp_mask.size)
    return {
        "fp_total_area": total,
        "fp_total_area_ratio": total / pixels,
        "fp_largest_8cc_area": int(largest),
        "fp_largest_8cc_area_ratio": largest / pixels,
        "fp_8cc_count": int(count),
        "serious_fp": int(total >= SERIOUS_FP_PIXELS or largest >= SERIOUS_FP_PIXELS),
    }


def per_image_metrics(
    prediction: torch.Tensor, target: torch.Tensor, tolerance: int = 2
) -> dict[str, float | int]:
    prediction = prediction.bool().cpu()
    target = target.bool().cpu()
    tp = int((prediction & target).sum())
    fp = int((prediction & ~target).sum())
    fn = int((~prediction & target).sum())
    tn = int((~prediction & ~target).sum())
    dice, iou = _overlap_metrics(prediction, target)
    bf1, biou = _boundary_metrics(prediction, target, tolerance)
    result = {
        "dice": dice,
        "iou": iou,
        "precision": 1.0 if tp + fp == 0 else tp / (tp + fp),
        "recall": 1.0 if tp + fn == 0 else tp / (tp + fn),
        "mae": (fp + fn) / prediction.numel(),
        "boundary_f1": bf1,
        "boundary_iou": biou,
        "gt_empty": float(not bool(target.any())),
        "gt_foreground_fraction": float(target.float().mean()),
        "predicted_foreground_fraction": float(prediction.float().mean()),
        "fp_pixels": float(fp),
        "specificity": 1.0 if tn + fp == 0 else tn / (tn + fp),
        "largest_fp_component": float(
            _largest_component(prediction.numpy()) if not target.any() else 0
        ),
    }
    result.update(fp_component_statistics(prediction, target))
    return result


@torch.inference_mode()
def evaluate_pairs(
    model, pairs: Iterable[CrossDomainPair], device: torch.device, image_size: int = 256
):
    rows = []
    for image, target, pair in CrossDomainDataset(pairs, image_size=image_size):
        logits = model(image.unsqueeze(0).to(device))
        if (
            logits.ndim != 4
            or logits.shape[0] != 1
            or tuple(logits.shape[-2:]) != tuple(target.shape)
        ):
            raise ValueError(
                "model logits must remain at the evaluation resolution: "
                f"expected (1, C, {target.shape[0]}, {target.shape[1]}), got {tuple(logits.shape)}"
            )
        prediction = logits.argmax(dim=1)[0].cpu()
        with Image.open(pair.image) as source:
            original_image_width, original_image_height = source.size
        with Image.open(pair.mask) as source:
            original_mask_width, original_mask_height = source.size
        row = {
            "dataset": pair.dataset,
            "sequence": pair.sequence or "",
            "name": pair.name,
            "original_image_width": int(original_image_width),
            "original_image_height": int(original_image_height),
            "original_mask_width": int(original_mask_width),
            "original_mask_height": int(original_mask_height),
        }
        row.update(per_image_metrics(prediction, target))
        rows.append(row)
    return rows


def summarize_rows(rows: list[dict]) -> dict:
    positives = [row for row in rows if not row["gt_empty"]]
    negatives = [row for row in rows if row["gt_empty"]]
    positive_summary = (
        {
            key: {
                "mean": float(np.mean([r[key] for r in positives])),
                "std": float(np.std([r[key] for r in positives])),
            }
            for key in SUMMARY_METRICS
        }
        if positives
        else {}
    )
    negative_summary = {}
    if negatives:
        negative_summary = {
            "false_positive_image_rate": float(
                np.mean([r["fp_pixels"] > 0 for r in negatives])
            ),
            "predicted_foreground_fraction_mean": float(
                np.mean([r["predicted_foreground_fraction"] for r in negatives])
            ),
            "fp_pixels_mean": float(np.mean([r["fp_pixels"] for r in negatives])),
            "specificity_mean": float(np.mean([r["specificity"] for r in negatives])),
            "largest_fp_component_mean": float(
                np.mean([r["largest_fp_component"] for r in negatives])
            ),
            "largest_fp_component_max": float(
                max(r["largest_fp_component"] for r in negatives)
            ),
            "fp_largest_8cc_area_mean": float(
                np.mean([r["fp_largest_8cc_area"] for r in negatives])
            ),
            "fp_8cc_count_mean": float(np.mean([r["fp_8cc_count"] for r in negatives])),
            "serious_fp_count": int(sum(int(r["serious_fp"]) for r in negatives)),
            "serious_fp_rate": float(np.mean([r["serious_fp"] for r in negatives])),
        }
    return {
        "total": len(rows),
        "positive": len(positives),
        "negative": len(negatives),
        "positive_metrics": positive_summary,
        "negative_metrics": negative_summary,
    }


def summarize_domains(summaries: dict[str, dict]) -> dict:
    """Aggregate positive-frame macro metrics with equal weight per target domain."""
    if not summaries:
        raise ValueError("target-domain summaries are empty")
    missing = [
        name
        for name, summary in summaries.items()
        if not summary.get("positive_metrics")
    ]
    if missing:
        raise ValueError(f"target domains contain no positive frames: {missing}")

    target_mean: dict[str, float] = {}
    domain_variance: dict[str, float] = {}
    worst_domain: dict[str, dict[str, float | str]] = {}
    for metric in SUMMARY_METRICS:
        values = {
            name: float(summary["positive_metrics"][metric]["mean"])
            for name, summary in summaries.items()
        }
        target_mean[metric] = float(np.mean(list(values.values())))
        domain_variance[metric] = float(np.var(list(values.values())))
        # Higher is better except for MAE, for which the largest value is worst.
        worst_name = (max if metric == "mae" else min)(values, key=values.get)
        worst_domain[metric] = {"dataset": worst_name, "value": values[worst_name]}
    return {
        "domain_count": len(summaries),
        "target_mean": target_mean,
        "worst_domain": worst_domain,
        "domain_variance": domain_variance,
    }
