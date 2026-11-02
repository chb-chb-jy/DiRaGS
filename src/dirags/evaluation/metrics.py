"""Frozen foreground region and tolerant boundary metrics."""

from __future__ import annotations

import torch
from torch.nn import functional as F


METRIC_KEYS = ("dice", "iou", "mae", "boundary_f1", "boundary_iou")


def _boundary(mask: torch.Tensor) -> torch.Tensor:
    values = mask.float().unsqueeze(0).unsqueeze(0)
    padded_inverse = F.pad(1.0 - values, (1, 1, 1, 1), value=1.0)
    eroded = 1.0 - F.max_pool2d(padded_inverse, kernel_size=3, stride=1)
    return (values - eroded).squeeze(0).squeeze(0).bool()


def _dilate(mask: torch.Tensor, tolerance: int) -> torch.Tensor:
    if tolerance == 0:
        return mask
    values = mask.float().unsqueeze(0).unsqueeze(0)
    dilated = F.max_pool2d(
        values,
        kernel_size=2 * tolerance + 1,
        stride=1,
        padding=tolerance,
    )
    return dilated.squeeze(0).squeeze(0).bool()


def _overlap_metrics(prediction: torch.Tensor, target: torch.Tensor):
    prediction_count = int(prediction.sum().item())
    target_count = int(target.sum().item())
    if prediction_count == 0 and target_count == 0:
        return 1.0, 1.0
    if prediction_count == 0 or target_count == 0:
        return 0.0, 0.0
    intersection = int((prediction & target).sum().item())
    dice = 2.0 * intersection / (prediction_count + target_count)
    union = prediction_count + target_count - intersection
    return dice, intersection / union


def _boundary_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    tolerance: int,
):
    predicted_boundary = _boundary(prediction)
    target_boundary = _boundary(target)
    predicted_count = int(predicted_boundary.sum().item())
    target_count = int(target_boundary.sum().item())
    if predicted_count == 0 and target_count == 0:
        return 1.0, 1.0
    if predicted_count == 0 or target_count == 0:
        return 0.0, 0.0

    target_band = _dilate(target_boundary, tolerance)
    predicted_band = _dilate(predicted_boundary, tolerance)
    matched_prediction = int((predicted_boundary & target_band).sum().item())
    matched_target = int((target_boundary & predicted_band).sum().item())
    precision = matched_prediction / predicted_count
    recall = matched_target / target_count
    boundary_f1 = (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )

    symmetric_intersection = 0.5 * (matched_prediction + matched_target)
    symmetric_union = predicted_count + target_count - symmetric_intersection
    boundary_iou = symmetric_intersection / symmetric_union
    return boundary_f1, boundary_iou


def _validate_inputs(
    logits: torch.Tensor,
    target: torch.Tensor,
    boundary_tolerance: int,
) -> None:
    if isinstance(boundary_tolerance, bool) or not isinstance(boundary_tolerance, int):
        raise ValueError("boundary_tolerance must be a nonnegative integer")
    if boundary_tolerance < 0:
        raise ValueError("boundary_tolerance must be a nonnegative integer")
    if logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError("expected two-class logits shaped [B, 2, H, W]")
    if target.ndim != 3:
        raise ValueError("expected target shaped [B, H, W]")
    if logits.shape[0] != target.shape[0] or logits.shape[2:] != target.shape[1:]:
        raise ValueError("logits and target shapes do not match")
    if logits.shape[0] == 0:
        raise ValueError("metrics require a nonempty batch")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("logits contain nonfinite values")
    if not torch.isfinite(target).all():
        raise FloatingPointError("target contains nonfinite values")
    if not torch.all((target == 0) | (target == 1)):
        raise ValueError("target must contain only binary labels 0 and 1")


@torch.no_grad()
def compute_batch_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    boundary_tolerance: int = 2,
) -> dict[str, float]:
    """Compute foreground-only per-image metrics and macro-average the batch."""

    _validate_inputs(logits, target, boundary_tolerance)
    predictions = logits.argmax(dim=1).eq(1)
    targets = target.eq(1)
    totals = {key: 0.0 for key in METRIC_KEYS}

    for prediction, sample_target in zip(predictions, targets):
        dice, iou = _overlap_metrics(prediction, sample_target)
        boundary_f1, boundary_iou = _boundary_metrics(
            prediction,
            sample_target,
            boundary_tolerance,
        )
        totals["dice"] += dice
        totals["iou"] += iou
        totals["mae"] += float(
            (prediction.float() - sample_target.float()).abs().mean().item()
        )
        totals["boundary_f1"] += boundary_f1
        totals["boundary_iou"] += boundary_iou

    batch_size = logits.shape[0]
    return {key: float(totals[key] / batch_size) for key in METRIC_KEYS}
