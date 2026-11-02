"""GCS gradient compatibility: project conflicting auxiliary updates.

The main segmentation gradient remains the fixed projection reference.
Only the final paper gradient rule is included.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import torch
from torch import nn

GCS_COMPOSITION = "project-conflicting-dst-gradient-v1"


@dataclass(frozen=True)
class GradientCompositionStats:
    dot: float
    segmentation_norm: float
    dst_norm: float
    cosine: float
    projected: bool


def _global_inner_product(
    left: Iterable[torch.Tensor | None],
    right: Iterable[torch.Tensor | None],
) -> torch.Tensor:
    terms = [
        a.detach().mul(b.detach()).sum()
        for a, b in zip(left, right)
        if a is not None and b is not None
    ]
    if not terms:
        return torch.tensor(0.0)
    return sum(terms)


def _global_squared_norm(grads: Iterable[torch.Tensor | None]) -> torch.Tensor:
    terms = [grad.detach().square().sum() for grad in grads if grad is not None]
    if not terms:
        return torch.tensor(0.0)
    return sum(terms)


def backward_gcs(
    model: nn.Module,
    *,
    segmentation_loss: torch.Tensor,
    probe_loss: torch.Tensor,
    stable_loss: torch.Tensor,
    stable_weight: float,
    epsilon: float = 1e-12,
) -> GradientCompositionStats:
    """Backpropagate DST-v1 after removing only its conflicting component.

    Projection is global across all trainable backbone parameters.  Probe
    gradients remain exactly those of DST-v1 and captured features remain
    detached from the probe objective.
    """

    if stable_weight < 0.0:
        raise ValueError("stable_weight must be nonnegative")
    if not hasattr(model, "base_model") or not hasattr(model, "probes"):
        raise TypeError("GCS requires a DST training wrapper")
    parameters = tuple(
        parameter
        for parameter in model.base_model.parameters()
        if parameter.requires_grad
    )
    if not parameters:
        raise ValueError("GCS backbone has no trainable parameters")

    # Use ordinary backward passes rather than autograd.grad.  SegmentationModel uses
    # reentrant activation checkpointing outside the axial blocks, and that
    # checkpoint implementation explicitly rejects autograd.grad/inputs-based
    # backward.  Saving and clearing .grad gives the same two gradient vectors
    # while remaining compatible with the frozen model execution path.
    segmentation_loss.backward(retain_graph=True)
    segmentation_grads = tuple(
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
    )
    for parameter in parameters:
        parameter.grad = None

    (float(stable_weight) * stable_loss).backward()
    dst_grads = tuple(
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
    )
    for parameter in parameters:
        parameter.grad = None

    dot = _global_inner_product(segmentation_grads, dst_grads)
    segmentation_sq = _global_squared_norm(segmentation_grads)
    dst_sq = _global_squared_norm(dst_grads)
    if not bool(torch.isfinite(dot)):
        raise FloatingPointError("GCS gradient inner product is nonfinite")
    if not bool(torch.isfinite(segmentation_sq)) or not bool(torch.isfinite(dst_sq)):
        raise FloatingPointError("GCS gradient norm is nonfinite")

    projected = bool(dot.item() < 0.0 and segmentation_sq.item() > epsilon)
    coefficient = dot / segmentation_sq.clamp_min(epsilon) if projected else None
    # Probe loss has no path to the backbone because the captured features are
    # detached in frozen DST-v1.  Backward therefore fills only probe grads.
    probe_loss.backward()
    if any(parameter.grad is not None for parameter in parameters):
        raise RuntimeError("frozen DST-v1 probe loss reached the backbone")

    for parameter, segmentation_grad, dst_grad in zip(
        parameters,
        segmentation_grads,
        dst_grads,
    ):
        if segmentation_grad is None and dst_grad is None:
            parameter.grad = None
            continue
        if segmentation_grad is None:
            composed = dst_grad
        elif dst_grad is None:
            composed = segmentation_grad
        elif coefficient is None:
            composed = segmentation_grad + dst_grad
        else:
            composed = segmentation_grad + dst_grad - coefficient * segmentation_grad
        parameter.grad = composed.detach().clone()

    segmentation_norm = float(segmentation_sq.sqrt().item())
    dst_norm = float(dst_sq.sqrt().item())
    denominator = max(segmentation_norm * dst_norm, epsilon)
    cosine = float(dot.item()) / denominator
    return GradientCompositionStats(
        dot=float(dot.item()),
        segmentation_norm=segmentation_norm,
        dst_norm=dst_norm,
        cosine=cosine,
        projected=projected,
    )
