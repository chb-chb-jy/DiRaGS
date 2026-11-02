"""ImageNet-normalized minibatch adapter for the formal PDC sampler."""

from __future__ import annotations
from typing import Sequence
import torch
from .pdc_common import IMAGENET_MEAN, IMAGENET_STD, PseudoDomainConfig
from .pdc import build_paired_raw_replay as build_d1_replay


def build_pdc_pseudo_domain(
    normalized_images: torch.Tensor,
    masks: torch.Tensor,
    sample_identities: Sequence[str],
    *,
    training_seed: int,
    epoch: int,
    config: PseudoDomainConfig,
) -> torch.Tensor:
    """Build one candidate pseudo batch while preserving the RPP-v1 operator."""

    config.validate()
    if normalized_images.ndim != 4 or normalized_images.shape[1] != 3:
        raise ValueError("images must have shape [B,3,H,W]")
    expected_masks = (
        normalized_images.shape[0],
        normalized_images.shape[2],
        normalized_images.shape[3],
    )
    if tuple(masks.shape) != expected_masks:
        raise ValueError("masks and images have incompatible shapes")
    if len(sample_identities) != normalized_images.shape[0]:
        raise ValueError("sample identity count does not match batch size")
    if len(set(map(str, sample_identities))) != len(sample_identities):
        raise ValueError("sample identities must be unique within a batch")

    mean = normalized_images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = normalized_images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    raw_images = (normalized_images * std + mean).clamp(0.0, 1.0)
    pseudo = []
    for index, identity in enumerate(sample_identities):
        common = {
            "raw_image": raw_images[index : index + 1],
            "mask": masks[index].detach().cpu(),
            "training_seed": training_seed,
            "epoch": epoch,
            "sample_identity": str(identity),
            "config": config,
        }
        replay = build_d1_replay(**common)
        candidate_raw = replay.d1
        pseudo.append(candidate_raw)
    result = (torch.cat(pseudo, dim=0) - mean) / std
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("RPP-v2 Module1 generated a nonfinite pseudo batch")
    return result
