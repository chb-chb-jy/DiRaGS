"""Shared PDC sampling primitives.

SAMPLING_PROTOCOL_VERSION and sampling slot literals are immutable: they
are inputs to the original stateless seed hash, not display names.
"""

from __future__ import annotations
import hashlib
from dataclasses import dataclass
from typing import Sequence
import torch
import torch.nn.functional as F

SAMPLING_PROTOCOL_VERSION = "CSTP-v1"


IMAGENET_MEAN = (0.485, 0.456, 0.406)


IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PseudoDomainConfig:
    gain_min: float = 0.80
    gain_max: float = 1.20
    bias_min: float = -0.08
    bias_max: float = 0.08
    gamma_min: float = 0.80
    gamma_max: float = 1.20
    illumination_amplitude: float = 0.15
    illumination_grid_size: int = 4
    soft_mask_kernel: int = 31
    soft_mask_passes: int = 2

    def validate(self) -> None:
        if not 0.0 < self.gain_min <= self.gain_max:
            raise ValueError("invalid pseudo-domain gain range")
        if not self.bias_min <= self.bias_max:
            raise ValueError("invalid pseudo-domain bias range")
        if not 0.0 < self.gamma_min <= self.gamma_max:
            raise ValueError("invalid pseudo-domain gamma range")
        if not 0.0 <= self.illumination_amplitude <= 1.0:
            raise ValueError("invalid pseudo-domain illumination amplitude")
        if self.illumination_grid_size < 2:
            raise ValueError("illumination_grid_size must be at least 2")
        if self.soft_mask_kernel <= 1 or self.soft_mask_kernel % 2 == 0:
            raise ValueError("soft_mask_kernel must be an odd integer greater than 1")
        if self.soft_mask_passes <= 0:
            raise ValueError("soft_mask_passes must be positive")


def stable_pseudo_seed(
    training_seed: int,
    epoch: int,
    sample_identity: str,
    slot: str,
    parameter_name: str,
) -> int:
    """Return a process-stable, domain-separated 63-bit seed."""

    parts = (
        SAMPLING_PROTOCOL_VERSION,
        str(int(training_seed)),
        str(int(epoch)),
        str(sample_identity),
        str(slot),
        str(parameter_name),
    )
    encoded = b"\x00".join(part.encode("utf-8") for part in parts)
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)


def _stateless_uniform(
    shape: Sequence[int],
    low: float,
    high: float,
    *,
    training_seed: int,
    epoch: int,
    sample_identity: str,
    slot: str,
    parameter_name: str,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        stable_pseudo_seed(
            training_seed,
            epoch,
            sample_identity,
            slot,
            parameter_name,
        )
    )
    return (
        torch.rand(tuple(shape), generator=generator, dtype=torch.float32)
        .mul(high - low)
        .add(low)
    )


def _soft_region(mask: torch.Tensor, config: PseudoDomainConfig) -> torch.Tensor:
    region = mask.float().unsqueeze(0).unsqueeze(0)
    padding = config.soft_mask_kernel // 2
    for _ in range(config.soft_mask_passes):
        region = F.avg_pool2d(
            region,
            kernel_size=config.soft_mask_kernel,
            stride=1,
            padding=padding,
        )
    return region.clamp(0.0, 1.0)
