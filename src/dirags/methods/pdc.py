"""PDC: coordinate-wise shared directions and independent regional magnitudes.

Replay field names retain their original spelling for parity checks.
Protocol string values are preserved to retain sample identity.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import torch
import torch.nn.functional as F
from .pdc_common import PseudoDomainConfig, _soft_region, _stateless_uniform

PDC_VERSION = "RPP-v2-D1-Directional-Copula-v1"


PDC_SHARED_DIRECTION_SLOT = "rpp_v2_d1_shared_direction"


REGION_SLOTS = ("foreground", "background")


@dataclass(frozen=True)
class ParameterSpec:
    shape: tuple[int, ...]
    low: float
    high: float
    key: str

    @property
    def span(self) -> float:
        return self.high - self.low

    @property
    def identity(self) -> float:
        return 0.5 * (self.low + self.high)


@dataclass(frozen=True)
class DirectionalParameter:
    rpp_v1_foreground: torch.Tensor
    rpp_v1_background: torch.Tensor
    d1_foreground: torch.Tensor
    d1_background: torch.Tensor
    shared_direction: torch.Tensor
    foreground_magnitude: torch.Tensor
    background_magnitude: torch.Tensor


@dataclass(frozen=True)
class PairedDirectionalReplay:
    rpp_v1: torch.Tensor
    d1: torch.Tensor
    rpp_v1_foreground: torch.Tensor
    rpp_v1_background: torch.Tensor
    d1_foreground: torch.Tensor
    d1_background: torch.Tensor
    soft_region: torch.Tensor
    parameters: Mapping[str, DirectionalParameter]


def parameter_specs(config: PseudoDomainConfig) -> dict[str, ParameterSpec]:
    """Return the unchanged RPP-v1 parameter shapes and ranges."""

    config.validate()
    return {
        "gain": ParameterSpec(
            (1, 3, 1, 1), config.gain_min, config.gain_max, "channel_gain"
        ),
        "bias": ParameterSpec(
            (1, 3, 1, 1), config.bias_min, config.bias_max, "channel_bias"
        ),
        "gamma": ParameterSpec(
            (1, 1, 1, 1), config.gamma_min, config.gamma_max, "gamma"
        ),
        "illumination": ParameterSpec(
            (1, 1, config.illumination_grid_size, config.illumination_grid_size),
            -config.illumination_amplitude,
            config.illumination_amplitude,
            "illumination_grid",
        ),
    }


def _unit_uniform(
    spec: ParameterSpec,
    *,
    training_seed: int,
    epoch: int,
    sample_identity: str,
    slot: str,
) -> torch.Tensor:
    return _stateless_uniform(
        spec.shape,
        0.0,
        1.0,
        training_seed=training_seed,
        epoch=epoch,
        sample_identity=sample_identity,
        slot=slot,
        parameter_name=spec.key,
    )


def directional_parameter(
    spec: ParameterSpec,
    *,
    training_seed: int,
    epoch: int,
    sample_identity: str,
) -> DirectionalParameter:
    """Draw RPP-v1 and D1 parameters from paired stateless random streams."""

    foreground_magnitude = _unit_uniform(
        spec,
        training_seed=training_seed,
        epoch=epoch,
        sample_identity=sample_identity,
        slot="foreground",
    )
    background_magnitude = _unit_uniform(
        spec,
        training_seed=training_seed,
        epoch=epoch,
        sample_identity=sample_identity,
        slot="background",
    )
    direction_uniform = _unit_uniform(
        spec,
        training_seed=training_seed,
        epoch=epoch,
        sample_identity=sample_identity,
        slot=PDC_SHARED_DIRECTION_SLOT,
    )
    shared_direction = direction_uniform.ge(0.5).to(torch.float32)
    d1_foreground_unit = 0.5 * (shared_direction + foreground_magnitude)
    d1_background_unit = 0.5 * (shared_direction + background_magnitude)

    def scale(unit: torch.Tensor) -> torch.Tensor:
        return unit.mul(spec.span).add(spec.low)

    result = DirectionalParameter(
        rpp_v1_foreground=scale(foreground_magnitude),
        rpp_v1_background=scale(background_magnitude),
        d1_foreground=scale(d1_foreground_unit),
        d1_background=scale(d1_background_unit),
        shared_direction=shared_direction,
        foreground_magnitude=foreground_magnitude,
        background_magnitude=background_magnitude,
    )
    if not all(bool(torch.isfinite(value).all()) for value in result.__dict__.values()):
        raise FloatingPointError("D1 parameter generation produced nonfinite values")
    for value in (
        result.rpp_v1_foreground,
        result.rpp_v1_background,
        result.d1_foreground,
        result.d1_background,
    ):
        if bool((value < spec.low).any()) or bool((value > spec.high).any()):
            raise RuntimeError("D1 parameter left its frozen RPP-v1 range")
    # Validate against the sampled direction itself.  Do not reconstruct a
    # binary side with ``>= identity``: after affine scaling in float32, a
    # value infinitesimally below the midpoint can round to the midpoint.
    # Equality is therefore a valid boundary value for either half.
    high_direction = shared_direction.to(torch.bool)
    fg_wrong_half = torch.where(
        high_direction,
        result.d1_foreground < spec.identity,
        result.d1_foreground > spec.identity,
    )
    bg_wrong_half = torch.where(
        high_direction,
        result.d1_background < spec.identity,
        result.d1_background > spec.identity,
    )
    if bool(fg_wrong_half.any()) or bool(bg_wrong_half.any()):
        raise RuntimeError("D1 parameter left its sampled shared direction half")
    if tuple(shared_direction.shape) != spec.shape:
        raise RuntimeError("D1 direction is not coordinate-wise")
    return result


def apply_appearance_parameters(
    image: torch.Tensor,
    parameters: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Apply the unchanged RPP-v1 pixel operator to explicit parameters."""

    required = {"gain", "bias", "gamma", "illumination"}
    if set(parameters) != required:
        raise ValueError("appearance parameter set mismatch")
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("D1 appearance input must have shape [1,3,H,W]")
    converted = {
        key: value.to(device=image.device, dtype=image.dtype)
        for key, value in parameters.items()
    }
    field = F.interpolate(
        converted["illumination"],
        size=image.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    transformed = (
        (image * converted["gain"] + converted["bias"])
        .clamp(1.0e-6, 1.0)
        .pow(converted["gamma"])
    )
    return (transformed * (1.0 + field)).clamp(0.0, 1.0)


def build_paired_raw_replay(
    raw_image: torch.Tensor,
    mask: torch.Tensor,
    *,
    training_seed: int,
    epoch: int,
    sample_identity: str,
    config: PseudoDomainConfig,
) -> PairedDirectionalReplay:
    """Build paired raw-RGB RPP-v1 and D1 views without model access."""

    if raw_image.ndim != 4 or raw_image.shape[0] != 1 or raw_image.shape[1] != 3:
        raise ValueError("raw_image must have shape [1,3,H,W]")
    if mask.shape != raw_image.shape[-2:]:
        raise ValueError("mask and raw image spatial shapes differ")
    if not bool(torch.isfinite(raw_image).all()):
        raise ValueError("raw image contains a nonfinite value")
    if bool((raw_image < 0.0).any()) or bool((raw_image > 1.0).any()):
        raise ValueError("raw image must be bounded in [0,1]")

    samples = {
        family: directional_parameter(
            spec,
            training_seed=training_seed,
            epoch=epoch,
            sample_identity=str(sample_identity),
        )
        for family, spec in parameter_specs(config).items()
    }

    def values(attribute: str) -> dict[str, torch.Tensor]:
        return {
            family: getattr(parameter, attribute)
            for family, parameter in samples.items()
        }

    rpp_v1_foreground = apply_appearance_parameters(
        raw_image, values("rpp_v1_foreground")
    )
    rpp_v1_background = apply_appearance_parameters(
        raw_image, values("rpp_v1_background")
    )
    d1_foreground = apply_appearance_parameters(raw_image, values("d1_foreground"))
    d1_background = apply_appearance_parameters(raw_image, values("d1_background"))
    soft = _soft_region(mask.detach().cpu(), config).to(
        device=raw_image.device, dtype=raw_image.dtype
    )
    rpp_v1 = soft * rpp_v1_foreground + (1.0 - soft) * rpp_v1_background
    d1 = soft * d1_foreground + (1.0 - soft) * d1_background
    for label, value in (("RPP-v1", rpp_v1), ("D1", d1)):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"{label} replay produced nonfinite values")
        if bool((value < 0.0).any()) or bool((value > 1.0).any()):
            raise RuntimeError(f"{label} replay left raw-RGB bounds")
    return PairedDirectionalReplay(
        rpp_v1=rpp_v1,
        d1=d1,
        rpp_v1_foreground=rpp_v1_foreground,
        rpp_v1_background=rpp_v1_background,
        d1_foreground=d1_foreground,
        d1_background=d1_background,
        soft_region=soft,
        parameters=samples,
    )
