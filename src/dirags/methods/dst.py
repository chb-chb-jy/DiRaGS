"""DST: shared positive-evidence support and selective P4 stabilization.

Loss operations, probe initialization, interpolation and detach paths
are copied from the formal original implementation.
"""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator
import torch
from torch import nn
from ..integrations.registry import evidence_axes, evidence_stages
import torch.nn.functional as F

DST_CHANNELS = 192


DST_SPATIAL_SIZE = 8


DST_CLASSES = 2


DST_LOSS_WEIGHT = 0.1


DST_TRAINABLE_PARAMETERS = 772


@dataclass(frozen=True)
class AxisStates:
    p2: torch.Tensor
    p4: torch.Tensor


@dataclass(frozen=True)
class CapturedStates:
    height: AxisStates
    width: AxisStates
    tsdf_prediction: torch.Tensor | None = None


def _restore_axis_state(sequence: torch.Tensor, axis: str) -> torch.Tensor:
    if sequence.ndim != 3:
        raise RuntimeError(f"{axis} DST hook expected a 3D sequence")
    folded_batch, length, channels = sequence.shape
    if length != DST_SPATIAL_SIZE or channels != DST_CHANNELS:
        raise RuntimeError(
            f"{axis} DST hook expected [B*8,8,192], got {tuple(sequence.shape)}"
        )
    if folded_batch % DST_SPATIAL_SIZE:
        raise RuntimeError("DST folded batch is not divisible by 8")
    batch = folded_batch // DST_SPATIAL_SIZE
    restored = sequence.reshape(batch, 8, 8, 192)
    if axis == "height":
        return restored.permute(0, 3, 2, 1).contiguous()
    if axis == "width":
        return restored.permute(0, 3, 1, 2).contiguous()
    raise ValueError(f"unsupported axis: {axis!r}")


class _StateCapture:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], torch.Tensor] = {}

    def record(self, axis: str, state: str, inputs: tuple[object, ...]) -> None:
        key = (axis, state)
        if key in self.values:
            raise RuntimeError(f"duplicate DST hook call for {axis} {state}")
        if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("unexpected DST hook interface")
        self.values[key] = _restore_axis_state(inputs[0], axis)

    def finish(self) -> CapturedStates:
        expected = {
            ("height", "p2"),
            ("height", "p4"),
            ("width", "p2"),
            ("width", "p4"),
        }
        if set(self.values) != expected:
            raise RuntimeError(f"incomplete DST capture: keys={sorted(self.values)}")
        return CapturedStates(
            height=AxisStates(
                self.values[("height", "p2")],
                self.values[("height", "p4")],
            ),
            width=AxisStates(
                self.values[("width", "p2")],
                self.values[("width", "p4")],
            ),
        )


@contextmanager
def capture_dst_states(base_model: nn.Module) -> Iterator[_StateCapture]:
    capture = _StateCapture()
    handles = []
    for axis, block in evidence_axes(base_model).items():
        handles.append(
            evidence_stages(block)["p2"].register_forward_pre_hook(
                lambda _module, inputs, axis=axis: capture.record(axis, "p2", inputs)
            )
        )
        handles.append(
            evidence_stages(block)["p4"].register_forward_pre_hook(
                lambda _module, inputs, axis=axis: capture.record(axis, "p4", inputs)
            )
        )
    try:
        yield capture
    finally:
        for handle in handles:
            handle.remove()


class DSTProbes(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.height = nn.Conv2d(192, 2, 1, bias=True)
            self.width = nn.Conv2d(192, 2, 1, bias=True)
        for parameter in self.parameters():
            nn.init.zeros_(parameter)
        if self.trainable_parameter_count != DST_TRAINABLE_PARAMETERS:
            raise RuntimeError("unexpected DST probe parameter count")

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


@dataclass(frozen=True)
class ClassBalancedHinge:
    total: torch.Tensor
    foreground: torch.Tensor
    background: torch.Tensor
    foreground_count: int
    background_count: int


@dataclass(frozen=True)
class DSTLossTerms:
    probe: torch.Tensor
    stable: torch.Tensor
    height_clean: ClassBalancedHinge
    height_pseudo: ClassBalancedHinge
    width_clean: ClassBalancedHinge
    width_pseudo: ClassBalancedHinge


def _probe_logits(
    probe: nn.Conv2d,
    feature: torch.Tensor,
    target: torch.Tensor,
    *,
    detach_feature: bool,
    detach_probe: bool,
) -> torch.Tensor:
    source = feature.detach() if detach_feature else feature
    if detach_probe:
        logits = F.conv2d(
            source,
            probe.weight.detach(),
            None if probe.bias is None else probe.bias.detach(),
        )
    else:
        logits = probe(source)
    return F.interpolate(
        logits,
        size=target.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )


def _masked_hinge(
    reference: torch.Tensor,
    current: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    count = int(mask.sum().item())
    if count == 0:
        return current.sum() * 0.0, 0
    return F.relu(reference[mask] - current[mask]).mean(), count


def _view_hinge(
    reference: torch.Tensor,
    current: torch.Tensor,
    protected: torch.Tensor,
    target: torch.Tensor,
) -> ClassBalancedHinge:
    foreground, foreground_count = _masked_hinge(
        reference,
        current,
        protected & (target == 1),
    )
    background, background_count = _masked_hinge(
        reference,
        current,
        protected & (target == 0),
    )
    return ClassBalancedHinge(
        0.5 * foreground + 0.5 * background,
        foreground,
        background,
        foreground_count,
        background_count,
    )


def compute_dst_losses(
    probes: DSTProbes,
    clean: CapturedStates,
    pseudo: CapturedStates,
    target: torch.Tensor,
    *,
    stable_probes: DSTProbes | None = None,
) -> DSTLossTerms:
    """Compute frozen DST-v1 losses with an optional slow evidence readout.

    ``probes`` always receives the original probe supervision.  When supplied,
    ``stable_probes`` is used consistently for every P2/P4 quantity in the
    stable branch.  Its parameters are detached by ``_probe_logits`` while the
    P4 features remain live, so the stable loss still backpropagates to the
    backbone.
    """

    target = target.long()
    signed = target.to(dtype=clean.height.p2.dtype).mul(2.0).sub(1.0)
    stable_probes = probes if stable_probes is None else stable_probes
    probe_losses = []
    stable_by_axis = {}
    for axis_name in ("height", "width"):
        probe = getattr(probes, axis_name)
        stable_probe = getattr(stable_probes, axis_name)
        clean_axis = getattr(clean, axis_name)
        pseudo_axis = getattr(pseudo, axis_name)
        axes = (clean_axis, pseudo_axis)
        for states in axes:
            for feature in (states.p2, states.p4):
                probe_losses.append(
                    F.cross_entropy(
                        _probe_logits(
                            probe,
                            feature,
                            target,
                            detach_feature=True,
                            detach_probe=False,
                        ),
                        target,
                    )
                )

        logits = []
        for states in axes:
            z2 = _probe_logits(
                stable_probe,
                states.p2,
                target,
                detach_feature=True,
                detach_probe=True,
            )
            z4 = _probe_logits(
                stable_probe,
                states.p4,
                target,
                detach_feature=False,
                detach_probe=True,
            )
            logits.append((z2, z4))
        margin2_clean = ((logits[0][0][:, 1] - logits[0][0][:, 0]) * signed).detach()
        margin2_pseudo = ((logits[1][0][:, 1] - logits[1][0][:, 0]) * signed).detach()
        reference = torch.minimum(margin2_clean, margin2_pseudo)
        protected = (margin2_clean > 0.0) & (margin2_pseudo > 0.0)
        view_losses = []
        for _z2, z4 in logits:
            margin4 = (z4[:, 1] - z4[:, 0]) * signed
            view_losses.append(_view_hinge(reference, margin4, protected, target))
        stable_by_axis[axis_name] = tuple(view_losses)

    probe_loss = sum(probe_losses) / 8.0
    height_clean, height_pseudo = stable_by_axis["height"]
    width_clean, width_pseudo = stable_by_axis["width"]
    height = 0.5 * height_clean.total + 0.5 * height_pseudo.total
    width = 0.5 * width_clean.total + 0.5 * width_pseudo.total
    return DSTLossTerms(
        probe=probe_loss,
        stable=0.5 * height + 0.5 * width,
        height_clean=height_clean,
        height_pseudo=height_pseudo,
        width_clean=width_clean,
        width_pseudo=width_pseudo,
    )


def dst_total_loss(
    clean_segmentation: torch.Tensor,
    pseudo_segmentation: torch.Tensor,
    terms: DSTLossTerms | None,
) -> torch.Tensor:
    segmentation = 0.5 * (clean_segmentation + pseudo_segmentation)
    if terms is None:
        return segmentation
    return segmentation + terms.probe + DST_LOSS_WEIGHT * terms.stable
