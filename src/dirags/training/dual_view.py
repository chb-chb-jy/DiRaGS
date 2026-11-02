"""Paired training forwards with original RNG and BatchNorm behavior.

The formal PDC-only and DST wrappers are retained. The TSDF-only branch
is omitted; unsupported capture requests still raise the original error.
"""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator
import torch
from torch import nn
from ..integrations.registry import evidence_axes
from torch.nn.modules.batchnorm import _BatchNorm
from ..methods.dst import CapturedStates, DSTProbes, capture_dst_states


class PseudoControlTrainingWrapper(nn.Module):
    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model

    def _disable_axial_checkpointing(self) -> None:
        for block in evidence_axes(self.base_model).values():
            block.eval()

    def train(self, mode: bool = True) -> "PseudoControlTrainingWrapper":
        super().train(mode)
        if mode:
            self._disable_axial_checkpointing()
        return self

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def training_forward(
        self,
        images: torch.Tensor,
        *,
        capture_tsdf: bool = False,
    ):
        if capture_tsdf:
            raise ValueError("pseudo-control wrapper cannot capture TSDF")
        if not self.training:
            raise RuntimeError("training_forward requires train mode")
        self._disable_axial_checkpointing()
        return self.base_model(images, return_aux=True), None


class DSTTrainingWrapper(PseudoControlTrainingWrapper):
    def __init__(self, base_model: nn.Module) -> None:
        super().__init__(base_model)
        self.probes = DSTProbes()

    def training_forward(
        self,
        images: torch.Tensor,
        *,
        capture_tsdf: bool = False,
    ):
        if capture_tsdf:
            raise ValueError("plain DST wrapper cannot capture TSDF")
        if not self.training:
            raise RuntimeError("training_forward requires train mode")
        self._disable_axial_checkpointing()
        with capture_dst_states(self.base_model) as capture:
            outputs = self.base_model(images, return_aux=True)
        states = capture.finish()
        expected = (images.shape[0], 192, 8, 8)
        for axis in (states.height, states.width):
            if axis.p2.shape != expected or axis.p4.shape != expected:
                raise RuntimeError("DST restored state shape mismatch")
        return outputs, states


def _capture_torch_rng() -> tuple[torch.Tensor, list[torch.Tensor]]:
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return torch.get_rng_state().clone(), [state.clone() for state in cuda]


def _restore_torch_rng(state: tuple[torch.Tensor, list[torch.Tensor]]) -> None:
    torch.set_rng_state(state[0])
    if state[1]:
        torch.cuda.set_rng_state_all(state[1])


def _capture_bn_buffers(model: nn.Module) -> dict[str, tuple[torch.Tensor | None, ...]]:
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, _BatchNorm):
            result[name] = tuple(
                None if value is None else value.detach().clone()
                for value in (
                    module.running_mean,
                    module.running_var,
                    module.num_batches_tracked,
                )
            )
    return result


def _restore_bn_buffers(
    model: nn.Module,
    snapshot: dict[str, tuple[torch.Tensor | None, ...]],
) -> None:
    modules = dict(model.named_modules())
    with torch.no_grad():
        for name, values in snapshot.items():
            module = modules[name]
            for attribute, saved in zip(
                ("running_mean", "running_var", "num_batches_tracked"), values
            ):
                current = getattr(module, attribute)
                if saved is None:
                    if current is not None:
                        raise RuntimeError("BN buffer None-state mismatch")
                else:
                    current.copy_(saved)


@dataclass(frozen=True)
class PairedForward:
    clean_outputs: object
    pseudo_outputs: object
    clean_states: CapturedStates | None
    pseudo_states: CapturedStates | None


@contextmanager
def paired_training_forward(
    model: PseudoControlTrainingWrapper,
    clean: torch.Tensor,
    pseudo: torch.Tensor,
) -> Iterator[PairedForward]:
    """Yield paired outputs and restore clean-view BN state after backward."""

    if not model.training:
        raise RuntimeError("paired training forward requires train mode")
    s0 = _capture_torch_rng()
    clean_outputs, clean_states = model.training_forward(clean)
    s1 = _capture_torch_rng()
    clean_bn = _capture_bn_buffers(model.base_model)
    _restore_torch_rng(s0)
    try:
        pseudo_outputs, pseudo_states = model.training_forward(pseudo)
    except Exception:
        _restore_bn_buffers(model.base_model, clean_bn)
        _restore_torch_rng(s1)
        raise

    # Preserve the random stream of a single clean-view forward.  BN buffers
    # must not be restored here: copy_ would change their version counters
    # while the clean/pseudo autograd graphs are still live.
    _restore_torch_rng(s1)
    try:
        yield PairedForward(
            clean_outputs,
            pseudo_outputs,
            clean_states,
            pseudo_states,
        )
    finally:
        _restore_bn_buffers(model.base_model, clean_bn)
