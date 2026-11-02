"""Original deterministic training and atomic checkpoint utilities."""

from __future__ import annotations
import hashlib
import inspect
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import torch
from torch import nn
from ..evaluation.metrics import METRIC_KEYS, compute_batch_metrics


def canonical_json_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def scheduler_position(epoch: int, batch_index: int, batches_per_epoch: int) -> float:
    if (
        epoch < 0
        or batch_index < 0
        or batches_per_epoch <= 0
        or batch_index >= batches_per_epoch
    ):
        raise ValueError("invalid scheduler position indices")
    return epoch + (batch_index + 1) / batches_per_epoch


def atomic_torch_save(payload: object, destination: Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_text(text: str, destination: Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(value: object, destination: Path) -> None:
    text = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    atomic_write_text(text + "\n", destination)


def _assert_finite_tensor(tensor: torch.Tensor, label: str) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{label} contains nonfinite values")


def _assert_finite_gradients(parameters: Iterable[nn.Parameter]) -> None:
    for parameter in parameters:
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("gradients contain nonfinite values")


def _capture_rng_state(train_loader) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    generator = getattr(train_loader, "generator", None)
    state["train_loader_generator"] = (
        generator.get_state() if generator is not None else None
    )
    return state


def _restore_rng_state(state: dict[str, Any], train_loader) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_state])
    generator_state = state.get("train_loader_generator")
    generator = getattr(train_loader, "generator", None)
    if generator_state is not None:
        if generator is None:
            raise ValueError("resume checkpoint requires a train loader generator")
        generator.set_state(generator_state.cpu())


def load_training_checkpoint(path, map_location):
    """Load a run-owned checkpoint including Python/NumPy optimizer RNG state."""
    options = (
        {"weights_only": False}
        if "weights_only" in inspect.signature(torch.load).parameters
        else {}
    )
    return torch.load(Path(path), map_location=map_location, **options)


@torch.no_grad()
def evaluate_model(model, dataloader, criterion, device, boundary_tolerance=2):
    """Original source validation/test averaging for the formal segmentation path."""
    model.eval()
    totals = {"loss": 0.0, **{key: 0.0 for key in METRIC_KEYS}}
    sample_count = 0
    for images, masks in dataloader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        outputs = model(images, return_aux=True)
        _assert_finite_tensor(outputs.logits, "logits")
        for deep_logits in outputs.deep_logits:
            _assert_finite_tensor(deep_logits, "deep logits")
        segmentation = criterion(outputs, masks)
        _assert_finite_tensor(segmentation, "segmentation loss")
        batch_size = images.shape[0]
        batch_metrics = compute_batch_metrics(
            outputs.logits, masks, boundary_tolerance=boundary_tolerance
        )
        totals["loss"] += float(segmentation.item()) * batch_size
        for key in METRIC_KEYS:
            totals[key] += batch_metrics[key] * batch_size
        sample_count += batch_size
    if sample_count == 0:
        raise ValueError("evaluation dataloader is empty")
    return {key: float(value / sample_count) for key, value in totals.items()}
