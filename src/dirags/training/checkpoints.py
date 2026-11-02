"""GCS training checkpoint and history helpers; original state schema retained."""
from __future__ import annotations
import csv
import io
from pathlib import Path
from typing import Any
import torch
from .dual_view import DSTTrainingWrapper, PseudoControlTrainingWrapper
from .common import _restore_rng_state, load_training_checkpoint

CSTP = "cstp"

def _history_csv(history: list[dict[str, Any]]) -> str:
    if not history:
        return ""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(history[0]))
    writer.writeheader()
    writer.writerows(history)
    return stream.getvalue()


def _checkpoint_payload(
    model: PseudoControlTrainingWrapper,
    optimizer,
    scheduler,
    *,
    mode: str,
    epoch: int,
    best_val_iou: float,
    history: list[dict[str, Any]],
    training_config_hash: str,
    inference_config_hash: str,
    split_hash: str,
    rng_state: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "epoch": epoch,
        "next_epoch": epoch + 1,
        "model_state_dict": model.base_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_iou": best_val_iou,
        "history": history,
        "config_hash": inference_config_hash,
        "cstp_training_config_hash": training_config_hash,
        "split_hash": split_hash,
        "rng_state": rng_state,
        "training_variant": mode,
    }
    if isinstance(model, DSTTrainingWrapper):
        payload["cstp_probe_state_dict"] = model.probes.state_dict()
        payload["training_only_parameters"] = model.probes.trainable_parameter_count
    else:
        payload["training_only_parameters"] = 0
    return payload


def _load_checkpoint(
    path: Path,
    model: PseudoControlTrainingWrapper,
    optimizer,
    scheduler,
    *,
    mode: str,
    training_config_hash: str,
    inference_config_hash: str,
    split_hash: str,
    train_loader,
) -> tuple[int, float, list[dict[str, Any]]]:
    checkpoint = load_training_checkpoint(
        Path(path), map_location=next(model.parameters()).device
    )
    expected = {
        "training_variant": mode,
        "cstp_training_config_hash": training_config_hash,
        "config_hash": inference_config_hash,
        "split_hash": split_hash,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"resume checkpoint identity mismatch: {key}")
    model.base_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if isinstance(model, DSTTrainingWrapper):
        model.probes.load_state_dict(checkpoint["cstp_probe_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if checkpoint.get("rng_state") is not None:
        _restore_rng_state(checkpoint["rng_state"], train_loader)
    return (
        int(checkpoint["next_epoch"]),
        float(checkpoint["best_val_iou"]),
        list(checkpoint.get("history", [])),
    )
