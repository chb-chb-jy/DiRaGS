"""Formal GCS-v1 epoch loop; source-IoU selection and resume schema preserved."""

from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Callable
import torch
from torch import nn
from . import checkpoints as frozen_training
from .dual_view import DSTTrainingWrapper, paired_training_forward
from ..methods.dst import DST_LOSS_WEIGHT, compute_dst_losses, dst_total_loss
from ..methods.pdc_common import PseudoDomainConfig
from ..methods.gcs import GCS_COMPOSITION, backward_gcs
from ..evaluation.metrics import METRIC_KEYS
from .common import (
    _assert_finite_tensor,
    _capture_rng_state,
    atomic_torch_save,
    atomic_write_json,
    atomic_write_text,
    evaluate_model,
    scheduler_position,
    load_training_checkpoint,
)


def _projected_dst_norm(
    *, dot: float, segmentation_norm: float, dst_norm: float
) -> float:
    """Return the norm after GC projection using already-computed scalars."""
    if segmentation_norm <= 0.0 or dst_norm < 0.0:
        raise ValueError("invalid gradient norm for temporal forensic logging")
    if dot >= 0.0:
        return float(dst_norm)
    projected_sq = dst_norm * dst_norm - dot * dot / (
        segmentation_norm * segmentation_norm
    )
    return max(0.0, projected_sq) ** 0.5


def train_gcs(
    *,
    model: DSTTrainingWrapper,
    train_loader,
    val_loader,
    test_loader,
    criterion: nn.Module,
    optimizer,
    scheduler,
    config: dict[str, Any],
    pseudo_config: PseudoDomainConfig,
    pseudo_builder: Callable,
    training_seed: int,
    training_config_hash: str,
    inference_config_hash: str,
    split_hash: str,
    output_dir: Path,
    device: torch.device,
    resume: Path | None = None,
    max_steps: int | None = None,
    pause_after_epochs: int | None = None,
) -> dict[str, Any]:
    """Train GCS with the shared dual-view trainer."""
    if not isinstance(model, DSTTrainingWrapper):
        raise TypeError("GCS requires DSTTrainingWrapper")
    if float(config["cstp_loss_weight"]) != DST_LOSS_WEIGHT:
        raise ValueError("GCS must retain the frozen DST-v1 loss weight")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")
    total_epochs = int(config["epochs"])
    if pause_after_epochs is not None and (not 0 < pause_after_epochs <= total_epochs):
        raise ValueError("pause_after_epochs is outside the frozen epoch range")
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise FileNotFoundError("GCS output directory was not prepared")
    composition = config.get("gradient_composition")
    if composition != GCS_COMPOSITION:
        raise ValueError("unsupported GCS gradient composition")
    if "gradient_reference_ema_rho" in config:
        raise ValueError("GC-v1 cannot include an EMA-reference coefficient")
    if "dst_leverage_floor_ratio" in config:
        raise ValueError("non-leverage-floor GCS cannot include a floor ratio")
    model.to(device)
    start_epoch = 0
    best_val_iou = float("-inf")
    history: list[dict[str, Any]] = []
    if resume is not None:
        start_epoch, best_val_iou, history = frozen_training._load_checkpoint(
            resume,
            model,
            optimizer,
            scheduler,
            mode=frozen_training.CSTP,
            training_config_hash=training_config_hash,
            inference_config_hash=inference_config_hash,
            split_hash=split_hash,
            train_loader=train_loader,
        )
    total_steps = 0
    patience_counter = 0
    stopped_by_limit = False
    stop_epoch = total_epochs if pause_after_epochs is None else pause_after_epochs
    for epoch in range(start_epoch, stop_epoch):
        model.train()
        running = {
            "total": 0.0,
            "clean_seg": 0.0,
            "pseudo_seg": 0.0,
            "probe": 0.0,
            "stable": 0.0,
            "gc_cosine": 0.0,
            "gc_conflicts": 0,
            "gc_batches": 0,
        }
        protected = {"h_fg": 0, "h_bg": 0, "w_fg": 0, "w_bg": 0}
        sample_count = 0
        for batch_index, (images, masks, identities) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            pseudo = pseudo_builder(
                images,
                masks,
                list(identities),
                training_seed=training_seed,
                epoch=epoch,
                config=pseudo_config,
            )
            optimizer.zero_grad()
            with paired_training_forward(model, images, pseudo) as paired:
                if paired.clean_states is None or paired.pseudo_states is None:
                    raise RuntimeError("GCS state capture is missing")
                clean_seg = criterion(paired.clean_outputs, masks)
                pseudo_seg = criterion(paired.pseudo_outputs, masks)
                terms = compute_dst_losses(
                    model.probes,
                    paired.clean_states,
                    paired.pseudo_states,
                    masks,
                    stable_probes=None,
                )
                segmentation = 0.5 * (clean_seg + pseudo_seg)
                total = dst_total_loss(clean_seg, pseudo_seg, terms)
                for name, value in (
                    ("clean segmentation", clean_seg),
                    ("pseudo segmentation", pseudo_seg),
                    ("probe", terms.probe),
                    ("stable", terms.stable),
                    ("total", total),
                ):
                    _assert_finite_tensor(value, f"{name} loss")
                stats = backward_gcs(
                    model,
                    segmentation_loss=segmentation,
                    probe_loss=terms.probe,
                    stable_loss=terms.stable,
                    stable_weight=float(config["cstp_loss_weight"]),
                )
            backbone_preclip = nn.utils.clip_grad_norm_(
                tuple(model.base_model.parameters()),
                float(config["grad_clip_norm"]),
                error_if_nonfinite=True,
            )
            nn.utils.clip_grad_norm_(
                tuple(model.probes.parameters()),
                float(config["grad_clip_norm"]),
                error_if_nonfinite=True,
            )
            optimizer.step()
            scheduler.step(scheduler_position(epoch, batch_index, len(train_loader)))
            batch_size = images.shape[0]
            running["total"] += float(total.item()) * batch_size
            running["clean_seg"] += float(clean_seg.item()) * batch_size
            running["pseudo_seg"] += float(pseudo_seg.item()) * batch_size
            running["probe"] += float(terms.probe.item()) * batch_size
            running["stable"] += float(terms.stable.item()) * batch_size
            if stats is None:
                raise RuntimeError("GC-v1 statistics are missing")
            actual_cosine = stats.cosine
            actual_projected = stats.projected
            segmentation_norm = stats.segmentation_norm
            dst_norm = stats.dst_norm
            projected_norm = _projected_dst_norm(
                dot=stats.dot, segmentation_norm=segmentation_norm, dst_norm=dst_norm
            )
            adjusted_norm = projected_norm
            running["gc_cosine"] += actual_cosine
            running["gc_conflicts"] += int(actual_projected)
            running["gc_batches"] += 1
            protected["h_fg"] += terms.height_clean.foreground_count
            protected["h_bg"] += terms.height_clean.background_count
            protected["w_fg"] += terms.width_clean.foreground_count
            protected["w_bg"] += terms.width_clean.background_count
            sample_count += batch_size
            total_steps += 1
            if max_steps is not None and total_steps >= max_steps:
                stopped_by_limit = True
                break
        if sample_count == 0 or running["gc_batches"] == 0:
            raise ValueError("training dataloader is empty")
        validation = evaluate_model(model, val_loader, criterion, device)
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": running["total"] / sample_count,
            "train_clean_seg_loss": running["clean_seg"] / sample_count,
            "train_pseudo_seg_loss": running["pseudo_seg"] / sample_count,
            "train_probe_loss": running["probe"] / sample_count,
            "train_stable_loss": running["stable"] / sample_count,
            "train_gc_conflict_fraction": running["gc_conflicts"]
            / running["gc_batches"],
            "train_gc_mean_cosine": running["gc_cosine"] / running["gc_batches"],
            **{f"train_protected_{key}": value for key, value in protected.items()},
            "val_loss": validation["loss"],
            **{f"val_{key}": validation[key] for key in METRIC_KEYS},
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        improved = validation["iou"] > best_val_iou
        if improved:
            best_val_iou = validation["iou"]
            patience_counter = 0
        else:
            patience_counter += 1
        payload = frozen_training._checkpoint_payload(
            model,
            optimizer,
            scheduler,
            mode=frozen_training.CSTP,
            epoch=epoch,
            best_val_iou=best_val_iou,
            history=history,
            training_config_hash=training_config_hash,
            inference_config_hash=inference_config_hash,
            split_hash=split_hash,
            rng_state=_capture_rng_state(train_loader),
        )
        atomic_torch_save(payload, output_dir / "last_checkpoint.pth")
        if improved:
            atomic_torch_save(payload, output_dir / "best_model.pth")
        atomic_write_text(
            frozen_training._history_csv(history), output_dir / "history.csv"
        )
        patience = config["early_stopping_patience"]
        if stopped_by_limit or (
            patience is not None and patience_counter >= int(patience)
        ):
            break
    training_only_parameters = model.probes.trainable_parameter_count
    if pause_after_epochs is not None and len(history) < total_epochs:
        result = {
            "status": "paused",
            "best_val_iou": float(best_val_iou),
            "completed_epochs": len(history),
            "training_only_parameters": training_only_parameters,
        }
        atomic_write_json(result, output_dir / "progress.json")
        return result
    best_path = output_dir / "best_model.pth"
    if not best_path.is_file():
        raise RuntimeError("GCS training produced no best checkpoint")
    best = load_training_checkpoint(best_path, map_location=device)
    model.base_model.load_state_dict(best["model_state_dict"], strict=True)
    model.probes.load_state_dict(best["cstp_probe_state_dict"], strict=True)
    test_metrics = evaluate_model(model, test_loader, criterion, device)
    result: dict[str, Any] = {
        "status": "max_steps" if stopped_by_limit else "complete",
        "best_val_iou": float(best["best_val_iou"]),
        "completed_epochs": len(history),
        "training_only_parameters": training_only_parameters,
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    progress = output_dir / "progress.json"
    if progress.exists():
        progress.unlink()
    atomic_write_json(result, output_dir / "metrics.json")
    return result
