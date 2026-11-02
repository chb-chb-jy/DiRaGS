"""Full-model DiRaGS configuration and original run identity mappings."""

from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

EXPERIMENTS = {
    "D_pdc_gcs": {
        "variant": "RPP-V2-D1-GC-DST-v1-CLEAN128",
        "training_mode": "cstp",
        "confirmation_protocol": "d1-compatible-dst-v2-candidate2-gc-seed42",
        "gradient_composition": "project-conflicting-dst-gradient-v1",
    }
}
TRAINING_FIELDS = (
    "dataset",
    "image_size",
    "batch_size",
    "epochs",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "grad_clip_norm",
    "scheduler",
    "scheduler_t0_epochs",
    "scheduler_tmult",
    "scheduler_eta_min",
    "deep_weights",
    "selection_metric",
    "early_stopping_patience",
    "num_workers",
    "pin_memory",
    "split_seed",
    "mask_threshold",
    "pseudo_domain_version",
    "pseudo_gain_min",
    "pseudo_gain_max",
    "pseudo_bias_min",
    "pseudo_bias_max",
    "pseudo_gamma_min",
    "pseudo_gamma_max",
    "pseudo_illumination_amplitude",
    "pseudo_illumination_grid_size",
    "pseudo_soft_mask_kernel",
    "pseudo_soft_mask_passes",
)


def canonical_hash(value):
    data = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("paper config must be a JSON object")
    if config.get("experiment") != "D_pdc_gcs":
        raise ValueError(
            "This edition supports the full DiRaGS configuration D_pdc_gcs"
        )
    expected = set(TRAINING_FIELDS) | {"experiment", "training_seed"}
    expected.add("dst_loss_weight")
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError("paper config fields are missing or unexpected")
    if config["experiment"] not in EXPERIMENTS:
        raise ValueError("This edition supports only the complete DiRaGS model")
    if config["dataset"] != "Kvasir-SEG" or config["optimizer"] != "AdamW":
        raise ValueError("the paper protocol uses Kvasir-SEG and AdamW")
    if (
        config["scheduler"] != "CosineAnnealingWarmRestarts"
        or config["selection_metric"] != "val_iou"
    ):
        raise ValueError(
            "scheduler or source validation selection rule differs from the paper protocol"
        )
    for key in [
        "image_size",
        "batch_size",
        "epochs",
        "scheduler_t0_epochs",
        "scheduler_tmult",
        "pseudo_illumination_grid_size",
        "pseudo_soft_mask_kernel",
        "pseudo_soft_mask_passes",
    ]:
        if (
            isinstance(config[key], bool)
            or not isinstance(config[key], int)
            or config[key] <= 0
        ):
            raise ValueError(f"{key} must be a positive integer")
    if config["image_size"] != 256 or config["mask_threshold"] != 128:
        raise ValueError("the paper uses image_size=256 and mask_threshold=128")
    if (
        type(config["training_seed"]) is not int
        or config["training_seed"] != 42
        or type(config["split_seed"]) is not int
        or (config["split_seed"] != 42)
    ):
        raise ValueError(
            "these released paper configurations use training seed and split seed 42"
        )
    if (
        type(config["num_workers"]) is not int
        or config["num_workers"] < 0
        or type(config["pin_memory"]) is not bool
    ):
        raise ValueError("invalid data loader settings")
    for key in [
        "learning_rate",
        "weight_decay",
        "grad_clip_norm",
        "scheduler_eta_min",
        "dst_loss_weight",
        "pseudo_gain_min",
        "pseudo_gain_max",
        "pseudo_bias_min",
        "pseudo_bias_max",
        "pseudo_gamma_min",
        "pseudo_gamma_max",
        "pseudo_illumination_amplitude",
    ]:
        value = config[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or (not math.isfinite(value))
        ):
            raise ValueError(f"{key} must be finite numeric")
    if (
        min(
            config["learning_rate"],
            config["grad_clip_norm"],
            config["scheduler_eta_min"],
        )
        <= 0
        or config["weight_decay"] < 0
    ):
        raise ValueError("invalid optimizer settings")
    weights = config["deep_weights"]
    if (
        not isinstance(weights, list)
        or len(weights) != 4
        or any(
            (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or (not math.isfinite(v))
                or (v < 0)
                for v in weights
            )
        )
        or (abs(sum(weights) - 1) > 1e-09)
    ):
        raise ValueError(
            "deep_weights must contain four nonnegative finite values summing to one"
        )
    patience = config["early_stopping_patience"]
    if patience is not None and (type(patience) is not int or patience <= 0):
        raise ValueError("early_stopping_patience must be null or a positive integer")
    weight = 0.1
    if config["dst_loss_weight"] != weight:
        raise ValueError("The complete DiRaGS model uses DST weight 0.1")
    if config["pseudo_domain_version"] != "regional-aligned-v1":
        raise ValueError("unknown pseudo-domain protocol version")
    if (
        not 0 < config["pseudo_gain_min"] <= config["pseudo_gain_max"]
        or not 0 < config["pseudo_gamma_min"] <= config["pseudo_gamma_max"]
        or config["pseudo_bias_min"] > config["pseudo_bias_max"]
    ):
        raise ValueError("invalid PDC parameter range")
    if (
        not 0 <= config["pseudo_illumination_amplitude"] <= 1
        or config["pseudo_illumination_grid_size"] < 2
        or config["pseudo_soft_mask_kernel"] <= 1
        or (config["pseudo_soft_mask_kernel"] % 2 != 1)
    ):
        raise ValueError("invalid illumination grid or soft mask")


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def legacy_training_config(config):
    """Preserve original scientific values and checkpoint identity hashes."""
    validate_config(config)
    return {
        **{key: config[key] for key in TRAINING_FIELDS},
        "base_variant": "A0-CLEAN128",
        "module1_candidate": "rpp_v2_d1_directional_copula",
        "training_seeds": [config["training_seed"]],
        "cstp_loss_weight": config["dst_loss_weight"],
        **EXPERIMENTS[config["experiment"]],
    }


def inference_config(config):
    runtime = legacy_training_config(config)
    keys = [key for key in TRAINING_FIELDS if not key.startswith("pseudo_")]
    return {
        **{key: runtime[key] for key in keys},
        "training_seeds": runtime["training_seeds"],
        "variant": "A0-CLEAN128",
    }


def pseudo_config(config):
    from .methods.pdc_common import PseudoDomainConfig

    result = PseudoDomainConfig(
        **{
            key.removeprefix("pseudo_"): value
            for key, value in config.items()
            if key.startswith("pseudo_") and key != "pseudo_domain_version"
        }
    )
    result.validate()
    return result
