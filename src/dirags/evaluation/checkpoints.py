"""Validate full-model DiRaGS training-to-inference identity."""

from __future__ import annotations
import json
from pathlib import Path
from ..config import canonical_hash, legacy_training_config, inference_config


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {Path(path).name}")
    return value


def validate_checkpoint_payload(checkpoint, config, split_hash, *, inference_hash=None):
    runtime = legacy_training_config(config)
    expected_hash = inference_hash or canonical_hash(inference_config(config))
    expected = {
        "cstp_training_config_hash": canonical_hash(runtime),
        "config_hash": canonical_hash(inference_config(config)),
        "split_hash": split_hash,
        "training_variant": runtime["training_mode"],
        "training_only_parameters": 772,
    }
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("model_state_dict"), dict
    ):
        raise ValueError("checkpoint must contain model_state_dict")
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"checkpoint identity mismatch: {key}")
    if ("cstp_probe_state_dict" in checkpoint) != True:
        raise ValueError("checkpoint DST probe presence changed")


def load_run_checkpoint(
    checkpoint_path,
    run_dir,
    config,
    *,
    device,
    allow_incomplete=False,
    require_paper_checkpoint=False,
):
    from ..training.common import load_training_checkpoint
    from ..datasets.splits import validate_split_manifest
    from .ood import checkpoint_sha256

    package = Path(__file__).resolve().parents[3]
    portable = read_json(package / "splits/kvasir_seed42.json")
    validate_split_manifest(portable)
    run = Path(run_dir).expanduser().resolve()
    file = Path(checkpoint_path).expanduser().resolve()
    required = [
        "best_model.pth",
        "config_snapshot.json",
        "split_snapshot.json",
        "environment.json",
        "metrics.json",
    ]
    for name in required:
        if not (run / name).is_file():
            raise FileNotFoundError(f"source run is missing {name}")
    if not file.is_file():
        raise FileNotFoundError("checkpoint file not found")
    weight_hash = checkpoint_sha256(file)
    if weight_hash != checkpoint_sha256(run / "best_model.pth"):
        raise ValueError("OOD evaluation requires the source-selected best checkpoint")
    reference = read_json(package / "results/dirags_seed42_reference.json")[
        "experiments"
    ][config["experiment"]]
    archived_match = weight_hash == reference["checkpoint_sha256"]
    if require_paper_checkpoint and (not archived_match):
        raise ValueError("checkpoint SHA256 differs from the archived paper checkpoint")
    saved = read_json(run / "config_snapshot.json")
    runtime = legacy_training_config(config)
    inference = inference_config(config)
    if not (run / "inference_config_snapshot.json").is_file():
        raise FileNotFoundError("source run is missing inference_config_snapshot.json")
    if read_json(run / "inference_config_snapshot.json") != inference:
        raise ValueError("source inference config differs from requested config")
    split = read_json(run / "split_snapshot.json")
    if split.get("version") == 2:
        validate_split_manifest(split)
        if split != portable:
            raise ValueError("source portable split differs from the paper split")
    elif split.get("version") == 1:
        if set(split) != {
            "version",
            "dataset",
            "root",
            "seed",
            "counts",
            "splits",
            "sha256",
        }:
            raise ValueError("legacy source split schema mismatch")
        if (
            canonical_hash({k: v for k, v in split.items() if k != "sha256"})
            != split["sha256"]
        ):
            raise ValueError("legacy split hash mismatch")
        if split["sha256"] != portable["source_manifest_sha256"] or any(
            (split[k] != portable[k] for k in ["dataset", "seed", "counts", "splits"])
        ):
            raise ValueError(
                "legacy split membership/identity differs from the paper split"
            )
    else:
        raise ValueError("unsupported source split version")
    environment = read_json(run / "environment.json")
    if "experiment" in saved:
        if (
            saved != config
            or read_json(run / "legacy_training_config_snapshot.json") != runtime
        ):
            raise ValueError("public source training config changed")
        settings = read_json(run / "run_settings.json")
        if (
            type(settings.get("training_seed")) is not int
            or settings["training_seed"] != 42
        ):
            raise ValueError("source training seed changed")
        if not allow_incomplete and settings.get("pretrained_encoder") is not True:
            raise ValueError(
                "diagnostic initialization requires --allow-incomplete-run"
            )
        run_format = "public"
    else:
        if saved != runtime:
            raise ValueError("legacy source training config changed")
        expected = {
            "training_seed": 42,
            "training_mode": runtime["training_mode"],
            "module1_candidate": runtime["module1_candidate"],
        }
        if "confirmation_protocol" in runtime:
            expected["confirmation_protocol"] = runtime["confirmation_protocol"]
        for key, value in expected.items():
            if environment.get(key) != value:
                raise ValueError(f"legacy source environment mismatch: {key}")
        run_format = "legacy"
    metrics = read_json(run / "metrics.json")
    complete = (
        metrics.get("status") == "complete" and metrics.get("completed_epochs") == 300
    )
    if not allow_incomplete and (not complete):
        raise ValueError(
            "paper OOD evaluation requires complete 300-epoch source training"
        )
    checkpoint = load_training_checkpoint(file, map_location=device)
    actual_inference_hash = canonical_hash(inference)
    validate_checkpoint_payload(
        checkpoint, config, split["sha256"], inference_hash=actual_inference_hash
    )
    return (
        checkpoint,
        dict(
            run_format=run_format,
            training_seed=42,
            source_training_complete_300_epochs=complete,
            diagnostic_override=bool(allow_incomplete),
            checkpoint_sha256=weight_hash,
            archived_paper_checkpoint_match=archived_match,
            training_config_hash=canonical_hash(runtime),
            inference_config_hash=actual_inference_hash,
            source_split_hash=split["sha256"],
        ),
    )
