"""Train the complete DiRaGS model with source-only checkpoint selection."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from .config import (
    load_config,
    canonical_hash,
    legacy_training_config,
    inference_config,
    pseudo_config,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--split",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "splits/kvasir_seed42.json",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--max-steps",
        type=int,
        help="diagnostic minibatch limit; not a full paper training run",
    )
    parser.add_argument(
        "--pause-after-epochs",
        type=int,
        help="pause at an epoch boundary; resume with last_checkpoint.pth",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="disable ImageNet initialization for an explicit diagnostic run",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate config and print legacy identity hashes without training",
    )
    return parser


def _seed_worker(worker_id):
    import random
    import numpy as np
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def prepare_run(output, config, split, settings, resume):
    from .training.common import atomic_write_json

    public_inference = inference_config(config)
    expected = {
        "config_snapshot.json": config,
        "split_snapshot.json": split,
        "run_settings.json": settings,
        "inference_config_snapshot.json": public_inference,
        "legacy_training_config_snapshot.json": legacy_training_config(config),
    }
    if resume is not None:
        if (
            not output.is_dir()
            or resume.resolve() != (output / "last_checkpoint.pth").resolve()
            or (not resume.is_file())
        ):
            raise ValueError(
                "resume must use this run directory and its last_checkpoint.pth"
            )
        for name, value in expected.items():
            saved = json.loads((output / name).read_text(encoding="utf-8"))
            if saved != value:
                raise ValueError(f"resume snapshot mismatch: {name}")
        return
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    for name, value in expected.items():
        atomic_write_json(value, output / name)


def run_training(args):
    import platform
    import torch
    import torchvision
    from torch.utils.data import DataLoader
    from .datasets.kvasir_seg import KvasirDataset, IdentifiedKvasirDataset
    from .datasets.splits import load_split_manifest
    from .integrations.segmentation import build_segmentation, ModelConfig
    from .integrations.segmentation import DeepSupervisionLoss
    from .training.common import set_determinism, atomic_write_json
    from .training.dual_view import DSTTrainingWrapper
    from .training.gcs_training import train_gcs
    from .methods.pdc_integration import build_pdc_pseudo_domain

    config = load_config(args.config)
    runtime = legacy_training_config(config)
    if args.data_root is None or args.output_dir is None:
        raise ValueError("--data-root and --output-dir are required for training")
    device = torch.device(args.device)
    if device.type == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.pause_after_epochs is not None and (
        not 0 < args.pause_after_epochs <= config["epochs"]
    ):
        raise ValueError("--pause-after-epochs is outside the configured epoch range")
    root = args.data_root.expanduser().resolve()
    split = load_split_manifest(args.split, root)
    output = args.output_dir.expanduser().resolve()
    resume = args.resume.expanduser().resolve() if args.resume else None
    settings = {
        "pretrained_encoder": not args.no_pretrained,
        "training_seed": config["training_seed"],
        "checkpoint_schema": "original-bcd-with-portable-split-v1",
    }
    prepare_run(output, config, split, settings, resume)
    if resume is None:
        atomic_write_json(
            dict(
                python=platform.python_version(),
                torch=torch.__version__,
                torchvision=torchvision.__version__,
                cuda_available=torch.cuda.is_available(),
                cuda_version=torch.version.cuda,
                device_count=torch.cuda.device_count(),
                requested_device=str(device),
            ),
            output / "environment.json",
        )
    set_determinism(config["training_seed"])
    subsets = split["splits"]
    opts = dict(
        image_size=config["image_size"], mask_threshold=config["mask_threshold"]
    )
    train_dataset = IdentifiedKvasirDataset
    train_data = train_dataset(root, subsets["train"], augment=True, **opts)
    val_data = KvasirDataset(root, subsets["val"], augment=False, **opts)
    test_data = KvasirDataset(root, subsets["test"], augment=False, **opts)
    common = dict(
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=bool(config["pin_memory"] and device.type == "cuda"),
        worker_init_fn=_seed_worker,
    )
    train_loader = DataLoader(
        train_data,
        shuffle=True,
        generator=torch.Generator().manual_seed(config["training_seed"]),
        **common,
    )
    val_loader = DataLoader(val_data, shuffle=False, **common)
    test_loader = DataLoader(test_data, shuffle=False, **common)
    base = build_segmentation(ModelConfig(pretrained_encoder=not args.no_pretrained))
    model = DSTTrainingWrapper(base)
    criterion = DeepSupervisionLoss(weights=config["deep_weights"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config["scheduler_t0_epochs"],
        T_mult=config["scheduler_tmult"],
        eta_min=config["scheduler_eta_min"],
    )
    kw = dict(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=runtime,
        pseudo_config=pseudo_config(config),
        training_seed=config["training_seed"],
        training_config_hash=canonical_hash(runtime),
        inference_config_hash=canonical_hash(inference_config(config)),
        split_hash=split["sha256"],
        output_dir=output,
        device=device,
        resume=resume,
        max_steps=args.max_steps,
        pause_after_epochs=args.pause_after_epochs,
    )
    return train_gcs(**kw, pseudo_builder=build_pdc_pseudo_domain)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.validate_only:
        print(
            json.dumps(
                dict(
                    experiment=config["experiment"],
                    training_seed=config["training_seed"],
                    training_config_hash=canonical_hash(legacy_training_config(config)),
                    inference_config_hash=canonical_hash(inference_config(config)),
                    status="CONFIG_VALID",
                ),
                indent=2,
            )
        )
        return 0
    result = run_training(args)
    print(json.dumps(result, indent=2))
    return 0
