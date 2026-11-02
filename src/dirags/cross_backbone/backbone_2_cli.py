'Frozen BackboneTwo baseline/PDC source-only training and selected-state readers.'

from dirags.integrations.registry import legacy_text as _legacy_text
import argparse
import csv
import json
import math
from pathlib import Path
import platform
import random
import time
import numpy as np
import torch
from . import backbone_2 as method
from .backbone_1_cli import save_weight, write_history, capture_rng, restore_rng
from ..evaluation.ood import checkpoint_sha256
from ..training.common import atomic_write_json

ROOT = Path(__file__).resolve().parents[3]
WEIGHT_SHA256 = "48b51a06db85a1fec0520028592561ffc29ea4e975f2a7f5b7acacbfca03f143"


def reference():
    return json.loads((ROOT / 'results/backbone_2_cross_backbone_seed42_reference.json').read_text(encoding="utf-8"))


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    matches = [label for label, value in reference()["experiments"].items() if value["config"] == config]
    if len(matches) != 1:
        raise ValueError('expected a frozen BackboneTwo baseline/PDC seed42 configuration')
    return config, matches[0]


def load_selected(checkpoint, config, *, run_dir=None, require_paper=False):
    checkpoint = Path(checkpoint)
    run = Path(run_dir or checkpoint.parent)
    value = method.load_checkpoint(checkpoint)
    identity = value.get("identity", {})
    label = "pdc" if "pdc" in config else "baseline"
    expected = reference()["experiments"][label]
    if (identity.get("config") != config or identity.get("source_identity_sha256") != expected["source_identity_sha256"]
            or identity.get("split_semantic_sha256") != expected["source_split_semantic_sha256"]
            or value.get("selection_metric") != "kvasir_source_val_iou"):
        raise ValueError("selected checkpoint configuration/source/selection identity differs")
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    with (run / "history.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if (metrics.get("status") != "complete" or metrics.get("completed_epochs") != 300
            or metrics.get("identity") != identity or metrics.get("ood_paths_read") != 0
            or metrics.get("optimizer_steps_total") != 15000
            or [int(r["epoch"]) for r in rows] != list(range(1, 301))
            or any(int(r["optimizer_steps"]) != 50 for r in rows)):
        raise ValueError("completed 300-epoch/15000-update source-only history is required")
    values = [float(r["val_iou"]) for r in rows]
    epoch = max(range(300), key=values.__getitem__) + 1
    if (not all(math.isfinite(v) for v in values) or value.get("epoch") != epoch or metrics.get("best_epoch") != epoch
            or not math.isclose(value["source_val"]["iou"], values[epoch-1], abs_tol=1e-12, rel_tol=0)
            or not math.isclose(metrics["best_source_val_iou"], values[epoch-1], abs_tol=1e-12, rel_tol=0)):
        raise ValueError("checkpoint is not the first maximum source-validation IoU")
    if not all(torch.isfinite(v).all().item() for v in value["model_state_dict"].values()):
        raise ValueError("nonfinite selected state")
    digest = checkpoint_sha256(checkpoint)
    if require_paper and digest != expected["checkpoint_sha256"]:
        raise ValueError("not the archived paper checkpoint")
    return value, {"checkpoint_sha256": digest, "best_epoch": epoch, "config": config,
                   "selection_metric": "kvasir_source_val_iou", "source_identity_sha256": expected["source_identity_sha256"]}


def make_scheduler(optimizer, config):
    s = config["scheduler"]
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=s["mode"], factor=s["factor"], patience=s["patience"],
        threshold=s["threshold"], threshold_mode=s["threshold_mode"], cooldown=s["cooldown"], min_lr=s["minimum_learning_rate"], eps=s["epsilon"])


def train_epoch(model, loader, optimizer, loss_fn, device, epoch, *, pdc=False):
    model.train()
    total, samples, steps = 0.0, 0, 0
    for batch in loader:
        images = batch[0].to(device, non_blocking=True)
        masks = batch[1].to(device, non_blocking=True)
        if pdc:
            stats = method.train_pdc_minibatch(model, optimizer, loss_fn, images, masks, batch[2], epoch=epoch)
            loss = stats["loss"]
        else:
            loss = method.train_baseline_minibatch(model, optimizer, loss_fn, images, masks)
        total += loss * len(images)
        samples += len(images)
        steps += 1
    if samples != 800 or steps != 50:
        raise ValueError("expected 800 source images and 50 optimizer updates per epoch")
    return total / samples, steps


def train(args):
    from .backbone_2_data import build_source_loaders, BackboneTwoDataset
    config, label = load_config(args.config)
    if args.data_root is None or args.pretrained is None or args.output_dir is None:
        raise ValueError("--data-root, --pretrained and --output-dir are required")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and not args.resume:
        raise FileExistsError("refusing an existing training directory")
    if args.resume and not output.is_dir():
        raise FileNotFoundError("resume requires an existing public-format run")
    if checkpoint_sha256(args.pretrained) != WEIGHT_SHA256:
        raise ValueError('ImageNet EncoderTwo-B3 SHA256 differs from the formal initialization')
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    train_loader, val_loader, manifest = build_source_loaders(args.split, args.data_root, pdc=label == "pdc", num_workers=4)
    expected = reference()["experiments"][label]
    if manifest["source_manifest_sha256"] != expected["source_split_semantic_sha256"]:
        raise ValueError("portable source split does not bind the paper membership")
    model, loss_fn = method.build_model(args.pretrained)
    model, loss_fn = model.to(device), loss_fn.to(device)
    oc = config["optimizer"]
    optimizer = torch.optim.Adam(model.parameters(), lr=oc["learning_rate"], betas=tuple(oc["betas"]), eps=oc["epsilon"], weight_decay=oc["weight_decay"])
    scheduler = make_scheduler(optimizer, config)
    identity = {"format": _legacy_text('text_62b36d9783820963'), "config": config, "split_semantic_sha256": expected["source_split_semantic_sha256"],
                "source_identity_sha256": expected["source_identity_sha256"], "portable_split_sha256": manifest["sha256"]}
    history, start, best_epoch, best_iou, best = [], 1, 0, -math.inf, None
    if args.resume:
        latest = method.load_checkpoint(output / "latest.pt")
        if latest["identity"] != identity:
            raise ValueError("resume identity differs; legacy trainer latest states are not resumed here")
        history = latest["history"]
        start = int(latest["epoch"]) + 1
        if not 1 <= start <= 301 or [int(r["epoch"]) for r in history] != list(range(1, start)):
            raise ValueError("incoherent resume history")
        model.load_state_dict(latest["model_state_dict"], strict=True)
        optimizer.load_state_dict(latest["optimizer_state_dict"])
        scheduler.load_state_dict(latest["scheduler_state_dict"])
        best = latest["selected_best_payload"]
        best_epoch, best_iou = best["epoch"], best["source_val"]["iou"]
        restore_rng(latest["rng_state"], train_loader)
        save_weight(output / "best_source_val_iou.pt", best)
        write_history(output / "history.csv", history)
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_write_json(identity, output / "identity.json")
        atomic_write_json(config, output / "config.json")
        atomic_write_json(manifest, output / "split_snapshot.json")
        atomic_write_json({"python": platform.python_version(), "torch": torch.__version__, "device": str(device)}, output / "environment.json")
    for epoch in range(start, 301):
        began = time.monotonic()
        loss, steps = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, pdc=label == "pdc")
        val = method.evaluate(model, val_loader, loss_fn, device)
        if val["samples"] != 100:
            raise ValueError("source validation must contain 100 samples")
        scheduler.step(val["loss"])
        lr = float(optimizer.param_groups[0]["lr"])
        if val["iou"] > best_iou:
            best_epoch, best_iou = epoch, val["iou"]
            best = {"epoch": epoch, "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "source_val": val, "identity": identity, "selection_metric": "kvasir_source_val_iou"}
            save_weight(output / "best_source_val_iou.pt", best)
        history.append({"epoch": epoch, "train_loss": loss, "val_loss": val["loss"], "val_iou": val["iou"], "val_dice": val["dice"],
                        "learning_rate": lr, "optimizer_steps": steps, "elapsed_seconds": time.monotonic()-began,
                        "best_epoch": best_epoch, "best_val_iou": best_iou})
        write_history(output / "history.csv", history)
        if epoch == 1 or epoch % 10 == 0 or epoch == 300:
            save_weight(output / "latest.pt", {"epoch": epoch, "identity": identity, "history": history,
                        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(), "rng_state": capture_rng(train_loader), "selected_best_payload": best})
        print('BackboneTwo/%s epoch=%d/300 val_iou=%.6f best_epoch=%d' % (label, epoch, val["iou"], best_epoch), flush=True)
    model.load_state_dict(best["model_state_dict"], strict=True)
    test_loader = torch.utils.data.DataLoader(BackboneTwoDataset(args.data_root, manifest["splits"]["test"], False), batch_size=16,
                                             shuffle=False, drop_last=False, num_workers=0)
    source_test = method.evaluate(model, test_loader, loss_fn, device)
    if source_test["samples"] != 100:
        raise ValueError("source test must contain 100 samples")
    atomic_write_json({"status": "complete", "completed_epochs": 300, "identity": identity, "best_epoch": best_epoch,
                       "best_source_val_iou": best_iou, "checkpoint_selection": "kvasir_source_val_iou", "source_test": source_test,
                       "optimizer_steps_total": sum(r["optimizer_steps"] for r in history), "ood_paths_read": 0}, output / "metrics.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", type=Path, default=ROOT / "splits/kvasir_seed42.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.check_config:
        _, label = load_config(args.config)
        print('BackboneTwo/%s seed42: batch16; 300 epochs; source-val IoU selection' % label)
        return 0
    train(args)
    return 0
