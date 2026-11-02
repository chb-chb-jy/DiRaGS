'Source-only BackboneOne baseline/PDC training, separate from SegmentationModel A-E.'
from __future__ import annotations

from dirags.integrations.registry import legacy_text as _legacy_text
import argparse
import copy
import csv
import json
import math
import platform
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from . import backbone_1 as method
from .backbone_1_data import build_source_loaders, BackboneOneDataset, seed_worker
from ..evaluation.ood import checkpoint_sha256
from ..training.common import atomic_write_json

ROOT = Path(__file__).resolve().parents[3]


def reference():
    return json.loads((ROOT / 'results/backbone_1_cross_backbone_seed42_reference.json').read_text(encoding="utf-8"))


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    candidates = reference()["experiments"]
    matches = [label for label, value in candidates.items() if config == value["config"]]
    if len(matches) != 1:
        raise ValueError('configuration must match a frozen BackboneOne baseline/PDC seed42 protocol')
    return config, matches[0]


def load_selected(checkpoint, config, *, run_dir=None, require_paper=False):
    checkpoint = Path(checkpoint)
    run = Path(run_dir or checkpoint.parent)
    value = method.load_checkpoint(checkpoint)
    ident = value.get("identity", {})
    if ident.get("config") != config or value.get("selection_metric") != "kvasir_source_val_iou":
        raise ValueError("checkpoint configuration or selection rule mismatch")
    label = "pdc" if "pdc" in config else "baseline"
    expected = reference()
    origin = ident.get("source_split_origin") if ident.get("format") == _legacy_text('text_6779c703914945ae') else ident.get("split_semantic_sha256")
    if origin != expected["experiments"][label]["source_split_semantic_sha256"] or ident.get("upstream_commit") != expected["upstream_commit"]:
        raise ValueError("checkpoint source membership or upstream identity differs")
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    with (run / "history.csv").open(encoding="utf-8-sig", newline="") as stream:
        history = list(csv.DictReader(stream))
    if (metrics.get("status") != "complete" or metrics.get("completed_epochs") != 300
            or metrics.get("identity") != ident or metrics.get("ood_paths_read") != 0
            or [int(row["epoch"]) for row in history] != list(range(1, 301))
            or any(int(row["optimizer_steps"]) != 600 for row in history)):
        raise ValueError("completed source-only 300-epoch training identity/history required")
    values = [float(row["val_iou"]) for row in history]
    selected = max(range(300), key=values.__getitem__) + 1
    if (not all(math.isfinite(v) for v in values) or value.get("epoch") != selected
            or metrics.get("best_epoch") != selected
            or not math.isclose(value["source_val"]["iou"], values[selected - 1], abs_tol=1e-10, rel_tol=0)
            or not math.isclose(metrics["best_source_val_iou"], values[selected - 1], abs_tol=1e-10, rel_tol=0)):
        raise ValueError("selected checkpoint is not the first source-validation IoU maximum")
    if not all(torch.isfinite(t).all().item() for t in value["model_state_dict"].values()):
        raise ValueError("nonfinite checkpoint state")
    digest = checkpoint_sha256(checkpoint)
    if require_paper and digest != reference()["experiments"][label]["checkpoint_sha256"]:
        raise ValueError("not the archived paper checkpoint")
    return value, {"checkpoint_sha256": digest, "best_epoch": selected,
                   "selection_metric": "kvasir_source_val_iou", "config": config}


def save_weight(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def write_history(path, history):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)


def capture_rng(loader):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
            "train_loader_generator": loader.generator.get_state()}


def restore_rng(state, loader):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])
    loader.generator.set_state(state["train_loader_generator"].cpu())


def train_epoch(model, loader, optimizer, config, device, epoch):
    model.train()
    totals, samples, steps = {}, 0, 0
    for batch in loader:
        images, masks = batch[0].to(device), batch[1].to(device)
        if "pdc" in config:
            stats = method.train_pdc_minibatch(model, optimizer, images, masks, batch[2], config, epoch=epoch)
        else:
            loss = method.train_baseline_minibatch(model, optimizer, images, masks, config)
            stats = {"loss": loss, "optimizer_steps": 3}
        samples += len(images)
        steps += stats["optimizer_steps"]
        for key, value in stats.items():
            if key != "optimizer_steps":
                totals[key] = totals.get(key, 0.0) + value * len(images)
    if samples != 800 or steps != 600:
        raise ValueError("expected 800 source images and 600 optimizer updates per epoch")
    return {key: value / samples for key, value in totals.items()}, steps


def train(args):
    config, label = load_config(args.config)
    if args.data_root is None or args.pretrained is None or args.output_dir is None:
        raise ValueError("--data-root, --pretrained, --output-dir are required")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and not args.resume:
        raise FileExistsError("refusing an existing training output directory")
    if args.resume and not output.is_dir():
        raise FileNotFoundError("resume requires an existing public-format run")
    if checkpoint_sha256(args.pretrained) != config["pretrained_sha256"]:
        raise ValueError('ImageNet EncoderOne weight hash differs from the frozen initialization')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    train_loader, val_loader, manifest = build_source_loaders(args.split, args.data_root, pdc=label == "pdc", num_workers=config["num_workers"])
    expected_split = reference()["experiments"][label]["source_split_semantic_sha256"]
    if manifest["source_manifest_sha256"] != expected_split:
        raise ValueError("portable source split does not bind the paper split")
    model = method.build_backbone_1(args.pretrained).to(device)
    oc = config["optimizer"]
    optimizer = torch.optim.Adam(model.parameters(), lr=oc["learning_rate"], betas=tuple(oc["betas"]), eps=oc["epsilon"], weight_decay=oc["weight_decay"])
    identity = {"format": _legacy_text('text_6779c703914945ae'), "config": config,
                "source_split_sha256": manifest["sha256"], "source_split_origin": expected_split,
                "upstream_commit": reference()["upstream_commit"]}
    history, start, best_epoch, best_iou, best = [], 1, 0, -math.inf, None
    if args.resume:
        latest = method.load_checkpoint(output / "latest.pt")
        if latest["identity"] != identity:
            raise ValueError("resume identity differs; original legacy runs cannot be resumed here")
        history = latest["history"]
        start = int(latest["epoch"]) + 1
        if [int(r["epoch"]) for r in history] != list(range(1, start)) or not 1 <= start <= 301:
            raise ValueError("incoherent resume history")
        model.load_state_dict(latest["model_state_dict"], strict=True)
        optimizer.load_state_dict(latest["optimizer_state_dict"])
        best = latest["selected_best_payload"]
        best_epoch, best_iou = best["epoch"], best["source_val"]["iou"]
        restore_rng(latest["rng_state"], train_loader)
        save_weight(output / "best_source_val_iou.pt", best)
        write_history(output / "history.csv", history)
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_write_json(identity, output / "identity.json")
        atomic_write_json(config, output / "config_snapshot.json")
        atomic_write_json(manifest, output / "split_snapshot.json")
        atomic_write_json({"python": platform.python_version(), "torch": torch.__version__, "device": str(device)}, output / "environment.json")
    for epoch in range(start, 301):
        began = time.monotonic()
        lr = method.learning_rate_for_epoch(config, epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr
        stats, steps = train_epoch(model, train_loader, optimizer, config, device, epoch)
        val = method.evaluate(model, val_loader, device)
        if val["iou"] > best_iou:
            best_epoch, best_iou = epoch, val["iou"]
            best = {"epoch": epoch, "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "source_val": val, "selection_metric": "kvasir_source_val_iou", "identity": identity}
            save_weight(output / "best_source_val_iou.pt", best)
        row = {"epoch": epoch, "train_loss": stats["loss"], "val_loss": val["loss"], "val_iou": val["iou"], "val_dice": val["dice"],
               "learning_rate": lr, "optimizer_steps": steps, "elapsed_seconds": time.monotonic() - began,
               "best_epoch": best_epoch, "best_val_iou": best_iou}
        if label == "pdc":
            row.update(train_clean_loss=stats["clean_loss"], train_pseudo_loss=stats["pseudo_loss"])
        history.append(row)
        write_history(output / "history.csv", history)
        if epoch == 1 or epoch % 10 == 0 or epoch == 300:
            save_weight(output / "latest.pt", {"epoch": epoch, "identity": identity, "history": history,
                        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                        "rng_state": capture_rng(train_loader), "selected_best_payload": best})
        print('BackboneOne/%s epoch=%d/300 val_iou=%.6f best_epoch=%d' % (label, epoch, val["iou"], best_epoch), flush=True)
    model.load_state_dict(best["model_state_dict"], strict=True)
    test_loader = torch.utils.data.DataLoader(BackboneOneDataset(args.data_root, manifest["splits"]["test"]), batch_size=1, shuffle=False,
                                             num_workers=config["num_workers"])  # Original source-test worker behavior.
    source_test = method.evaluate(model, test_loader, device)
    atomic_write_json({"status": "complete", "completed_epochs": 300, "identity": identity,
                       "best_epoch": best_epoch, "best_source_val_iou": best_iou, "source_test": source_test,
                       "optimizer_steps_total": sum(r["optimizer_steps"] for r in history), "ood_paths_read": 0}, output / "metrics.json")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--split", type=Path, default=ROOT / "splits/kvasir_seed42.json")
    p.add_argument("--pretrained", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--check-config", action="store_true")
    args = p.parse_args(argv)
    if args.check_config:
        config, label = load_config(args.config)
        print('BackboneOne/%s seed42: 300 epochs; 192/256/320; source-val IoU selection' % label)
        return 0
    train(args)
    return 0
