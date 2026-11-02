'BackboneOne checkpoint bridge for the common frozen OOD4 evaluator.'

from dirags.integrations.registry import legacy_text as _legacy_text
import argparse
import csv
import json
from pathlib import Path
import platform
import torch
from .backbone_1 import build_backbone_1, inference_logits
from .backbone_1_cli import ROOT, load_config, load_selected
from ..evaluation.ood import evaluate_pairs, summarize_rows, summarize_domains
from ..evaluation.ood_manifest import DATASETS, SEQUENCES, MANIFEST_SHA256, load_manifest, verify_target_pairs, validate_rows
from ..training.common import atomic_write_json


class BackboneOneBridge(torch.nn.Module):
    """Convert the exact sigmoid>=0.5 decision into shared two-channel scores."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        logits = inference_logits(self.model(images))
        if tuple(logits.shape) != (images.shape[0], 1, *images.shape[-2:]) or not torch.isfinite(logits).all().item():
            raise ValueError('invalid BackboneOne inference output')
        foreground = (torch.sigmoid(logits) >= 0.5).float()
        return torch.cat((1 - foreground, foreground), dim=1)


def evaluate(args):
    config, label = load_config(args.config)
    checkpoint, identity = load_selected(args.checkpoint, config, run_dir=args.run_dir, require_paper=args.require_paper_checkpoint)
    model = build_backbone_1()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    if args.check_only:
        print('BackboneOne/%s strict CPU restore PASS; selected source epoch=%d' % (label, identity["best_epoch"]))
        return identity
    roots = {d: getattr(args, d.lower()) for d in DATASETS}
    if args.output_dir is None or any(root is None for root in roots.values()):
        raise ValueError("--output-dir and all four target roots are required")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite an evaluation output directory")
    records = load_manifest(args.manifest)
    pairs = verify_target_pairs(records, roots)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # TF32 flags stay at runtime defaults, matching the original evaluator.
    model = BackboneOneBridge(model).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    rows, domains = [], {}
    for d in DATASETS:
        current = evaluate_pairs(model, pairs[d], device)
        validate_rows(current, [r for r in records if r["dataset"] == d])
        domains[d] = summarize_rows(current)
        if d == "PolypGen":
            domains[d]["sequences"] = {s: summarize_rows([r for r in current if r["sequence"] == s]) for s in SEQUENCES}
        rows.extend(current)
    result = {"status": "complete", "method": 'BackboneOne+PDC' if label == "pdc" else _legacy_text('text_d030064689ed7cb6'),
              "domains": domains, "aggregate": summarize_domains(domains),
              "empty62": {"count": 62, **domains["PolypGen"]["negative_metrics"]}}
    output.mkdir(parents=True, exist_ok=False)
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_write_json({"records": records}, output / "manifest.json")
    atomic_write_json({**identity, "torch": torch.__version__, "python": platform.python_version(),
                       "target_manifest_sha256": MANIFEST_SHA256, "target_mask_threshold": 128,
                       "prediction_rule": "sigmoid(last_lateral_map)>=0.5; no minmax or postprocessing",
                       "target_domain_tuning": False, "polypgen_main_positive_only": True,
                       "image_size": 256, "device": str(device),
                       "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                       "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32)}, output / "evaluation_identity.json")
    atomic_write_json(result, output / "metrics.json")  # Completion marker last.
    print('BackboneOne/%s Macro Dice=%.6f IoU=%.6f' % (label, result["aggregate"]["target_mean"]["dice"], result["aggregate"]["target_mean"]["iou"]))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--require-paper-checkpoint", action="store_true")
    p.add_argument("--check-only", action="store_true", help="Strict CPU restore only; no forward or output writes")
    p.add_argument("--manifest", type=Path, default=ROOT / "splits/ood4_seed42.json")
    p.add_argument("--device", default="cuda:0")
    for d in DATASETS:
        p.add_argument("--" + d.lower(), type=Path)
    p.add_argument("--output-dir", type=Path)
    evaluate(p.parse_args(argv))
    return 0
