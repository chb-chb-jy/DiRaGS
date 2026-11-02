'OpenCV BGR BackboneTwo inference with canonical PIL masks and common metrics.'

from dirags.integrations.registry import legacy_text as _legacy_text
import argparse
import csv
from pathlib import Path
import platform
import cv2
import numpy as np
from PIL import Image
import torch
from .backbone_2 import build_model
from .backbone_2_cli import ROOT, load_config, load_selected
from ..evaluation.ood import per_image_metrics, summarize_rows, summarize_domains
from ..evaluation.ood_manifest import DATASETS, SEQUENCES, MANIFEST_SHA256, load_manifest, verify_target_pairs, validate_rows
from ..training.common import atomic_write_json


@torch.inference_mode()
def evaluate_pairs(model, pairs, device):
    rows = []
    nearest = getattr(Image, "Resampling", Image).NEAREST
    for pair in pairs:
        image = cv2.imread(str(pair.image), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("OpenCV image read failed")
        height, width = image.shape[:2]
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0).to(device)
        logits = model(tensor)
        if tuple(logits.shape) != (1, 1, 256, 256) or not torch.isfinite(logits).all().item():
            raise ValueError('invalid BackboneTwo output')
        prediction = (torch.sigmoid(logits[0, 0]) >= 0.5).cpu()
        with Image.open(pair.mask) as source:
            mask_width, mask_height = source.size
            target = torch.from_numpy((np.asarray(source.convert("L").resize((256, 256), nearest)) >= 128).copy()).bool()
        row = {"dataset": pair.dataset, "sequence": pair.sequence or "", "name": pair.name,
               "original_image_width": width, "original_image_height": height,
               "original_mask_width": mask_width, "original_mask_height": mask_height}
        row.update(per_image_metrics(prediction, target))
        rows.append(row)
    return rows


def evaluate(args):
    config, label = load_config(args.config)
    checkpoint, identity = load_selected(args.checkpoint, config, run_dir=args.run_dir, require_paper=args.require_paper_checkpoint)
    model, _ = build_model()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    if args.check_only:
        print('BackboneTwo/%s strict CPU restore PASS; epoch=%d' % (label, identity["best_epoch"]))
        return identity
    roots = {d: getattr(args, d.lower()) for d in DATASETS}
    if args.output_dir is None or any(p is None for p in roots.values()):
        raise ValueError("--output-dir and all four target roots are required")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite an evaluation output directory")
    records = load_manifest(args.manifest)
    pairs = verify_target_pairs(records, roots)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Preserve inference's runtime TF32 defaults; training explicitly disables it.
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    domains, rows = {}, []
    for d in DATASETS:
        current = evaluate_pairs(model, pairs[d], device)
        validate_rows(current, [r for r in records if r["dataset"] == d])
        domains[d] = summarize_rows(current)
        if d == "PolypGen":
            domains[d]["sequences"] = {s: summarize_rows([r for r in current if r["sequence"] == s]) for s in SEQUENCES}
        rows.extend(current)
    metrics = {"status": "complete", "method": 'BackboneTwo+PDC' if label == "pdc" else _legacy_text('text_db59475f53f92714'),
               "gt_rule": "PIL.convert_L.resize_256_NEAREST_then_gte128", "domains": domains,
               "aggregate": summarize_domains(domains), "empty62": {"count": 62, **domains["PolypGen"]["negative_metrics"]}}
    output.mkdir(parents=True, exist_ok=False)
    with (output / "per_image.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    atomic_write_json({"records": records}, output / "manifest.json")
    atomic_write_json({**identity, "python": platform.python_version(), "torch": torch.__version__, "opencv": cv2.__version__,
                       "image_preprocessing": "OpenCV BGR/255 INTER_LINEAR256", "gt_rule": metrics["gt_rule"],
                       "prediction": "sigmoid(single_logit)>=0.5; no minmax/TTA/postprocessing", "target_domain_tuning": False,
                       "target_manifest_sha256": MANIFEST_SHA256, "device": str(device),
                       "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32), "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32)}, output / "evaluation_identity.json")
    atomic_write_json(metrics, output / "metrics.json")
    print('BackboneTwo/%s Macro Dice=%.6f IoU=%.6f' % (label, metrics["aggregate"]["target_mean"]["dice"], metrics["aggregate"]["target_mean"]["iou"]))
    return metrics


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--require-paper-checkpoint", action="store_true")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--manifest", type=Path, default=ROOT / "splits/ood4_seed42.json")
    p.add_argument("--device", default="cuda:0")
    for d in DATASETS:
        p.add_argument("--" + d.lower(), type=Path)
    p.add_argument("--output-dir", type=Path)
    evaluate(p.parse_args(argv))
    return 0
