"""Command-line OOD4 evaluation with frozen targets and verified checkpoint identity."""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from .config import load_config, inference_config
from .evaluation.ood_manifest import (
    DATASETS,
    SEQUENCES,
    COUNTS,
    MANIFEST_SHA256,
    load_manifest,
    verify_target_pairs,
    validate_rows,
)


def build_parser():
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument(
        "--run-dir",
        type=Path,
        help="source run metadata directory; defaults to checkpoint parent",
    )
    p.add_argument("--manifest", type=Path, default=root / "splits/ood4_seed42.json")
    for d in ["clinicdb", "colondb", "etis", "polypgen"]:
        p.add_argument("--" + d, type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="validate config and frozen manifest without loading checkpoints/data",
    )
    p.add_argument(
        "--require-paper-checkpoint",
        action="store_true",
        help="require the archived paper checkpoint SHA256",
    )
    p.add_argument(
        "--allow-incomplete-run",
        action="store_true",
        help="explicit diagnostic override; results are marked diagnostic",
    )
    return p


def run_evaluation(args):
    import platform
    import torch
    from .integrations.segmentation import build_segmentation, ModelConfig
    from .training.common import atomic_write_json
    from .evaluation.checkpoints import load_run_checkpoint
    from .evaluation.ood import evaluate_pairs, summarize_rows, summarize_domains

    config = load_config(args.config)
    records = load_manifest(args.manifest)
    if (
        args.checkpoint is None
        or args.output_dir is None
        or any(
            getattr(args, d) is None
            for d in ["clinicdb", "colondb", "etis", "polypgen"]
        )
    ):
        raise ValueError(
            "--checkpoint, --output-dir and all four target roots are required"
        )
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite an evaluation output directory")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    roots = {d: getattr(args, d.lower()) for d in DATASETS}
    pairs = verify_target_pairs(records, roots)
    run = args.run_dir or args.checkpoint.parent
    checkpoint, identity = load_run_checkpoint(
        args.checkpoint,
        run,
        config,
        device=device,
        allow_incomplete=args.allow_incomplete_run,
        require_paper_checkpoint=args.require_paper_checkpoint,
    )
    model = build_segmentation(ModelConfig(pretrained_encoder=False))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    all_rows = []
    domains = {}
    for d in DATASETS:
        rows = evaluate_pairs(model, pairs[d], device)
        validate_rows(rows, [r for r in records if r["dataset"] == d])
        all_rows.extend(rows)
        domains[d] = summarize_rows(rows)
        if d == "PolypGen":
            domains[d]["sequences"] = {
                s: summarize_rows([row for row in rows if row["sequence"] == s])
                for s in SEQUENCES
            }
    metrics = dict(domains=domains, aggregate=summarize_domains(domains))
    output.mkdir(parents=True, exist_ok=False)
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    atomic_write_json(metrics, output / "metrics.json")
    atomic_write_json({"records": records}, output / "manifest.json")
    atomic_write_json(
        dict(
            source_config=(
                json.loads((run / "config_snapshot.json").read_text(encoding="utf-8"))
                if identity["run_format"] == "legacy_a"
                else inference_config(config)
            ),
            paper_config=config,
            evaluation=dict(
                image_size=256,
                mask_threshold=128,
                prediction_rule="argmax",
                image_resize="PIL bilinear",
                mask_resize="PIL nearest",
                score_resolution="256x256",
                domain_weighting="equal_positive_frame_domain_means",
                target_manifest_sha256=MANIFEST_SHA256,
            ),
        ),
        output / "config_snapshot.json",
    )
    atomic_write_json(
        dict(
            python=platform.python_version(),
            torch=torch.__version__,
            device=str(device),
            **identity,
        ),
        output / "environment.json",
    )
    return metrics


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    load_manifest(args.manifest)
    if args.validate_only:
        print(
            json.dumps(
                dict(
                    status="CONFIG_AND_MANIFEST_VALID",
                    experiment=config["experiment"],
                    manifest_sha256=MANIFEST_SHA256,
                    domains=COUNTS,
                ),
                indent=2,
            )
        )
        return 0
    print(json.dumps(run_evaluation(args), indent=2))
    return 0
