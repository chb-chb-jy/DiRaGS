# PDC cross-backbone experiments

This group evaluates PDC paired-view training on BackboneOne and BackboneTwo, separately
from the complete DiRaGS training workflow. DST and GCS are disabled. The first sections
describe BackboneOne; BackboneTwo's input, loss and optimization differences are below.

Obtain and configure the separately distributed backbone sources first;
see [external backbone setup](external_backbones.md). Network and component
loss definitions are loaded from those external locations at runtime.

## Files

| Role | File |
|---|---|
| Baseline config | `configs/cross_backbone/backbone_1_baseline_seed42.json` |
| PDC config | `configs/cross_backbone/backbone_1_pdc_seed42.json` |
| External BackboneOne / Res2Net interface | `src/dirags/integrations/backbone_1.py` |
| Loss and three-scale training steps | `src/dirags/cross_backbone/backbone_1.py` |
| Source data / portable split | `src/dirags/cross_backbone/backbone_1_data.py` |
| Training and checkpoint readers | `src/dirags/cross_backbone/backbone_1_cli.py` |
| Shared-metric inference bridge | `src/dirags/cross_backbone/backbone_1_ood.py` |
| Archived identities and results | `results/backbone_1_cross_backbone_seed42_reference.json` |

PDC itself is reused from `methods/pdc.py` and `methods/pdc_integration.py`.
BackboneOne sources and their research-use statement are documented in
[source notices](../THIRD_PARTY_NOTICES.md) and
[BackboneOne notice](BACKBONE_1_UPSTREAM_NOTICE.md).

## Source training

Use the same dependency environment as the main package. Obtain the
[official Res2Net-50 initialization](https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth)
separately; its SHA256 must be
`3cf99910a0f1a4c9c43e707958179bf928b9cc81ed9297dc960fe07d1c46139c`.

Validate configs without data access or training:

```bash
python scripts/train_backbone_1.py --config configs/cross_backbone/backbone_1_baseline_seed42.json --check-config
python scripts/train_backbone_1.py --config configs/cross_backbone/backbone_1_pdc_seed42.json --check-config
```

Train each run independently from the same ImageNet initialization; the PDC
run is not fine-tuned from the selected baseline checkpoint:

```bash
KVASIR_ROOT="/path/to/Kvasir-SEG"
RES2NET_WEIGHT="/path/to/res2net50_v1b_26w_4s-3cf99910.pth"
python scripts/train_backbone_1.py --config configs/cross_backbone/backbone_1_baseline_seed42.json --data-root "$KVASIR_ROOT" --pretrained "$RES2NET_WEIGHT" --output-dir runs/backbone_1_baseline_seed42 --device cuda:0
python scripts/train_backbone_1.py --config configs/cross_backbone/backbone_1_pdc_seed42.json --data-root "$KVASIR_ROOT" --pretrained "$RES2NET_WEIGHT" --output-dir runs/backbone_1_pdc_seed42 --device cuda:0
```

The ordered 800/100/100 source split is shared with the complete DiRaGS model. Training uses seed42,
batch4, 300 epochs, Adam, and three scales (192/256/320), with three optimizer
updates per minibatch. Checkpoints are selected by the first maximum source
validation IoU. Source masks retain the formal BackboneOne grayscale threshold >=1;
multiscale supervision uses bilinear soft masks. PDC is generated once on the
hard 256px base grid before the shared resizes; sampling epoch is training epoch
minus one. Paired forwards replay RNG and restore clean BN buffers after backward.

The formal loss preserves the scalar mean BCE produced by the upstream's
legacy `reduce="none"` argument. Changing it to modern `reduction="none"`
changes the actual experiment. The initial-LR-based decay is assigned at each
epoch start, including epoch50; it is not compounded from the previous LR.

`--resume` accepts this public trainer's `latest.pt` and restores model,
optimizer, RNG/loader state, history and the coherent selected best. It does
not resume the older server trainer's different rolling-checkpoint format.
Existing output directories require explicit `--resume`.

## OOD4 evaluation

These commands use the trained weights produced above. Original selected paper
weights are not included in this source package. Both legacy completed server
runs and completed public runs are accepted when their identities match.
`--check-only` performs strict CPU restore without forward passes or output
writes; optional `--require-paper-checkpoint` checks the archived weight SHA256.

```bash
python scripts/evaluate_backbone_1.py --config configs/cross_backbone/backbone_1_pdc_seed42.json --checkpoint runs/backbone_1_pdc_seed42/best_source_val_iou.pt --check-only
TARGETS=(--clinicdb /path/to/CVC-ClinicDB --colondb /path/to/CVC-ColonDB --etis /path/to/ETIS-LaribPolypDB --polypgen /path/to/PolypGen)
python scripts/evaluate_backbone_1.py --config configs/cross_backbone/backbone_1_baseline_seed42.json --checkpoint runs/backbone_1_baseline_seed42/best_source_val_iou.pt "${TARGETS[@]}" --output-dir runs/backbone_1_baseline_ood4 --device cuda:0
python scripts/evaluate_backbone_1.py --config configs/cross_backbone/backbone_1_pdc_seed42.json --checkpoint runs/backbone_1_pdc_seed42/best_source_val_iou.pt "${TARGETS[@]}" --output-dir runs/backbone_1_pdc_ood4 --device cuda:0
```

The evaluator reuses the main package's PIL RGB bilinear/ImageNet processing,
nearest-neighbor GT >=128, 256x256 scoring, 1461-sample frozen target membership,
and metrics. It selects the final BackboneOne lateral map, applies sigmoid>=0.5
(including the zero-logit tie), and performs no per-image minmax or postprocessing.
PolypGen overlap uses 211 positive frames; the 62 empty frames are separate.
Macro scores equally average the four positive-frame domain means. TF32 settings
are recorded without overriding runtime defaults, matching the old evaluator.

## Archived seed42 result

| Quantity | BackboneOne baseline | BackboneOne + PDC |
|---|---:|---:|
| Source-selected epoch (one-based) | 63 | 118 |
| Macro Dice (%) | 67.887918 | 72.183901 |
| Macro IoU (%) | 60.610564 | 64.438954 |
| Empty62 ordinary false-positive images | 10/62 | 30/62 |
| Empty62 serious false-positive images | 4/62 | 18/62 |

The unrounded improvements are +4.2959826404 Dice percentage points and
+3.8283894529 IoU percentage points. The archived JSON preserves all domains,
checkpoint/CSV hashes and both favorable and unfavorable observations. It is
an archived reference, not a new real-target run performed during packaging.


## BackboneTwo dependencies and source training

BackboneTwo reuses the same PDC sampler and scoring functions, with its own model,
loss and data handling under `src/dirags/cross_backbone/backbone_2*.py`.
Its configs are `configs/cross_backbone/backbone_2_baseline_seed42.json` and
`configs/cross_backbone/backbone_2_pdc_seed42.json`.

After installing the chosen core environment:

```bash
pip install -r requirements/backbone_2.txt
pip install --no-deps opencv-python==4.12.0.88
```

The separate OpenCV command retains the captured NumPy 1.26.4/OpenCV 4.12
runtime pairing. OpenCV's wheel metadata requires NumPy>=2 for Python>=3.9;
the recorded pairing is intentionally retained and tested without upgrading
the existing scientific stack. SciPy, scikit-learn and
Albumentations are not needed by the public model/data path: unrelated original
imports are omitted, and augmentation follows the original qualified adapter.

Obtain the [PVT-B3 ImageNet initialization](https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b3.pth)
separately. Required SHA256:
`48b51a06db85a1fec0520028592561ffc29ea4e975f2a7f5b7acacbfca03f143`.
Both runs initialize independently from it, not from the selected baseline.
The original qualifier constructs a reference PVT to validate pretrained keys
before constructing BackboneTwo. That reference construction is retained because it
consumes RNG and determines the subsequent decoder initialization.

```bash
python scripts/train_backbone_2.py --config configs/cross_backbone/backbone_2_baseline_seed42.json --check-config
python scripts/train_backbone_2.py --config configs/cross_backbone/backbone_2_pdc_seed42.json --check-config
KVASIR_ROOT="/path/to/Kvasir-SEG"
PVT_WEIGHT="/path/to/pvt_v2_b3.pth"
python scripts/train_backbone_2.py --config configs/cross_backbone/backbone_2_baseline_seed42.json --data-root "$KVASIR_ROOT" --pretrained "$PVT_WEIGHT" --output-dir runs/backbone_2_baseline_seed42 --device cuda:0
python scripts/train_backbone_2.py --config configs/cross_backbone/backbone_2_pdc_seed42.json --data-root "$KVASIR_ROOT" --pretrained "$PVT_WEIGHT" --output-dir runs/backbone_2_pdc_seed42 --device cuda:0
```

The source protocol keeps batch16, 300 epochs and 50 optimizer updates per
epoch. Images are OpenCV BGR/255. Both images and source masks use linear
256px resizing; masks remain soft. Rotation, flips and image-only coarse dropout
follow the frozen original adapter. PDC alone converts BGR to RGB and uses
soft-target>=0.5 to define regions, then returns to BGR; supervision is unchanged.
Two half-weighted backward passes share one optimizer update, replaying the
clean-forward RNG and restoring clean BN/RNG after both backwards. There is no
gradient clipping or AMP. The original batch-flattened sigmoid-probability
BCE-plus-Dice loss and validation-loss ReduceLROnPlateau are preserved.

`--resume` restores this public trainer's model, optimizer, scheduler, RNG/loader
state, history and coherent best state. It does not resume the different legacy
server rolling-state format. Selection always uses the first maximum source
validation IoU; the fixed budget runs to epoch300. Training disables both TF32
flags, as the formal original did.

## BackboneTwo OOD4 evaluation

```bash
python scripts/evaluate_backbone_2.py --config configs/cross_backbone/backbone_2_pdc_seed42.json --checkpoint runs/backbone_2_pdc_seed42/best_source_val_iou.pt --check-only
TARGETS=(--clinicdb /path/to/CVC-ClinicDB --colondb /path/to/CVC-ColonDB --etis /path/to/ETIS-LaribPolypDB --polypgen /path/to/PolypGen)
python scripts/evaluate_backbone_2.py --config configs/cross_backbone/backbone_2_baseline_seed42.json --checkpoint runs/backbone_2_baseline_seed42/best_source_val_iou.pt "${TARGETS[@]}" --output-dir runs/backbone_2_baseline_ood4 --device cuda:0
python scripts/evaluate_backbone_2.py --config configs/cross_backbone/backbone_2_pdc_seed42.json --checkpoint runs/backbone_2_pdc_seed42/best_source_val_iou.pt "${TARGETS[@]}" --output-dir runs/backbone_2_pdc_ood4 --device cuda:0
```

Both original completed server checkpoints and completed public runs are
accepted. `--check-only` restores on CPU without forward passes or target
access; `--require-paper-checkpoint` optionally pins the archived weight hash.
Full strict restore does not require the separate ImageNet initialization.

BackboneTwo inference retains OpenCV BGR/255 image input and uses the canonical
PIL grayscale/nearest256 GT >=128. It applies sigmoid(single_logit)>=0.5 without
minmax, TTA, postprocessing or PDC at inference. It preserves runtime TF32
defaults and records their values. Macro scoring and Empty62 connectivity/area
rules are the same as above. Legacy OpenCV-GT scores are not substituted for
the common-PIL results.

| Quantity | BackboneTwo baseline | BackboneTwo + PDC |
|---|---:|---:|
| Source-selected epoch (one-based) | 19 | 22 |
| Common-PIL Macro Dice (%) | 69.233273 | 73.469625 |
| Common-PIL Macro IoU (%) | 61.954738 | 65.415550 |
| Empty62 ordinary false-positive images | 19/62 | 24/62 |
| Empty62 serious false-positive images | 7/62 | 8/62 |

The unrounded gains are +4.2363512898 Dice and +3.4608128716 IoU percentage
points. The archived reference, selected weights/CSV/prediction hashes and
paired changes are in `results/backbone_2_cross_backbone_seed42_reference.json`.
These are PDC paired-training results, not GCS cross-backbone evidence.
