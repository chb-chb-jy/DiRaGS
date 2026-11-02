# Data preparation

Obtain the five datasets separately. This repository provides fixed source
membership and target relative paths; original dataset images and masks stay
outside the repository. Use the providers' access conditions and citation
instructions linked below.

## Acquisition resources and exact study scope

| Dataset | Provider / acquisition resource | Scope used in this study |
|---|---|---|
| Kvasir-SEG | [Simula dataset page and download](https://datasets.simula.no/kvasir-seg/) | All 1000 pairs; bundled seed42 source split: 800 train, 100 validation, 100 test |
| CVC-ClinicDB | [CVC database directory](https://pages.cvc.uab.es/CVC-Colon/index.php/databases/) and its [ClinicDB download page](https://polyp.grand-challenge.org/site/Polyp/CVCClinicDB/) | All 612 target pairs |
| CVC-ColonDB | [CVC database directory](https://pages.cvc.uab.es/CVC-Colon/index.php/databases/); the study recorded the [longvil Kaggle mirror](https://www.kaggle.com/datasets/longvil/cvc-colondb) | All 380 target pairs |
| ETIS-Larib Polyp DB | [Legacy project page](https://polyp.grand-challenge.org/EtisLarib/) and [TIB catalogue entry](https://service.tib.eu/ldmservice/dataset/etis-larib-polyp-db) | All 196 target pairs |
| PolypGen | [Dataset DOI / Synapse](https://doi.org/10.7303/syn26376615); [dataset paper](https://www.nature.com/articles/s41597-023-01981-y) | C6 seq18–seq22: 273 target frames, 211 positive and 62 empty-GT frames |

Kvasir-SEG is the segmented-polyp dataset, rather than the broader Kvasir
classification collection. The CVC provider lists access conditions in its
[data-use agreement](https://pages.cvc.uab.es/ai4polypnet/wp-content/uploads/2024/05/AgreementData_CVC-1.pdf).
Synapse access follows its account and dataset access requirements.

The legacy ETIS page was not retrievable during this packaging check. A
separately maintained [Zenodo benchmark collection](https://zenodo.org/records/5579392)
describes a `TestDataset` containing 196 ETIS and 380 ColonDB test images.
This is a secondary collection, not the ETIS original publisher. Its default
training/test split differs from this study, and it must not replace the
bundled source split or the full 612-image ClinicDB evaluation. Check any
retrieved copy against the bundled target names, paths and original dimensions.
This package does not download, redistribute or certify that collection.

## Directory layout

Choose one parent directory, for example `/path/to/datasets`. The names below
are the roots passed to the commands in README. Keep source and target data
at their original resolutions; the loaders perform the documented resizing.

```text
datasets/
├── Kvasir-SEG/
│   ├── images/
│   │   └── cju5ekty5ckzf07550c9u3ckk.jpg
│   └── masks/
│       └── cju5ekty5ckzf07550c9u3ckk.jpg
├── CVC-ClinicDB/
│   └── PNG/
│       ├── Original/
│       │   └── 1.png
│       └── Ground Truth/
│           └── 1.png
├── CVC-ColonDB/
│   ├── images/
│   │   └── 1.png
│   └── masks/
│       └── 1.png
├── ETIS-LaribPolypDB/
│   ├── original/
│   │   └── 1.tif
│   └── ground_truth/
│       └── p1.tif
└── PolypGen/
    └── PolypGen2021_MultiCenterData_v3/
        └── sequenceData/
            └── positive/
                ├── seq18/
                │   ├── images_seq18/
                │   │   └── seq3_C6_0.jpg
                │   └── masks_seq18/
                │       └── seq3_C6_0_mask.jpg
                ├── seq19/   # images_seq19/ and masks_seq19/
                ├── seq20/   # images_seq20/ and masks_seq20/
                ├── seq21/   # images_seq21/ and masks_seq21/
                └── seq22/   # images_seq22/ and masks_seq22/
```

The shown filenames are examples from the bundled manifests, not a minimal
dataset. Prepare all listed pairs. `--polypgen` points to the parent `PolypGen/`
directory, not directly to `sequenceData/` or `positive/`. Empty62 is selected
from these same sequence directories using GT status; no separate negative
download or image-moving step is required.

The loaders accept the documented masks and matching stems; retain the
bundled relative paths when arranging target datasets. Different benchmark
archives may rename or resize images. Use original-resolution files rather
than changing the manifest to force another archive to match.

## Fixed membership and evaluation

- `splits/kvasir_seed42.json` fixes source train/validation/test identities
  and ordering. Keep a single source `images/` and `masks/` pair of folders;
  the loader selects split members without copying them into split directories.
- `splits/ood4_seed42.json` lists 1461 target image/mask pairs with domain,
  sequence, relative paths, original dimensions and expected empty-GT status.
- The paper's overlap averages use 1399 positive frames: 612 ClinicDB,
  380 ColonDB, 196 ETIS and 211 PolypGen. The remaining 62 PolypGen frames
  are used for negative-response statistics, outside overlap averages.

Configuration-only commands validate the configs/manifests without reading
image pixels. Full training/evaluation require dataset files and perform
the file/pair checks in the loaders. The commands are in
[README](../README.md#quick-start-train-and-evaluate-d) and
[four-domain evaluation](../README.md#evaluate-on-four-target-domains).

## ImageNet initialization and paper checkpoints

Dataset downloads do not include the model initialization required here.
SegmentationModel training obtains EfficientNet-B0 ImageNet weights through torchvision;
BackboneOne and BackboneTwo use the separately supplied initialization files and hashes
in [cross-backbone experiments](cross_backbone.md).

Selected trained paper checkpoints are not distributed in this candidate.
Train the desired configuration before evaluation, retaining the entire
run directory. `results/*.json` contains archived metrics and checkpoint
identities, not tensors or a download substitute for model weights.
