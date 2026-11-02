# Separately obtained backbone sources

This repository contains the study's methods and reproduction adapters. It
does not contain the BAMPolyp, PraNet/Res2Net or MRANet/PVT implementations or
their component-loss implementations. Obtain those sources from their authors,
following their applicable terms, and keep them outside this repository.
Dependencies, datasets and initialization weights are installed separately too.

## BAMPolyp: complete DiRaGS model

```bash
git clone https://github.com/farhad324/BAMPolyp.git /path/to/upstream/BAMPolyp
git -C /path/to/upstream/BAMPolyp checkout --detach bebb67be60b13b4c61c613fe046a920956629548
export DIRAGS_MAIN_SOURCE=/path/to/upstream/BAMPolyp/bampolyp.py
python scripts/check_external_backbones.py --backbone main
```

The loader checks LF-normalized SHA256
`6d0d40ed5c59fceaeadbd5d332b8f6b305380253b61c1b39c36075a42fba29d6`
and executes only the required definitions from the local file, in their
original order. It preserves the final IoULoss definition. It does not run
upstream dataset-download, directory-creation or training statements. This
definition loader is not a security sandbox; use the documented upstream source.

The interface changes `return_deep` into the study's structured `return_aux`
output and controls encoder weight loading. It registers no extra network
module or parameter prefix. The component losses remain external. The study
aggregates four supervised outputs with weights 0.6/0.2/0.1/0.1 and passes its
fixed component coefficients to the upstream loss. These settings reproduce
the study's baseline; they are not an additional method contribution.

Then use the training/evaluation commands in README. Configuration
checks (`--validate-only`) do not need the upstream source.

## PraNet: baseline and PDC cross-backbone experiment

```bash
git clone https://github.com/DengPingFan/PraNet.git /path/to/upstream/PraNet
git -C /path/to/upstream/PraNet checkout --detach 95f12bd4e32ec954b5069654abae7de9473fd1f5
export DIRAGS_BACKBONE_1_ROOT=/path/to/upstream/PraNet
python scripts/check_external_backbones.py --backbone backbone_1
```

Required files are `lib/PraNet_Res2Net.py`, `lib/Res2Net_v1b.py` and `MyTrain.py`.
Their LF-normalized hashes and definition names are recorded in `provenance/upstream_sources.json`. The adapter
loads the original `structure_loss`, including the legacy `reduce="none"`
behavior. The training adapter supplies the separately obtained initialization
state instead of using the upstream hard-coded local weight path. A copied
model implementation or copied loss is not needed in this repository.

## MRANet: baseline and PDC cross-backbone experiment

```bash
git clone https://github.com/hpguo1982/MRANet.git /path/to/upstream/MRANet
git -C /path/to/upstream/MRANet checkout --detach 9dada235c8f96587588c25eaa288481808b98c56
export DIRAGS_BACKBONE_2_SOURCE_DIR=/path/to/upstream/MRANet/codes
python scripts/check_external_backbones.py --backbone backbone_2
```

The recorded training archive did not include a Git commit. The commit above
was checked during packaging: all seven original code-file hashes match the
archived source hashes. It is an obtainable matching snapshot, not a claim
about the historical checkout. Original hashes remain in
`results/backbone_2_cross_backbone_seed42_reference.json`.

The loader checks and loads `model.py`, `pvtv2.py`, `utils.py` (`cus_sample`)
and `metrics.py` (`DiceBCELoss`). It omits unused script imports and global
timm registration, and retains model calculations. The adapter supplies
initialization at runtime and retains the extra PVT construction used by the
formal trainer before decoder initialization.

Install `requirements/backbone_2.txt` for these experiments. Full commands and
preprocessing details are in [cross-backbone experiments](cross_backbone.md).

## Windows PowerShell

Set the same locations using environment variables in the current terminal:

```powershell
$env:DIRAGS_MAIN_SOURCE = 'C:\upstream\BAMPolyp\bampolyp.py'
$env:DIRAGS_BACKBONE_1_ROOT = 'C:\upstream\PraNet'
$env:DIRAGS_BACKBONE_2_SOURCE_DIR = 'C:\upstream\MRANet\codes'
python scripts/check_external_backbones.py --backbone main
```

These examples are placeholders for your chosen checkout paths. Hash checking
accepts Git's LF or CRLF checkout. Other source changes fail explicitly.
