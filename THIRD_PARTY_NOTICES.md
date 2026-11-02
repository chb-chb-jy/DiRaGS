# External sources and study adaptations

This edition contains complete DiRaGS training and evaluation, and the study's
PDC cross-backbone adapters, configurations and data/evaluation protocols. It does not bundle
BAMPolyp, PraNet/Res2Net, MRANet/PVT or their component-loss implementations.
Those sources and pretrained weights are obtained separately from their authors.

Source-loading interfaces are in `src/dirags/integrations/`. They check pinned
source hashes before loading the required definitions from a user-supplied
local checkout. No upstream model source is embedded or downloaded by these
interfaces. Setup is documented in [external backbones](docs/external_backbones.md).
This notice records attribution and settings; it does not grant permissions
on behalf of upstream authors or license the entire project.

## BAMPolyp

Official repository: [farhad324/BAMPolyp](https://github.com/farhad324/BAMPolyp).
Inspected commit: `bebb67be60b13b4c61c613fe046a920956629548`.
Authors: Md. Farhadul Islam and coauthors.
[Paper: BAMPolyp: Bi-Axial Mamba Bottleneck for Gastrointestinal Polyp Segmentation](https://openaccess.thecvf.com/content/ICCV2025W/CVAMD/html/Islam_BAMPolyp_Bi-Axial_Mamba_Bottleneck_for_Gastrointestinal_Polyp_Segmentation_ICCVW_2025_paper.html).

The inspected upstream source has no explicit source-code license. That status
is recorded without treating public visibility or citation as a redistribution
grant. The network, selective scan and component-loss definitions remain in
the separately obtained upstream file, under their authors' rights.

The study's interface exposes the existing main plus three auxiliary outputs
through `return_aux` and controls encoder initialization. The formal study
uses four-output weights 0.6/0.2/0.1/0.1; the upstream deep-supervision helper
consumes only three outputs. DiRaGS uses the study's four-output aggregation.
The external ComboLoss receives effective coefficients 0.2 CE, 0.3 Dice,
0.2 Focal, 0.2 Tversky, 0.3 IoU and 0.1 Boundary. Its `bce_weight` argument
multiplies CrossEntropyLoss; the interface uses the study's `ce_weight` name.
These are disclosed reproduction settings, not additional method contributions.

## PraNet and Res2Net

Official repository: [DengPingFan/PraNet](https://github.com/DengPingFan/PraNet).
Pinned commit: `95f12bd4e32ec954b5069654abae7de9473fd1f5`.
The upstream README contains a research/education-use statement, recorded in
[PraNet source notice](docs/BACKBONE_1_UPSTREAM_NOTICE.md).

The external files provide the original network and `structure_loss`.
The study retains the loss's effective scalar BCE behavior from the legacy
`reduce` argument, four supervised maps, multi-scale updates and elementwise
gradient clipping. The initialization state is supplied at runtime instead of
using the upstream hard-coded local path. The study assigns decay from the
initial learning rate rather than compounding it. Its PDC paired-view steps
and configuration records remain in this repository.

## MRANet and PVTv2

Official repository: [hpguo1982/MRANet](https://github.com/hpguo1982/MRANet).
Seven original source hashes are preserved in
`results/backbone_2_cross_backbone_seed42_reference.json`. The historical training
archive did not record a Git commit. During packaging, publicly obtainable
commit `9dada235c8f96587588c25eaa288481808b98c56` was checked against all seven
archived source hashes and matched. It is a matching retrieval snapshot;
it is not asserted to be the historical checkout.

The inspected archive provides no explicit source-code license. The source
implementations remain external, including MRANet, PVTv2, `cus_sample` and
DiceBCELoss. The study interface preserves BGR inputs, soft labels, the original
loss, and the reference PVT construction that affects initialization RNG.
PDC uses sequential half-weighted view backwards with one optimizer step.
Initialization file I/O is supplied at runtime; selected states load strictly.

## Installed dependencies

PyTorch, torchvision, einops, NumPy, Pillow and optional OpenCV/timm are installed
separately and retain their own terms. Their source and pretrained weights are
not bundled. Refer to the corresponding official projects for their licenses.
Dependency licenses do not grant rights in external backbone projects.

The study authors' choice of license for their own distributed code is separate
from these upstream terms; this edition does not apply a blanket MIT license.

## Public naming and compatibility mapping

The public package is `dirags`. Its main connection is
`integrations/segmentation.py`; cross-backbone adapters are named `backbone_1`
and `backbone_2`. The correspondence is BAMPolyp, PraNet and MRANet,
respectively. Network and component-loss implementations remain external.
`provenance/upstream_sources.json` records the exact source symbols, paths,
hashes and feature-tap locations required to connect those implementations.
`provenance/compatibility.json` preserves legacy checkpoint-format values and
defines current generic export labels. These are readable metadata, not
embedded model code. Original checkpoint parameter names are not rewritten.
Naming changes do not change source attribution or the upstream terms above.
