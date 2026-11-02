# Installation

Use Python 3.10.18 and the recorded Linux/CUDA stack for paper runs.
The root requirements pin PyTorch 1.12.1+cu113, torchvision 0.13.1+cu113,
NumPy 1.26.4, Pillow 12.0.0 and einops 0.8.1.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

The recorded CUDA version is 11.3. The requirements capture the direct
dependencies, rather than an OS/driver lock. The local CPU packaging checks
used a separate environment and do not represent a new paper training run.

Obtain and configure model sources using [external backbone setup](external_backbones.md).
The project entrypoints load the local `src/dirags/` package directly.
Datasets and weights are obtained separately from their providers.

PraNet uses the core dependencies. For MRANet also follow the exact optional
dependency commands in [cross-backbone experiments](cross_backbone.md#backbonetwo-dependencies-and-source-training).
They retain the recorded NumPy/OpenCV runtime pairing.

Run commands from the repository root; keep datasets and training outputs
outside the source directories. [README](../README.md) gives the main workflow.
