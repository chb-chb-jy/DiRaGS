"""Original Kvasir-SEG pairing, preprocessing and source augmentation."""

from __future__ import annotations
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


MASK_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


IMAGENET_MEAN = (0.485, 0.456, 0.406)


IMAGENET_STD = (0.229, 0.224, 0.225)


LEGACY_MASK_THRESHOLD = 1


@dataclass(frozen=True)
class Pair:
    name: str
    image: Path
    mask: Path


def _files_with_suffix(directory: Path, suffixes: set[str]) -> list[Path]:
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in suffixes
        ),
        key=lambda path: path.name,
    )


def build_pairs(root: Path) -> list[Pair]:
    """Return exact one-to-one image/mask pairs, rejecting ambiguous data."""

    root = Path(root)
    image_dir = root / "images"
    mask_dir = root / "masks"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"images directory not found: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"masks directory not found: {mask_dir}")

    image_paths = _files_with_suffix(image_dir, IMAGE_SUFFIXES)
    mask_paths = _files_with_suffix(mask_dir, MASK_SUFFIXES)
    if not image_paths:
        raise ValueError(f"no supported images found in {image_dir}")

    image_stems = [path.stem for path in image_paths]
    duplicate_image_stems = sorted(
        stem for stem in set(image_stems) if image_stems.count(stem) > 1
    )
    if duplicate_image_stems:
        raise ValueError(f"duplicate image stems: {duplicate_image_stems}")

    masks_by_stem: dict[str, list[Path]] = {}
    for path in mask_paths:
        masks_by_stem.setdefault(path.stem, []).append(path)

    pairs: list[Pair] = []
    for image in image_paths:
        candidates = sorted(
            masks_by_stem.get(image.stem, []), key=lambda path: path.name
        )
        if not candidates:
            raise FileNotFoundError(f"no mask for {image.name}")
        if len(candidates) != 1:
            raise ValueError(f"expected exactly one mask for {image.name}")
        pairs.append(Pair(name=image.name, image=image, mask=candidates[0]))

    orphan_stems = sorted(set(masks_by_stem) - set(image_stems))
    if orphan_stems:
        raise ValueError(f"orphan masks: {orphan_stems}")
    return pairs


def create_split(names: Sequence[str], seed: int = 42) -> dict[str, list[str]]:
    """Reproduce the original sorted NumPy 80/10/10 split deterministically."""

    names = [str(name) for name in names]
    if len(names) != len(set(names)):
        raise ValueError("duplicate image names are not allowed")
    if not names:
        raise ValueError("cannot split an empty dataset")

    ordered = np.asarray(sorted(names), dtype=object)
    indices = np.random.RandomState(seed).permutation(len(ordered))
    train_end = int(0.8 * len(ordered))
    val_end = int(0.9 * len(ordered))
    return {
        "train": ordered[indices[:train_end]].tolist(),
        "val": ordered[indices[train_end:val_end]].tolist(),
        "test": ordered[indices[val_end:]].tolist(),
    }


class KvasirDataset(Dataset):
    'Load a named subset using the frozen source preprocessing protocol.'

    def __init__(
        self,
        root: Path,
        names: Sequence[str],
        image_size: int = 256,
        augment: bool = False,
        mask_threshold: int = LEGACY_MASK_THRESHOLD,
    ) -> None:
        if int(image_size) <= 0:
            raise ValueError("image_size must be positive")
        if (
            isinstance(mask_threshold, bool)
            or not isinstance(mask_threshold, int)
            or not 1 <= mask_threshold <= 255
        ):
            raise ValueError("mask_threshold must be an integer in [1, 255]")
        names = [str(name) for name in names]
        if len(names) != len(set(names)):
            raise ValueError("duplicate dataset names are not allowed")

        pairs = {pair.name: pair for pair in build_pairs(Path(root))}
        missing = sorted(set(names) - set(pairs))
        if missing:
            raise KeyError(f"split contains unknown images: {missing}")
        self.pairs = [pairs[name] for name in names]
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.mask_threshold = int(mask_threshold)

    def __len__(self) -> int:
        return len(self.pairs)

    def _augment(self, image: Image.Image, mask: Image.Image):
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            image = TF.rotate(
                image,
                angle,
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )
            mask = TF.rotate(
                mask,
                angle,
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )
        if random.random() > 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
            image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))
        return image, mask

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.pairs[index]
        with Image.open(pair.image) as source:
            image = source.convert("RGB")
        with Image.open(pair.mask) as source:
            mask = source.convert("L")

        if self.augment:
            image, mask = self._augment(image, mask)

        size = [self.image_size, self.image_size]
        image = TF.resize(image, size, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, size, interpolation=InterpolationMode.NEAREST)
        image_tensor = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        mask_array = (np.array(mask, copy=True) >= self.mask_threshold).astype(np.int64)
        mask_tensor = torch.from_numpy(mask_array).long()
        return image_tensor, mask_tensor


class IdentifiedKvasirDataset(KvasirDataset):
    """Return the unchanged source tensors with the image identity for PDC."""

    def __getitem__(self, index):
        image, mask = super().__getitem__(index)
        return image, mask, self.pairs[index].name
