"""Original OpenCV BGR/255, released augmentation and soft-label preprocessing."""
import random
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from ..datasets.splits import load_split_manifest
from .backbone_2 import require
IMAGE_SIZE = 256
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MASK_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

def resolve_pair(root: Path, name: str) -> tuple[Path, Path]:
    image = root / "images" / name
    require(image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES,
            f"image not found: {image}")
    masks = sorted(
        candidate for candidate in (root / "masks").glob(f"{image.stem}.*")
        if candidate.is_file() and candidate.suffix.lower() in MASK_SUFFIXES
    )
    require(len(masks) == 1,
            f"expected exactly one mask for {name}, found {len(masks)}")
    return image, masks[0]

class BackboneTwoDataset(Dataset):
    def __init__(self, root, names, augment, identified=False):
        self.identified = identified
        self.pairs = [resolve_pair(root, str(name)) for name in names]
        self.augment = bool(augment)

    @staticmethod
    def apply_upstream_augmentation(image, mask):
        # Dependency-free equivalent of the released Albumentations recipe.
        if random.random() < 0.3:
            height, width = image.shape[:2]
            angle = random.uniform(-35.0, 35.0)
            matrix = cv2.getRotationMatrix2D(
                (width / 2.0, height / 2.0), angle, 1.0
            )
            image = cv2.warpAffine(
                image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            mask = cv2.warpAffine(
                mask, matrix, (width, height), flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
        if random.random() < 0.3:
            image, mask = cv2.flip(image, 1), cv2.flip(mask, 1)
        if random.random() < 0.3:
            image, mask = cv2.flip(image, 0), cv2.flip(mask, 0)
        # Released source includes CoarseDropout although the paper omits it.
        # In the old Albumentations API mask_fill_value defaults to None, so
        # holes affect only the image, not the segmentation mask.
        if random.random() < 0.3:
            height, width = image.shape[:2]
            for _ in range(random.randint(1, 10)):
                hole_h = random.randint(1, min(32, height))
                hole_w = random.randint(1, min(32, width))
                top = random.randint(0, height - hole_h)
                left = random.randint(0, width - hole_w)
                image[top:top + hole_h, left:left + hole_w] = 0
        return image, mask

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        require(image is not None and mask is not None,
                f"OpenCV read failed: {image_path}")
        if self.augment:
            image, mask = self.apply_upstream_augmentation(image, mask)
        # Preserve released preprocessing exactly: BGR, /255, and the
        # released default linear interpolation for both image and mask.
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE),
                           interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE),
                          interpolation=cv2.INTER_LINEAR)
        image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
        mask = np.expand_dims(mask.astype(np.float32) / 255.0, axis=0)
        item = torch.from_numpy(image), torch.from_numpy(mask)
        return (*item, image_path.name) if self.identified else item


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def build_source_loaders(split, root, *, pdc=False, num_workers=4):
    manifest = load_split_manifest(split, root)
    train = DataLoader(BackboneTwoDataset(root, manifest["splits"]["train"], True, identified=pdc),
                       batch_size=16, shuffle=True, drop_last=True, num_workers=num_workers,
                       worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(42))
    val = DataLoader(BackboneTwoDataset(root, manifest["splits"]["val"], False), batch_size=16,
                     shuffle=False, drop_last=False, num_workers=num_workers, worker_init_fn=seed_worker)
    return train, val, manifest
