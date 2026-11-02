"BackboneOne's frozen source preprocessing, using the shared portable membership."
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from ..datasets.kvasir_seg import KvasirDataset
from ..datasets.splits import load_split_manifest


class BackboneOneDataset(Dataset):
    def __init__(self, root, names, *, augment=False, identified=False):
        self.original = KvasirDataset(root, names, image_size=256, augment=augment, mask_threshold=1)
        self.identified = identified

    def __len__(self):
        return len(self.original)

    def __getitem__(self, index):
        image, mask = self.original[index]
        pair = image, mask.unsqueeze(0).float()
        return (*pair, self.original.pairs[index].name) if self.identified else pair


def seed_worker(worker_id):
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def build_source_loaders(split, root, *, pdc, num_workers=4):
    manifest = load_split_manifest(split, root)
    train = DataLoader(BackboneOneDataset(root, manifest["splits"]["train"], augment=True, identified=pdc),
                       batch_size=4, shuffle=True, drop_last=True, num_workers=num_workers,
                       worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(42))
    val = DataLoader(BackboneOneDataset(root, manifest["splits"]["val"]), batch_size=1,
                     shuffle=False, drop_last=False, num_workers=num_workers, worker_init_fn=seed_worker)
    return train, val, manifest
