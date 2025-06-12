import os
from typing import Callable, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

__all__ = ["BloodMNISTDataset", "BLOODMNIST_CLASSES"]

BLOODMNIST_CLASSES = [
    "Neutrophil",
    "Eosinophil",
    "Basophil",
    "Lymphocyte",
    "Monocyte",
    "Platelet",
    "Erythroblast",
    "Immature Granulocytes",
]


class BloodMNISTDataset(Dataset):
    """Dataset for the BloodMNIST collection of peripheral blood cell images."""

    def __init__(
        self,
        root: os.PathLike,
        split: str,
        img_transform: Callable | None = None,
        lbl_transform: Callable | None = None,
        com_transform: Callable | None = None,
        verbose: bool = True,
    ) -> None:
        assert split in {"train", "val", "test"}
        self.root = os.path.join(root, split)
        self.classes = BLOODMNIST_CLASSES
        self.img_transform = img_transform
        self.lbl_transform = lbl_transform
        self.com_transform = com_transform
        self.verbose = verbose
        self.make_dataset()

    def make_dataset(self) -> None:
        samples = []
        for class_name in sorted(os.listdir(self.root)):
            class_dir = os.path.join(self.root, class_name)
            if not os.path.isdir(class_dir):
                continue
            try:
                lbl_idx = int(class_name)
            except ValueError:
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    samples.append((os.path.join(class_dir, fname), lbl_idx))
        self.samples = samples

    def __getitem__(self, idx: int):
        img_path, lbl = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        sample = {"img": img, "lbl": lbl}

        if callable(self.img_transform):
            sample["img"] = self.img_transform(sample["img"])

        if callable(self.lbl_transform):
            sample["lbl"] = self.lbl_transform(sample["lbl"])

        if callable(self.com_transform):
            return self.com_transform(sample)

        return sample

    def __len__(self) -> int:
        return len(self.samples)

    def compute_normalization_constants(self, limit: int | None = None) -> Tuple[float, float]:
        """Compute mean and std for the dataset."""
        arr = []
        length = len(self) if limit is None else limit
        for idx in tqdm(range(length), disable=not self.verbose):
            img = self[idx]["img"]
            arr.append(np.ravel(np.array(img)))
        arr = np.concatenate(arr)
        return float(np.mean(arr)), float(np.std(arr))
