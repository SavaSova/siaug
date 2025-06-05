import os
from typing import Callable, List, Tuple

import numpy as np
import polars as pl
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

__all__ = ["NIHDataset", "NIH_PATHOLOGIES"]

NIH_PATHOLOGIES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
    "No Finding",
]


class NIHDataset(Dataset):
    """The NIH ChestX-ray14 dataset.

    Args:
        csv_path (os.PathLike): Path to ``Data_Entry_2017_v2020.csv``.
        images_dir (os.PathLike): Directory with image files.
        list_path (os.PathLike | None): Optional path to a text file with image
            names for the desired split. If ``None`` the entire csv is used.
        img_transform (Callable, optional): Image transform. Defaults to ``None``.
        lbl_transform (Callable, optional): Label transform. Defaults to ``None``.
        com_transform (Callable, optional): Composite transform. Defaults to
            ``None``.
        columns (List[str], optional): Label columns to use. Defaults to
            ``NIH_PATHOLOGIES``.
        verbose (bool, optional): Show progress bar for ``compute_normalization_constants``.
    """

    def __init__(
        self,
        csv_path: os.PathLike,
        images_dir: os.PathLike,
        list_path: os.PathLike | None = None,
        img_transform: Callable | None = None,
        lbl_transform: Callable | None = None,
        com_transform: Callable | None = None,
        columns: List[str] = NIH_PATHOLOGIES,
        verbose: bool = True,
    ) -> None:
        self.img_transform = img_transform
        self.lbl_transform = lbl_transform
        self.com_transform = com_transform
        self.verbose = verbose
        self.make_dataset(csv_path, images_dir, list_path, columns)

    def make_dataset(
        self,
        csv_path: os.PathLike,
        images_dir: os.PathLike,
        list_path: os.PathLike | None,
        columns: List[str],
    ) -> None:
        df = pl.read_csv(csv_path, infer_schema_length=1000)
        if list_path is not None:
            with open(list_path) as f:
                names = {line.strip() for line in f if line.strip()}
            df = df.filter(pl.col("Image Index").is_in(names))

        # create label matrix
        def encode_labels(labels: str) -> list[int]:
            lbls = [lb.strip() for lb in labels.split("|")]
            return [1 if c in lbls else 0 for c in columns]

        lbls = df.get_column("Finding Labels").apply(encode_labels).to_list()
        imgs = df.get_column("Image Index").apply(lambda x: os.path.join(images_dir, x)).to_list()
        self.samples = list(zip(imgs, lbls))

    def __getitem__(self, idx: int):
        img_path, lbl = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        sample = {"img": img, "lbl": np.array(lbl, dtype=np.float32)}

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
        arr = []
        length = len(self) if limit is None else limit
        for idx in tqdm(range(length), disable=not self.verbose):
            arr.append(np.ravel(self[idx]["img"]))
        arr = np.concatenate(arr)
        return float(np.mean(arr)), float(np.std(arr))
