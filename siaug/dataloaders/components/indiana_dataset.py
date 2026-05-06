import os
from typing import Callable, Tuple

import numpy as np
import polars as pl
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

__all__ = ["IndianaDataset", "IndianaBinaryDataset"]


class IndianaDataset(Dataset):
    """Indiana University chest X-ray dataset.

    The Kaggle version is expected to contain ``indiana_reports.csv``,
    ``indiana_projections.csv`` and an image directory, usually
    ``images/images_normalized``.
    """

    def __init__(
        self,
        reports_csv: os.PathLike,
        projections_csv: os.PathLike,
        images_dir: os.PathLike,
        list_path: os.PathLike | None = None,
        projection: str | None = "frontal",
        img_transform: Callable | None = None,
        lbl_transform: Callable | None = None,
        com_transform: Callable | None = None,
        verbose: bool = True,
    ) -> None:
        self.img_transform = img_transform
        self.lbl_transform = lbl_transform
        self.com_transform = com_transform
        self.verbose = verbose
        self.make_dataset(reports_csv, projections_csv, images_dir, list_path, projection)

    def make_dataset(
        self,
        reports_csv: os.PathLike,
        projections_csv: os.PathLike,
        images_dir: os.PathLike,
        list_path: os.PathLike | None,
        projection: str | None,
    ) -> None:
        reports = pl.read_csv(reports_csv, infer_schema_length=1000)
        projections = pl.read_csv(projections_csv, infer_schema_length=1000)

        reports = reports.with_columns(pl.col("uid").cast(pl.Int64))
        projections = projections.with_columns(pl.col("uid").cast(pl.Int64))

        if projection is not None:
            projections = projections.filter(
                pl.col("projection").str.to_lowercase() == projection.lower()
            )

        df = projections.join(reports, on="uid", how="inner")

        if list_path is not None:
            with open(list_path, encoding="utf-8") as f:
                names = {line.strip() for line in f if line.strip()}
            df = df.filter(
                pl.col("filename").is_in(names) | pl.col("uid").cast(pl.Utf8).is_in(names)
            )

        def encode_binary_label(problems: str | None) -> list[int]:
            if problems is None:
                return [1]
            normalized = problems.strip().lower()
            tokens = {token.strip() for token in normalized.replace(",", ";").split(";")}
            return [0 if tokens == {"normal"} else 1]

        lbls = (
            df.get_column("Problems")
            .map_elements(encode_binary_label, return_dtype=pl.List(pl.Int8))
            .to_list()
        )
        imgs = (
            df.get_column("filename")
            .map_elements(lambda x: os.path.join(images_dir, x), return_dtype=pl.Utf8)
            .to_list()
        )
        uids = df.get_column("uid").to_list()
        problems = df.get_column("Problems").to_list()
        self.samples = list(zip(imgs, lbls, uids, problems))

    def __getitem__(self, idx: int):
        img_path, lbl, uid, problems = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        sample = {
            "img": img,
            "lbl": np.array(lbl, dtype=np.float32),
            "uid": uid,
            "problems": problems,
            "img_path": img_path,
        }

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


class IndianaBinaryDataset(IndianaDataset):
    """Binary Indiana dataset alias for Hydra configs."""
