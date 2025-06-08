import argparse
import polars as pl
import torch
from siaug.dataloaders.components.nih_dataset import NIH_PATHOLOGIES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute pos_weight for NIH dataset"
    )
    parser.add_argument("csv_path", type=str, help="Path to Data_Entry_2017_v2020.csv")
    parser.add_argument("images_dir", type=str, help="Directory with images")
    parser.add_argument(
        "--list_path",
        type=str,
        default=None,
        help="Optional list of images for split",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pl.read_csv(args.csv_path, infer_schema_length=1000)
    if args.list_path is not None:
        with open(args.list_path) as f:
            names = {line.strip() for line in f if line.strip()}
        df = df.filter(pl.col("Image Index").is_in(names))

    def encode_labels(labels: str) -> list[int]:
        lbls = [lb.strip() for lb in labels.split("|")]
        return [1 if c in lbls else 0 for c in NIH_PATHOLOGIES]

    lbls = (
        df.get_column("Finding Labels")
        .map_elements(encode_labels, return_dtype=pl.List(pl.Int8))
        .to_list()
    )
    lbls = torch.tensor(lbls)
    pos_counts = lbls.sum(dim=0)
    total = lbls.shape[0]
    pos_weight = (total - pos_counts) / pos_counts
    print("pos_weight:")
    print("[" + ", ".join(f"{w:.4f}" for w in pos_weight.tolist()) + "]")


if __name__ == "__main__":
    main()
