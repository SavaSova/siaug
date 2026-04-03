import csv
import os
from pathlib import Path

import hydra
import pyrootutils
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from siaug.utils.extras import sanitize_dataloader_kwargs, set_seed

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
config_dir = os.path.join(root, "configs")


@hydra.main(version_base="1.2", config_path=config_dir, config_name="train.yaml")
def main(cfg: DictConfig):
    """Export per-image predictions for binary validation error analysis."""

    print(f"=> Starting error analysis export [experiment={cfg['task_name']}]")
    cfg = instantiate(cfg)

    seed = cfg.get("seed", None)
    if seed is not None:
        set_seed(seed)

    output_csv = cfg.get("output_csv", None)
    if output_csv is None:
        raise ValueError("`output_csv` must be specified.")

    threshold = float(cfg.get("threshold", 0.5))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=> Using device [device={device}]")

    print("=> Instantiating valid dataloader")
    valid_dataloader = DataLoader(**sanitize_dataloader_kwargs(cfg["dataloader"]["valid"]))

    print("=> Creating model")
    model = cfg["model"].to(device)
    model.eval()

    rows = []
    print("=> Exporting validation predictions")
    with torch.no_grad():
        for batch in tqdm(valid_dataloader):
            images = batch["img"].to(device)
            targets = batch["lbl"].to(device)
            img_paths = batch["img_path"]

            logits = model(images).view(-1)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long()
            targets = targets.view(-1).long()

            for img_path, target, prob, pred in zip(
                img_paths, targets.cpu(), probs.cpu(), preds.cpu()
            ):
                target_int = int(target.item())
                pred_int = int(pred.item())

                if pred_int == 1 and target_int == 0:
                    error_type = "false_positive"
                elif pred_int == 0 and target_int == 1:
                    error_type = "false_negative"
                elif pred_int == target_int == 1:
                    error_type = "true_positive"
                else:
                    error_type = "true_negative"

                rows.append(
                    {
                        "img_path": str(img_path),
                        "file_name": Path(img_path).name,
                        "target": target_int,
                        "probability": float(prob.item()),
                        "prediction": pred_int,
                        "error_type": error_type,
                    }
                )

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "img_path",
                "file_name",
                "target",
                "probability",
                "prediction",
                "error_type",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"=> Saved predictions CSV [path={output_csv}]")
    print(f"=> Total rows exported: {len(rows)}")


if __name__ == "__main__":
    main()
