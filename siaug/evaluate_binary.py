import csv
import os
from pathlib import Path
from typing import Any

import hydra
import pyrootutils
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import (
    binary_accuracy,
    binary_auroc,
    binary_average_precision,
    binary_f1_score,
    binary_precision,
    binary_recall,
)

from siaug.utils.extras import sanitize_dataloader_kwargs, set_seed

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
config_dir = os.path.join(root, "configs")


def _as_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _confusion_counts(
    probs: torch.Tensor, targets: torch.Tensor, threshold: float
) -> dict[str, int]:
    preds = probs >= threshold
    positives = targets == 1
    negatives = targets == 0

    return {
        "tn": int((~preds & negatives).sum().item()),
        "fp": int((preds & negatives).sum().item()),
        "fn": int((~preds & positives).sum().item()),
        "tp": int((preds & positives).sum().item()),
    }


def _metrics_at_threshold(
    probs: torch.Tensor, targets: torch.Tensor, threshold: float
) -> dict[str, Any]:
    counts = _confusion_counts(probs, targets, threshold)
    return {
        "threshold": threshold,
        "accuracy": _as_float(binary_accuracy(probs, targets, threshold=threshold)),
        "f1": _as_float(binary_f1_score(probs, targets, threshold=threshold)),
        "precision": _as_float(binary_precision(probs, targets, threshold=threshold)),
        "recall": _as_float(binary_recall(probs, targets, threshold=threshold)),
        **counts,
    }


def _write_dicts(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@hydra.main(version_base="1.2", config_path=config_dir, config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    """Evaluate a binary classification checkpoint and export threshold diagnostics."""

    OmegaConf.set_struct(cfg, False)
    if "eval" not in cfg or cfg.eval.get("checkpoint") is None:
        raise ValueError("Pass +eval.checkpoint=/path/to/model.safetensors")

    seed = cfg.get("seed", None)
    if seed is not None:
        set_seed(seed)

    checkpoint = Path(str(cfg.eval.checkpoint))
    output_dir = Path(str(cfg.eval.get("output_dir", Path(cfg.paths.output_dir) / "binary_eval")))
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg.model.ckpt_path = str(checkpoint)
    cfg.model.prefix = None
    cfg.model.reset_head = False
    cfg.model.freeze = False

    cfg.dataloader.valid.num_workers = int(cfg.dataloader.valid.get("num_workers", 0))
    cfg.dataloader.valid.shuffle = False
    cfg.dataloader.valid.drop_last = False

    dataloader = DataLoader(**sanitize_dataloader_kwargs(instantiate(cfg.dataloader.valid)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = instantiate(cfg.model).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.eval()

    criterion = instantiate(cfg.criterion).to(device)

    rows: list[dict[str, Any]] = []
    logits_all: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    losses: list[float] = []
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["img"].to(device)
            targets = batch["lbl"].to(device)
            logits = model(images)
            loss = criterion(logits, targets.float())

            batch_size = images.size(0)
            losses.append(float(loss.item()) * batch_size)
            total += batch_size

            logits_cpu = logits.detach().cpu().reshape(-1)
            targets_cpu = targets.detach().cpu().long().reshape(-1)
            probs_cpu = torch.sigmoid(logits_cpu)
            logits_all.append(logits_cpu)
            targets_all.append(targets_cpu)

            uids = batch.get("uid", [None] * batch_size)
            problems = batch.get("problems", [None] * batch_size)
            img_paths = batch.get("img_path", [None] * batch_size)
            for idx in range(batch_size):
                rows.append(
                    {
                        "uid": uids[idx],
                        "img_path": img_paths[idx],
                        "problems": problems[idx],
                        "target": int(targets_cpu[idx].item()),
                        "logit": float(logits_cpu[idx].item()),
                        "probability": float(probs_cpu[idx].item()),
                    }
                )

    logits = torch.cat(logits_all)
    targets = torch.cat(targets_all)
    probs = torch.sigmoid(logits)

    threshold_rows = [_metrics_at_threshold(probs, targets, i / 100) for i in range(1, 100)]
    best_f1 = max(threshold_rows, key=lambda row: row["f1"])
    best_balanced = max(
        threshold_rows,
        key=lambda row: (row["recall"] + (row["tn"] / max(1, row["tn"] + row["fp"]))) / 2,
    )
    recall_candidates = [row for row in threshold_rows if row["recall"] >= 0.9]
    recall_oriented = (
        max(recall_candidates, key=lambda row: row["precision"]) if recall_candidates else best_f1
    )

    summary_rows = [
        {
            "checkpoint": str(checkpoint),
            "loss": sum(losses) / max(1, total),
            "auroc": _as_float(binary_auroc(probs, targets)),
            "auprc": _as_float(binary_average_precision(probs, targets)),
            "positive_rate": float(targets.float().mean().item()),
            "mean_probability": float(probs.mean().item()),
            "best_f1_threshold": best_f1["threshold"],
            "best_f1": best_f1["f1"],
            "best_f1_accuracy": best_f1["accuracy"],
            "best_f1_precision": best_f1["precision"],
            "best_f1_recall": best_f1["recall"],
            "best_balanced_threshold": best_balanced["threshold"],
            "recall_oriented_threshold": recall_oriented["threshold"],
        }
    ]

    _write_dicts(output_dir / "predictions.csv", rows)
    _write_dicts(output_dir / "threshold_table.csv", threshold_rows)
    _write_dicts(output_dir / "metrics_summary.csv", summary_rows)
    _write_dicts(
        output_dir / "confusion_threshold_0_50.csv", [_metrics_at_threshold(probs, targets, 0.5)]
    )
    _write_dicts(output_dir / "confusion_best_f1.csv", [best_f1])
    _write_dicts(output_dir / "confusion_recall_oriented.csv", [recall_oriented])

    summary_md = output_dir / "metrics_summary.md"
    summary_md.write_text(
        "\n".join(
            [
                "# Binary Evaluation Summary",
                "",
                f"Checkpoint: `{checkpoint}`",
                f"Samples: {total}",
                f"Loss: {summary_rows[0]['loss']:.6f}",
                f"AUROC: {summary_rows[0]['auroc']:.6f}",
                f"AUPRC: {summary_rows[0]['auprc']:.6f}",
                f"Positive rate: {summary_rows[0]['positive_rate']:.6f}",
                f"Mean probability: {summary_rows[0]['mean_probability']:.6f}",
                "",
                "## Threshold 0.50",
                "",
                str(_metrics_at_threshold(probs, targets, 0.5)),
                "",
                "## Best F1 Threshold",
                "",
                str(best_f1),
                "",
                "## Recall-Oriented Threshold",
                "",
                str(recall_oriented),
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"=> Binary evaluation saved to {output_dir}")
    print(f"=> AUROC={summary_rows[0]['auroc']:.6f} AUPRC={summary_rows[0]['auprc']:.6f}")
    print(f"=> Best F1 threshold={best_f1['threshold']:.2f} F1={best_f1['f1']:.6f}")


if __name__ == "__main__":
    main()
