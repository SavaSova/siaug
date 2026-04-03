# -*- coding: utf-8 -*-
import json
from pathlib import Path

p = Path(r"D:\Code\siaug-2025\siaug\notebooks\nih-binary-error-analysis.ipynb")

cells = []


def add_md(text: str):
    text = text.strip("\n")
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.split("\n")],
        }
    )


def add_code(text: str):
    text = text.strip("\n")
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.split("\n")],
        }
    )


add_md(
    """
# Анализ ошибок для NIH Binary

Этот ноутбук нужен для практического завершения раздела 5.6, подготовки рисунков для Word и заполнения приложения Е.

Что он делает:
- загружает лучший checkpoint выбранного режима обучения;
- использует уже существующий CSV с предсказаниями, если он был создан ранее;
- при необходимости заново прогоняет модель по validation-выборке;
- показывает ложноположительные, ложноотрицательные, истинно положительные и истинно отрицательные примеры;
- сохраняет готовые изображения, которые можно вставлять в диплом.

Рекомендуемый порядок работы:
1. Сначала работать с `fine_tuning`.
2. Для основной главы использовать два рисунка: ложноположительные и ложноотрицательные ошибки `fine_tuning`.
3. Для приложения Е отдельно сохранить дополнительные ошибочные и правильные примеры.
"""
)

add_code(
    """
from pathlib import Path
import importlib
import subprocess
import sys

AUTO_INSTALL_MISSING = False

REQUIRED_PACKAGES = {
    "matplotlib": "matplotlib",
    "polars": "polars",
    "PIL": "pillow",
    "tqdm": "tqdm",
}

missing = []
for module_name, package_name in REQUIRED_PACKAGES.items():
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError:
        missing.append(package_name)

if missing and AUTO_INSTALL_MISSING:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
elif missing:
    raise RuntimeError(
        "Не хватает пакетов для ноутбука: "
        + ", ".join(missing)
        + ". Установите их командой: python -m pip install "
        + " ".join(missing)
        + ". Если хотите, можно поставить AUTO_INSTALL_MISSING = True."
    )

import matplotlib.pyplot as plt
import polars as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm.auto import tqdm

from siaug.dataloaders.components.nih_dataset import NIHBinaryDataset
from siaug.models.lincls import create_lincls

plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams["axes.grid"] = False
"""
)

add_code(
    """
if Path.cwd().name == "notebooks":
    PROJECT_DIR = Path.cwd().resolve().parent
else:
    PROJECT_DIR = Path.cwd().resolve()

DATA_DIR = PROJECT_DIR / "data" / "cxr8"
CSV_PATH = DATA_DIR / "Data_Entry_2017_v2020.csv"
IMAGES_DIR = DATA_DIR / "images_png"
VALID_LIST = PROJECT_DIR / "data_splits" / "nih_binary" / "valid_list.txt"
OUTPUT_DIR = PROJECT_DIR / "error_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cpu")
BATCH_SIZE = 8
NUM_WORKERS = 0
THRESHOLD = 0.5
FORCE_REGENERATE_CSV = False

EXPERIMENTS = {
    "baseline": {
        "checkpoint": PROJECT_DIR / "logs" / "lcls_nih_binary_baseline" / "runs" / "2026-03-27_14-35-14" / "checkpoints" / "best.pt" / "model.safetensors",
        "title": "Базовый вариант",
    },
    "linear_eval": {
        "checkpoint": PROJECT_DIR / "logs" / "lcls_nih_binary_finetune" / "runs" / "2026-03-27_19-27-59" / "checkpoints" / "best.pt" / "model.safetensors",
        "title": "Оценка фиксированных признаков",
    },
    "fine_tuning": {
        "checkpoint": PROJECT_DIR / "logs" / "lcls_nih_binary_finetune" / "runs" / "2026-03-28_09-22-20" / "checkpoints" / "best.pt" / "model.safetensors",
        "title": "Полное дообучение",
    },
}

PROJECT_DIR, OUTPUT_DIR
"""
)

add_code(
    """
valid_transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.50551915, 0.50551915, 0.50551915], std=[0.2895694, 0.2895694, 0.2895694]),
])


def build_valid_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    dataset = NIHBinaryDataset(
        csv_path=CSV_PATH,
        images_dir=IMAGES_DIR,
        list_path=VALID_LIST,
        img_transform=valid_transform,
        com_transform=None,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def build_model(checkpoint_path: Path, device=DEVICE):
    model = create_lincls(
        backbone="resnet50",
        num_classes=1,
        num_channels=3,
        freeze=False,
        ckpt_path=checkpoint_path,
        prefix=None,
        reset_head=False,
    )
    model = model.to(device)
    model.eval()
    return model


def collect_predictions(
    experiment_key: str,
    threshold=THRESHOLD,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    device=DEVICE,
    force_regenerate: bool = FORCE_REGENERATE_CSV,
):
    config = EXPERIMENTS[experiment_key]
    output_csv = OUTPUT_DIR / f"{experiment_key}_valid_predictions.csv"

    if output_csv.exists() and not force_regenerate:
        print(f"Используется уже существующий CSV: {output_csv}")
        return pl.read_csv(output_csv)

    loader = build_valid_loader(batch_size=batch_size, num_workers=num_workers)
    model = build_model(config["checkpoint"], device=device)

    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{experiment_key}: inference"):
            images = batch["img"].to(device)
            targets = batch["lbl"].view(-1).long()
            img_paths = batch["img_path"]

            logits = model(images).view(-1).cpu()
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long()

            for img_path, target, prob, pred in zip(img_paths, targets, probs, preds):
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

                rows.append({
                    "experiment": experiment_key,
                    "experiment_title": config["title"],
                    "img_path": str(img_path),
                    "file_name": Path(img_path).name,
                    "target": target_int,
                    "probability": float(prob.item()),
                    "prediction": pred_int,
                    "error_type": error_type,
                })

    df = pl.DataFrame(rows)
    df.write_csv(output_csv)
    print(f"Сохранён CSV: {output_csv}")
    return df


def summarize_errors(df: pl.DataFrame):
    return df.group_by("error_type").len().sort("error_type")


def show_examples(
    df: pl.DataFrame,
    error_type: str,
    n: int = 6,
    sort_desc: bool = True,
    offset: int = 0,
    save_path: str | None = None,
    figure_title: str | None = None,
):
    if error_type == "false_positive":
        subset = df.filter(pl.col("error_type") == error_type).sort("probability", descending=sort_desc).slice(offset, n)
    elif error_type == "false_negative":
        subset = df.filter(pl.col("error_type") == error_type).sort("probability", descending=not sort_desc).slice(offset, n)
    elif error_type == "true_positive":
        subset = df.filter(pl.col("error_type") == error_type).sort("probability", descending=sort_desc).slice(offset, n)
    else:
        subset = df.filter(pl.col("error_type") == error_type).sort("probability", descending=not sort_desc).slice(offset, n)

    rows = subset.to_dicts()
    if not rows:
        print(f"Нет примеров для {error_type}")
        return

    cols = 3
    plot_rows = (len(rows) + cols - 1) // cols
    fig, axes = plt.subplots(plot_rows, cols, figsize=(15, 5 * plot_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, row in zip(axes, rows):
        image = Image.open(row["img_path"]).convert("L")
        ax.imshow(image, cmap="gray")
        ax.set_title(
            f"{row['file_name']}\n"
            f"истинная метка={row['target']}, предсказание={row['prediction']}\n"
            f"вероятность={row['probability']:.4f}",
            fontsize=10,
        )
        ax.axis("off")

    for ax in axes[len(rows):]:
        ax.axis("off")

    final_title = figure_title or f"Примеры: {error_type}"
    fig.suptitle(final_title, fontsize=16)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Сохранён рисунок: {save_path}")

    plt.show()
"""
)

add_md(
    """
## Прогон одного режима

Ниже используется `fine_tuning` как основной режим для анализа ошибок в тексте диплома.
Если CSV уже существует, ноутбук возьмёт его повторно.
"""
)

add_code(
    """
EXPERIMENT_KEY = "fine_tuning"  # baseline | linear_eval | fine_tuning

df = collect_predictions(
    experiment_key=EXPERIMENT_KEY,
    threshold=THRESHOLD,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    device=DEVICE,
)

df.head()
"""
)

add_code(
    """
summarize_errors(df)
"""
)

add_md(
    """
## Рисунки для Word

Ниже идут две ключевые ячейки для основной главы диплома.

Используемые рисунки:
- `Рисунок 5` — примеры ложноположительных классификаций модели полного дообучения;
- `Рисунок 6` — примеры ложноотрицательных классификаций модели полного дообучения.

Сохраняемые файлы:
- `error_analysis/figure_5_false_positive_fine_tuning.png`
- `error_analysis/figure_6_false_negative_fine_tuning.png`
"""
)

add_code(
    """
show_examples(
    df,
    error_type="false_positive",
    n=6,
    save_path=OUTPUT_DIR / "figure_5_false_positive_fine_tuning.png",
    figure_title="Рисунок 5. Примеры ложноположительных классификаций модели полного дообучения",
)
"""
)

add_code(
    """
show_examples(
    df,
    error_type="false_negative",
    n=6,
    save_path=OUTPUT_DIR / "figure_6_false_negative_fine_tuning.png",
    figure_title="Рисунок 6. Примеры ложноотрицательных классификаций модели полного дообучения",
)
"""
)

add_md(
    """
## Приложение Е: примеры изображений и предсказаний модели

Ниже идут дополнительные примеры для приложения Е.
CSV при этом не пересчитывается, если он уже создан.
Чтобы получить другой набор примеров, меняйте `APPENDIX_OFFSET`.
"""
)

add_code(
    """
APPENDIX_OFFSET = 6
APPENDIX_COUNT = 6
"""
)

add_code(
    """
show_examples(
    df,
    error_type="false_positive",
    n=APPENDIX_COUNT,
    offset=APPENDIX_OFFSET,
    save_path=OUTPUT_DIR / "appendix_e_false_positive_fine_tuning.png",
    figure_title="Приложение Е. Дополнительные ложноположительные примеры полного дообучения",
)
"""
)

add_code(
    """
show_examples(
    df,
    error_type="false_negative",
    n=APPENDIX_COUNT,
    offset=APPENDIX_OFFSET,
    save_path=OUTPUT_DIR / "appendix_e_false_negative_fine_tuning.png",
    figure_title="Приложение Е. Дополнительные ложноотрицательные примеры полного дообучения",
)
"""
)

add_code(
    """
show_examples(
    df,
    error_type="true_positive",
    n=APPENDIX_COUNT,
    offset=APPENDIX_OFFSET,
    save_path=OUTPUT_DIR / "appendix_e_true_positive_fine_tuning.png",
    figure_title="Приложение Е. Истинно положительные примеры полного дообучения",
)
"""
)

add_code(
    """
show_examples(
    df,
    error_type="true_negative",
    n=APPENDIX_COUNT,
    offset=APPENDIX_OFFSET,
    save_path=OUTPUT_DIR / "appendix_e_true_negative_fine_tuning.png",
    figure_title="Приложение Е. Истинно отрицательные примеры полного дообучения",
)
"""
)

add_md(
    """
## При необходимости: прогон всех трёх режимов

Этот блок нужен только в том случае, если вы захотите получить CSV для всех режимов.
Если соответствующий CSV уже существует, ноутбук использует его повторно.
Чтобы принудительно пересчитать предсказания, установите `FORCE_REGENERATE_CSV = True`.
"""
)

add_code(
    """
RUN_ALL = False

all_results = {}
if RUN_ALL:
    for key in EXPERIMENTS:
        print(f"\n=== {key} ===")
        all_results[key] = collect_predictions(
            experiment_key=key,
            threshold=THRESHOLD,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            device=DEVICE,
        )
        display(summarize_errors(all_results[key]))
"""
)

add_md(
    """
## Что дальше писать в диплом

После просмотра примеров полезно коротко зафиксировать наблюдения:
- какие ложноположительные случаи встречаются чаще всего;
- какие ложноотрицательные случаи оказываются наиболее трудными;
- какие истинно положительные и истинно отрицательные примеры выглядят наиболее показательными;
- согласуется ли характер ошибок с балансом `Precision` и `Recall`.

Для основной главы обычно достаточно двух рисунков:
- `Рисунок 5` — ложноположительные случаи `fine_tuning`;
- `Рисунок 6` — ложноотрицательные случаи `fine_tuning`.

Для приложения Е можно использовать четыре типа примеров: false positive, false negative, true positive и true negative.
"""
)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook rewritten successfully")
