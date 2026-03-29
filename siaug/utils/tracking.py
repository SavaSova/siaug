from __future__ import annotations

import json
from copy import deepcopy
from typing import Any


def resolve_tracker(
    cfg: dict[str, Any],
) -> tuple[bool, str | None, dict[str, dict[str, Any]], str | None]:
    """Return accelerator tracker settings derived from the Hydra config."""

    logger_cfg = cfg.get("logger")
    if logger_cfg is None:
        return False, None, {}, None

    tracker_cfg = deepcopy(logger_cfg)
    tracker_name = tracker_cfg.pop("backend", "wandb")
    project_dir = cfg["paths"]["output_dir"]

    return True, tracker_name, {tracker_name: tracker_cfg}, project_dir


def sanitize_tracker_config(
    cfg: dict[str, Any], tracker_name: str | None
) -> dict[str, Any] | None:
    """Return a tracker-compatible config payload for experiment initialization."""

    if tracker_name is None:
        return None

    if tracker_name != "tensorboard":
        return cfg

    flat_cfg: dict[str, Any] = {}
    _flatten_for_tensorboard(cfg, flat_cfg)
    return flat_cfg


def _flatten_for_tensorboard(value: Any, out: dict[str, Any], prefix: str = "") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_for_tensorboard(item, out, child_prefix)
        return

    if isinstance(value, list):
        out[prefix] = json.dumps(value, ensure_ascii=True, sort_keys=True)
        return

    if value is None or isinstance(value, (bool, int, float, str)):
        out[prefix] = value
        return

    out[prefix] = str(value)
