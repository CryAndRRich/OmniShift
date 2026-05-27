"""Checkpoint save/load helpers."""

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    state_dict: dict,
    result_meta: dict,
    path: Path,
) -> None:
    """Save best model state_dict + result metadata to a .pt file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict, "result": result_meta}, path)


def load_checkpoint(model: nn.Module, path: Path, device="cpu") -> dict:
    """Load state_dict from checkpoint into model. Returns result metadata."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    return ckpt.get("result", {})


def save_log(log_data: dict, path: Path) -> None:
    """Save training log as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(log_data, f, indent=2, default=str)
