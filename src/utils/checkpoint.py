import json
from pathlib import Path

import torch
import torch.nn as nn

def save_checkpoint(
    model: nn.Module,
    state_dict: dict,
    result_meta: dict,
    path: Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict, "result": result_meta}, path)

def load_checkpoint(model: nn.Module, path: Path, device="cpu") -> dict:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    return ckpt.get("result", {})

def save_log(log_data: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(log_data, f, indent=2, default=str)