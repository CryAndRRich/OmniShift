"""Sparsity regularization: L1 penalty on conv weights for learnable sparse mode."""

import torch
import torch.nn as nn

from src.quantize.sparse_shift import SparseShiftConv2d
from src.quantize.ewgs import SparseShiftConv2dEWGS

_SPARSE_TYPES = (SparseShiftConv2d, SparseShiftConv2dEWGS)


def compute_sparsity_reg(model: nn.Module, lambda_l1: float = 0.0) -> torch.Tensor:
    """L1 penalty on weights of learnable sparse shift conv layers."""
    if lambda_l1 <= 0:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device)

    reg = torch.tensor(0.0, device=next(model.parameters()).device)
    for m in model.modules():
        if isinstance(m, _SPARSE_TYPES) and m.sparse_mode == "learnable":
            reg = reg + m.weight.abs().mean()
    return lambda_l1 * reg
