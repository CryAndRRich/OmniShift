"""Sparsity regularization: L1 penalty on conv weights for learnable sparse mode."""

import torch
import torch.nn as nn

from src.quantize.sparse_shift import SparseShiftConv2d


def compute_sparsity_reg(model: nn.Module, lambda_l1: float = 0.0) -> torch.Tensor:
    """L1 penalty on weights of learnable SparseShiftConv2d layers.

    Encourages weight magnitudes to fall below the learned threshold,
    increasing sparsity without fixing the ratio explicitly.
    """
    if lambda_l1 <= 0:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device)

    reg = torch.tensor(0.0, device=next(model.parameters()).device)
    for m in model.modules():
        if isinstance(m, SparseShiftConv2d) and m.sparse_mode == "learnable":
            reg = reg + m.weight.abs().mean()
    return lambda_l1 * reg
