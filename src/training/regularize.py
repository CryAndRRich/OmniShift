import torch
import torch.nn as nn

from src.quantize.sparse_shift import SparseShiftConv2d
from src.quantize.ewgs import SparseShiftConv2dEWGS
from src.quantize.s3shift import S3ShiftConv2d
from src.quantize.aptq_ternary import APTQTernaryConv2d

_SPARSE_TYPES = (SparseShiftConv2d, SparseShiftConv2dEWGS, S3ShiftConv2d,
                 APTQTernaryConv2d)

def compute_sparsity_reg(model: nn.Module, lambda_l1: float = 0.0) -> torch.Tensor:
    if lambda_l1 <= 0:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device)

    reg = torch.tensor(0.0, device=next(model.parameters()).device)
    for m in model.modules():
        if not isinstance(m, _SPARSE_TYPES) or m.sparse_mode != "learnable":
            continue
        if isinstance(m, S3ShiftConv2d):
            reg = reg + (2.0 ** m.exp_param).abs().mean()
        elif isinstance(m, APTQTernaryConv2d):
            reg = reg + m.w_pos.abs().mean()
            reg = reg + m.w_neg.abs().mean()
        else:
            reg = reg + m.weight.abs().mean()
    return lambda_l1 * reg