"""Unified ops counter for all OmniShift model variants.

Energy model (45nm CMOS):
  mul   = 3.7 pJ
  add   = 0.9 pJ
  shift = 0.13 pJ  (~28× cheaper than mul)

BN ops (folded form: y = scale * x + bias):
  std BN  → 1 mul + 1 add per activation element
  PoT-BN  → 1 shift + 1 add per activation element

Sparse conv (skip-zero hardware):
  shift+add counts are scaled by (1 - sparsity).
"""

from typing import Optional
import torch
import torch.nn as nn

# Energy costs (pJ, 45nm CMOS)
_MUL_PJ   = 3.7
_ADD_PJ   = 0.9
_SHIFT_PJ = 0.13


def _conv_macs(c_in, c_out, kh, kw, h_out, w_out) -> int:
    return c_in * c_out * kh * kw * h_out * w_out


def count_mul_add_shift(
    model: nn.Module,
    input_size: tuple = (1, 3, 32, 32),
    sparsity: Optional[float] = None,
) -> dict:
    """Count mul / add / shift ops and compute energy for any OmniShift model.

    Detects model type from layer attributes and module classes.
    Returns dict with keys: mul, add, shift (absolute counts),
    mul_G, add_G, shift_G (GigaOps), energy_pJ, energy_GpJ, sparsity.
    """
    from src.quantize.sparse_shift import SparseShiftConv2d
    from src.quantize.shift import ShiftConv2d
    from src.quantize.pot_bn import PoTBatchNorm2d

    # EWGS modules (same forward ops as their Phase 4 counterparts)
    try:
        from src.quantize.ewgs import SparseShiftConv2dEWGS, PoTBatchNorm2dEWGS
        _sparse_types = (SparseShiftConv2d, SparseShiftConv2dEWGS)
        _pot_bn_types = (PoTBatchNorm2d, PoTBatchNorm2dEWGS)
    except ImportError:
        _sparse_types = (SparseShiftConv2d,)
        _pot_bn_types = (PoTBatchNorm2d,)

    # PoT activation (Phase 6)
    try:
        from src.quantize.pot_act import PoTActivation
        has_pot_act = any(isinstance(m, PoTActivation) for m in model.modules())
    except ImportError:
        has_pot_act = False

    # Detect conv and BN types from first interior block
    sample_conv = model.stage1[0].conv1
    sample_bn = model.stage1[0].bn1

    is_sparse = isinstance(sample_conv, _sparse_types)
    is_shift  = isinstance(sample_conv, ShiftConv2d)
    is_pot_bn = isinstance(sample_bn, _pot_bn_types)

    # APoT / DenseShift only exist in Phase 1 models (src.models.resnet20)
    try:
        from src.models.resnet20 import APoTConv2d as _APoT, DenseShiftConv2d as _Dense
        is_apot  = isinstance(sample_conv, _APoT)
        is_dense = isinstance(sample_conv, _Dense)
    except Exception:
        is_apot = is_dense = False

    # Resolve sparsity
    if is_sparse:
        if sparsity is None:
            if hasattr(model, 'sparse_mode') and model.sparse_mode == "fixed":
                sparsity = model.sparsity_ratio
            else:
                sparsity = model.get_global_sparsity()
        nonzero_ratio = 1.0 - sparsity
    else:
        sparsity = 0.0
        nonzero_ratio = 1.0

    counts: dict = {"mul": 0, "add": 0, "shift": 0}

    def add_bn_ops(planes, h_o, w_o):
        n = planes * h_o * w_o
        if is_pot_bn:
            counts["shift"] += n
        else:
            counts["mul"] += n
        counts["add"] += n

    def add_interior_conv_ops(in_c, out_c, kh, kw, h_o, w_o):
        """All interior convs (stage 1-3, including shortcuts)."""
        m = _conv_macs(in_c, out_c, kh, kw, h_o, w_o)
        if is_sparse:
            counts["shift"] += int(m * nonzero_ratio)
            counts["add"]   += int(m * nonzero_ratio)
        elif is_shift:
            counts["shift"] += m
            counts["add"]   += m
        elif is_apot:
            K = getattr(sample_conv, 'K', 2)
            counts["shift"] += K * m
            counts["add"]   += m + (K - 1) * m
        elif is_dense:
            counts["shift"] += m
            counts["add"]   += m
        else:  # mul (vanilla Conv2d)
            counts["mul"] += m
            counts["add"] += m

    h, w = input_size[2], input_size[3]
    H, W = h, w

    # First conv: always nn.Conv2d (mul)
    m = _conv_macs(input_size[1], 16, 3, 3, H, W)
    counts["mul"] += m
    counts["add"] += m
    add_bn_ops(16, H, W)

    in_planes = 16
    for planes, num_blocks, stride in [(16, 3, 1), (32, 3, 2), (64, 3, 2)]:
        strides = [stride] + [1] * (num_blocks - 1)
        for s in strides:
            H_out = H // s
            W_out = W // s

            add_interior_conv_ops(in_planes, planes, 3, 3, H_out, W_out)
            add_bn_ops(planes, H_out, W_out)
            add_interior_conv_ops(planes, planes, 3, 3, H_out, W_out)
            add_bn_ops(planes, H_out, W_out)

            if s != 1 or in_planes != planes:
                add_interior_conv_ops(in_planes, planes, 1, 1, H_out, W_out)
                add_bn_ops(planes, H_out, W_out)

            in_planes = planes
            H, W = H_out, W_out

    counts["add"] += in_planes * H * W  # AvgPool

    # FC: always mul
    num_classes = model.fc.out_features
    counts["mul"] += in_planes * num_classes
    counts["add"] += in_planes * num_classes

    # PoT activation quantization overhead (Phase 6):
    # Each PoTActivation element costs ~1 shift (log2-round to PoT grid).
    if has_pot_act:
        h_act, w_act = input_size[2], input_size[3]
        counts["shift"] += 16 * h_act * w_act   # stem act
        ah, aw = h_act, w_act
        for planes_a, n_blocks_a, stride_a in [(16, 3, 1), (32, 3, 2), (64, 3, 2)]:
            strides_a = [stride_a] + [1] * (n_blocks_a - 1)
            for s_a in strides_a:
                ah, aw = ah // s_a, aw // s_a
                counts["shift"] += planes_a * ah * aw  # act1 (after conv1 relu)
                counts["shift"] += planes_a * ah * aw  # act2 (block output relu)

    energy_pj = (_MUL_PJ * counts["mul"]
                 + _ADD_PJ * counts["add"]
                 + _SHIFT_PJ * counts["shift"])

    return {
        "mul":        counts["mul"],
        "add":        counts["add"],
        "shift":      counts["shift"],
        "mul_G":      counts["mul"]   / 1e9,
        "add_G":      counts["add"]   / 1e9,
        "shift_G":    counts["shift"] / 1e9,
        "energy_pJ":  energy_pj,
        "energy_GpJ": energy_pj / 1e9,
        "sparsity":   sparsity,
    }


def count_params(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
