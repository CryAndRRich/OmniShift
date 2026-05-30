"""Factory functions that produce (make_conv, make_bn, make_act) triples.

Given a quantization config dict, make_factories() returns three callables
that a backbone model uses to instantiate its conv, BN, and activation layers.
This decouples backbone architecture from quantization choice.

Quantization config keys (all optional, defaults shown):
    use_sparse    : bool  = True   — sparse shift weights W ∈ {0, ±2^p}
    sparse_mode   : str   = "learnable"  — "fixed" | "learnable"
    sparsity_ratio: float = 0.5    — target sparsity (fixed mode only)
    use_pot_bn    : bool  = True   — PoT-quantize BN scale γ/σ → ±2^q
    bn_warmup     : int   = 30     — epoch at which PoT-BN activates
    use_pot_act   : bool  = True   — PoT-quantize post-ReLU activations
    act_levels    : int   = 8      — number of non-zero PoT levels
    act_alpha_init: float = 4.0    — initial clip value for PoTActivation
    use_ewgs      : bool  = True   — EWGS backward instead of STE
    ewgs_lambda   : float = 0.02   — EWGS scaling factor λ
"""

import torch.nn as nn


def make_factories(quant_cfg: dict):
    """Return (make_conv, make_bn, make_act) factory callables.

    make_conv(in_c, out_c, k, stride, padding, bias=False) -> nn.Module
    make_bn(num_features) -> nn.Module
    make_act() -> nn.Module
    """
    use_sparse     = quant_cfg.get('use_sparse',     True)
    sparse_mode    = quant_cfg.get('sparse_mode',    'learnable')
    sparsity_ratio = quant_cfg.get('sparsity_ratio', 0.5)
    use_pot_bn     = quant_cfg.get('use_pot_bn',     True)
    bn_warmup      = quant_cfg.get('bn_warmup',      30)
    use_pot_act    = quant_cfg.get('use_pot_act',    True)
    act_levels     = quant_cfg.get('act_levels',     8)
    act_alpha_init = quant_cfg.get('act_alpha_init', 4.0)
    use_ewgs       = quant_cfg.get('use_ewgs',       True)
    ewgs_lambda    = quant_cfg.get('ewgs_lambda',    0.02)

    # ── Conv factory ─────────────────────────────────────────────────────────
    if use_sparse and use_ewgs:
        from src.quantize.ewgs import SparseShiftConv2dEWGS
        def make_conv(in_c, out_c, k=3, s=1, p=1, bias=False):
            return SparseShiftConv2dEWGS(
                in_c, out_c, kernel_size=k, stride=s, padding=p, bias=bias,
                sparse_mode=sparse_mode, sparsity_ratio=sparsity_ratio,
                ewgs_lambda=ewgs_lambda)

    elif use_sparse:
        from src.quantize.sparse_shift import SparseShiftConv2d
        def make_conv(in_c, out_c, k=3, s=1, p=1, bias=False):
            return SparseShiftConv2d(
                in_c, out_c, kernel_size=k, stride=s, padding=p, bias=bias,
                sparse_mode=sparse_mode, sparsity_ratio=sparsity_ratio)

    else:
        def make_conv(in_c, out_c, k=3, s=1, p=1, bias=False):
            return nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=bias)

    # ── BN factory ───────────────────────────────────────────────────────────
    if use_pot_bn and use_ewgs:
        from src.quantize.ewgs import PoTBatchNorm2dEWGS
        def make_bn(c):
            return PoTBatchNorm2dEWGS(c, use_pot_after_epoch=bn_warmup,
                                      ewgs_lambda=ewgs_lambda)

    elif use_pot_bn:
        from src.quantize.pot_bn import PoTBatchNorm2d
        def make_bn(c):
            return PoTBatchNorm2d(c, use_pot_after_epoch=bn_warmup)

    else:
        def make_bn(c):
            return nn.BatchNorm2d(c)

    # ── Activation factory ───────────────────────────────────────────────────
    if use_pot_act and use_ewgs:
        from src.quantize.ewgs import PoTActivationEWGS
        def make_act():
            return PoTActivationEWGS(n_levels=act_levels, alpha_init=act_alpha_init,
                                     use_pot_after_epoch=bn_warmup,
                                     ewgs_lambda=ewgs_lambda)

    elif use_pot_act:
        from src.quantize.pot_act import PoTActivation
        def make_act():
            return PoTActivation(n_levels=act_levels, alpha_init=act_alpha_init,
                                 use_pot_after_epoch=bn_warmup)

    else:
        def make_act():
            return nn.Identity()

    return make_conv, make_bn, make_act
