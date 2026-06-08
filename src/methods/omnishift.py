import torch.nn as nn

def make_factories(sparse_mode='learnable', sparsity_ratio=0.5, bn_warmup=30,
                   act_levels=8, act_alpha_init=4.0, ewgs_lambda=0.02, **opts):
    from src.quantize.ewgs import (SparseShiftConv2dEWGS,
                                    PoTBatchNorm2dEWGS,
                                    PoTActivationEWGS)

    def make_conv(in_c, out_c, k=3, s=1, p=1, bias=False):
        return SparseShiftConv2dEWGS(
            in_c, out_c, kernel_size=k, stride=s, padding=p, bias=bias,
            sparse_mode=sparse_mode, sparsity_ratio=sparsity_ratio,
            ewgs_lambda=ewgs_lambda)

    def make_bn(c):
        return PoTBatchNorm2dEWGS(c, use_pot_after_epoch=bn_warmup,
                                   ewgs_lambda=ewgs_lambda)

    def make_act():
        return PoTActivationEWGS(n_levels=act_levels, alpha_init=act_alpha_init,
                                  use_pot_after_epoch=bn_warmup,
                                  ewgs_lambda=ewgs_lambda)

    return make_conv, make_bn, make_act