def make_factories(sparse_mode='learnable', sparsity_ratio=0.5, bn_warmup=30,
                   act_warmup=None, act_levels=8, act_alpha_init=4.0,
                   ewgs_lambda=0.02, mask_hysteresis=0.1, exp_hysteresis=0.1,
                   mask_freeze_epoch=160, freeze_after_epoch=100,
                   flip_freeze_th=0.02, **opts):
    from src.quantize.ewgs import (SparseShiftConv2dEWGS,
                                    PoTBatchNorm2dEWGS,
                                    PoTActivationEWGS)

    # Staggered quantization: activating PoT-BN and PoT-Act in the same epoch
    # (while LR is still high) caused an 80% -> 28% val-acc crash on CIFAR-10.
    if act_warmup is None:
        act_warmup = bn_warmup + 30

    def make_conv(in_c, out_c, k=3, s=1, p=1, bias=False):
        return SparseShiftConv2dEWGS(
            in_c, out_c, kernel_size=k, stride=s, padding=p, bias=bias,
            sparse_mode=sparse_mode, sparsity_ratio=sparsity_ratio,
            ewgs_lambda=ewgs_lambda,
            mask_hysteresis=mask_hysteresis,
            mask_freeze_epoch=mask_freeze_epoch,
            freeze_after_epoch=freeze_after_epoch,
            flip_freeze_th=flip_freeze_th)

    def make_bn(c):
        return PoTBatchNorm2dEWGS(c, use_pot_after_epoch=bn_warmup,
                                   ewgs_lambda=ewgs_lambda,
                                   exp_hysteresis=exp_hysteresis)

    def make_act():
        return PoTActivationEWGS(n_levels=act_levels, alpha_init=act_alpha_init,
                                  use_pot_after_epoch=act_warmup,
                                  ewgs_lambda=ewgs_lambda,
                                  exp_hysteresis=exp_hysteresis)

    return make_conv, make_bn, make_act
