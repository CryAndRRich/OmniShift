import torch.nn as nn

def make_factories(n_bits=3, alpha_init=1.0, include_zero=True, **opts):
    from src.quantize.apot import APoTConv2d

    def make_conv(in_c, out_c, k=3, s=1, p=1, bias=False):
        return APoTConv2d(
            in_c, out_c, kernel_size=k, stride=s, padding=p, bias=bias,
            n_bits=n_bits, alpha_init=alpha_init, include_zero=include_zero)

    def make_bn(c):
        return nn.BatchNorm2d(c)

    def make_act():
        return nn.Identity()

    return make_conv, make_bn, make_act