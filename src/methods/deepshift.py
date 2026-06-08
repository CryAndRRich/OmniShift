import torch.nn as nn

def make_factories(**opts):
    from src.quantize.shift import ShiftConv2d

    def make_conv(in_c, out_c, k=3, s=1, p=1, bias=False):
        return ShiftConv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=bias)

    def make_bn(c):
        return nn.BatchNorm2d(c)

    def make_act():
        return nn.Identity()

    return make_conv, make_bn, make_act