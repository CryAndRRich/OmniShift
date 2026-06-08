import torch.nn as nn

def make_factories(sparse_mode="learnable", sparsity_ratio=0.5,
                   p_min=-8, p_max=1, init_threshold=0.05, **opts):
    from src.quantize.s3shift import S3ShiftConv2d

    def make_conv(in_c, out_c, k=3, s=1, p=1, bias=False):
        return S3ShiftConv2d(
            in_c, out_c, kernel_size=k, stride=s, padding=p, bias=bias,
            sparse_mode=sparse_mode, p_min=p_min, p_max=p_max,
            sparsity_ratio=sparsity_ratio, init_threshold=init_threshold)

    def make_bn(c):
        return nn.BatchNorm2d(c)

    def make_act():
        return nn.Identity()

    return make_conv, make_bn, make_act