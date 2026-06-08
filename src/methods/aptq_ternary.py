import torch.nn as nn

def make_factories(sparse_mode: str = "learnable",
                   sparsity_ratio: float = 0.5,
                   p_min: int = -8,
                   p_max: int = 0,
                   init_threshold: float = 0.05,
                   **opts):
    from src.quantize.aptq_ternary import APTQTernaryConv2d

    def make_conv(in_c: int, out_c: int, k: int = 3,
                  s: int = 1, p: int = 1, bias: bool = False):
        return APTQTernaryConv2d(
            in_c, out_c,
            kernel_size=k, stride=s, padding=p, bias=bias,
            sparse_mode=sparse_mode,
            sparsity_ratio=sparsity_ratio,
            p_min=p_min,
            p_max=p_max,
            init_threshold=init_threshold,
        )

    def make_bn(c: int):
        return nn.BatchNorm2d(c)

    def make_act():
        return nn.Identity()

    return make_conv, make_bn, make_act