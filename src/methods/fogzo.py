import torch.nn as nn

def make_factories(n_perturbations: int = 4,
                   sigma: float = 0.01,
                   fogzo_lambda: float = 0.1,
                   grad_clip: float = 10.0,
                   p_min: int = -15,
                   p_max: int = 0,
                   **opts):
    from src.quantize.fogzo import FogzoShiftConv2d

    def make_conv(in_c, out_c, k=3, s=1, p=1, bias=False):
        return FogzoShiftConv2d(
            in_c, out_c,
            kernel_size=k, stride=s, padding=p, bias=bias,
            n_perturbations=n_perturbations,
            sigma=sigma,
            fogzo_lambda=fogzo_lambda,
            grad_clip=grad_clip,
            p_min=p_min,
            p_max=p_max,
        )

    def make_bn(c):
        return nn.BatchNorm2d(c)

    def make_act():
        return nn.Identity()

    return make_conv, make_bn, make_act