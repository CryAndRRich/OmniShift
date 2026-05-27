"""PoT-BatchNorm: batch normalization whose scale is quantized to power-of-two.

Forward (folded form): y = round_to_PoT(γ/σ) * (x - μ) + β

At inference, the scale multiplication becomes a single bit shift, eliminating
multipliers from the BN layer entirely.
"""

import torch
import torch.nn as nn


class ScaleToPoT(torch.autograd.Function):
    """Round BN scale (γ/σ) to nearest signed PoT, STE backward.

    p is clamped to [-15, 15] to avoid extreme shifts on real hardware.
    """

    @staticmethod
    def forward(ctx, scale):
        sign = torch.sign(scale)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        abs_s = scale.abs().clamp(min=1e-8)
        log2_s = torch.log2(abs_s)
        p = torch.round(log2_s)
        p = torch.clamp(p, min=-15, max=15)
        return sign * (2.0 ** p)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class PoTBatchNorm2d(nn.Module):
    """BatchNorm with scale γ/σ quantized to nearest power-of-two.

    Args:
        num_features: number of channels C.
        momentum: running stats EMA momentum.
        eps: numerical stability for std.
        use_pot_after_epoch: epoch from which PoT quantization activates.
            0 = PoT from start; N > 0 = N epochs of standard BN warmup.
    """

    def __init__(self, num_features, momentum=0.1, eps=1e-5,
                 use_pot_after_epoch=0):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.use_pot_after_epoch = use_pot_after_epoch
        self.current_epoch = 0

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def _should_use_pot(self):
        return self.current_epoch >= self.use_pot_after_epoch

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(
                    self.momentum * mean.detach())
                self.running_var.mul_(1 - self.momentum).add_(
                    self.momentum * var.detach())
        else:
            mean = self.running_mean
            var = self.running_var

        std = torch.sqrt(var + self.eps)
        scale_fp = self.weight / std

        if self._should_use_pot():
            scale = ScaleToPoT.apply(scale_fp)
        else:
            scale = scale_fp

        return (scale.view(1, -1, 1, 1) * (x - mean.view(1, -1, 1, 1))
                + self.bias.view(1, -1, 1, 1))

    def extra_repr(self):
        return (f'{self.num_features}, momentum={self.momentum}, '
                f'eps={self.eps}, use_pot_after_epoch={self.use_pot_after_epoch}')


def set_bn_epoch(model: nn.Module, epoch: int) -> None:
    """Set current_epoch on all epoch-gated quantization modules.

    Handles PoTBatchNorm2d and any other module with a current_epoch attribute
    (e.g., PoTActivation), so a single call covers all warmup-controlled layers.
    """
    for m in model.modules():
        if hasattr(m, 'current_epoch'):
            m.current_epoch = epoch
