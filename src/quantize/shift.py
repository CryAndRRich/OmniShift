"""DeepShift-PS: power-of-two weight quantization with STE.

Weights are constrained to ±2^p (signed power-of-two). At inference, each
multiplication is replaced by a single bit-shift operation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RoundToPoT(torch.autograd.Function):
    """Round weight to nearest signed power-of-two: ±2^p, p ∈ Z. STE backward."""

    @staticmethod
    def forward(ctx, w):
        sign = torch.sign(w)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        abs_w = w.abs().clamp(min=1e-8)
        log2_w = torch.log2(abs_w)
        p = torch.round(log2_w)
        return sign * (2.0 ** p)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class ShiftConv2d(nn.Module):
    """DeepShift-PS: weights constrained to ±2^p (signed power-of-two).

    Inference: multiplication replaced by bit shift.
    Training: full-precision latent weight, quantize to PoT in forward (STE).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size if isinstance(kernel_size, tuple)
                            else (kernel_size, kernel_size))
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        w_pot = RoundToPoT.apply(self.weight)
        return F.conv2d(x, w_pot, self.bias,
                        stride=self.stride, padding=self.padding)

    def extra_repr(self):
        return (f'{self.in_channels}, {self.out_channels}, '
                f'kernel_size={self.kernel_size}, stride={self.stride}, '
                f'padding={self.padding}, bias={self.bias is not None}')
