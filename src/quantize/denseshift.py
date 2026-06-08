import torch
import torch.nn as nn
import torch.nn.functional as F

class _SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s):
        out = s.sign()
        return torch.where(out == 0, torch.ones_like(out), out)

    @staticmethod
    def backward(ctx, grad):
        return grad

class _ShiftSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, p_min, p_max):
        return p.round().clamp(p_min, p_max)

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None

class DenseShiftConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=False, p_min=-7, p_max=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size if isinstance(kernel_size, tuple)
                            else (kernel_size, kernel_size))
        self.stride = stride
        self.padding = padding
        self.p_min = p_min
        self.p_max = p_max

        shape = (out_channels, in_channels, *self.kernel_size)

        self.sign_param = nn.Parameter(torch.empty(shape))
        self.exp_param = nn.Parameter(torch.zeros(shape))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

        nn.init.normal_(self.sign_param, mean=0.0, std=1e-3)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        s_q = _SignSTE.apply(self.sign_param)
        p_q = _ShiftSTE.apply(self.exp_param, self.p_min, self.p_max)
        w_q = s_q * (2.0 ** p_q)
        return F.conv2d(x, w_q, self.bias, stride=self.stride, padding=self.padding)

    def extra_repr(self):
        return (f"{self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, "
                f"p=[{self.p_min},{self.p_max}]")