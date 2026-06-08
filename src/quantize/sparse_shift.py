import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class FixedSparseShiftQuantize(autograd.Function):
    @staticmethod
    def forward(ctx, w, sparsity_ratio):
        abs_w = w.abs()
        N = w.numel()
        k = int(sparsity_ratio * N)

        if k > 0 and k < N:
            threshold = abs_w.flatten().kthvalue(k).values
            mask = (abs_w > threshold).float()
        elif k == 0:
            mask = torch.ones_like(w)
        else:
            mask = torch.zeros_like(w)

        sign = torch.sign(w)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        abs_w_clamp = abs_w.clamp(min=1e-8)
        p = torch.round(torch.log2(abs_w_clamp))
        w_pot = sign * (2.0 ** p)

        return mask * w_pot

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class LearnableSparseShiftQuantize(autograd.Function):
    @staticmethod
    def forward(ctx, w, threshold):
        abs_w = w.abs()
        mask = (abs_w > threshold).float()

        sign = torch.sign(w)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        abs_w_clamp = abs_w.clamp(min=1e-8)
        p = torch.round(torch.log2(abs_w_clamp))
        w_pot = sign * (2.0 ** p)

        return mask * w_pot

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class SparseShiftConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=False, sparse_mode="fixed",
                 sparsity_ratio=0.5, init_threshold=0.05):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size if isinstance(kernel_size, tuple)
                            else (kernel_size, kernel_size))
        self.stride = stride
        self.padding = padding
        self.sparse_mode = sparse_mode
        self.sparsity_ratio = sparsity_ratio

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

        if sparse_mode == "learnable":
            self.register_buffer("log_threshold", torch.tensor(math.log(init_threshold)))

        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.sparse_mode == "fixed":
            w_q = FixedSparseShiftQuantize.apply(self.weight, self.sparsity_ratio)
        elif self.sparse_mode == "learnable":
            threshold = self.log_threshold.exp()
            w_q = LearnableSparseShiftQuantize.apply(self.weight, threshold)
        else:
            raise ValueError(f"Unknown sparse_mode: {self.sparse_mode}")

        return F.conv2d(x, w_q, self.bias,
                        stride=self.stride, padding=self.padding)

    @torch.no_grad()
    def get_actual_sparsity(self):
        if self.sparse_mode == "fixed":
            w_q = FixedSparseShiftQuantize.apply(self.weight, self.sparsity_ratio)
        else:
            threshold = self.log_threshold.exp()
            w_q = LearnableSparseShiftQuantize.apply(self.weight, threshold)
        return (w_q == 0).float().mean().item()

    def extra_repr(self):
        s = (f"{self.in_channels}, {self.out_channels}, "
             f"kernel_size={self.kernel_size}, stride={self.stride}, "
             f"padding={self.padding}, bias={self.bias is not None}, "
             f"sparse_mode={self.sparse_mode}")
        if self.sparse_mode == "fixed":
            s += f", sparsity_ratio={self.sparsity_ratio}"
        return s