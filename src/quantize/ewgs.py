"""EWGS (Element-Wise Gradient Scaling) quantizers for Phase 5.

Replaces STE backward with:
  g = g_STE ⊙ (1 + λ · sign(g_STE) ⊙ sign(w − Q(w)))

λ=0.02 improves gradient flow near quantization boundaries without
changing the forward pass — Phase 4 and Phase 5 have identical inference ops.

Reference: Lee et al., "Network Quantization with Element-wise Gradient
Scaling", CVPR 2021. https://arxiv.org/abs/2104.00903
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.quantize.pot_bn import PoTBatchNorm2d

EWGS_LAMBDA = 0.02


# ---------------------------------------------------------------------------
# Autograd functions
# ---------------------------------------------------------------------------

class RoundToPoTEWGS(torch.autograd.Function):
    """Round weight to nearest signed PoT with EWGS backward."""

    @staticmethod
    def forward(ctx, w, ewgs_lambda):
        sign = torch.sign(w)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        abs_w = w.abs().clamp(min=1e-8)
        p = torch.round(torch.log2(abs_w))
        w_pot = sign * (2.0 ** p)
        ctx.save_for_backward(w, w_pot)
        ctx.ewgs_lambda = ewgs_lambda
        return w_pot

    @staticmethod
    def backward(ctx, grad_output):
        w, w_pot = ctx.saved_tensors
        lam = ctx.ewgs_lambda
        grad = grad_output * (1.0 + lam * grad_output.sign() * (w - w_pot).sign())
        return grad, None


class ScaleToPoTEWGS(torch.autograd.Function):
    """Round BN scale to nearest signed PoT (p clamped ±15) with EWGS backward."""

    @staticmethod
    def forward(ctx, scale, ewgs_lambda):
        sign = torch.sign(scale)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        abs_s = scale.abs().clamp(min=1e-8)
        p = torch.round(torch.log2(abs_s))
        p = torch.clamp(p, min=-15, max=15)
        s_pot = sign * (2.0 ** p)
        ctx.save_for_backward(scale, s_pot)
        ctx.ewgs_lambda = ewgs_lambda
        return s_pot

    @staticmethod
    def backward(ctx, grad_output):
        scale, s_pot = ctx.saved_tensors
        lam = ctx.ewgs_lambda
        grad = grad_output * (1.0 + lam * grad_output.sign() * (scale - s_pot).sign())
        return grad, None


class FixedSparseShiftEWGS(torch.autograd.Function):
    """W ∈ {0, ±2^p} with fixed sparsity ratio — EWGS backward."""

    @staticmethod
    def forward(ctx, w, sparsity_ratio, ewgs_lambda):
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
        p = torch.round(torch.log2(abs_w.clamp(min=1e-8)))
        w_pot = sign * (2.0 ** p)
        w_q = mask * w_pot

        ctx.save_for_backward(w, w_q)
        ctx.ewgs_lambda = ewgs_lambda
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        w, w_q = ctx.saved_tensors
        lam = ctx.ewgs_lambda
        grad = grad_output * (1.0 + lam * grad_output.sign() * (w - w_q).sign())
        return grad, None, None


class LearnableSparseShiftEWGS(torch.autograd.Function):
    """W ∈ {0, ±2^p} with learnable threshold — EWGS backward."""

    @staticmethod
    def forward(ctx, w, threshold, ewgs_lambda):
        abs_w = w.abs()
        mask = (abs_w > threshold).float()

        sign = torch.sign(w)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        p = torch.round(torch.log2(abs_w.clamp(min=1e-8)))
        w_pot = sign * (2.0 ** p)
        w_q = mask * w_pot

        ctx.save_for_backward(w, w_q)
        ctx.ewgs_lambda = ewgs_lambda
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        w, w_q = ctx.saved_tensors
        lam = ctx.ewgs_lambda
        grad = grad_output * (1.0 + lam * grad_output.sign() * (w - w_q).sign())
        return grad, None, None


# ---------------------------------------------------------------------------
# Module wrappers
# ---------------------------------------------------------------------------

class SparseShiftConv2dEWGS(nn.Module):
    """SparseShiftConv2d with EWGS backward — identical forward to SparseShiftConv2d.

    Drop-in replacement: forward pass and inference energy are unchanged from
    Phase 4; only the gradient estimator is replaced (STE → EWGS).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=False, sparse_mode="fixed",
                 sparsity_ratio=0.5, init_threshold=0.05,
                 ewgs_lambda: float = EWGS_LAMBDA):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size if isinstance(kernel_size, tuple)
                            else (kernel_size, kernel_size))
        self.stride = stride
        self.padding = padding
        self.sparse_mode = sparse_mode
        self.sparsity_ratio = sparsity_ratio
        self.ewgs_lambda = ewgs_lambda

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

        if sparse_mode == "learnable":
            self.log_threshold = nn.Parameter(
                torch.tensor(math.log(init_threshold)))

        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.sparse_mode == "fixed":
            w_q = FixedSparseShiftEWGS.apply(
                self.weight, self.sparsity_ratio, self.ewgs_lambda)
        elif self.sparse_mode == "learnable":
            threshold = self.log_threshold.exp()
            w_q = LearnableSparseShiftEWGS.apply(
                self.weight, threshold, self.ewgs_lambda)
        else:
            raise ValueError(f"Unknown sparse_mode: {self.sparse_mode!r}")

        return F.conv2d(x, w_q, self.bias,
                        stride=self.stride, padding=self.padding)

    @torch.no_grad()
    def get_actual_sparsity(self) -> float:
        if self.sparse_mode == "fixed":
            w_q = FixedSparseShiftEWGS.apply(
                self.weight, self.sparsity_ratio, self.ewgs_lambda)
        else:
            threshold = self.log_threshold.exp()
            w_q = LearnableSparseShiftEWGS.apply(
                self.weight, threshold, self.ewgs_lambda)
        return (w_q == 0).float().mean().item()

    def extra_repr(self):
        s = (f'{self.in_channels}, {self.out_channels}, '
             f'kernel_size={self.kernel_size}, stride={self.stride}, '
             f'padding={self.padding}, sparse_mode={self.sparse_mode}, '
             f'ewgs_lambda={self.ewgs_lambda}')
        if self.sparse_mode == "fixed":
            s += f', sparsity_ratio={self.sparsity_ratio}'
        return s


class PoTBatchNorm2dEWGS(PoTBatchNorm2d):
    """PoTBatchNorm2d with EWGS backward for the ScaleToPoT quantization step."""

    def __init__(self, num_features, momentum=0.1, eps=1e-5,
                 use_pot_after_epoch=0, ewgs_lambda: float = EWGS_LAMBDA):
        super().__init__(num_features, momentum=momentum, eps=eps,
                         use_pot_after_epoch=use_pot_after_epoch)
        self.ewgs_lambda = ewgs_lambda

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
            scale = ScaleToPoTEWGS.apply(scale_fp, self.ewgs_lambda)
        else:
            scale = scale_fp

        return (scale.view(1, -1, 1, 1) * (x - mean.view(1, -1, 1, 1))
                + self.bias.view(1, -1, 1, 1))
