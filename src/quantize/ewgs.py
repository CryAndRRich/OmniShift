import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from src.quantize.pot_bn import PoTBatchNorm2d

EWGS_LAMBDA = 0.02

class RoundToPoTEWGS(autograd.Function):
    @staticmethod
    def forward(ctx, w, ewgs_lambda):
        sign = torch.sign(w)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        abs_w = w.abs().clamp(min=1e-8)
        p = torch.round(torch.log2(abs_w))
        p = p.clamp(-15, 0)
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

class ScaleToPoTEWGS(autograd.Function):
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

class FixedSparseShiftEWGS(autograd.Function):
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
        p = p.clamp(-15, 0)
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

class LearnableSparseShiftEWGS(autograd.Function):
    @staticmethod
    def forward(ctx, w, threshold, ewgs_lambda):
        abs_w = w.abs()
        mask = (abs_w > threshold).float()

        sign = torch.sign(w)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        p = torch.round(torch.log2(abs_w.clamp(min=1e-8)))
        p = p.clamp(-15, 0)
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

class SparseShiftConv2dEWGS(nn.Module):
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

            self.register_buffer('log_threshold',
                                  torch.tensor(math.log(init_threshold)))

        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
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
        s = (f"{self.in_channels}, {self.out_channels}, "
             f"kernel_size={self.kernel_size}, stride={self.stride}, "
             f"padding={self.padding}, sparse_mode={self.sparse_mode}, "
             f"ewgs_lambda={self.ewgs_lambda}")
        if self.sparse_mode == "fixed":
            s += f", sparsity_ratio={self.sparsity_ratio}"
        return s

class RoundActivToPoTEWGS(autograd.Function):
    @staticmethod
    def forward(ctx, x, log_alpha, n_levels, ewgs_lambda):
        alpha = log_alpha.exp()
        x_clip = x.clamp(0.0, alpha.item())
        eps = 1e-8
        active = x_clip > eps
        p_max = torch.floor(torch.log2(alpha.clamp(min=eps)))
        p_min = p_max - (n_levels - 1)
        p = torch.round(torch.log2(x_clip.clamp(min=eps)))
        p = p.clamp(p_min.item(), p_max.item())
        x_q = torch.where(active, 2.0 ** p, torch.zeros_like(x_clip))
        ctx.save_for_backward(x, log_alpha, x_q)
        ctx.ewgs_lambda = ewgs_lambda
        return x_q

    @staticmethod
    def backward(ctx, grad_output):
        x, log_alpha, x_q = ctx.saved_tensors
        lam = ctx.ewgs_lambda
        alpha = log_alpha.exp()
        in_range = ((x >= 0) & (x <= alpha)).float()
        ewgs_scale = 1.0 + lam * grad_output.sign() * (x - x_q).sign()
        grad_x = grad_output * in_range * ewgs_scale
        at_clip = (x > alpha).float()
        grad_log_alpha = (grad_output * at_clip * alpha).sum().reshape(log_alpha.shape)
        return grad_x, grad_log_alpha, None, None

class PoTActivationEWGS(nn.Module):
    def __init__(self, n_levels: int = 8, alpha_init: float = 4.0,
                 use_pot_after_epoch: int = 0,
                 ewgs_lambda: float = EWGS_LAMBDA):
        super().__init__()
        self.n_levels = n_levels
        self.use_pot_after_epoch = use_pot_after_epoch
        self.current_epoch = 0
        self.ewgs_lambda = ewgs_lambda
        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(alpha_init), dtype=torch.float32))

    def _should_use_pot(self) -> bool:
        return self.current_epoch >= self.use_pot_after_epoch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._should_use_pot():
            return x
        return RoundActivToPoTEWGS.apply(
            x, self.log_alpha, self.n_levels, self.ewgs_lambda)

    def extra_repr(self) -> str:
        return (f"n_levels={self.n_levels}, "
                f"use_pot_after_epoch={self.use_pot_after_epoch}, "
                f"ewgs_lambda={self.ewgs_lambda}")

class PoTBatchNorm2dEWGS(PoTBatchNorm2d):
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