import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class _SignSTE(autograd.Function):
    @staticmethod
    def forward(ctx, s):
        out = s.sign()
        return torch.where(out == 0, torch.ones_like(out), out)

    @staticmethod
    def backward(ctx, grad):
        return grad

class _ExpRoundSTE(autograd.Function):
    @staticmethod
    def forward(ctx, e, p_min, p_max):
        return e.round().clamp(p_min, p_max)

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None

class _SparseMaskSTE(autograd.Function):
    @staticmethod
    def forward(ctx, w_eff, threshold):
        mask = (w_eff.abs() > threshold).float()
        return w_eff * mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class S3ShiftConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=False, sparse_mode="learnable",
                 p_min=-8, p_max=1, sparsity_ratio=0.5, init_threshold=0.05):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size if isinstance(kernel_size, tuple)
                            else (kernel_size, kernel_size))
        self.stride = stride
        self.padding = padding
        self.p_min = p_min
        self.p_max = p_max
        self.sparse_mode = sparse_mode
        self.sparsity_ratio = sparsity_ratio

        if sparse_mode not in ("fixed", "learnable"):
            raise ValueError(f"sparse_mode must be 'fixed' or 'learnable', got {sparse_mode!r}")

        shape = (out_channels, in_channels, *self.kernel_size)

        self.sign_param = nn.Parameter(torch.empty(shape))

        self.exp_param = nn.Parameter(torch.zeros(shape))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

        if sparse_mode == "learnable":
            self.register_buffer("log_threshold", torch.tensor(math.log(init_threshold)))

        nn.init.normal_(self.sign_param, mean=0.0, std=1e-3)
        nn.init.normal_(self.exp_param, mean=0.0, std=0.01)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _quantize(self):
        s = _SignSTE.apply(self.sign_param)
        p = _ExpRoundSTE.apply(self.exp_param, self.p_min, self.p_max)

        w_eff = s * (2.0 ** p)

        if self.sparse_mode == "learnable":
            threshold = self.log_threshold.exp()
            return _SparseMaskSTE.apply(w_eff, threshold)
        else:
            exp_det = self.exp_param.detach()
            N = exp_det.numel()
            k = int(self.sparsity_ratio * N)
            if k == 0:
                return w_eff
            if k >= N:
                return torch.zeros_like(w_eff)
            threshold_exp = exp_det.flatten().kthvalue(k).values

            mask = (exp_det > threshold_exp).float().detach()

            return w_eff + (w_eff * mask - w_eff).detach()

    def forward(self, x):
        w_q = self._quantize()
        return F.conv2d(x, w_q, self.bias,
                        stride=self.stride, padding=self.padding)

    @torch.no_grad()
    def get_actual_sparsity(self) -> float:
        w_q = self._quantize()
        return (w_q == 0).float().mean().item()

    def extra_repr(self) -> str:
        s = (f"{self.in_channels}, {self.out_channels}, "
             f"kernel_size={self.kernel_size}, stride={self.stride}, "
             f"padding={self.padding}, sparse_mode={self.sparse_mode}, "
             f"p=[{self.p_min},{self.p_max}]")
        if self.sparse_mode == "fixed":
            s += f", sparsity_ratio={self.sparsity_ratio}"
        return s