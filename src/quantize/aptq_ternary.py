import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class _PoTQuantSTE(autograd.Function):
    @staticmethod
    def forward(ctx, x, p_min: int, p_max: int):
        log2x = torch.log2(x.clamp(min=1e-8))
        p = log2x.round().clamp(p_min, p_max)
        return 2.0 ** p

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

class _SparseMaskSTE(autograd.Function):
    @staticmethod
    def forward(ctx, w_eff, threshold):
        mask = (w_eff.abs() > threshold).float()
        return w_eff * mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class APTQTernaryConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, bias: bool = False,
                 sparse_mode: str = "learnable", sparsity_ratio: float = 0.5,
                 p_min: int = -8, p_max: int = 0,
                 init_threshold: float = 0.05):
        super().__init__()

        if sparse_mode not in ("fixed", "learnable"):
            raise ValueError(
                f"sparse_mode must be 'fixed' or 'learnable', got {sparse_mode!r}")

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

        shape = (out_channels, in_channels, *self.kernel_size)

        self.w_pos = nn.Parameter(torch.empty(shape))
        self.w_neg = nn.Parameter(torch.empty(shape))
        self.bias_param = nn.Parameter(torch.empty(out_channels)) if bias else None

        if sparse_mode == "learnable":
            self.register_buffer("log_threshold", torch.tensor(math.log(init_threshold)))

        nn.init.normal_(self.w_pos, mean=0.5, std=0.1)
        nn.init.normal_(self.w_neg, mean=0.5, std=0.1)
        if self.bias_param is not None:
            nn.init.zeros_(self.bias_param)

    def _pot_quantize(self, w: torch.Tensor) -> torch.Tensor:
        w_abs = w.abs().clamp(min=1e-8)
        return _PoTQuantSTE.apply(w_abs, self.p_min, self.p_max)

    def _effective_weight(self) -> torch.Tensor:
        q_pos = self._pot_quantize(self.w_pos)
        q_neg = self._pot_quantize(self.w_neg)
        return q_pos - q_neg

    def _quantize(self) -> torch.Tensor:
        w_eff = self._effective_weight()

        if self.sparse_mode == "fixed":
            abs_eff = w_eff.abs()
            N = w_eff.numel()
            k = int(self.sparsity_ratio * N)
            if k <= 0:
                return w_eff
            if k >= N:
                return torch.zeros_like(w_eff)
            threshold = abs_eff.detach().flatten().kthvalue(k).values
            return _SparseMaskSTE.apply(w_eff, threshold)
        else:
            threshold = self.log_threshold.exp()
            return _SparseMaskSTE.apply(w_eff, threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = self._quantize()
        return F.conv2d(x, w_q, self.bias_param,
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