import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

def _build_apot_pos_levels(alpha: torch.Tensor, n_bits: int, device, dtype) -> torch.Tensor:
    n_levels = 1 << (n_bits - 1)
    delta = alpha / float(n_levels)
    indices = torch.arange(n_levels, dtype=dtype, device=device)
    return indices * delta

class APoTQuantize(autograd.Function):
    @staticmethod
    def forward(ctx, w, alpha, n_bits, include_zero):
        pos_levels = _build_apot_pos_levels(alpha, n_bits, w.device, w.dtype)

        sign = w.sign()
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        abs_w = w.abs()

        if include_zero:
            diff = (abs_w.unsqueeze(-1) - pos_levels).abs()
            idx = diff.argmin(dim=-1)
            q_abs = pos_levels[idx]
            q = torch.where(q_abs == 0, torch.zeros_like(w), sign * q_abs)
        else:
            nonzero = pos_levels[1:]
            diff = (abs_w.unsqueeze(-1) - nonzero).abs()
            idx = diff.argmin(dim=-1)
            q_abs = nonzero[idx]
            q = sign * q_abs

        ctx.save_for_backward(w, alpha)
        ctx.include_zero = include_zero
        ctx.n_bits = n_bits
        return q

    @staticmethod
    def backward(ctx, grad):
        w, alpha = ctx.saved_tensors

        grad_w = grad

        n_levels = 1 << (ctx.n_bits - 1)
        max_level = alpha * float(n_levels - 1) / float(n_levels)
        at_clip = (w.abs() >= max_level).float()
        grad_alpha = (grad * w.sign() * at_clip).sum().reshape(alpha.shape)
        return grad_w, grad_alpha, None, None

class APoTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=False, n_bits=3, alpha_init=1.0,
                 include_zero=True):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = (kernel_size if isinstance(kernel_size, tuple)
                             else (kernel_size, kernel_size))
        self.stride = stride
        self.padding = padding
        self.n_bits = n_bits
        self.include_zero = include_zero

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        alpha = self.alpha.abs().clamp(min=1e-4)
        w_q = APoTQuantize.apply(self.weight, alpha, self.n_bits, self.include_zero)
        return F.conv2d(x, w_q, self.bias, stride=self.stride, padding=self.padding)

    def extra_repr(self):
        n_levels = 1 << (self.n_bits - 1)
        delta_str = f"{self.alpha.item():.4f}/{n_levels}"
        return (f"{self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, n_bits={self.n_bits}, "
                f"step={delta_str}, include_zero={self.include_zero}")