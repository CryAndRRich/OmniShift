import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

_P_MIN = -15
_P_MAX = 0

class _FogzoPoTFunction(autograd.Function):
    @staticmethod
    def forward(ctx, w, n_perturbations, sigma, fogzo_lambda, grad_clip, p_min, p_max):
        sign = torch.sign(w)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        abs_w = w.abs().clamp(min=1e-8)
        log2_w = torch.log2(abs_w)
        p = torch.round(log2_w).clamp(p_min, p_max)
        w_q = sign * (2.0 ** p)

        ctx.save_for_backward(w, w_q)
        ctx.n_perturbations = n_perturbations
        ctx.sigma = sigma
        ctx.fogzo_lambda = fogzo_lambda
        ctx.grad_clip = grad_clip
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        w, _ = ctx.saved_tensors
        n = ctx.n_perturbations
        lam = ctx.fogzo_lambda
        clip = ctx.grad_clip

        g_ste = grad_output

        correction = torch.zeros_like(g_ste)
        if n > 0 and lam != 0.0:

            g_flat = g_ste.flatten()
            for _ in range(n):
                u_i = torch.randn_like(w)
                u_flat = u_i.flatten()
                dot = torch.dot(g_flat, u_flat)
                correction = correction + dot.sign() * u_i
            correction = correction / n

        g_final = g_ste + lam * correction
        g_final = g_final.clamp(-clip, clip)

        return g_final, None, None, None, None, None, None

class FogzoShiftConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=False,
                 n_perturbations: int = 4,
                 sigma: float = 0.01,
                 fogzo_lambda: float = 0.1,
                 grad_clip: float = 10.0,
                 p_min: int = _P_MIN,
                 p_max: int = _P_MAX):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size if isinstance(kernel_size, tuple)
                            else (kernel_size, kernel_size))
        self.stride = stride
        self.padding = padding
        self.n_perturbations = n_perturbations
        self.sigma = sigma
        self.fogzo_lambda = fogzo_lambda
        self.grad_clip = grad_clip
        self.p_min = p_min
        self.p_max = p_max

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = _FogzoPoTFunction.apply(
            self.weight,
            self.n_perturbations,
            self.sigma,
            self.fogzo_lambda,
            self.grad_clip,
            self.p_min,
            self.p_max,
        )
        return F.conv2d(x, w_q, self.bias,
                        stride=self.stride, padding=self.padding)

    def get_quantized_weight(self) -> torch.Tensor:
        with torch.no_grad():
            sign = torch.sign(self.weight)
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            abs_w = self.weight.abs().clamp(min=1e-8)
            p = torch.round(torch.log2(abs_w)).clamp(self.p_min, self.p_max)
            return sign * (2.0 ** p)

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, bias={self.bias is not None}, "
            f"n_perturbations={self.n_perturbations}, sigma={self.sigma}, "
            f"fogzo_lambda={self.fogzo_lambda}, grad_clip={self.grad_clip}, "
            f"p=[{self.p_min},{self.p_max}]"
        )
