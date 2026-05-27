"""PoT Activation Quantization for Phase 6.

Post-ReLU activations are rounded to a signed power-of-two grid:
  {0} ∪ {2^p : p = p_min, …, p_max}

where p_max = floor(log2(α)) and α is a learnable per-layer clip parameter.
Values below 2^p_min are mapped to 0 (treated as zero by skip-zero hardware).

This makes activations multiply-free: downstream PoT-BN scales and PoT
conv weights are applied as shifts → fully multiply-free inference end-to-end.

Grid: n_levels non-zero PoT values + one zero = n_levels+1 distinct outputs
      (default n_levels=8, equivalent to a 4-bit unsigned quantization).
"""

import math
import torch
import torch.nn as nn


class RoundActivToPoT(torch.autograd.Function):
    """Quantize non-negative activation to {0} ∪ PoT grid, STE backward."""

    @staticmethod
    def forward(ctx, x, log_alpha, n_levels):
        alpha = log_alpha.exp()
        # Clip to [0, alpha]
        x_clip = x.clamp(0.0, alpha.item())

        eps = 1e-8
        active = x_clip > eps

        # Grid bounds: p_max = floor(log2(alpha)), p_min = p_max - (n_levels-1)
        p_max = torch.floor(torch.log2(alpha.clamp(min=eps)))
        p_min = p_max - (n_levels - 1)

        p = torch.round(torch.log2(x_clip.clamp(min=eps)))
        p = p.clamp(p_min.item(), p_max.item())
        x_q = torch.where(active, 2.0 ** p, torch.zeros_like(x_clip))

        ctx.save_for_backward(x, log_alpha)
        return x_q

    @staticmethod
    def backward(ctx, grad_output):
        x, log_alpha = ctx.saved_tensors
        alpha = log_alpha.exp()

        # STE: gradient passes through where 0 ≤ x ≤ alpha
        in_range = ((x >= 0) & (x <= alpha)).float()
        grad_x = grad_output * in_range

        # Gradient for log_alpha via chain rule: dL/d(log_alpha) = dL/dalpha * alpha
        at_clip = (x > alpha).float()
        grad_log_alpha = (grad_output * at_clip * alpha).sum().reshape(log_alpha.shape)

        return grad_x, grad_log_alpha, None


class PoTActivation(nn.Module):
    """Quantize post-ReLU activations to a PoT grid.

    Insert after each F.relu() call in the residual blocks to make activations
    power-of-two quantized. Together with PoT-BN scales and SparseShift
    weights, this achieves a fully multiply-free forward pass.

    Args:
        n_levels: non-zero PoT grid points (default 8, covering 2^-5 … 2^2
                  when alpha_init=4.0).
        alpha_init: initial learnable clip value α_max. Start ≥ max expected
                    activation magnitude so clipping doesn't hurt accuracy
                    before training adjusts α.
        use_pot_after_epoch: number of warmup epochs to run with identity
                    (no PoT quantization). Matches BN warmup convention.
    """

    def __init__(self, n_levels: int = 8, alpha_init: float = 4.0,
                 use_pot_after_epoch: int = 0):
        super().__init__()
        self.n_levels = n_levels
        self.use_pot_after_epoch = use_pot_after_epoch
        self.current_epoch = 0
        # log-parameterized so alpha is always positive
        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(alpha_init), dtype=torch.float32))

    def _should_use_pot(self) -> bool:
        return self.current_epoch >= self.use_pot_after_epoch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._should_use_pot():
            return x
        return RoundActivToPoT.apply(x, self.log_alpha, self.n_levels)

    def extra_repr(self) -> str:
        alpha = math.exp(self.log_alpha.item()) if not self.log_alpha.requires_grad \
            else float('nan')
        return (f'n_levels={self.n_levels}, '
                f'use_pot_after_epoch={self.use_pot_after_epoch}')
