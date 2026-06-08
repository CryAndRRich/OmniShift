import math
import torch
import torch.nn as nn
import torch.autograd as autograd

class RoundActivToPoT(autograd.Function):
    @staticmethod
    def forward(ctx, x, log_alpha, n_levels):
        alpha = log_alpha.exp()

        x_clip = x.clamp(0.0, alpha.item())

        eps = 1e-8
        active = x_clip > eps

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

        in_range = ((x >= 0) & (x <= alpha)).float()
        grad_x = grad_output * in_range

        at_clip = (x > alpha).float()
        grad_log_alpha = (grad_output * at_clip * alpha).sum().reshape(log_alpha.shape)

        return grad_x, grad_log_alpha, None

class PoTActivation(nn.Module):

    def __init__(self, n_levels: int = 8, alpha_init: float = 4.0,
                 use_pot_after_epoch: int = 0):
        super().__init__()
        self.n_levels = n_levels
        self.use_pot_after_epoch = use_pot_after_epoch
        self.current_epoch = 0

        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(alpha_init), dtype=torch.float32))

    def _should_use_pot(self) -> bool:
        return self.current_epoch >= self.use_pot_after_epoch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._should_use_pot():
            return x
        return RoundActivToPoT.apply(x, self.log_alpha, self.n_levels)

    def extra_repr(self) -> str:
        alpha = math.exp(self.log_alpha.item()) if self.log_alpha.requires_grad else float("nan")
        return (f"n_levels={self.n_levels}, "
                f"alpha={alpha:.3f}, "
                f"use_pot_after_epoch={self.use_pot_after_epoch}")