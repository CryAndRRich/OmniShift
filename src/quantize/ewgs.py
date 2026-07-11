import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from src.quantize.pot_bn import PoTBatchNorm2d

EWGS_LAMBDA = 0.02

# Hysteresis margins: a stored exponent/mask only changes when the continuous
# value drifts beyond round-boundary + margin. On a PoT grid a boundary flip
# is a 2x value change (vs 1 LSB on a uniform grid), so flap suppression
# matters far more here than in standard QAT.
EXP_HYSTERESIS = 0.1
MASK_HYSTERESIS = 0.1


def _buffers_mutable(module: nn.Module) -> bool:
    # Hysteresis state must not advance during no-grad passes
    # (BN re-estimation, sparsity probes) — only during real training steps.
    # NOTE: must be evaluated BEFORE entering a no_grad block.
    return module.training and torch.is_grad_enabled()


class QuantEWGSSTE(autograd.Function):
    """Generic EWGS straight-through: forward returns a precomputed quantized
    tensor, backward rescales the gradient by (1 + lam*sign(g)*sign(x - x_q))."""

    @staticmethod
    def forward(ctx, x, x_q, ewgs_lambda):
        ctx.save_for_backward(x, x_q)
        ctx.ewgs_lambda = ewgs_lambda
        return x_q

    @staticmethod
    def backward(ctx, grad_output):
        x, x_q = ctx.saved_tensors
        lam = ctx.ewgs_lambda
        grad = grad_output * (1.0 + lam * grad_output.sign() * (x - x_q).sign())
        return grad, None, None


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


class SparseShiftConv2dEWGS(nn.Module):
    """Sparse shift conv with oscillation-aware training.

    Learnable mode stabilizers (vs plain threshold masking):
      * hysteresis gate: a weight turns ON above t*(1+rho) and OFF below
        t*(1-rho) — in between it keeps its previous state, so SGD noise
        around the threshold no longer churns the mask every step;
      * mask freezing: mask updates stop at `mask_freeze_epoch`;
      * iterative value freezing (Nagel et al., ICML 2022, adapted to the
        PoT grid): per-weight EMA of quantized-value flips; from
        `freeze_after_epoch` on, weights whose flip rate exceeds
        `flip_freeze_th` are pinned to their current quantized value.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=False, sparse_mode="fixed",
                 sparsity_ratio=0.5, init_threshold=0.05,
                 ewgs_lambda: float = EWGS_LAMBDA,
                 mask_hysteresis: float = MASK_HYSTERESIS,
                 mask_freeze_epoch: int = 160,
                 freeze_after_epoch: int = 100,
                 flip_freeze_th: float = 0.02,
                 flip_ema_momentum: float = 0.01):
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
        self.mask_hysteresis = mask_hysteresis
        self.mask_freeze_epoch = mask_freeze_epoch
        self.freeze_after_epoch = freeze_after_epoch
        self.flip_freeze_th = flip_freeze_th
        self.flip_ema_momentum = flip_ema_momentum
        self.current_epoch = 0  # set externally via set_bn_epoch

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        if sparse_mode == "learnable":
            self.register_buffer('log_threshold',
                                 torch.tensor(math.log(init_threshold)))
            with torch.no_grad():
                init_mask = (self.weight.abs() > init_threshold)
            self.register_buffer('mask', init_mask)
            self.register_buffer('prev_q', torch.zeros_like(self.weight))
            self.register_buffer('flip_ema', torch.zeros_like(self.weight))
            self.register_buffer('frozen',
                                 torch.zeros_like(self.weight, dtype=torch.bool))
            self.register_buffer('frozen_val', torch.zeros_like(self.weight))

    def _quantize_learnable(self, update: bool) -> torch.Tensor:
        with torch.no_grad():
            w = self.weight
            abs_w = w.abs()
            threshold = self.log_threshold.exp()

            if update and self.current_epoch < self.mask_freeze_epoch:
                rho = self.mask_hysteresis
                turn_on = abs_w > threshold * (1.0 + rho)
                turn_off = abs_w < threshold * (1.0 - rho)
                self.mask.copy_((self.mask | turn_on) & ~turn_off)

            sign = torch.sign(w)
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            p = torch.round(torch.log2(abs_w.clamp(min=1e-8))).clamp(-15, 0)
            w_q = self.mask.float() * sign * (2.0 ** p)

            if update:
                flips = (w_q != self.prev_q).float()
                m = self.flip_ema_momentum
                self.flip_ema.mul_(1.0 - m).add_(m * flips)
                self.prev_q.copy_(w_q)
                if self.current_epoch >= self.freeze_after_epoch:
                    newly = (self.flip_ema > self.flip_freeze_th) & ~self.frozen
                    if newly.any():
                        self.frozen_val.copy_(
                            torch.where(newly, w_q, self.frozen_val))
                        self.frozen |= newly

            return torch.where(self.frozen, self.frozen_val, w_q)

    def forward(self, x):
        if self.sparse_mode == "fixed":
            w_q = FixedSparseShiftEWGS.apply(
                self.weight, self.sparsity_ratio, self.ewgs_lambda)
        elif self.sparse_mode == "learnable":
            w_q = QuantEWGSSTE.apply(
                self.weight, self._quantize_learnable(_buffers_mutable(self)),
                self.ewgs_lambda)
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
            w_q = self._quantize_learnable(update=False)
        return (w_q == 0).float().mean().item()

    @torch.no_grad()
    def get_flip_rate(self) -> float:
        if self.sparse_mode != "learnable":
            return 0.0
        return self.flip_ema.mean().item()

    @torch.no_grad()
    def get_frozen_frac(self) -> float:
        if self.sparse_mode != "learnable":
            return 0.0
        return self.frozen.float().mean().item()

    def extra_repr(self):
        s = (f"{self.in_channels}, {self.out_channels}, "
             f"kernel_size={self.kernel_size}, stride={self.stride}, "
             f"padding={self.padding}, sparse_mode={self.sparse_mode}, "
             f"ewgs_lambda={self.ewgs_lambda}")
        if self.sparse_mode == "fixed":
            s += f", sparsity_ratio={self.sparsity_ratio}"
        else:
            s += (f", mask_hysteresis={self.mask_hysteresis}, "
                  f"freeze_after_epoch={self.freeze_after_epoch}")
        return s


class RoundActivToPoTEWGS(autograd.Function):
    @staticmethod
    def forward(ctx, x, log_alpha, p_max, n_levels, ewgs_lambda):
        alpha = log_alpha.exp()
        x_clip = x.clamp(0.0, alpha.item())
        eps = 1e-8
        active = x_clip > eps
        p_min = p_max - (n_levels - 1)
        p = torch.round(torch.log2(x_clip.clamp(min=eps)))
        p = p.clamp(p_min, p_max)
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
        return grad_x, grad_log_alpha, None, None, None


class PoTActivationEWGS(nn.Module):
    """PoT activation with a hysteresis-anchored grid.

    The grid anchor p_max = floor(log2(alpha)) is stored in a buffer and only
    re-anchored when log2(alpha) drifts outside [p_max - h, p_max + 1 + h].
    Without this, alpha crossing a power of two shifts the entire activation
    grid by 2x from one step to the next.
    """

    def __init__(self, n_levels: int = 8, alpha_init: float = 4.0,
                 use_pot_after_epoch: int = 0,
                 ewgs_lambda: float = EWGS_LAMBDA,
                 exp_hysteresis: float = EXP_HYSTERESIS):
        super().__init__()
        self.n_levels = n_levels
        self.use_pot_after_epoch = use_pot_after_epoch
        self.current_epoch = 0
        self.ewgs_lambda = ewgs_lambda
        self.exp_hysteresis = exp_hysteresis
        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(alpha_init), dtype=torch.float32))
        self.register_buffer('grid_pmax',
                             torch.tensor(math.floor(math.log2(alpha_init)),
                                          dtype=torch.float32))

    def _should_use_pot(self) -> bool:
        return self.current_epoch >= self.use_pot_after_epoch

    def _update_grid_anchor(self):
        with torch.no_grad():
            log2a = self.log_alpha / math.log(2.0)
            h = self.exp_hysteresis
            lo = self.grid_pmax - h
            hi = self.grid_pmax + 1.0 + h
            if log2a < lo or log2a > hi:
                self.grid_pmax.fill_(math.floor(log2a.item()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._should_use_pot():
            return x
        if _buffers_mutable(self):
            self._update_grid_anchor()
        return RoundActivToPoTEWGS.apply(
            x, self.log_alpha, self.grid_pmax.item(), self.n_levels,
            self.ewgs_lambda)

    def extra_repr(self) -> str:
        return (f"n_levels={self.n_levels}, "
                f"use_pot_after_epoch={self.use_pot_after_epoch}, "
                f"ewgs_lambda={self.ewgs_lambda}, "
                f"exp_hysteresis={self.exp_hysteresis}")


class PoTBatchNorm2dEWGS(PoTBatchNorm2d):
    """PoT batch norm with a stored, hysteresis-updated scale exponent.

    The PoT exponent q per channel lives in a buffer. During training it only
    changes when log2|gamma/sigma_batch| drifts more than 0.5 + h away from q.
    Eval uses the SAME stored exponent instead of re-quantizing from running
    stats — previously a channel near a rounding boundary could use scale 2^q
    in training and 2^(q+1) at eval, a silent 2x mismatch that dominated
    val-accuracy noise.
    """

    def __init__(self, num_features, momentum=0.1, eps=1e-5,
                 use_pot_after_epoch=0, ewgs_lambda: float = EWGS_LAMBDA,
                 exp_hysteresis: float = EXP_HYSTERESIS):
        super().__init__(num_features, momentum=momentum, eps=eps,
                         use_pot_after_epoch=use_pot_after_epoch)
        self.ewgs_lambda = ewgs_lambda
        self.exp_hysteresis = exp_hysteresis
        self.register_buffer('pot_exp', torch.zeros(num_features))
        self.register_buffer('pot_sign', torch.ones(num_features))
        self.register_buffer('pot_init', torch.tensor(False))

    def _update_pot_scale(self, scale_fp: torch.Tensor, update: bool):
        with torch.no_grad():
            sign = torch.sign(scale_fp)
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            log2s = torch.log2(scale_fp.abs().clamp(min=1e-8))
            q_round = torch.round(log2s).clamp(-15, 15)

            if not bool(self.pot_init):
                self.pot_exp.copy_(q_round)
                self.pot_sign.copy_(sign)
                self.pot_init.fill_(True)
            elif update:
                drift = (log2s - self.pot_exp).abs() > 0.5 + self.exp_hysteresis
                flipped = sign != self.pot_sign
                upd = drift | flipped
                self.pot_exp.copy_(torch.where(upd, q_round, self.pot_exp))
                self.pot_sign.copy_(torch.where(upd, sign, self.pot_sign))

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
            self._update_pot_scale(scale_fp, _buffers_mutable(self))
            s_pot = self.pot_sign * (2.0 ** self.pot_exp)
            scale = QuantEWGSSTE.apply(scale_fp, s_pot, self.ewgs_lambda)
        else:
            scale = scale_fp

        return (scale.view(1, -1, 1, 1) * (x - mean.view(1, -1, 1, 1))
                + self.bias.view(1, -1, 1, 1))
