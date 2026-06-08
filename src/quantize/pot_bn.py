import torch
import torch.nn as nn
import torch.autograd as autograd

class ScaleToPoT(autograd.Function):
    @staticmethod
    def forward(ctx, scale):
        sign = torch.sign(scale)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        abs_s = scale.abs().clamp(min=1e-8)
        log2_s = torch.log2(abs_s)
        p = torch.round(log2_s)
        p = torch.clamp(p, min=-15, max=15)
        return sign * (2.0 ** p)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class PoTBatchNorm2d(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-5,
                 use_pot_after_epoch=0):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.use_pot_after_epoch = use_pot_after_epoch
        self.current_epoch = 0

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def _should_use_pot(self):
        return self.current_epoch >= self.use_pot_after_epoch

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
            scale = ScaleToPoT.apply(scale_fp)
        else:
            scale = scale_fp

        return (scale.view(1, -1, 1, 1) * (x - mean.view(1, -1, 1, 1))
                + self.bias.view(1, -1, 1, 1))

    def extra_repr(self):
        return (f"{self.num_features}, momentum={self.momentum}, "
                f"eps={self.eps}, use_pot_after_epoch={self.use_pot_after_epoch}")

def set_bn_epoch(model: nn.Module, epoch: int) -> None:
    for m in model.modules():
        if hasattr(m, "current_epoch"):
            m.current_epoch = epoch