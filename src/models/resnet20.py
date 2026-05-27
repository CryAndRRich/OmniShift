"""ResNet-20 Phase 1 baselines: vanilla mul, DeepShift, APoT, DenseShift.

Convention: first conv (3→16) and final FC always use standard multiplication.
All interior convs use the specified conv_type.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.quantize.shift import ShiftConv2d


# ── APoT helpers ─────────────────────────────────────────────────────────────

def _build_apot_levels(bit: int = 3, K: int = 2):
    n = max((bit - 1) // K, 1)
    sub_dicts = []
    for i in range(K):
        term_values = [0.0]
        for j in range(1, 2 ** n):
            term_values.append(2.0 ** (-(i * n + j - 1)))
        sub_dicts.append(term_values)

    levels = set()
    if K == 1:
        levels = set(sub_dicts[0])
    elif K == 2:
        for a in sub_dicts[0]:
            for b in sub_dicts[1]:
                levels.add(a + b)
    else:
        from itertools import product as iprod
        for combo in iprod(*sub_dicts):
            levels.add(sum(combo))
    return torch.tensor(sorted(levels), dtype=torch.float32)


class _APoTQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, levels, alpha):
        w_clipped = torch.clamp(w / alpha, -1.0, 1.0)
        sign = torch.sign(w_clipped)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        mag = w_clipped.abs()
        mag_flat = mag.reshape(-1, 1)
        idx = (mag_flat - levels.view(1, -1)).abs().argmin(dim=1)
        mag_q = levels[idx].reshape(mag.shape)
        ctx.save_for_backward(w, alpha)
        return sign * mag_q * alpha

    @staticmethod
    def backward(ctx, grad_output):
        w, alpha = ctx.saved_tensors
        mask_inside = (w.abs() <= alpha).float()
        grad_w = grad_output * mask_inside
        outside_sign = torch.sign(w) * (w.abs() > alpha).float()
        grad_alpha = (grad_output * outside_sign).sum().reshape(alpha.shape)
        return grad_w, None, grad_alpha


class APoTConv2d(nn.Module):
    """APoT-quantized conv (additive PoT weights). Default: 3-bit, K=2."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=False, bit=3, K=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size if isinstance(kernel_size, tuple)
                            else (kernel_size, kernel_size))
        self.stride = stride
        self.padding = padding
        self.bit = bit
        self.K = K

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.alpha = nn.Parameter(torch.tensor(3.0))
        self.register_buffer('levels', _build_apot_levels(bit=bit, K=K))

        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        w_mean = self.weight.mean(dim=(1, 2, 3), keepdim=True)
        w_std = self.weight.std(dim=(1, 2, 3), keepdim=True) + 1e-5
        w_norm = (self.weight - w_mean) / w_std
        w_quant = _APoTQuantize.apply(w_norm, self.levels, self.alpha.abs() + 1e-4)
        return F.conv2d(x, w_quant, self.bias, stride=self.stride, padding=self.padding)


# ── DenseShift helpers ────────────────────────────────────────────────────────

def _build_denseshift_levels(bit: int = 3):
    num_levels = 2 ** bit
    return torch.tensor(sorted(2.0 ** (-i) for i in range(num_levels)),
                        dtype=torch.float32)


class _DenseShiftQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w_mag, levels, alpha):
        w_clipped = torch.clamp(w_mag / alpha, 1e-8, 1.0)
        idx = (w_clipped.reshape(-1, 1) - levels.view(1, -1)).abs().argmin(dim=1)
        w_q = levels[idx].reshape(w_mag.shape)
        ctx.save_for_backward(w_mag, alpha)
        return w_q * alpha

    @staticmethod
    def backward(ctx, grad_output):
        w_mag, alpha = ctx.saved_tensors
        mask = (w_mag <= alpha).float()
        grad_w = grad_output * mask
        grad_alpha = (grad_output * (w_mag > alpha).float()).sum().reshape(alpha.shape)
        return grad_w, None, grad_alpha


class _SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        sign = torch.sign(w)
        return torch.where(sign == 0, torch.ones_like(sign), sign)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class DenseShiftConv2d(nn.Module):
    """DenseShift conv (ICCV 2023): sign-scale decomposition, zero-free PoT mag."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=False, bit=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size if isinstance(kernel_size, tuple)
                            else (kernel_size, kernel_size))
        self.stride = stride
        self.padding = padding
        self.bit = bit

        self.weight_mag = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size))
        self.weight_sign = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.register_buffer('levels', _build_denseshift_levels(bit=bit))
        self._init_weights()

    def _init_weights(self):
        fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        std = math.sqrt(2.0 / fan_in)
        nn.init.normal_(self.weight_mag, mean=std * 1.5, std=std * 0.3)
        self.weight_mag.data.clamp_(min=1e-4)
        nn.init.normal_(self.weight_sign, mean=0.0, std=1.0)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        sign = _SignSTE.apply(self.weight_sign)
        mag_q = _DenseShiftQuantize.apply(
            self.weight_mag.abs(), self.levels, self.alpha.abs() + 1e-4)
        w_quant = sign * mag_q
        return F.conv2d(x, w_quant, self.bias, stride=self.stride, padding=self.padding)


# ── Conv factory ─────────────────────────────────────────────────────────────

def _make_conv(conv_type: str, in_ch, out_ch, kernel_size=3,
               stride=1, padding=1, bias=False):
    """Factory for conv layers. conv_type ∈ {'mul','shift','apot','denseshift'}."""
    if conv_type == "mul":
        return nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                         padding=padding, bias=bias)
    elif conv_type == "shift":
        return ShiftConv2d(in_ch, out_ch, kernel_size, stride=stride,
                           padding=padding, bias=bias)
    elif conv_type == "apot":
        return APoTConv2d(in_ch, out_ch, kernel_size, stride=stride,
                          padding=padding, bias=bias)
    elif conv_type == "denseshift":
        return DenseShiftConv2d(in_ch, out_ch, kernel_size, stride=stride,
                                padding=padding, bias=bias)
    else:
        raise ValueError(f"Unknown conv_type: {conv_type}")


# ── Model ────────────────────────────────────────────────────────────────────

class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, conv_type="mul"):
        super().__init__()
        self.conv1 = _make_conv(conv_type, in_planes, planes, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = _make_conv(conv_type, planes, planes, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                _make_conv(conv_type, in_planes, planes, 1, stride=stride, padding=0),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet20(nn.Module):
    """ResNet-20 for CIFAR (Phase 1 baselines): mul, shift, apot, denseshift."""

    def __init__(self, num_classes=10, conv_type="mul", in_channels=3):
        super().__init__()
        self.conv_type = conv_type
        self.in_planes = 16

        self.conv1 = nn.Conv2d(in_channels, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.stage1 = self._make_stage(16, 3, stride=1, conv_type=conv_type)
        self.stage2 = self._make_stage(32, 3, stride=2, conv_type=conv_type)
        self.stage3 = self._make_stage(64, 3, stride=2, conv_type=conv_type)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        self._init_weights()

    def _make_stage(self, planes, num_blocks, stride, conv_type):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(_BasicBlock(self.in_planes, planes, stride=s,
                                      conv_type=conv_type))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.avgpool(out).flatten(1)
        return self.fc(out)


_NAME_TO_CONV = {
    "resnet20": "mul",
    "deepshift": "shift",
    "apot": "apot",
    "denseshift": "denseshift",
}


def build_model(model_name: str, num_classes: int, in_channels: int = 3) -> ResNet20:
    """Build a Phase 1 baseline by name."""
    if model_name not in _NAME_TO_CONV:
        raise ValueError(
            f"Unknown model: {model_name!r}. Choose from {list(_NAME_TO_CONV)}.")
    return ResNet20(num_classes=num_classes, conv_type=_NAME_TO_CONV[model_name],
                    in_channels=in_channels)
