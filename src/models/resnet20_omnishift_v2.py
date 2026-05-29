"""ResNet-20: SparseShift + PoT-BN + PoT Activations, all with EWGS (Phase 7).

Combines Phase 5 (EWGS gradient estimator) and Phase 6 (PoT activation
quantization) into a single fully multiply-free model.

Forward pass: all multiplications in the conv stack replaced by bit-shifts.
  - Weights     : W ∈ {0, ±2^p} via SparseShiftConv2dEWGS
  - BN scales   : γ/σ → ±2^q   via PoTBatchNorm2dEWGS
  - Activations : post-ReLU → {0} ∪ {2^p} via PoTActivationEWGS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.quantize.ewgs import (
    SparseShiftConv2dEWGS,
    PoTBatchNorm2dEWGS,
    PoTActivationEWGS,
    EWGS_LAMBDA,
)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,
                 sparse_mode="fixed", sparsity_ratio=0.5,
                 bn_warmup=30, ewgs_lambda=EWGS_LAMBDA,
                 act_levels=8, act_alpha_init=4.0):
        super().__init__()

        def make_conv(in_c, out_c, k, s, p):
            return SparseShiftConv2dEWGS(
                in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False,
                sparse_mode=sparse_mode, sparsity_ratio=sparsity_ratio,
                ewgs_lambda=ewgs_lambda)

        def make_bn(c):
            return PoTBatchNorm2dEWGS(c, use_pot_after_epoch=bn_warmup,
                                      ewgs_lambda=ewgs_lambda)

        def make_act():
            return PoTActivationEWGS(n_levels=act_levels, alpha_init=act_alpha_init,
                                     use_pot_after_epoch=bn_warmup,
                                     ewgs_lambda=ewgs_lambda)

        self.conv1 = make_conv(in_planes, planes, 3, stride, 1)
        self.bn1   = make_bn(planes)
        self.act1  = make_act()
        self.conv2 = make_conv(planes, planes, 3, 1, 1)
        self.bn2   = make_bn(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                make_conv(in_planes, planes, 1, stride, 0),
                make_bn(planes),
            )

        self.act2 = make_act()

    def forward(self, x):
        out = self.act1(F.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.act2(F.relu(out))


class ResNet20OmniShiftV2(nn.Module):
    """ResNet-20 fully multiply-free: SparseShift + PoT-BN + PoT-Act + EWGS."""

    def __init__(self, num_classes=10, in_channels=3,
                 sparse_mode="fixed", sparsity_ratio=0.5,
                 bn_warmup=30, ewgs_lambda=EWGS_LAMBDA,
                 act_levels=8, act_alpha_init=4.0):
        super().__init__()
        self.sparse_mode    = sparse_mode
        self.sparsity_ratio = sparsity_ratio
        self.bn_warmup      = bn_warmup
        self.ewgs_lambda    = ewgs_lambda
        self.act_levels     = act_levels
        self.in_planes      = 16

        self.conv1    = nn.Conv2d(in_channels, 16, 3, stride=1, padding=1, bias=False)
        self.bn1      = PoTBatchNorm2dEWGS(16, use_pot_after_epoch=bn_warmup,
                                           ewgs_lambda=ewgs_lambda)
        self.act_stem = PoTActivationEWGS(n_levels=act_levels, alpha_init=act_alpha_init,
                                          use_pot_after_epoch=bn_warmup,
                                          ewgs_lambda=ewgs_lambda)

        self.stage1 = self._make_stage(16, 3, stride=1,
                                        act_levels=act_levels,
                                        act_alpha_init=act_alpha_init)
        self.stage2 = self._make_stage(32, 3, stride=2,
                                        act_levels=act_levels,
                                        act_alpha_init=act_alpha_init)
        self.stage3 = self._make_stage(64, 3, stride=2,
                                        act_levels=act_levels,
                                        act_alpha_init=act_alpha_init)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Linear(64, num_classes)
        self._init_weights()

    def _make_stage(self, planes, num_blocks, stride,
                    act_levels=8, act_alpha_init=4.0):
        strides = [stride] + [1] * (num_blocks - 1)
        layers  = []
        for s in strides:
            layers.append(_BasicBlock(
                self.in_planes, planes, stride=s,
                sparse_mode=self.sparse_mode,
                sparsity_ratio=self.sparsity_ratio,
                bn_warmup=self.bn_warmup,
                ewgs_lambda=self.ewgs_lambda,
                act_levels=act_levels,
                act_alpha_init=act_alpha_init,
            ))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, PoTBatchNorm2dEWGS)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.act_stem(F.relu(self.bn1(self.conv1(x))))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.avgpool(out).flatten(1)
        return self.fc(out)

    @torch.no_grad()
    def get_global_sparsity(self) -> float:
        sparsities = [m.get_actual_sparsity()
                      for m in self.modules()
                      if isinstance(m, SparseShiftConv2dEWGS)]
        return sum(sparsities) / len(sparsities) if sparsities else 0.0


def build_model(model_name: str, num_classes: int,
                in_channels: int = 3) -> ResNet20OmniShiftV2:
    """Build a Phase 7 (OmniShift v2) model by config name.

    Supported names:
      omnishift_v2_fixed50
      omnishift_v2_learnable
    """
    configs = {
        "omnishift_v2_fixed50": dict(
            sparse_mode="fixed", sparsity_ratio=0.5,
            bn_warmup=30, ewgs_lambda=EWGS_LAMBDA,
            act_levels=8, act_alpha_init=4.0),
        "omnishift_v2_learnable": dict(
            sparse_mode="learnable", sparsity_ratio=0.5,
            bn_warmup=30, ewgs_lambda=EWGS_LAMBDA,
            act_levels=8, act_alpha_init=4.0),
    }
    if model_name not in configs:
        raise ValueError(f"Unknown Phase 7 config: {model_name!r}. "
                         f"Valid: {list(configs)}")
    return ResNet20OmniShiftV2(num_classes=num_classes, in_channels=in_channels,
                               **configs[model_name])
