"""ResNet-20 with SparseShift + PoT-BN + PoT Activation Quantization (Phase 6).

Extends Phase 4 by inserting PoTActivation after every ReLU in the residual
stack. Post-ReLU activations are rounded to {0} ∪ {2^p : p_min ≤ p ≤ p_max},
where p_max = floor(log2(α)) and α is a learnable per-layer clip parameter.

With SparseShift weights (PoT) + PoT-BN scales (PoT) + PoT activations (PoT),
every multiply in the conv stack becomes a bit-shift → fully multiply-free.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.quantize.sparse_shift import SparseShiftConv2d
from src.quantize.pot_bn import PoTBatchNorm2d
from src.quantize.pot_act import PoTActivation


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,
                 sparse_mode="fixed", sparsity_ratio=0.5,
                 bn_warmup=30, act_levels=8, act_alpha_init=4.0):
        super().__init__()

        def make_conv(in_c, out_c, k, s, p):
            return SparseShiftConv2d(in_c, out_c, kernel_size=k, stride=s,
                                     padding=p, bias=False,
                                     sparse_mode=sparse_mode,
                                     sparsity_ratio=sparsity_ratio)

        def make_bn(c):
            return PoTBatchNorm2d(c, use_pot_after_epoch=bn_warmup)

        def make_act():
            return PoTActivation(n_levels=act_levels, alpha_init=act_alpha_init,
                                 use_pot_after_epoch=bn_warmup)

        self.conv1 = make_conv(in_planes, planes, 3, stride, 1)
        self.bn1   = make_bn(planes)
        self.act1  = make_act()   # after first relu in block
        self.conv2 = make_conv(planes, planes, 3, 1, 1)
        self.bn2   = make_bn(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                make_conv(in_planes, planes, 1, stride, 0),
                make_bn(planes),
            )

        self.act2 = make_act()    # after block output relu

    def forward(self, x):
        out = self.act1(F.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.act2(F.relu(out))


class ResNet20PoTAct(nn.Module):
    """ResNet-20: SparseShiftConv2d + PoTBatchNorm2d + PoTActivation (Phase 6).

    Fully multiply-free forward pass: weights (PoT), BN scales (PoT),
    and activations (PoT) — all downstream multiplications become bit-shifts.
    """

    def __init__(self, num_classes=10, in_channels=3,
                 sparse_mode="fixed", sparsity_ratio=0.5,
                 bn_warmup=30, act_levels=8, act_alpha_init=4.0):
        super().__init__()
        self.sparse_mode    = sparse_mode
        self.sparsity_ratio = sparsity_ratio
        self.bn_warmup      = bn_warmup
        self.act_levels     = act_levels
        self.in_planes      = 16

        # Stem: standard conv + PoT-BN + PoT activation
        self.conv1    = nn.Conv2d(in_channels, 16, 3, stride=1, padding=1, bias=False)
        self.bn1      = PoTBatchNorm2d(16, use_pot_after_epoch=bn_warmup)
        self.act_stem = PoTActivation(n_levels=act_levels, alpha_init=act_alpha_init,
                                      use_pot_after_epoch=bn_warmup)

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
                act_levels=act_levels,
                act_alpha_init=act_alpha_init,
            ))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, PoTBatchNorm2d)):
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
                      if isinstance(m, SparseShiftConv2d)]
        return sum(sparsities) / len(sparsities) if sparsities else 0.0


def build_model(model_name: str, num_classes: int,
                in_channels: int = 3) -> ResNet20PoTAct:
    """Build a Phase 6 (PoT Activation) model by config name.

    Supported names:
      sparseshift_fixed50_potbn_w30_act
      sparseshift_learnable_potbn_w30_act
    """
    configs = {
        "sparseshift_fixed50_potbn_w30_act": dict(
            sparse_mode="fixed", sparsity_ratio=0.5,
            bn_warmup=30, act_levels=8, act_alpha_init=4.0),
        "sparseshift_learnable_potbn_w30_act": dict(
            sparse_mode="learnable", sparsity_ratio=0.5,
            bn_warmup=30, act_levels=8, act_alpha_init=4.0),
    }
    if model_name not in configs:
        raise ValueError(f"Unknown Phase 6 config: {model_name!r}. "
                         f"Valid: {list(configs)}")
    return ResNet20PoTAct(num_classes=num_classes, in_channels=in_channels,
                          **configs[model_name])
