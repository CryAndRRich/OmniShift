"""ResNet-20 with SparseShift + PoT-BN + EWGS gradient estimator (Phase 5).

Identical forward pass / inference energy to Phase 4 (resnet20_full.py).
Only the backward pass differs: STE replaced by EWGS in all quantizers,
improving gradient flow through PoT and sparsity rounding boundaries.

EWGS: g = g_STE ⊙ (1 + λ · sign(g_STE) ⊙ sign(w − Q(w))), λ=0.02
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.quantize.ewgs import (
    SparseShiftConv2dEWGS,
    PoTBatchNorm2dEWGS,
    EWGS_LAMBDA,
)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,
                 sparse_mode="fixed", sparsity_ratio=0.5,
                 bn_warmup=30, ewgs_lambda=EWGS_LAMBDA):
        super().__init__()

        def make_conv(in_c, out_c, k, s, p):
            return SparseShiftConv2dEWGS(
                in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False,
                sparse_mode=sparse_mode, sparsity_ratio=sparsity_ratio,
                ewgs_lambda=ewgs_lambda)

        def make_bn(c):
            return PoTBatchNorm2dEWGS(c, use_pot_after_epoch=bn_warmup,
                                      ewgs_lambda=ewgs_lambda)

        self.conv1 = make_conv(in_planes, planes, 3, stride, 1)
        self.bn1   = make_bn(planes)
        self.conv2 = make_conv(planes, planes, 3, 1, 1)
        self.bn2   = make_bn(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                make_conv(in_planes, planes, 1, stride, 0),
                make_bn(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet20EWGS(nn.Module):
    """ResNet-20: SparseShiftConv2dEWGS + PoTBatchNorm2dEWGS (Phase 5).

    Forward pass is byte-for-byte identical to ResNet20SparsePoTBN (Phase 4).
    Energy numbers are therefore identical; the improvement shows up as
    higher accuracy from better-behaved gradients during training.
    """

    def __init__(self, num_classes=10, in_channels=3,
                 sparse_mode="fixed", sparsity_ratio=0.5,
                 bn_warmup=30, ewgs_lambda=EWGS_LAMBDA):
        super().__init__()
        self.sparse_mode   = sparse_mode
        self.sparsity_ratio = sparsity_ratio
        self.bn_warmup     = bn_warmup
        self.ewgs_lambda   = ewgs_lambda
        self.in_planes     = 16

        def make_bn(c):
            return PoTBatchNorm2dEWGS(c, use_pot_after_epoch=bn_warmup,
                                      ewgs_lambda=ewgs_lambda)

        # Stem: standard conv + EWGS PoT-BN
        self.conv1 = nn.Conv2d(in_channels, 16, 3, stride=1, padding=1, bias=False)
        self.bn1   = make_bn(16)

        self.stage1 = self._make_stage(16, 3, stride=1)
        self.stage2 = self._make_stage(32, 3, stride=2)
        self.stage3 = self._make_stage(64, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Linear(64, num_classes)
        self._init_weights()

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers  = []
        for s in strides:
            layers.append(_BasicBlock(
                self.in_planes, planes, stride=s,
                sparse_mode=self.sparse_mode,
                sparsity_ratio=self.sparsity_ratio,
                bn_warmup=self.bn_warmup,
                ewgs_lambda=self.ewgs_lambda,
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
        out = F.relu(self.bn1(self.conv1(x)))
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
                in_channels: int = 3) -> ResNet20EWGS:
    """Build a Phase 5 (EWGS) model by config name.

    Supported names:
      sparseshift_fixed50_potbn_w30_ewgs
      sparseshift_learnable_potbn_w30_ewgs
    """
    configs = {
        "sparseshift_fixed50_potbn_w30_ewgs": dict(
            sparse_mode="fixed", sparsity_ratio=0.5,
            bn_warmup=30, ewgs_lambda=EWGS_LAMBDA),
        "sparseshift_learnable_potbn_w30_ewgs": dict(
            sparse_mode="learnable", sparsity_ratio=0.5,
            bn_warmup=30, ewgs_lambda=EWGS_LAMBDA),
    }
    if model_name not in configs:
        raise ValueError(f"Unknown Phase 5 config: {model_name!r}. "
                         f"Valid: {list(configs)}")
    return ResNet20EWGS(num_classes=num_classes, in_channels=in_channels,
                        **configs[model_name])
