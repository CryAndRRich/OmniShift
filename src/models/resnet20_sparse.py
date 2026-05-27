"""ResNet-20 with Sparse Shift convolutions (Phase 4).

Interior convs use SparseShiftConv2d: W ∈ {0, ±2^p}.
BN remains standard nn.BatchNorm2d.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.quantize.sparse_shift import SparseShiftConv2d


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,
                 sparse_mode="fixed", sparsity_ratio=0.5):
        super().__init__()

        def make_conv(in_c, out_c, k, s, p):
            return SparseShiftConv2d(in_c, out_c, kernel_size=k, stride=s,
                                     padding=p, bias=False,
                                     sparse_mode=sparse_mode,
                                     sparsity_ratio=sparsity_ratio)

        self.conv1 = make_conv(in_planes, planes, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = make_conv(planes, planes, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                make_conv(in_planes, planes, 1, stride, 0),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet20Sparse(nn.Module):
    """ResNet-20 with SparseShiftConv2d (W ∈ {0, ±2^p}) and std BN."""

    def __init__(self, num_classes=10, in_channels=3,
                 sparse_mode="fixed", sparsity_ratio=0.5):
        super().__init__()
        self.sparse_mode = sparse_mode
        self.sparsity_ratio = sparsity_ratio
        self.in_planes = 16

        self.conv1 = nn.Conv2d(in_channels, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.stage1 = self._make_stage(16, 3, stride=1, sparse_mode=sparse_mode,
                                       sparsity_ratio=sparsity_ratio)
        self.stage2 = self._make_stage(32, 3, stride=2, sparse_mode=sparse_mode,
                                       sparsity_ratio=sparsity_ratio)
        self.stage3 = self._make_stage(64, 3, stride=2, sparse_mode=sparse_mode,
                                       sparsity_ratio=sparsity_ratio)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        self._init_weights()

    def _make_stage(self, planes, num_blocks, stride, sparse_mode, sparsity_ratio):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(_BasicBlock(self.in_planes, planes, stride=s,
                                      sparse_mode=sparse_mode,
                                      sparsity_ratio=sparsity_ratio))
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

    @torch.no_grad()
    def get_global_sparsity(self) -> float:
        """Average sparsity across all SparseShiftConv2d layers."""
        sparsities = [m.get_actual_sparsity()
                      for m in self.modules()
                      if isinstance(m, SparseShiftConv2d)]
        return sum(sparsities) / len(sparsities) if sparsities else 0.0


def parse_config_name(model_name: str) -> dict:
    """Parse Phase 4 config names into constructor kwargs.

    'sparseshift_fixed50'   → fixed 50% sparsity
    'sparseshift_learnable' → learnable threshold + L1 reg
    """
    if model_name == "sparseshift_fixed50":
        return dict(sparse_mode="fixed", sparsity_ratio=0.5)
    if model_name == "sparseshift_learnable":
        return dict(sparse_mode="learnable", sparsity_ratio=0.5)
    raise ValueError(f"Unknown Phase 4 config: {model_name!r}")


def build_model(model_name: str, num_classes: int, in_channels: int = 3) -> ResNet20Sparse:
    """Build a Phase 4 model by config name."""
    cfg = parse_config_name(model_name)
    return ResNet20Sparse(num_classes=num_classes, in_channels=in_channels, **cfg)
