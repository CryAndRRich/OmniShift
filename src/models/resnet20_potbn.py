"""ResNet-20 with PoT-BatchNorm (Phase 3).

Supports configurable conv type (mul or shift) and BN type (std or PoT),
with optional warmup epochs before PoT-BN activates.
"""

import torch.nn as nn
import torch.nn.functional as F

from src.quantize.shift import ShiftConv2d
from src.quantize.pot_bn import PoTBatchNorm2d


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,
                 conv_type="shift", bn_type="std", bn_warmup=0):
        super().__init__()
        Conv = ShiftConv2d if conv_type == "shift" else nn.Conv2d

        def make_bn(c):
            return (PoTBatchNorm2d(c, use_pot_after_epoch=bn_warmup)
                    if bn_type == "pot" else nn.BatchNorm2d(c))

        self.conv1 = Conv(in_planes, planes, kernel_size=3, stride=stride,
                          padding=1, bias=False)
        self.bn1 = make_bn(planes)
        self.conv2 = Conv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = make_bn(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                Conv(in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                make_bn(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet20PoTBN(nn.Module):
    """ResNet-20 with configurable conv type and PoT-BN.

    conv_type: 'mul' (nn.Conv2d) or 'shift' (ShiftConv2d / DeepShift)
    bn_type:   'std' (nn.BatchNorm2d) or 'pot' (PoTBatchNorm2d)
    bn_warmup: epochs of standard BN warmup before switching to PoT-BN
    """

    def __init__(self, num_classes=10, in_channels=3,
                 conv_type="shift", bn_type="std", bn_warmup=0):
        super().__init__()
        self.conv_type = conv_type
        self.bn_type = bn_type
        self.bn_warmup = bn_warmup
        self.in_planes = 16

        def make_bn(c):
            return (PoTBatchNorm2d(c, use_pot_after_epoch=bn_warmup)
                    if bn_type == "pot" else nn.BatchNorm2d(c))

        self.conv1 = nn.Conv2d(in_channels, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = make_bn(16)

        self.stage1 = self._make_stage(16, 3, stride=1, conv_type=conv_type,
                                       bn_type=bn_type)
        self.stage2 = self._make_stage(32, 3, stride=2, conv_type=conv_type,
                                       bn_type=bn_type)
        self.stage3 = self._make_stage(64, 3, stride=2, conv_type=conv_type,
                                       bn_type=bn_type)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        self._init_weights()

    def _make_stage(self, planes, num_blocks, stride, conv_type, bn_type):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(_BasicBlock(self.in_planes, planes, stride=s,
                                      conv_type=conv_type, bn_type=bn_type,
                                      bn_warmup=self.bn_warmup))
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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.avgpool(out).flatten(1)
        return self.fc(out)


def parse_config_name(model_name: str) -> dict:
    """Parse Phase 3 config names into constructor kwargs.

    'resnet20_std'             → mul conv + std BN
    'deepshift_std'            → shift conv + std BN (= Phase 1 DeepShift)
    'deepshift_potbn'          → shift conv + PoT-BN (no warmup)
    'deepshift_potbn_warmupN'  → shift conv + PoT-BN with N-epoch warmup
    """
    if model_name == "resnet20_std":
        return dict(conv_type="mul", bn_type="std", bn_warmup=0)
    if model_name == "deepshift_std":
        return dict(conv_type="shift", bn_type="std", bn_warmup=0)
    if model_name == "deepshift_potbn":
        return dict(conv_type="shift", bn_type="pot", bn_warmup=0)
    if model_name.startswith("deepshift_potbn_warmup"):
        warmup = int(model_name.replace("deepshift_potbn_warmup", ""))
        return dict(conv_type="shift", bn_type="pot", bn_warmup=warmup)
    raise ValueError(f"Unknown Phase 3 config: {model_name!r}")


def build_model(model_name: str, num_classes: int, in_channels: int = 3) -> ResNet20PoTBN:
    """Build a Phase 3 model by config name."""
    cfg = parse_config_name(model_name)
    return ResNet20PoTBN(num_classes=num_classes, in_channels=in_channels, **cfg)
