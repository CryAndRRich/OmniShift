"""Parameterized CIFAR ResNet (20/32/56/110) and VGG-11 for OmniShift.

Each backbone accepts factory callables (make_conv, make_bn, make_act)
from src.quantize.wrap.make_factories(), so quantization is fully decoupled
from architecture.

Supported backbones:
    resnet20  — 3 stages × [3, 3, 3] blocks, 16/32/64 channels
    resnet32  — 3 stages × [5, 5, 5] blocks
    resnet56  — 3 stages × [9, 9, 9] blocks
    resnet110 — 3 stages × [18, 18, 18] blocks
    vgg11     — VGG-11 adapted for 32×32 input, 4 MaxPool, AdaptiveAvgPool
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.quantize.sparse_shift import SparseShiftConv2d
from src.quantize.ewgs import SparseShiftConv2dEWGS

_SPARSE_TYPES = (SparseShiftConv2d, SparseShiftConv2dEWGS)


# ─── ResNet ──────────────────────────────────────────────────────────────────

class _BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride, make_conv, make_bn, make_act):
        super().__init__()
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


class ResNetCIFAR(nn.Module):
    """Parameterized CIFAR ResNet with pluggable quantization factories.

    Stem conv (3→16) always uses standard nn.Conv2d — first layer stays FP.
    Interior convs use make_conv. FC stays standard nn.Linear.
    """

    STAGE_CONFIGS = {
        'resnet20':  [3, 3, 3],
        'resnet32':  [5, 5, 5],
        'resnet56':  [9, 9, 9],
        'resnet110': [18, 18, 18],
    }

    def __init__(self, blocks_per_stage, make_conv, make_bn, make_act,
                 num_classes=10, in_channels=3):
        super().__init__()
        self._in_planes = 16

        self.conv1    = nn.Conv2d(in_channels, 16, 3, stride=1, padding=1, bias=False)
        self.bn1      = make_bn(16)
        self.act_stem = make_act()

        self.stage1 = self._make_stage(16,  blocks_per_stage[0], 1, make_conv, make_bn, make_act)
        self.stage2 = self._make_stage(32,  blocks_per_stage[1], 2, make_conv, make_bn, make_act)
        self.stage3 = self._make_stage(64,  blocks_per_stage[2], 2, make_conv, make_bn, make_act)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Linear(64, num_classes)

        self._init_weights()

    def _make_stage(self, planes, n_blocks, stride, make_conv, make_bn, make_act):
        strides = [stride] + [1] * (n_blocks - 1)
        layers  = []
        for s in strides:
            layers.append(_BasicBlock(self._in_planes, planes, s,
                                      make_conv, make_bn, make_act))
            self._in_planes = planes
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
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
        mods = [m for m in self.modules() if isinstance(m, _SPARSE_TYPES)]
        return sum(m.get_actual_sparsity() for m in mods) / len(mods) if mods else 0.0


# ─── VGG ─────────────────────────────────────────────────────────────────────

class VGG_CIFAR(nn.Module):
    """VGG-11 for 32×32 CIFAR input with pluggable quantization factories.

    Architecture: 4 MaxPool(2,2) → 32→16→8→4→2, then AdaptiveAvgPool(1).
    First conv stays standard nn.Conv2d (FP). Remaining convs use make_conv.
    """

    _CFG = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M']

    def __init__(self, make_conv, make_bn, make_act, num_classes=10, in_channels=3):
        super().__init__()
        layers = []
        in_c   = in_channels
        first  = True
        for v in self._CFG:
            if v == 'M':
                layers.append(nn.MaxPool2d(2, 2))
            else:
                conv = (nn.Conv2d(in_c, v, 3, 1, 1, bias=False)
                        if first else make_conv(in_c, v, 3, 1, 1))
                first = False
                layers += [conv, make_bn(v), nn.ReLU(inplace=False), make_act()]
                in_c = v

        self.features = nn.Sequential(*layers)
        self.avgpool  = nn.AdaptiveAvgPool2d(1)
        self.fc       = nn.Linear(512, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out).flatten(1)
        return self.fc(out)

    @torch.no_grad()
    def get_global_sparsity(self) -> float:
        mods = [m for m in self.modules() if isinstance(m, _SPARSE_TYPES)]
        return sum(m.get_actual_sparsity() for m in mods) / len(mods) if mods else 0.0


# ─── Builder ─────────────────────────────────────────────────────────────────

SUPPORTED_BACKBONES = list(ResNetCIFAR.STAGE_CONFIGS) + ['vgg11']


def build_model(backbone: str, quant_cfg: dict,
                num_classes: int, in_channels: int = 3) -> nn.Module:
    """Build backbone with OmniShift quantization applied.

    Args:
        backbone:    'resnet20' | 'resnet32' | 'resnet56' | 'resnet110' | 'vgg11'
        quant_cfg:   dict from config YAML 'quantize' section
        num_classes: number of output classes
        in_channels: input channels (3 for RGB)
    """
    from src.quantize.wrap import make_factories
    make_conv, make_bn, make_act = make_factories(quant_cfg)

    if backbone in ResNetCIFAR.STAGE_CONFIGS:
        blocks = ResNetCIFAR.STAGE_CONFIGS[backbone]
        return ResNetCIFAR(blocks, make_conv, make_bn, make_act,
                           num_classes=num_classes, in_channels=in_channels)
    elif backbone == 'vgg11':
        return VGG_CIFAR(make_conv, make_bn, make_act,
                         num_classes=num_classes, in_channels=in_channels)
    else:
        raise ValueError(f"Unknown backbone: {backbone!r}. "
                         f"Supported: {SUPPORTED_BACKBONES}")
