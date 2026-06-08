from typing import Optional
import torch
import torch.nn as nn

_MUL_PJ = 3.7
_ADD_PJ = 0.9
_SHIFT_PJ = 0.13

def _get_padding_stride(module):
    pad = getattr(module, "padding", 0)
    stride = getattr(module, "stride", 1)
    if isinstance(pad, (list, tuple)):
        pad = pad[0]
    if isinstance(stride, (list, tuple)):
        stride = stride[0]
    return pad, stride

def count_mul_add_shift(
    model: nn.Module,
    input_size: tuple = (1, 3, 32, 32),
    sparsity: Optional[float] = None,
) -> dict:
    from src.quantize.sparse_shift import SparseShiftConv2d
    from src.quantize.s3shift import S3ShiftConv2d
    from src.quantize.aptq_ternary import APTQTernaryConv2d
    from src.quantize.ewgs import (SparseShiftConv2dEWGS,
                                              PoTBatchNorm2dEWGS,
                                              PoTActivationEWGS)
    from src.quantize.pot_bn import PoTBatchNorm2d
    from src.quantize.pot_act import PoTActivation
    from src.quantize.shift import ShiftConv2d
    from src.quantize.fogzo import FogzoShiftConv2d
    from src.quantize.apot import APoTConv2d
    from src.quantize.xnor import XNORConv2d
    from src.quantize.denseshift import DenseShiftConv2d

    _sparse_types = (SparseShiftConv2d, SparseShiftConv2dEWGS, S3ShiftConv2d, APTQTernaryConv2d)
    _dense_shift_types = (ShiftConv2d, FogzoShiftConv2d, APoTConv2d, DenseShiftConv2d)
    _binary_types = (XNORConv2d,)
    _pot_bn_types = (PoTBatchNorm2d, PoTBatchNorm2dEWGS)
    _pot_act_types = (PoTActivation, PoTActivationEWGS)

    sparse_mods = [m for m in model.modules() if isinstance(m, _sparse_types)]
    if sparse_mods and sparsity is None:
        sparsity = sum(m.get_actual_sparsity() for m in sparse_mods) / len(sparse_mods)
    sparsity = sparsity or 0.0
    nonzero_ratio = 1.0 - sparsity

    counts = {"mul": 0, "add": 0, "shift": 0}
    hooks  = []

    def _conv_macs(mod, inp):
        x = inp[0]
        _, _, H_in, W_in = x.shape
        pad, stride = _get_padding_stride(mod)

        if hasattr(mod, "out_channels"):
            C_out = mod.out_channels
            C_in = mod.in_channels
            kH, kW = mod.kernel_size if isinstance(mod.kernel_size, tuple) \
                     else (mod.kernel_size, mod.kernel_size)
        else:

            C_out = mod.weight.shape[0]
            C_in = mod.weight.shape[1]
            kH, kW = mod.weight.shape[2], mod.weight.shape[3]

        H_out = (H_in + 2 * pad - kH) // stride + 1
        W_out = (W_in + 2 * pad - kW) // stride + 1
        return C_in * C_out * kH * kW * H_out * W_out

    for m in model.modules():
        if isinstance(m, _sparse_types):
            def sparse_hook(mod, inp, out, _nr=nonzero_ratio):
                macs = _conv_macs(mod, inp)
                counts["shift"] += int(macs * _nr)
                counts["add"] += int(macs * _nr)
            hooks.append(m.register_forward_hook(sparse_hook))

        elif isinstance(m, _dense_shift_types):
            def dense_shift_hook(mod, inp, out):
                macs = _conv_macs(mod, inp)
                counts["shift"] += macs
                counts["add"] += macs
            hooks.append(m.register_forward_hook(dense_shift_hook))

        elif isinstance(m, _binary_types):
            def binary_hook(mod, inp, out):
                macs = _conv_macs(mod, inp)
                counts["add"] += macs
            hooks.append(m.register_forward_hook(binary_hook))

        elif isinstance(m, nn.Conv2d):
            def conv_hook(mod, inp, out):
                macs = _conv_macs(mod, inp)
                counts["mul"] += macs
                counts["add"] += macs
            hooks.append(m.register_forward_hook(conv_hook))

        elif isinstance(m, nn.Linear):
            def linear_hook(mod, inp, out):
                x = inp[0]
                macs = x.shape[-1] * mod.out_features
                counts["mul"] += macs
                counts["add"] += macs
            hooks.append(m.register_forward_hook(linear_hook))

        elif isinstance(m, _pot_bn_types):
            def pot_bn_hook(mod, inp, out):
                x = inp[0]
                n = x.shape[1] * x.shape[2] * x.shape[3]
                if mod._should_use_pot():
                    counts["shift"] += n
                else:
                    counts["mul"] += n
                counts["add"] += n
            hooks.append(m.register_forward_hook(pot_bn_hook))

        elif isinstance(m, nn.BatchNorm2d):
            def bn_hook(mod, inp, out):
                x = inp[0]
                n = x.shape[1] * x.shape[2] * x.shape[3]
                counts["mul"] += n
                counts["add"] += n
            hooks.append(m.register_forward_hook(bn_hook))

        elif isinstance(m, _pot_act_types):
            def pot_act_hook(mod, inp, out):
                x = inp[0]
                if hasattr(mod, "_should_use_pot") and mod._should_use_pot():
                    counts["shift"] += x.numel()
            hooks.append(m.register_forward_hook(pot_act_hook))

        elif isinstance(m, nn.AdaptiveAvgPool2d):
            def avgpool_hook(mod, inp, out):
                x = inp[0]
                counts["add"] += x.shape[1] * x.shape[2] * x.shape[3]
            hooks.append(m.register_forward_hook(avgpool_hook))

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    dummy  = torch.zeros(input_size, device=device)
    with torch.no_grad():
        model(dummy)
    if was_training:
        model.train()

    for h in hooks:
        h.remove()

    energy_pj = (_MUL_PJ * counts["mul"] + _ADD_PJ * counts["add"] + _SHIFT_PJ * counts["shift"])

    return {
        "mul": counts["mul"],
        "add": counts["add"],
        "shift": counts["shift"],
        "mul_G": counts["mul"] / 1e9,
        "add_G": counts["add"] / 1e9,
        "shift_G": counts["shift"] / 1e9,
        "energy_pJ": energy_pj,
        "energy_GpJ": energy_pj / 1e9,
        "sparsity": sparsity,
    }

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)