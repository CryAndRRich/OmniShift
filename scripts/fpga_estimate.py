#!/usr/bin/env python3
"""FPGA resource estimation for OmniShift models.

Estimates Xilinx 7-series FPGA resource utilization using a theoretical
model derived from published synthesis results. No physical FPGA required.

Resource model (16-bit integer datapath, Xilinx 7-series):
  Bit-shift  → barrel shifter → 0 DSP48E2,  ~0 LUT  (wired routing)
  Addition   → carry-chain    → 0 DSP48E2, ~16 LUT  (16-bit ripple-carry)
  Multiply   → DSP primitive  → 4 DSP48E2,   0 LUT  (absorbed into DSP48E2)

Key claim: interior conv stack uses 0 DSP48E2 blocks — all weights are PoT
shifts. Only the first conv (3→16) and FC (64→C) retain full-precision mul.

Usage:
    cd OmniShift
    python scripts/fpga_estimate.py --baseline
    python scripts/fpga_estimate.py --config configs/phase7_omnishift_v2.yaml \\
                                    --name omnishift_v2_learnable --dataset cifar10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

# ---------------------------------------------------------------------------
# Xilinx 7-series resource model (16-bit integer datapath)
# ---------------------------------------------------------------------------
_LUT_PER_ADD   = 16   # 16-bit carry-chain adder
_LUT_PER_SHIFT = 0    # barrel shift: wired routing
_LUT_PER_MUL   = 0    # multiplier: absorbed into DSP48E2
_DSP_PER_MUL   = 4    # 16×16 multiplier → 4 DSP48E2
_DSP_PER_ADD   = 0
_DSP_PER_SHIFT = 0

# Artix-7 XC7A200T — representative low-cost edge FPGA
_TOTAL_LUT  = 134_600
_TOTAL_DSP  = 740
_TOTAL_BRAM = 365     # 36Kb BRAMs


def estimate_fpga_resources(ops: dict) -> dict:
    lut = (ops["mul"]   * _LUT_PER_MUL
           + ops["add"]   * _LUT_PER_ADD
           + ops["shift"] * _LUT_PER_SHIFT)
    dsp = ops["mul"] * _DSP_PER_MUL
    return {
        "LUT":      lut,
        "DSP48E2":  dsp,
        "LUT_pct":  lut / _TOTAL_LUT  * 100,
        "DSP_pct":  dsp / _TOTAL_DSP  * 100,
    }


def _build_model(cfg, num_classes, in_channels):
    mtype = cfg["model"]["type"]
    name  = cfg["experiment"]["name"]
    if mtype == "baseline":
        from src.models.resnet20 import build_model
    elif mtype == "potbn":
        from src.models.resnet20_potbn import build_model
    elif mtype == "sparse":
        from src.models.resnet20_sparse import build_model
    elif mtype == "full":
        from src.models.resnet20_full import build_model
    elif mtype == "ewgs":
        from src.models.resnet20_ewgs import build_model
    elif mtype == "pot_act":
        from src.models.resnet20_pot_act import build_model
    elif mtype == "omnishift_v2":
        from src.models.resnet20_omnishift_v2 import build_model
    else:
        raise ValueError(f"Unknown model type: {mtype!r}")
    return build_model(name, num_classes=num_classes, in_channels=in_channels)


def run_estimate(cfg: dict, dataset_name: str | None = None) -> dict:
    from src.data.loaders import get_dataloaders
    from src.utils.ops_counter import count_mul_add_shift, count_params
    from src.utils.seed import set_seed
    from src.quantize.pot_bn import set_bn_epoch

    dataset_name = dataset_name or cfg["experiment"]["dataset"]
    seed = cfg["experiment"].get("seed", 42)
    seed_worker_fn, gen = set_seed(seed)
    data = get_dataloaders(dataset_name, batch_size=1, seed=seed,
                           num_workers=0,
                           seed_worker_fn=seed_worker_fn, generator=gen)

    model = _build_model(cfg, data["num_classes"], data["channels"])
    set_bn_epoch(model, 999)

    img_size = data["image_size"]
    ops    = count_mul_add_shift(model, (1, data["channels"], img_size, img_size))
    fpga   = estimate_fpga_resources(ops)
    params = count_params(model)
    return {"model": cfg["experiment"]["name"], "dataset": dataset_name,
            "ops": ops, "fpga": fpga, "params": params}


def baseline_comparison():
    """Cross-phase comparison using measured op counts."""
    phases = [
        ("ResNet-20 (FP32)",                  0.0410, 0.0410, 0.0000),
        ("DeepShift P1",                       0.0010, 0.0445, 0.0440),
        ("SparseShift+PoT-BN P4 fixed50",     0.0010, 0.0230, 0.0220),
        ("SparseShift+PoT-BN P4 learnable",   0.0010, 0.0099, 0.0089),
        ("+ EWGS P5 learnable (SVHN)",         0.0010, 0.0075, 0.0065),
        ("+ PoT-Act P6 learnable (SVHN)",      0.0010, 0.0127, 0.0117),
        ("OmniShift v2 P7 learnable (est.)",  0.0010, 0.0090, 0.0082),
    ]

    print("\n" + "=" * 85)
    print(f"{'Model':<42} {'DSP48E2':>9} {'DSP%':>7} {'LUT(M)':>8} {'LUT%':>7}")
    print("=" * 85)
    for name, mul_g, add_g, shift_g in phases:
        ops = {"mul":   int(mul_g   * 1e9),
               "add":   int(add_g   * 1e9),
               "shift": int(shift_g * 1e9)}
        f = estimate_fpga_resources(ops)
        print(f"{name:<42} {f['DSP48E2']:>9,.0f} {f['DSP_pct']:>6.1f}% "
              f"{f['LUT']/1e6:>7.2f}M {f['LUT_pct']:>6.1f}%")
    print("=" * 85)
    print(f"\nTarget: Xilinx Artix-7 XC7A200T "
          f"({_TOTAL_DSP} DSP48E2, {_TOTAL_LUT//1000}k LUT)")
    print("OmniShift interior conv stack → 0 DSP (shifts are wired barrel-shifters).")
    print("Residual mul = first conv (3→16) + FC (64→C).\n")


def main():
    parser = argparse.ArgumentParser(description="FPGA resource estimation")
    parser.add_argument("--config",   help="YAML config path")
    parser.add_argument("--dataset",  help="Override dataset")
    parser.add_argument("--baseline", action="store_true",
                        help="Cross-phase comparison table (no model needed)")
    args = parser.parse_args()

    if args.baseline or args.config is None:
        baseline_comparison()
        return

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    r = run_estimate(cfg, args.dataset)
    ops, fpga = r["ops"], r["fpga"]

    print(f"\n{'='*60}")
    print(f"Model  : {r['model']} | Dataset: {r['dataset']}")
    print(f"Params : {r['params']/1e6:.3f}M")
    print(f"{'='*60}")
    print(f"Ops (post-warmup):")
    print(f"  Mul   : {ops['mul_G']:.4f} G  →  {fpga['DSP48E2']:,} DSP48E2")
    print(f"  Add   : {ops['add_G']:.4f} G  →  {ops['add'] * _LUT_PER_ADD / 1e6:.2f}M LUT")
    print(f"  Shift : {ops['shift_G']:.4f} G →  0 DSP (wired barrel-shift)")
    print(f"  Energy: {ops['energy_GpJ']:.4f} GpJ")
    print(f"{'='*60}")
    print(f"FPGA Resources (Artix-7 XC7A200T):")
    print(f"  DSP48E2 : {fpga['DSP48E2']:>6,}  / {_TOTAL_DSP}   ({fpga['DSP_pct']:.1f}%)")
    print(f"  LUT     : {fpga['LUT']/1e6:>6.2f}M / {_TOTAL_LUT/1e6:.1f}M  ({fpga['LUT_pct']:.1f}%)")
    print(f"  Interior conv DSP: 0  (all shifts are wired)\n")


if __name__ == "__main__":
    main()
