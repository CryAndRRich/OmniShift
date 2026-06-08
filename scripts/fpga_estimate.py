import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

_LUT_PER_ADD = 16
_DSP_PER_MUL = 4
_TOTAL_LUT = 134600
_TOTAL_DSP = 740

def estimate_fpga_resources(ops: dict) -> dict:
    lut = ops["add"] * _LUT_PER_ADD
    dsp = ops["mul"] * _DSP_PER_MUL
    return {
        "LUT": lut,
        "DSP48E2": dsp,
        "LUT_pct": lut / _TOTAL_LUT * 100,
        "DSP_pct": dsp / _TOTAL_DSP * 100,
    }

def run_estimate(cfg: dict, dataset_name: str | None = None) -> dict:
    from src.data.loaders import get_dataloaders
    from src.utils.ops_counter import count_mul_add_shift, count_params
    from src.utils.seed import set_seed
    from src.quantize.pot_bn import set_bn_epoch
    from src.models.resnet_cifar import build_model

    dataset_name = dataset_name or cfg["experiment"].get("dataset", "cifar10")
    seed = cfg["experiment"].get("seed", 42)
    backbone = cfg["experiment"].get("backbone", "resnet20")
    method = cfg["experiment"].get("method", "omnishift")
    method_opts = cfg["experiment"].get("method_opts", {})

    seed_worker_fn, gen = set_seed(seed)
    data = get_dataloaders(dataset_name, batch_size=1, seed=seed,
                           num_workers=0, seed_worker_fn=seed_worker_fn, generator=gen)

    model = build_model(backbone, method, data["num_classes"], data["channels"],
                        **method_opts)
    set_bn_epoch(model, 999)

    img_size = data["image_size"]
    ops = count_mul_add_shift(model, (1, data["channels"], img_size, img_size))
    fpga = estimate_fpga_resources(ops)
    params = count_params(model)

    return {"backbone": backbone, "dataset": dataset_name,
            "ops": ops, "fpga": fpga, "params": params}

def main():
    parser = argparse.ArgumentParser(description="FPGA resource estimation")
    parser.add_argument("--config", help="YAML config path")
    parser.add_argument("--name", help="Override experiment.name")
    parser.add_argument("--dataset", help="Override dataset")
    parser.add_argument("--baseline", action="store_true", help="Cross-config comparison table")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.name:
        cfg["experiment"]["name"] = args.name

    r = run_estimate(cfg, args.dataset)
    ops, fpga = r["ops"], r["fpga"]

    print(f"\n{'='*60}")
    print(f"Backbone : {r['backbone']} | Dataset: {r['dataset']}")
    print(f"Params   : {r['params']/1e6:.3f}M")
    print(f"{'='*60}")
    print(f"Mul  : {ops['mul_G']:.4f}G  ->  {fpga['DSP48E2']:,} DSP48E2")
    print(f"Add  : {ops['add_G']:.4f}G  ->  {ops['add'] * _LUT_PER_ADD / 1e6:.2f}M LUT")
    print(f"Shift: {ops['shift_G']:.4f}G ->  0 DSP (wired)")
    print(f"Energy: {ops['energy_GpJ']:.4f} GpJ")
    print(f"{'='*60}")
    print(f"FPGA (Artix-7 XC7A200T):")
    print(f"  DSP48E2 : {fpga['DSP48E2']:>6,} / {_TOTAL_DSP}  ({fpga['DSP_pct']:.1f}%)")
    print(f"  LUT     : {fpga['LUT']/1e6:>6.2f}M / {_TOTAL_LUT/1e6:.1f}M  ({fpga['LUT_pct']:.1f}%)\n")

if __name__ == "__main__":
    main()