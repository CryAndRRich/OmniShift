import argparse
import json
import sys
from pathlib import Path

_DSP_PER_MUL = 4
_TOTAL_DSP   = 740
_TOTAL_LUT   = 134_600

METHOD_ORDER  = ["fp32", "deepshift", "apot", "denseshift", "s3shift", "fogzo", "aptq", "omnishift"]
DATASET_ORDER = ["cifar10", "svhn", "stl10"]
BACKBONE_ORDER = ["resnet20", "resnet56"]


def load_results(log_root: Path) -> list[dict]:
    rows = []
    for json_file in sorted(log_root.rglob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
            meta = data.get("meta", data)
            ops  = meta.get("final_ops") or {}
            rows.append({
                "file":     json_file.name,
                "name":     meta.get("run_name") or meta.get("model_name", "?"),
                "method":   meta.get("method", "?"),
                "backbone": meta.get("backbone", "?"),
                "dataset":  meta.get("dataset_name", "?"),
                "test_acc": meta.get("test_acc", float("nan")),
                "sparsity": meta.get("final_sparsity", 0.0),
                "energy":   ops.get("energy_GpJ", float("nan")),
                "params":   meta.get("n_params", 0),
                "mul_G":    ops.get("mul_G", 0.0),
                "add_G":    ops.get("add_G", 0.0),
                "shift_G":  ops.get("shift_G", 0.0),
                "mul_raw":  ops.get("mul", 0),
            })
        except Exception as e:
            print(f"[warn] {json_file}: {e}", file=sys.stderr)
    return rows


def print_table(rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: (r["dataset"], r["energy"]))
    hdr = (f"{'Name':<36} {'Backbone':<10} {'Dataset':<10} "
           f"{'TestAcc':>8} {'Sparsity':>10} {'Energy(GpJ)':>12} {'Params':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:<36} {r['backbone']:<10} {r['dataset']:<10} "
              f"{r['test_acc']:>8.4f} {r['sparsity']:>10.2%} "
              f"{r['energy']:>12.4f} {r['params']:>10,}")


def print_dsp_table(rows: list[dict]) -> None:
    print(f"=== FPGA Multiply Analysis - Artix-7 XC7A200T  "
          f"(total DSP={_TOTAL_DSP}, LUT={_TOTAL_LUT:,}) ===")
    print("    DSP demand = mul_raw x 4 (unrolled estimate; ratio vs fp32 is the key metric)")
    print()

    backbones = [b for b in BACKBONE_ORDER if any(r["backbone"] == b for r in rows)]
    methods   = [m for m in METHOD_ORDER   if any(r["method"]   == m for r in rows)]

    fp32_mul = {}
    for bb in backbones:
        ref = next((r for r in rows if r["method"] == "fp32"
                    and r["backbone"] == bb and r["dataset"] == "cifar10"), None)
        fp32_mul[bb] = ref["mul_raw"] if ref else 1

    col_w = 34
    print(f"{'Method':<14}", end="")
    for bb in backbones:
        print(f"  {bb+':':>{col_w}}", end="")
    print()
    sub_hdr = f"  {'mul_G':>8} {'DSP(xk)':>10} {'vs FP32':>9} {'source':>12}"
    print(f"{'':14}" + sub_hdr * len(backbones))
    print("-" * (14 + len(backbones) * (len(sub_hdr))))

    for method in methods:
        print(f"{method:<14}", end="")
        for bb in backbones:
            r = next((r for r in rows if r["method"] == method
                      and r["backbone"] == bb and r["dataset"] == "cifar10"), None)
            if r is None:
                print(f"  {'-':>8} {'-':>10} {'-':>9} {'-':>12}", end="")
                continue
            mul_raw  = r["mul_raw"]
            dsp_k    = mul_raw * _DSP_PER_MUL / 1000
            ratio    = fp32_mul[bb] / mul_raw if mul_raw > 0 else float("inf")
            source   = "weights+BN" if method == "fp32" else ("BN only" if mul_raw > 0 else "none")
            ratio_s  = f"{ratio:.1f}x↓" if ratio != float("inf") else "∞ (0 mul)"
            print(f"  {r['mul_G']:>8.4f} {dsp_k:>10.1f} {ratio_s:>9} {source:>12}", end="")
        print()

    print()
    print("  Notes: DSP(xk) = unrolled estimate (each mul -> 4 DSPs). In practice, DSP reuse reduces")
    print("         this by 100-1000x. Key takeaway: shift-only methods (deepshift→omnishift) use")
    print("         only BN multiplications - 40-100x less DSP pressure than fp32.")


def plot_pareto(rows: list[dict], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("[pareto] matplotlib not available - skipping plot.", file=sys.stderr)
        return

    COLORS = {
        "fp32": "#555555",
        "deepshift": "#1f77b4",
        "apot": "#ff7f0e",
        "denseshift": "#2ca02c",
        "s3shift": "#d62728",
        "fogzo": "#9467bd",
        "aptq": "#8c564b",
        "omnishift": "#e377c2",
    }
    MARKERS = {"resnet20": "o", "resnet56": "^"}
    MS = {"resnet20": 90,  "resnet56": 110}

    datasets = [d for d in DATASET_ORDER  if any(r["dataset"]  == d for r in rows)]
    backbones = [b for b in BACKBONE_ORDER if any(r["backbone"] == b for r in rows)]
    methods = [m for m in METHOD_ORDER   if any(r["method"]   == m for r in rows)]

    ncols = len(datasets)
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5.2))
    if ncols == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        ds_rows = [r for r in rows if r["dataset"] == ds]

        for bb in backbones:
            for r in [r for r in ds_rows if r["backbone"] == bb]:
                method = r["method"]
                ax.scatter(
                    r["energy"], r["test_acc"] * 100,
                    c=COLORS.get(method, "#333"),
                    marker=MARKERS[bb], s=MS[bb],
                    zorder=3, edgecolors="white", linewidths=0.6,
                )
                bb_short = bb[-2:]  
                ax.annotate(
                    f"{method}\n({bb_short})",
                    (r["energy"], r["test_acc"] * 100),
                    textcoords="offset points", xytext=(6, 2),
                    fontsize=6.2, color=COLORS.get(method, "#333"),
                )

        ax.set_xscale("log")
        ax.set_xlabel("Energy per inference (GpJ, log scale)", fontsize=9)
        ax.set_ylabel("Test Accuracy (%)", fontsize=9)
        ax.set_title(ds.upper(), fontsize=11, fontweight="bold")
        ax.grid(True, which="both", ls="--", alpha=0.35)
        ax.tick_params(labelsize=8)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COLORS[m], markersize=8, label=m)
        for m in methods
    ] + [
        Line2D([0], [0], marker=MARKERS[b], color="#666",
               markersize=8, linestyle="none", label=b)
        for b in backbones
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=len(legend_handles),
        fontsize=7.5, bbox_to_anchor=(0.5, -0.07),
        framealpha=0.9,
    )

    fig.suptitle("OmniShift - Accuracy vs Energy Pareto Frontier", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[pareto] Saved -> {out_path}")


def print_efficiency_table(rows: list[dict]) -> None:
    print("=== Accuracy-Energy Efficiency Score  (acc% / energy_GpJ, higher = better) ===")
    print()

    datasets = [d for d in DATASET_ORDER if any(r["dataset"] == d for r in rows)]
    backbones = [b for b in BACKBONE_ORDER if any(r["backbone"] == b for r in rows)]
    methods = [m for m in METHOD_ORDER if any(r["method"]  == m for r in rows)]

    col = 18
    print(f"{'Method':<14} {'Backbone':<10}", end="")
    for ds in datasets:
        print(f"  {ds:>{col}}", end="")
    print()
    print("-" * (24 + len(datasets) * (col + 2)))

    for bb in backbones:
        for method in methods:
            r_list = [r for r in rows if r["method"] == method and r["backbone"] == bb]
            if not r_list:
                continue
            print(f"{method:<14} {bb:<10}", end="")
            for ds in datasets:
                r = next((r for r in r_list if r["dataset"] == ds), None)
                if r and r["energy"] > 0:
                    score = r["test_acc"] * 100 / r["energy"]
                    print(f"  {score:>{col},.0f}", end="")
                else:
                    print(f"  {'-':>{col}}", end="")
            print()
        print()


def main():
    parser = argparse.ArgumentParser(description="Summarize OmniShift results")
    parser.add_argument("--log-root", default="logs")
    parser.add_argument("--pareto", metavar="FILE.png", nargs="?", const="pareto.png",
                        help="Generate Pareto plot PNG (default name: pareto.png)")
    parser.add_argument("--no-dsp", action="store_true", help="Skip DSP table")
    parser.add_argument("--no-efficiency", action="store_true", help="Skip efficiency-score table")
    args = parser.parse_args()

    log_root = Path(args.log_root)
    if not log_root.exists():
        print(f"No log directory at {log_root}. Run some experiments first.")
        return

    rows = load_results(log_root)
    if not rows:
        print("No result JSON files found.")
        return

    print(f"\n=== OmniShift Results ({len(rows)} runs) - sorted by dataset + energy ===\n")
    print_table(rows)
    print()

    if not args.no_dsp:
        print_dsp_table(rows)
        print()

    if not args.no_efficiency:
        print_efficiency_table(rows)
        print()

    if args.pareto is not None:
        plot_pareto(rows, Path(args.pareto))

if __name__ == "__main__":
    main()
