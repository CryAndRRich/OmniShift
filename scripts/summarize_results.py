import argparse
import json
import sys
from pathlib import Path

def load_results(log_root: Path) -> list[dict]:
    rows = []
    for json_file in sorted(log_root.rglob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
            meta = data.get("meta", data)
            final_ops = meta.get("final_ops") or {}
            rows.append({
                "file": json_file.name,
                "name": meta.get("run_name") or meta.get("model_name", "?"),
                "backbone": meta.get("backbone", "?"),
                "dataset": meta.get("dataset_name", "?"),
                "test_acc": meta.get("test_acc", float("nan")),
                "sparsity": meta.get("final_sparsity", 0.0),
                "energy": final_ops.get("energy_GpJ", float("nan")),
                "params": meta.get("n_params", 0)
            })
        except Exception as e:
            print(f"[warn] Could not parse {json_file}: {e}", file=sys.stderr)
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

def main():
    parser = argparse.ArgumentParser(description="Summarize OmniShift results")
    parser.add_argument("--log-root", default="logs")
    args = parser.parse_args()

    log_root = Path(args.log_root)
    if not log_root.exists():
        print(f"No log directory at {log_root}. Run some experiments first.")
        return

    rows = load_results(log_root)
    if not rows:
        print("No result JSON files found.")
        return

    print(f"\n=== OmniShift Results ({len(rows)} runs) — sorted by dataset + energy ===\n")
    print_table(rows)
    print()

if __name__ == "__main__":
    main()
