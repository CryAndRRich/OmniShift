#!/usr/bin/env python3
"""Scan all logs/phase*/*.json files and print a sorted results table.

Usage:
    cd OmniShift
    python scripts/summarize_results.py [--log-root logs]
"""

import argparse
import json
import sys
from pathlib import Path


def load_results(log_root: Path) -> list[dict]:
    rows = []
    for json_file in sorted(log_root.glob("phase*/*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
            meta = data.get("meta", data)
            final_ops = meta.get("final_ops") or meta.get("ops") or {}
            rows.append({
                "file":       json_file.name,
                "model":      meta.get("model_name", "?"),
                "dataset":    meta.get("dataset_name", "?"),
                "test_acc":   meta.get("test_acc", float("nan")),
                "sparsity":   meta.get("final_sparsity", 0.0),
                "energy":     final_ops.get("energy_GpJ", float("nan")),
                "params":     meta.get("n_params", 0),
            })
        except Exception as e:
            print(f"[warn] Could not parse {json_file}: {e}", file=sys.stderr)
    return rows


def print_table(rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: r["energy"])
    header = f"{'Model':<40} {'Dataset':<10} {'TestAcc':>8} {'Sparsity':>10} {'Energy(GpJ)':>12} {'Params':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['model']:<40} {r['dataset']:<10} "
              f"{r['test_acc']:>8.4f} {r['sparsity']:>10.2%} "
              f"{r['energy']:>12.4f} {r['params']:>10,}")


def main():
    parser = argparse.ArgumentParser(description="Summarize OmniShift results")
    parser.add_argument("--log-root", default="logs",
                        help="Root directory containing phase*/  log folders")
    args = parser.parse_args()

    log_root = Path(args.log_root)
    if not log_root.exists():
        print(f"No log directory found at {log_root}. Run some experiments first.")
        return

    rows = load_results(log_root)
    if not rows:
        print("No result JSON files found.")
        return

    print(f"\n=== OmniShift Results ({len(rows)} runs) — sorted by energy ===\n")
    print_table(rows)
    print()


if __name__ == "__main__":
    main()
