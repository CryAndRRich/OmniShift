#!/usr/bin/env python3
"""Regenerate the Results section in README.md from logs/phase*/*.json.

Called automatically by run_experiment.py after each run.
Safe to run manually at any time.

Usage:
    cd OmniShift
    python scripts/update_readme.py [--log-root logs] [--readme README.md]
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

START_MARKER = "<!-- RESULTS_TABLE_START -->"
END_MARKER   = "<!-- RESULTS_TABLE_END -->"

PHASE_LABEL = {
    "resnet20":                            1,
    "deepshift":                           1,
    "apot":                                1,
    "denseshift":                          1,
    "deepshift_std":                       2,
    "deepshift_potbn":                     2,
    "deepshift_potbn_warmup10":            2,
    "deepshift_potbn_warmup30":            2,
    "sparseshift_fixed50":                 3,
    "sparseshift_learnable":               3,
    "sparseshift_fixed50_potbn_w30":       4,
    "sparseshift_learnable_potbn_w30":     4,
    "sparseshift_fixed50_potbn_w30_ewgs":  5,
    "sparseshift_learnable_potbn_w30_ewgs": 5,
    "sparseshift_fixed50_potbn_w30_act":   6,
    "sparseshift_learnable_potbn_w30_act": 6,
    "omnishift_v2_fixed50":                7,
    "omnishift_v2_learnable":              7,
}

BASELINE_ENERGY = {
    "cifar10": 0.1887,
    "svhn":    0.1887,
}


def load_results(log_root: Path) -> list[dict]:
    rows = []
    seen = set()
    for jf in sorted(log_root.glob("phase*/*.json")):
        try:
            data = json.loads(jf.read_text())
            meta = data.get("meta", data)
            ops = meta.get("final_ops") or meta.get("ops") or {}
            model   = meta.get("model_name", "?")
            dataset = meta.get("dataset_name", "?")
            key = (model, dataset)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "phase":    PHASE_LABEL.get(model, "?"),
                "model":    model,
                "dataset":  dataset,
                "test_acc": meta.get("test_acc"),
                "sparsity": meta.get("final_sparsity"),
                "energy":   ops.get("energy_GpJ"),
            })
        except Exception as e:
            print(f"[warn] {jf}: {e}", file=sys.stderr)
    return rows


def _acc(v) -> str:
    return f"{v:.2%}" if v is not None else "?"


def _sp(v) -> str:
    return f"{v:.2%}" if v is not None else "—"


def _eng(v) -> str:
    return f"{v:.4f}" if v is not None else "?"


def _ratio(energy, dataset) -> str:
    base = BASELINE_ENERGY.get(dataset)
    if energy is None or base is None:
        return "?"
    return f"{base / energy:.1f}×"


def build_table(rows: list[dict], dataset: str) -> str:
    ds_rows = [r for r in rows if r["dataset"] == dataset]
    ds_rows.sort(key=lambda r: (r["phase"] if isinstance(r["phase"], int) else 99,
                                 r["energy"] if r["energy"] is not None else 999))

    lines = [
        f"| Phase | Model | Test Acc | Sparsity | Energy (GpJ) | vs ResNet-20 |",
        "|-------|-------|:--------:|:--------:|:------------:|:------------:|",
    ]
    for r in ds_rows:
        bold = r["phase"] == 5
        model = f"**{r['model']}**" if bold else r["model"]
        lines.append(
            f"| {r['phase']} | {model} | {_acc(r['test_acc'])} | "
            f"{_sp(r['sparsity'])} | {_eng(r['energy'])} | "
            f"{_ratio(r['energy'], r['dataset'])} |"
        )
    return "\n".join(lines)


def build_ladder(rows: list[dict], dataset: str) -> str:
    ds_rows = [r for r in rows if r["dataset"] == dataset and r["energy"] is not None]
    ds_rows.sort(key=lambda r: r["energy"], reverse=True)
    base = BASELINE_ENERGY.get(dataset, 0.1887)

    lines = [
        "| Stage | Energy (GpJ) | vs ResNet-20 |",
        "|-------|:------------:|:------------:|",
    ]
    if not any(r["model"] == "resnet20" for r in ds_rows):
        lines.append(f"| ResNet-20 (full precision) | {base:.4f} | 1.0× |")
    for r in ds_rows:
        bold = r["phase"] == 5
        label = f"**{r['model']}**" if bold else r["model"]
        lines.append(
            f"| {label} | {_eng(r['energy'])} | {_ratio(r['energy'], r['dataset'])} |"
        )
    return "\n".join(lines)


def generate_section(rows: list[dict]) -> str:
    today = date.today().isoformat()
    datasets = sorted({r["dataset"] for r in rows}) or ["cifar10", "svhn"]

    parts = [f"Last updated: {today}\n"]
    for ds in datasets:
        parts.append(f"### {ds.upper()}\n")
        parts.append(build_table(rows, ds))
        parts.append("")

    parts.append("### Energy Ladder\n")
    primary = "cifar10" if "cifar10" in datasets else datasets[0]
    parts.append(f"*Dataset: {primary.upper()}*\n")
    parts.append(build_ladder(rows, primary))
    parts.append("")

    return "\n".join(parts)


def update_readme(readme_path: Path, log_root: Path) -> None:
    text = readme_path.read_text()
    if START_MARKER not in text or END_MARKER not in text:
        print(f"[error] Markers not found in {readme_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_results(log_root)
    if not rows:
        print("No log files found — README left unchanged.")
        return

    new_section = generate_section(rows)
    before = text[:text.index(START_MARKER) + len(START_MARKER)]
    after  = text[text.index(END_MARKER):]
    readme_path.write_text(before + "\n" + new_section + after)
    print(f"README updated with {len(rows)} run(s) → {readme_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", default="logs")
    parser.add_argument("--readme",   default="README.md")
    args = parser.parse_args()

    readme_path = Path(args.readme)
    log_root    = Path(args.log_root)

    if not readme_path.exists():
        print(f"[error] {readme_path} not found", file=sys.stderr)
        sys.exit(1)

    update_readme(readme_path, log_root)


if __name__ == "__main__":
    main()
