import argparse
import json
import sys
from datetime import date
from pathlib import Path

START_MARKER = "<!-- RESULTS_TABLE_START -->"
END_MARKER = "<!-- RESULTS_TABLE_END -->"

_DS_LABELS = {"cifar10": "CIFAR-10", "svhn": "SVHN", "stl10": "STL-10"}
_DS_ORDER = ["cifar10", "svhn", "stl10"]


def load_results(log_root: Path) -> list[dict]:
    rows = []
    seen = set()
    for jf in sorted(log_root.rglob("*.json")):
        try:
            data = json.loads(jf.read_text())
            meta = data.get("meta", data)
            ops = meta.get("final_ops") or {}
            method   = meta.get("method", "?")
            backbone = meta.get("backbone", "?")
            dataset  = meta.get("dataset_name", "?")
            name     = meta.get("run_name") or meta.get("model_name", "?")
            key = (name, dataset)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "method":   method,
                "backbone": backbone,
                "dataset":  dataset,
                "test_acc": meta.get("test_acc"),
                "sparsity": meta.get("final_sparsity"),
                "energy":   ops.get("energy_GpJ"),
            })
        except Exception as e:
            print(f"[warn] {jf}: {e}", file=sys.stderr)
    return rows


def _build_baselines(rows: list[dict]) -> dict:
    baselines = {}
    for r in rows:
        if r["method"] == "fp32":
            baselines[(r["backbone"], r["dataset"])] = (r["energy"], r["test_acc"])
    return baselines


def _acc(v):
    return f"{v:.2%}" if v is not None else "?"

def _sp(v):
    return f"{v:.2%}" if v is not None else "—"

def _eng(v):
    return f"{v:.4f}" if v is not None else "?"

def _ratio(energy, fp32_energy):
    if not energy or not fp32_energy or energy == 0:
        return "?"
    r = fp32_energy / energy
    s = f"{r:.1f}x"
    return f"**{s}**" if r >= 4.0 else s


def build_table(rows: list[dict], backbone: str, dataset: str, fp32_energy) -> str:
    ds_rows = sorted(
        [r for r in rows if r["backbone"] == backbone and r["dataset"] == dataset],
        key=lambda r: r["energy"] if r["energy"] is not None else 999,
    )
    lines = [
        "| Method | Test Acc | Sparsity | Energy (GpJ) | vs FP32 |",
        "|--------|:--------:|:--------:|:------------:|:-------:|",
    ]
    for r in ds_rows:
        label = "fp32 (baseline)" if r["method"] == "fp32" else r["method"]
        lines.append(
            f"| {label} | {_acc(r['test_acc'])} | "
            f"{_sp(r['sparsity'])} | {_eng(r['energy'])} | {_ratio(r['energy'], fp32_energy)} |"
        )
    return "\n".join(lines)


def generate_section(rows: list[dict]) -> str:
    today = date.today().isoformat()
    baselines = _build_baselines(rows)
    backbones = sorted({r["backbone"] for r in rows})

    parts = [f"Last updated: {today}\n"]
    for bb in backbones:
        fp32_e = next(
            (baselines[(bb, ds)][0] for ds in _DS_ORDER if (bb, ds) in baselines),
            None
        )
        acc_parts = [
            f"{_DS_LABELS[ds]} {baselines[(bb, ds)][1]*100:.2f}%"
            for ds in _DS_ORDER if (bb, ds) in baselines
        ]
        parts.append(f"### {bb}\n")
        if fp32_e:
            parts.append(f"> FP32 baseline: {fp32_e:.4f} GpJ | {' | '.join(acc_parts)}\n")
        for ds in _DS_ORDER:
            fp32_e_ds = (baselines.get((bb, ds)) or (None, None))[0]
            parts.append(f"#### {_DS_LABELS[ds]}\n")
            parts.append(build_table(rows, bb, ds, fp32_e_ds))
            parts.append("")
        parts.append("---\n")

    return "\n".join(parts)


def update_readme(readme_path: Path, log_root: Path) -> None:
    text = readme_path.read_text()
    if START_MARKER not in text or END_MARKER not in text:
        print(f"[error] Markers not found in {readme_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_results(log_root)
    if not rows:
        print("No log files found - README left unchanged.")
        return

    new_section = generate_section(rows)
    before = text[:text.index(START_MARKER) + len(START_MARKER)]
    after = text[text.index(END_MARKER):]
    readme_path.write_text(before + "\n" + new_section + after)
    print(f"README updated ({len(rows)} run(s)) -> {readme_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", default="logs")
    parser.add_argument("--readme", default="README.md")
    args = parser.parse_args()
    update_readme(Path(args.readme), Path(args.log_root))


if __name__ == "__main__":
    main()
