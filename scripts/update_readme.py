import argparse
import json
import sys
from datetime import date
from pathlib import Path

START_MARKER = "<!-- RESULTS_TABLE_START -->"
END_MARKER   = "<!-- RESULTS_TABLE_END -->"

BASELINE_ENERGY = 0.1887

def load_results(log_root: Path) -> list[dict]:
    rows = []
    seen = set()
    for jf in sorted(log_root.rglob("*.json")):
        try:
            data = json.loads(jf.read_text())
            meta = data.get("meta", data)
            ops = meta.get("final_ops") or {}
            name = meta.get("run_name") or meta.get("model_name", "?")
            backbone = meta.get("backbone", "?")
            dataset = meta.get("dataset_name", "?")
            key = (name, dataset)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "name": name,
                "backbone": backbone,
                "dataset": dataset,
                "test_acc": meta.get("test_acc"),
                "sparsity": meta.get("final_sparsity"),
                "energy": ops.get("energy_GpJ"),
            })
        except Exception as e:
            print(f"[warn] {jf}: {e}", file=sys.stderr)
    return rows

def _acc(v):
    return f"{v:.2%}" if v is not None else "?"

def _sp(v):
    return f"{v:.2%}" if v is not None else "—"

def _eng(v):
    return f"{v:.4f}" if v is not None else "?"

def _ratio(energy):
    if energy is None or energy == 0:
        return "?"
    return f"{BASELINE_ENERGY / energy:.1f}x"

def build_table(rows: list[dict], dataset: str) -> str:
    ds_rows = sorted(
        [r for r in rows if r["dataset"] == dataset],
        key=lambda r: r["energy"] if r["energy"] is not None else 999,
    )
    lines = [
        "| Name | Backbone | Test Acc | Sparsity | Energy (GpJ) | vs ResNet-20 FP32 |",
        "|------|----------|:--------:|:--------:|:------------:|:-----------------:|",
    ]
    for r in ds_rows:
        lines.append(
            f"| {r['name']} | {r['backbone']} | {_acc(r['test_acc'])} | "
            f"{_sp(r['sparsity'])} | {_eng(r['energy'])} | {_ratio(r['energy'])} |"
        )
    return "\n".join(lines)

def generate_section(rows: list[dict]) -> str:
    today = date.today().isoformat()
    datasets = sorted({r["dataset"] for r in rows})
    parts = [f"Last updated: {today}\n"]
    for ds in datasets:
        parts.append(f"### {ds.upper()}\n")
        parts.append(build_table(rows, ds))
        parts.append("")
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
    after  = text[text.index(END_MARKER):]
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