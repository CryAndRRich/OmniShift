import json
import sys
import subprocess
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

METHODS = ["fp32", "deepshift", "apot", "denseshift", "s3shift", "fogzo", "aptq", "omnishift"]
BACKBONES = ["resnet20", "resnet56"]
DATASETS = ["cifar10", "svhn", "stl10"]
DS_LABELS = {"cifar10": "CIFAR-10", "svhn": "SVHN", "stl10": "STL-10"}
_COLORS = [
    "#333333", "#1f77b4", "#ff7f0e", "#2ca02c",
    "#d62728", "#9467bd", "#8c564b", "#e377c2",
]
METHOD_COLOR = dict(zip(METHODS, _COLORS))


def load_logs(log_root: Path) -> dict:
    data = {}
    for jf in sorted(log_root.rglob("*.json")):
        try:
            d = json.loads(jf.read_text())
            meta = d.get("meta", {})
            log = d.get("log", [])
            if not log:
                continue
            key = (meta.get("backbone"), meta.get("dataset_name"), meta.get("method"))
            if all(key):
                data[key] = log
        except Exception:
            pass
    return data


def save_plot(data: dict, backbone: str, dataset: str, metric: str, out_path: Path):
    train_key = "tr_loss" if metric == "loss" else "tr_acc"
    val_key = "val_loss" if metric == "loss" else "val_acc"
    ylabel = "Loss" if metric == "loss" else "Accuracy"

    fig, ax = plt.subplots(figsize=(10, 6))
    for method in METHODS:
        log = data.get((backbone, dataset, method))
        if not log:
            continue
        epochs = [e["epoch"] + 1 for e in log]
        color = METHOD_COLOR[method]
        ax.plot(epochs, [e[train_key] for e in log],
                color=color, linewidth=1.0, alpha=0.35, linestyle="--")
        ax.plot(epochs, [e[val_key] for e in log],
                color=color, linewidth=1.5, label=method)

    ds_label = DS_LABELS.get(dataset, dataset)
    ax.set_title(f"{backbone} | {ds_label} | {ylabel}  (dashed = train, solid = val)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    root = Path(__file__).resolve().parent.parent
    log_root = root / "logs"
    assets_dir = root / "assets"

    if not log_root.exists():
        print(f"No logs directory found at {log_root}")
        return

    data = load_logs(log_root)
    if not data:
        print("No log files found.")
        return

    for backbone in BACKBONES:
        for dataset in DATASETS:
            for metric in ("loss", "acc"):
                out_path = assets_dir / f"{backbone}_{dataset}_{metric}.png"
                save_plot(data, backbone, dataset, metric, out_path)


if __name__ == "__main__":
    main()