#!/usr/bin/env python3
"""Run a single OmniShift experiment from a YAML config file.

Usage:
    cd OmniShift
    python scripts/run_experiment.py --config configs/phase5_combine.yaml [--dataset svhn]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

# Ensure project root is on path when running from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loaders import get_dataloaders
from src.training.train import train_one_epoch, evaluate, EarlyStopping
from src.training.scheduler import cosine_lr_schedule
from src.utils.ops_counter import count_mul_add_shift, count_params
from src.utils.seed import set_seed, clear_memory
from src.utils.checkpoint import save_checkpoint, save_log
from src.quantize.pot_bn import set_bn_epoch


def _build_model(cfg: dict, num_classes: int, in_channels: int):
    mtype = cfg["model"]["type"]
    name = cfg["experiment"]["name"]
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


def run(cfg: dict, dataset_override: str | None = None) -> dict:
    exp = cfg["experiment"]
    tr = cfg["training"]
    out = cfg["output"]

    dataset_name = dataset_override or exp["dataset"]
    seed = exp.get("seed", 42)
    epochs = tr["epochs"]
    batch_size = tr["batch_size"]
    lr = tr["lr"]
    momentum = tr["momentum"]
    weight_decay = tr["weight_decay"]
    warmup_epochs = tr.get("warmup_epochs", 0)
    clip_grad = tr.get("clip_grad", 0.0)
    sparsity_lambda = tr.get("sparsity_lambda", 0.0)

    ckpt_dir = Path(out["checkpoint_dir"])
    log_dir = Path(out["log_dir"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_worker_fn, gen = set_seed(seed)

    data = get_dataloaders(dataset_name, batch_size=batch_size, seed=seed,
                           num_workers=4, seed_worker_fn=seed_worker_fn,
                           generator=gen)
    train_loader = data["train_loader"]
    val_loader   = data["val_loader"]
    test_loader  = data["test_loader"]
    num_classes  = data["num_classes"]
    in_channels  = data["channels"]
    img_size     = data["image_size"]

    model_name = exp["name"]
    print(f"\n{'='*70}")
    print(f"Model: {model_name} | Dataset: {dataset_name} | Seed: {seed}")
    print(f"Classes: {num_classes} | Image size: {img_size} | Device: {device}")
    print(f"{'='*70}\n")

    model = _build_model(cfg, num_classes, in_channels).to(device)

    # Estimate ops pre-run (simulate post-warmup BN)
    set_bn_epoch(model, 999)
    ops = count_mul_add_shift(model, (1, in_channels, img_size, img_size))
    set_bn_epoch(model, 0)
    n_params = count_params(model)

    print(f"Params: {n_params/1e6:.3f}M")
    print(f"Ops (post-warmup): Mul={ops['mul_G']:.4f}G | Add={ops['add_G']:.4f}G | "
          f"Shift={ops['shift_G']:.4f}G | Energy={ops['energy_GpJ']:.4f} GpJ | "
          f"Sparsity={ops['sparsity']:.1%}\n")

    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                 weight_decay=weight_decay)
    scheduler = cosine_lr_schedule(optimizer, epochs, len(train_loader),
                                    warmup_epochs=warmup_epochs)

    best_val = 0.0
    best_state = None
    best_epoch = -1
    log = []

    for epoch in range(epochs):
        set_bn_epoch(model, epoch)
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            scaler=scaler, clip_grad=clip_grad, sparsity_lambda=sparsity_lambda)
        val_loss, val_acc = evaluate(model, val_loader, device, use_amp=(device == "cuda"))

        sparsity_tag = ""
        if hasattr(model, 'get_global_sparsity'):
            sp = model.get_global_sparsity()
            sparsity_tag = f" | sp={sp:.2%}"

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            best_epoch = epoch
            star = " *"
        else:
            star = ""

        dt = time.time() - t0
        log.append(dict(epoch=epoch, tr_loss=tr_loss, tr_acc=tr_acc,
                        val_loss=val_loss, val_acc=val_acc, time=dt))
        print(f"[E{epoch+1:3d}] tr={tr_loss:.4f}/{tr_acc:.4f} | "
              f"val={val_loss:.4f}/{val_acc:.4f} | best={best_val:.4f}{star}"
              f"{sparsity_tag} | {dt:.1f}s")
        clear_memory()

    # Final eval
    model.load_state_dict(best_state)
    set_bn_epoch(model, 999)
    _, test_acc = evaluate(model, test_loader, device)

    final_sparsity = model.get_global_sparsity() if hasattr(model, 'get_global_sparsity') else 0.0
    final_ops = count_mul_add_shift(model, (1, in_channels, img_size, img_size),
                                     sparsity=final_sparsity)

    print(f"\n[{model_name} @ {dataset_name}] "
          f"best_val={best_val:.4f} (ep {best_epoch+1}) | test_acc={test_acc:.4f} | "
          f"energy={final_ops['energy_GpJ']:.4f} GpJ\n")

    result_meta = dict(
        model_name=model_name, dataset_name=dataset_name, seed=seed,
        best_val=best_val, best_epoch=best_epoch, test_acc=test_acc,
        final_sparsity=final_sparsity, final_ops=final_ops, n_params=n_params,
    )

    tag = f"{model_name}_{dataset_name}_seed{seed}"
    save_checkpoint(model, best_state, result_meta, ckpt_dir / f"{tag}.pt")
    save_log({"meta": result_meta, "log": log}, log_dir / f"{tag}.json")
    print(f"Checkpoint → {ckpt_dir / f'{tag}.pt'}")
    print(f"Log        → {log_dir / f'{tag}.json'}\n")

    _update_readme(log_dir.parent)
    return result_meta


def _update_readme(log_root: Path) -> None:
    script = Path(__file__).parent / "update_readme.py"
    readme = Path(__file__).parent.parent / "README.md"
    if not script.exists() or not readme.exists():
        return
    import subprocess
    result = subprocess.run(
        [sys.executable, str(script),
         "--log-root", str(log_root),
         "--readme", str(readme)],
        capture_output=True, text=True
    )
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.returncode != 0 and result.stderr.strip():
        print(f"[warn] update_readme: {result.stderr.strip()}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Run an OmniShift experiment")
    parser.add_argument("--config",  required=True, help="Path to YAML config")
    parser.add_argument("--dataset", default=None,  help="Override experiment.dataset")
    parser.add_argument("--name",    default=None,  help="Override experiment.name (model)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.name:
        cfg["experiment"]["name"] = args.name

    result = run(cfg, dataset_override=args.dataset)
    print("Done.")
    print(f"  test_acc    : {result['test_acc']:.4f}")
    print(f"  energy_GpJ  : {result['final_ops']['energy_GpJ']:.4f}")
    print(f"  sparsity    : {result['final_sparsity']:.2%}")


if __name__ == "__main__":
    main()
