#!/usr/bin/env python3
"""TensorRT inference benchmark for OmniShift models.

Exports a trained model to ONNX, builds a TensorRT FP16 engine,
and measures latency / throughput on the available GPU (e.g., Tesla T4).

TensorRT validates deployment speed independently from the theoretical 45nm
energy model. GPU execution does not replicate bit-shift hardware, but provides
concrete latency numbers for edge deployment discussion.

Fallback: if TensorRT is unavailable, runs a PyTorch FP32 baseline instead.

Usage:
    cd OmniShift
    python scripts/trt_benchmark.py --config configs/phase7_omnishift_v2.yaml \\
                                    --name omnishift_v2_learnable --dataset cifar10
    python scripts/trt_benchmark.py --config configs/phase7_omnishift_v2.yaml \\
                                    --checkpoint checkpoints/phase7/omnishift_v2_learnable_cifar10_seed42.pt \\
                                    --batch 64 --n-runs 1000
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml


def _build_model(cfg, num_classes, in_channels):
    mtype = cfg["model"]["type"]
    name  = cfg["experiment"]["name"]
    if mtype == "baseline":
        from src.models.resnet20 import build_model
    elif mtype == "potbn":
        from src.models.resnet20_potbn import build_model
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


def _warmup_sync(fn, n, device):
    for _ in range(n):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()


def pytorch_benchmark(model, dummy_input, n_warmup=100, n_runs=1000):
    device = "cuda" if dummy_input.is_cuda else "cpu"
    model.eval()
    _warmup_sync(lambda: model(dummy_input), n_warmup, device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(dummy_input)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    batch   = dummy_input.shape[0]
    return elapsed / n_runs * 1000, batch * n_runs / elapsed


def trt_benchmark(model, dummy_input, n_warmup=100, n_runs=1000):
    """Build TensorRT FP16 engine and benchmark. Returns (lat_ms, ips) or None."""
    try:
        import tensorrt as trt
        import numpy as np
    except ImportError:
        return None

    import io
    buf = io.BytesIO()
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model, dummy_input, buf,
            opset_version=13,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
    buf.seek(0)

    logger  = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser  = trt.OnnxParser(network, logger)
    if not parser.parse(buf.read()):
        return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        return None

    runtime = trt.Runtime(logger)
    engine  = runtime.deserialize_cuda_engine(serialized)
    ctx     = engine.create_execution_context()

    out_shape = (dummy_input.shape[0], network.get_output(0).shape[1])
    inp_dev   = dummy_input.contiguous()
    out_dev   = torch.zeros(out_shape, device="cuda")

    def run_trt():
        ctx.execute_v2([inp_dev.data_ptr(), out_dev.data_ptr()])

    _warmup_sync(run_trt, n_warmup, "cuda")
    t0 = time.perf_counter()
    for _ in range(n_runs):
        run_trt()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    batch   = dummy_input.shape[0]
    return elapsed / n_runs * 1000, batch * n_runs / elapsed


def main():
    parser = argparse.ArgumentParser(description="TensorRT / PyTorch benchmark")
    parser.add_argument("--config",     required=True)
    parser.add_argument("--dataset",    default=None)
    parser.add_argument("--checkpoint", default=None, help=".pt checkpoint")
    parser.add_argument("--batch",      type=int, default=1)
    parser.add_argument("--n-runs",     type=int, default=1000)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from src.data.loaders import get_dataloaders
    from src.utils.seed import set_seed
    from src.quantize.pot_bn import set_bn_epoch

    dataset_name = args.dataset or cfg["experiment"]["dataset"]
    seed = cfg["experiment"].get("seed", 42)
    seed_worker_fn, gen = set_seed(seed)
    data = get_dataloaders(dataset_name, batch_size=1, seed=seed,
                           num_workers=0,
                           seed_worker_fn=seed_worker_fn, generator=gen)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = _build_model(cfg, data["num_classes"], data["channels"]).to(device)
    set_bn_epoch(model, 999)
    model.eval()

    if args.checkpoint:
        ckpt  = torch.load(args.checkpoint, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"Loaded: {args.checkpoint}")

    img_size = data["image_size"]
    dummy    = torch.randn(args.batch, data["channels"], img_size, img_size,
                           device=device)

    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"\n{'='*60}")
    print(f"Model  : {cfg['experiment']['name']} | Dataset: {dataset_name}")
    print(f"Device : {gpu_name} | Batch: {args.batch} | Runs: {args.n_runs}")
    print(f"{'='*60}")

    with torch.no_grad():
        lat_pt, ips_pt = pytorch_benchmark(model, dummy, n_runs=args.n_runs)
    print(f"PyTorch FP32  : {lat_pt:7.3f} ms/batch | {ips_pt:>10,.0f} img/s")

    trt_result = trt_benchmark(model, dummy, n_runs=args.n_runs)
    if trt_result:
        lat_trt, ips_trt = trt_result
        print(f"TensorRT FP16 : {lat_trt:7.3f} ms/batch | {ips_trt:>10,.0f} img/s")
        print(f"TRT speedup   : {lat_pt/lat_trt:.2f}×")
    else:
        print("TensorRT      : not available (pip install tensorrt)")
    print()


if __name__ == "__main__":
    main()
