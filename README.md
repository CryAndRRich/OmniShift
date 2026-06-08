# OmniShift

> **A multiply-free CNN training framework for edge/IoT devices**  
> Apply coordinated PoT quantization to any backbone → fully multiply-free inference with zero DSP usage on FPGA.

---

## Overview

OmniShift is a **framework**, not a model. It converts any supported CNN backbone into a multiply-free network by applying four independently toggleable quantization techniques:

| Component | Description | Effect |
|-----------|-------------|--------|
| **Sparse Shift** | W ∈ {0, ±2^p} | Conv multiplications → bit-shifts + skip-zero |
| **PoT-BN** | γ/σ → ±2^q | BN scale multiplication → shift |
| **PoT-Act** | Post-ReLU → {0} ∪ {2^p} | Activation quantization to log-uniform grid |
| **EWGS** | Element-Wise Gradient Scaling | Replaces STE backward → smoother training |

**Energy model (45nm CMOS):** `mul = 3.7 pJ`, `add = 0.9 pJ`, `shift = 0.13 pJ`

---

## Key Results (ResNet-20, all 4 components ON)

| Dataset | Test Acc | Sparsity | Energy (GpJ) | vs ResNet-20 FP32 |
|---------|:--------:|:--------:|:------------:|:-----------------:|
| CIFAR-10 (learnable) | 81.99% | 90.98% | 0.0060 | **31.5×** |
| CIFAR-10 (fixed 50%) | 86.46% | 50.00% | 0.0230 | 8.2× |
| SVHN (learnable) | 95.38% | 93.64% | 0.0049 | **38.5×** |
| SVHN (fixed 50%) | 96.20% | 50.00% | 0.0230 | 8.2× |

ResNet-20 FP32 baseline: 92.23% CIFAR-10 / 96.49% SVHN / 0.1887 GpJ

---

## Quick Start

```bash
pip install torch torchvision pyyaml

cd OmniShift

# Sanity check — all 9 methods
python3 -c "
from src.models.resnet_cifar import build_model
from src.utils.ops_counter import count_mul_add_shift
from src.quantize.pot_bn import set_bn_epoch
import torch

for method in ['fp32', 'deepshift', 'apot', 'xnor', 'denseshift', 's3shift', 'fogzo', 'aptq', 'omnishift']:
    m = build_model('resnet20', method, num_classes=10)
    set_bn_epoch(m, 999)
    out = m(torch.randn(2, 3, 32, 32))
    ops = count_mul_add_shift(m)
    print(f'{method:12s}  shape={out.shape}  energy={ops[\"energy_GpJ\"]:.4f} GpJ')
"

# Run experiment
python scripts/run_experiment.py --config configs/omnishift.yaml
python scripts/run_experiment.py --config configs/omnishift.yaml --method deepshift --dataset svhn

# Print results table
python scripts/summarize_results.py
```

---

## Supported Backbones & Datasets

**Backbones:**
- `resnet20` — ResNet-20 (3×[3,3,3] blocks, 16/32/64 channels)
- `resnet32` — ResNet-32 (3×[5,5,5] blocks)
- `resnet56` — ResNet-56 (3×[9,9,9] blocks)
- `resnet110` — ResNet-110 (3×[18,18,18] blocks)
- `vgg11` — VGG-11 adapted for 32×32 input

**Datasets:** `cifar10`, `cifar100`, `svhn`, `stl10`, `tiny_imagenet`

---

## Baselines

| Method key | Paper | Authors | ArXiv | Venue |
|------------|-------|---------|-------|-------|
| `fp32` | — | — | — | — |
| `deepshift` | DeepShift: Towards Multiplication-Less Neural Networks | Elhoushi et al. | [1905.13298](https://arxiv.org/abs/1905.13298) | CVPR-W 2021 |
| `apot` | Additive Power-of-Two Quantization | Li et al. | [1909.13144](https://arxiv.org/abs/1909.13144) | ICLR 2020 |
| `xnor` | XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks | Rastegari et al. | [1603.05279](https://arxiv.org/abs/1603.05279) | ECCV 2016 |
| `denseshift` | DenseShift: Towards Accurate and Efficient Low-Bit Power-of-Two Quantization | Li et al. | [2208.09708](https://arxiv.org/abs/2208.09708) | ICCV 2023 |
| `s3shift` | S³: Sign-Sparse-Shift Reparametrization for Effective Training of Low-Bit Shift Networks | Li et al. | [2107.03453](https://arxiv.org/abs/2107.03453) | NeurIPS 2021 |
| `fogzo` | FOGZO: First-Order-Guided Zeroth-Order Gradient Descent for Quantization-Aware Training | Yang & Aamodt | [2510.23926](https://arxiv.org/abs/2510.23926) | NeurIPS 2025 |
| `aptq` | APTQ: Adaptive Global Power-of-Two Ternary Quantization | Liu et al. | — | Sensors (MDPI) 2024 |
| `omnishift` | OmniShift (this work) | — | — | — |

> `apot` implements the correct additive PoT grid (δ = α/2^(n_bits−1) step, uniformly-spaced levels). `xnor` implements full XNOR-Net (weight + activation binarization). `aptq` has no public arXiv preprint; DOI: 10.3390/s24010181.

---

## Project Structure

```
OmniShift/
├── src/
│   ├── quantize/
│   │   ├── shift.py           # ShiftConv2d — W ∈ {±2^p} (DeepShift-PS)
│   │   ├── sparse_shift.py    # SparseShiftConv2d — W ∈ {0, ±2^p}
│   │   ├── s3shift.py         # S3ShiftConv2d — sign×sparse×shift
│   │   ├── fogzo.py           # FogzoShiftConv2d — ZO-augmented STE
│   │   ├── aptq_ternary.py    # APTQTernaryConv2d — two-sub-filter PoT ternary
│   │   ├── apot.py            # APoTConv2d — additive PoT grid
│   │   ├── xnor.py            # XNORConv2d — binary weights + activations
│   │   ├── denseshift.py      # DenseShiftConv2d — sign×shift, no zero
│   │   ├── pot_bn.py          # PoTBatchNorm2d, set_bn_epoch
│   │   ├── pot_act.py         # PoTActivation
│   │   └── ewgs.py            # EWGS variants of all quantizers
│   ├── methods/
│   │   ├── __init__.py        # get_factories(), METHODS list
│   │   ├── fp32.py            # FP32 baseline
│   │   ├── deepshift.py       # DeepShift-PS
│   │   ├── apot.py            # APoT
│   │   ├── xnor.py            # XNOR-Net / BWN
│   │   ├── denseshift.py      # DenseShift
│   │   ├── s3shift.py         # S³
│   │   ├── fogzo.py           # FOGZO
│   │   ├── aptq_ternary.py    # APTQ
│   │   └── omnishift.py       # OmniShift full pipeline
│   ├── models/
│   │   └── resnet_cifar.py    # ResNetCIFAR, VGG_CIFAR, build_model()
│   ├── data/
│   │   └── loaders.py         # get_dataloaders (5 datasets)
│   ├── training/
│   │   ├── train.py           # train_one_epoch, evaluate, EarlyStopping
│   │   ├── scheduler.py       # cosine_lr_schedule
│   │   └── regularize.py      # L1 sparsity regularization
│   └── utils/
│       ├── ops_counter.py     # hook-based backbone-agnostic op counter
│       ├── seed.py            # set_seed, clear_memory
│       └── checkpoint.py      # save_checkpoint, save_log
├── configs/
│   └── omnishift.yaml         # unified config (edit method/backbone/dataset)
├── scripts/
│   ├── run_experiment.py      # training entry point (CLI + Python API)
│   ├── summarize_results.py   # print results table from logs/
│   ├── update_readme.py       # auto-update Results section from logs/
│   ├── fpga_estimate.py       # Xilinx 7-series resource estimation
│   └── trt_benchmark.py       # TensorRT FP16 benchmark
└── notebooks/
    └── omnishift.ipynb        # Kaggle notebook (setup / config / train / results)
```

---

## Configuration

Edit `configs/omnishift.yaml` or pass `--method`/`--dataset` flags:

```yaml
experiment:
  method:   "omnishift"   # fp32 | deepshift | apot | xnor | denseshift | s3shift | fogzo | aptq | omnishift
  backbone: "resnet20"    # resnet20 | resnet32 | resnet56 | resnet110 | vgg11
  dataset:  "cifar10"     # cifar10 | cifar100 | svhn | stl10 | tiny_imagenet
  seed:     42

training:
  epochs:          200
  batch_size:      256
  lr:              0.1      # cosine decay
  sparsity_lambda: 0.0001   # L1 regularization (learnable sparse mode)
```

Method-specific opts via `method_opts` key or Python API:

```python
from scripts.run_experiment import build_cfg, run

cfg = build_cfg('omnishift', 'resnet20', 'cifar10', seed=42, epochs=200,
                sparse_mode='learnable', bn_warmup=30)
result = run(cfg)
```

---

## Outputs

Each run saves two files:

```
checkpoints/{run_name}_{dataset}_seed{seed}.pt   # best weights + metadata
logs/{run_name}_{dataset}_seed{seed}.json        # per-epoch loss/acc log
```

---

## Results

<!-- RESULTS_TABLE_START -->
Last updated: 2026-05-30

*Run `python scripts/summarize_results.py` to see results from your own runs.*
*Run `python scripts/update_readme.py` to auto-populate this section from logs/.*
<!-- RESULTS_TABLE_END -->

---

## Hyperparameters

| Param | Default |
|-------|---------|
| Epochs | 200 |
| Batch size | 256 |
| LR | 0.1 (cosine decay) |
| Momentum | 0.9 |
| Weight decay | 5×10⁻⁴ |
| Sparsity λ | 10⁻⁴ (learnable mode) |
| BN warmup | 30 epochs |
| EWGS λ | 0.02 |
| PoT-Act levels | 8 |

Val split: 10% of train, `torch.Generator(seed=42)`.

---

## References

- [DeepShift](https://arxiv.org/abs/1905.13298) — Elhoushi et al., CVPR-W 2021
- [APoT](https://arxiv.org/abs/1909.13144) — Li et al., ICLR 2020
- [XNOR-Net / BWN](https://arxiv.org/abs/1603.05279) — Rastegari et al., ECCV 2016
- [DenseShift](https://arxiv.org/abs/2208.09708) — Li et al., ICCV 2023
- [S³](https://arxiv.org/abs/2107.03453) — Li et al., NeurIPS 2021
- [EWGS](https://arxiv.org/abs/2104.00903) — Lee et al., CVPR 2021
- [FOGZO](https://arxiv.org/abs/2510.23926) — Yang & Aamodt, NeurIPS 2025
- [APTQ](https://doi.org/10.3390/s24010181) — Liu et al., Sensors 2024
