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

# Sanity check
python -c "
from src.models.resnet_cifar import build_model
from src.utils.ops_counter import count_mul_add_shift
from src.quantize.pot_bn import set_bn_epoch
import torch

qcfg = {'use_sparse': True, 'sparse_mode': 'learnable',
         'use_pot_bn': True, 'use_pot_act': True, 'use_ewgs': True}
m = build_model('resnet20', qcfg, num_classes=10)
set_bn_epoch(m, 999)
out = m(torch.randn(2, 3, 32, 32))
ops = count_mul_add_shift(m)
print(f'OK — shape={out.shape}, energy={ops[\"energy_GpJ\"]:.4f} GpJ')
"

# Run experiment (edit configs/omnishift.yaml to change backbone/dataset/toggles)
python scripts/run_experiment.py --config configs/omnishift.yaml
python scripts/run_experiment.py --config configs/omnishift.yaml --dataset svhn

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

## Configuration

All options are in `configs/omnishift.yaml`:

```yaml
experiment:
  backbone: "resnet20"     # backbone to use
  dataset:  "cifar10"      # dataset
  name:     "omnishift"    # run name for checkpoint/log files
  seed:     42

quantize:
  use_sparse:     true     # W ∈ {0, ±2^p}
  sparse_mode:    "learnable"   # "fixed" | "learnable"
  use_pot_bn:     true     # BN scale → ±2^q
  bn_warmup:      30       # epoch to activate PoT-BN and PoT-Act
  use_pot_act:    true     # activations → {0} ∪ {2^p}
  use_ewgs:       true     # EWGS backward

training:
  epochs: 200
  batch_size: 256
  lr: 0.1                  # cosine decay
  sparsity_lambda: 0.0001  # L1 regularization (learnable mode)
```

---

## Project Structure

```
OmniShift/
├── src/
│   ├── quantize/
│   │   ├── sparse_shift.py  # SparseShiftConv2d (fixed/learnable)
│   │   ├── pot_bn.py        # PoTBatchNorm2d, set_bn_epoch
│   │   ├── pot_act.py       # PoTActivation
│   │   ├── ewgs.py          # EWGS variants of all quantizers
│   │   └── wrap.py          # make_factories() — backbone-agnostic entry point
│   ├── models/
│   │   └── resnet_cifar.py  # ResNetCIFAR, VGG_CIFAR, build_model()
│   ├── data/
│   │   └── loaders.py       # get_dataloaders (5 datasets)
│   ├── training/
│   │   ├── train.py         # train_one_epoch, evaluate, EarlyStopping
│   │   ├── scheduler.py     # cosine_lr_schedule
│   │   └── regularize.py    # L1 sparsity regularization
│   └── utils/
│       ├── ops_counter.py   # hook-based backbone-agnostic op counter
│       ├── seed.py          # set_seed, clear_memory
│       └── checkpoint.py    # save_checkpoint, save_log
├── configs/
│   └── omnishift.yaml       # unified config (edit backbone/dataset/toggles)
├── scripts/
│   ├── run_experiment.py    # training entry point
│   ├── summarize_results.py # print results table
│   └── update_readme.py     # auto-update this README
└── notebooks/
    └── omnishift.ipynb      # unified Kaggle notebook (single run + ablation)
```

---

## Outputs

Each run saves two files under `checkpoints/` and `logs/`:

```
checkpoints/{run_name}_{dataset}_seed{seed}.pt   # best weights + metadata
logs/{run_name}_{dataset}_seed{seed}.json        # per-epoch loss/acc log
```

Log JSON format:
```json
{
  "meta": {
    "run_name": "omnishift", "backbone": "resnet20", "dataset_name": "cifar10",
    "test_acc": 0.8199, "final_sparsity": 0.9098,
    "final_ops": {"energy_GpJ": 0.006, "mul_G": 0.001, ...},
    "n_params": 272513
  },
  "log": [
    {"epoch": 0, "tr_loss": 1.23, "tr_acc": 0.45, "val_loss": 1.18, "val_acc": 0.48, "time": 12.3},
    ...
  ]
}
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

- [DeepShift](https://arxiv.org/abs/1905.13298) — Elhoushi et al., CVPR 2021
- [EWGS](https://arxiv.org/abs/2104.00903) — Lee et al., CVPR 2021
- [APoT](https://arxiv.org/abs/1909.13144) — Li et al., ICLR 2020
