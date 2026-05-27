# OmniShift

> **Multiplier-less ResNet-20 for edge/IoT devices**
> Replaces all multiplications with bit-shifts using DeepShift + Sparse Shift + PoT-BN.

---

## Overview

OmniShift trains a standard ResNet-20 on CIFAR-10 / SVHN with **zero multiplications** in the convolutional stack, targeting ultra-low-power edge inference (45nm CMOS).

Three techniques are stacked progressively:

| Phase | Technique | Key idea |
|-------|-----------|----------|
| 1 | DeepShift | W ∈ {±2^p} — shift replaces conv mul |
| 3 | PoT-BN | γ/σ → ±2^q — shift replaces BN scale mul |
| 4 | Sparse Shift | W ∈ {0, ±2^p} — skip-zero saves further energy |
| **5** | **OmniShift (combined)** | **All three → maximum energy reduction** |

Energy model (45nm CMOS): `mul = 3.7 pJ`, `add = 0.9 pJ`, `shift = 0.13 pJ`

---

## Quick Start

```bash
# Install dependencies
pip install torch torchvision pyyaml

# Sanity check — verifies all imports and forward pass
cd OmniShift
python -c "
from src.models.resnet20_full import build_model
from src.utils.ops_counter import count_mul_add_shift
from src.quantize.pot_bn import set_bn_epoch
import torch
m = build_model('sparseshift_learnable_potbn_w30', num_classes=10)
set_bn_epoch(m, 999)
out = m(torch.randn(2, 3, 32, 32))
ops = count_mul_add_shift(m)
print(f'OK — shape={out.shape}, energy={ops[\"energy_GpJ\"]:.4f} GpJ')
"

# Run a single experiment
python scripts/run_experiment.py --config configs/phase5_combine.yaml
python scripts/run_experiment.py --config configs/phase5_combine.yaml --dataset svhn

# Print results table to stdout
python scripts/summarize_results.py

# Regenerate results table in this README
python scripts/update_readme.py
```

---

## Results

*Auto-updated after each experiment run. Refresh manually: `python scripts/update_readme.py`*

<!-- RESULTS_TABLE_START -->
Last updated: —

### CIFAR-10

| Phase | Model | Test Acc | Sparsity | Energy (GpJ) | vs ResNet-20 |
|-------|-------|:--------:|:--------:|:------------:|:------------:|
| 1 | resnet20 | 92.23% | — | 0.1887 | 1.0× |
| 1 | deepshift | 92.13% | — | 0.0445 | 4.2× |
| 1 | apot | 91.90% | — | 0.0861 | 2.2× |
| 1 | denseshift | 91.22% | — | 0.0445 | 4.2× |
| 2 | deepshift_potbn_warmup30 | 92.02% | — | 0.0438 | 4.3× |
| 3 | sparseshift_fixed50 | 91.94% | 50.00% | 0.0238 | 7.9× |
| 3 | sparseshift_learnable | 92.06% | 62.91% | 0.0184 | 10.3× |
| **4** | **sparseshift_fixed50_potbn_w30** | **91.33%** | 50.00% | **0.0230** | **8.2×** |
| **4** | **sparseshift_learnable_potbn_w30** | **91.21%** | 64.67% | **0.0169** | **11.2×** |

### SVHN

| Phase | Model | Test Acc | Sparsity | Energy (GpJ) | vs ResNet-20 |
|-------|-------|:--------:|:--------:|:------------:|:------------:|
| 1 | resnet20 | 96.49% | — | 0.1887 | 1.0× |
| 1 | deepshift | 96.71% | — | 0.0445 | 4.2× |
| 2 | deepshift_potbn_warmup30 | 96.42% | — | 0.0438 | 4.3× |
| 3 | sparseshift_fixed50 | 96.74% | 50.00% | 0.0238 | 7.9× |
| 3 | sparseshift_learnable | 96.46% | 81.46% | 0.0107 | 17.6× |
| **4** | **sparseshift_fixed50_potbn_w30** | **96.65%** | 50.00% | **0.0230** | **8.2×** |
| **4** | **sparseshift_learnable_potbn_w30** | **96.45%** | 81.66% | **0.0099** | **19.1×** |

### Energy Ladder (CIFAR-10, progressive)

| Stage | Energy (GpJ) | vs ResNet-20 |
|-------|:------------:|:------------:|
| P1 ResNet-20 (full precision) | 0.1887 | 1× |
| P1 + DeepShift | 0.0445 | 4.2× |
| P2 + PoT-BN | 0.0438 | 4.3× |
| P3 + Sparse fixed 50% | 0.0238 | 7.9× |
| P4 + Sparse fixed 50% + PoT-BN | 0.0230 | 8.2× |
| **P4 Sparse learnable + PoT-BN (current best)** | **0.0169** | **11.2×** |
| **P7 OmniShift v2 (+ EWGS + PoT activations)** | **TBD** | **TBD** |

<!-- RESULTS_TABLE_END -->

---

## Project Structure

```
OmniShift/
├── src/
│   ├── quantize/
│   │   ├── shift.py             # RoundToPoT, ShiftConv2d (DeepShift)
│   │   ├── sparse_shift.py      # SparseShiftConv2d — fixed/learnable modes
│   │   └── pot_bn.py            # PoTBatchNorm2d, set_bn_epoch
│   ├── models/
│   │   ├── resnet20.py          # Phase 1 baselines (resnet20/deepshift/apot/denseshift)
│   │   ├── resnet20_potbn.py    # Phase 3: DeepShift + PoT-BN
│   │   ├── resnet20_sparse.py   # Phase 4: Sparse Shift
│   │   └── resnet20_full.py     # Phase 5: OmniShift (FINAL MODEL)
│   ├── data/loaders.py          # get_dataloaders (CIFAR-10, SVHN, …)
│   ├── training/
│   │   ├── train.py             # train_one_epoch, evaluate, EarlyStopping
│   │   ├── scheduler.py         # cosine_lr_schedule
│   │   └── regularize.py        # L1 sparsity regularization
│   └── utils/
│       ├── ops_counter.py       # count_mul_add_shift, energy formula
│       ├── seed.py              # set_seed, clear_memory
│       └── checkpoint.py        # save/load helpers
├── configs/                     # YAML experiment configs
├── scripts/
│   ├── run_experiment.py        # Main training entry point
│   ├── summarize_results.py     # Print results table to stdout
│   └── update_readme.py         # Regenerate README results table from logs
├── notebooks/                   # Phase training scripts (import from src/)
│   ├── phase1_baselines.py
│   ├── phase3_potbn.py
│   ├── phase4_sparse_shift.py
│   └── phase5_combine.py
└── docs/
    ├── RESULTS.md               # Full results reference (all phases)
    ├── ARCHITECTURE.md          # Class descriptions + I/O shapes
    └── ENERGY_MODEL.md          # Energy formula derivation
```

---

## Key Design Decisions

**Why shift instead of multiply?**
A bit-shift costs 0.13 pJ vs 3.7 pJ for multiply (28× cheaper). On a ResNet-20 inference pass (~41M multiply-accumulates), replacing muls with shifts reduces energy from 0.189 GpJ to 0.044 GpJ.

**Why PoT-BN?**
Batch Normalization's `y = (γ/σ)·x + β` contains a multiply per activation element. Quantizing `γ/σ` to ±2^q replaces this with a shift — eliminating the last significant multiplication source.

**Why sparsity?**
Zero weights → skip computation entirely. With 50% sparsity, shift count halves. With learnable sparsity on SVHN, the model discovers ~81% of weights are unnecessary, yielding 17.6× energy reduction.

**First conv + FC always use standard multiply** (3→16 channels and 64→10) — their parameter counts are negligible and they are excluded from energy savings claims.

---

## Hyperparameters

| Param | Value |
|-------|-------|
| SEED | 42 |
| EPOCHS | 200 |
| BATCH_SIZE | 256 |
| LR | 0.1 (cosine decay) |
| MOMENTUM | 0.9 |
| WEIGHT_DECAY | 5e-4 |
| SPARSITY_LAMBDA | 1e-4 (learnable mode L1) |
| BN_WARMUP | 30 epochs |

Val split: 10% of train, `torch.Generator(seed=42)`.

---

## Citation / Reference

Based on techniques from:
- [DeepShift](https://arxiv.org/abs/1905.13298) — Elhoushi et al., 2021
- [APoT](https://arxiv.org/abs/1909.13144) — Li et al., 2020
- [DenseShift](https://arxiv.org/abs/2208.09708) — Pan et al., 2022
