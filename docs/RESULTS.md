# OmniShift — Experiment Results

## Phase 1 — Baselines

| Method | CIFAR-10 | SVHN | Energy (GpJ) | Mul (G) | Notes |
|---|---|---|---|---|---|
| ResNet-20 (baseline) | 92.23% | 96.49% | 0.189 | 0.041 | Full precision |
| DeepShift | **92.13%** | **96.71%** | **0.045** | 0.001 | W=±2^p, best baseline |
| APoT (K=2, bit=3) | 91.90% | 94.95% | 0.087 | 0.001 | Higher bit, higher energy |
| DenseShift | 91.22% | 94.72% | 0.045 | 0.001 | Worse accuracy |

**Winner Phase 1**: DeepShift — best accuracy + lowest energy.

> *Note: Adaptive bit per stage was tested and failed — does not improve over DeepShift uniform. Dropped, not included in numbering.*

---

## Phase 2 — PoT-BN (Idea A)

`y = γ/σ * (x-μ) + β` → `y = round_to_PoT(γ/σ) * (x-μ) + β`

| Config | CIFAR-10 | SVHN | Energy (GpJ) | Mul (G) |
|---|---|---|---|---|
| DeepShift + std BN (Phase 1) | 92.13% | 96.71% | 0.045 | 0.001 |
| + PoT-BN (no warmup) | 91.52% | **96.58%** | **0.044** | **0.0004** |
| + PoT-BN warmup10 | 91.93% | 96.30% | **0.044** | **0.0004** |
| **+ PoT-BN warmup30** | **92.02%** | 96.42% | **0.044** | **0.0004** |

**Best config**: `deepshift_potbn_warmup30` — balanced across both datasets.
**Claim**: Residual mul = 0.0004G (first conv 3→16 + FC 64→10) = <0.001% of baseline.

---

## Phase 3 — Sparse Shift (Idea E)

| Config | CIFAR-10 | SVHN | Sparsity | Energy (GpJ) |
|---|---|---|---|---|
| DeepShift (Phase 1 baseline) | 92.13% | 96.71% | 0% | 0.045 |
| **sparseshift_fixed50** | **91.94%** | **96.74%** | 50.00% | **0.0238** |
| **sparseshift_learnable** | **92.06%** | **96.46%** | 62.91% / **81.46%** | **0.0184 / 0.0107** |

**Key findings**:
- Fixed 50%: exact 50% sparsity, stable, near-zero accuracy drop
- Learnable: CIFAR-10 → 62.91%, **SVHN → 81.46%** (SVHN is easier, model prunes aggressively)
- Learnable SVHN: energy 0.0107 GpJ = **17.7× reduction** vs ResNet-20 baseline

---

## Phase 4 — Combine: Sparse Shift + PoT-BN

| Config | CIFAR-10 | SVHN | Sparsity (C10 / SVHN) | Energy (GpJ) |
|---|---|---|---|---|
| sparseshift_fixed50_potbn_w30 | **91.33%** | **96.65%** | 50.00% / 50.00% | **0.0230** |
| sparseshift_learnable_potbn_w30 | **91.21%** | **96.45%** | 64.67% / 81.66% | **0.0169 / 0.0099** |

**Achieved energy reduction vs ResNet-20**: 8.2× (fixed) — **19.1× (learnable, SVHN)**

---

## Phase 5 — P4 + EWGS Gradient Estimator

*Base: Phase 4 | Thay STE → EWGS (λ=0.02) trong tất cả quantizers*

| Config | CIFAR-10 | SVHN | Sparsity | Energy (GpJ) | vs P4 |
|---|---|---|---|---|---|
| sparseshift_fixed50_potbn_w30 + EWGS | TBD | TBD | ~50% | TBD | — |
| sparseshift_learnable_potbn_w30 + EWGS | TBD | TBD | TBD | TBD | — |

---

## Phase 6 — P4 + PoT Activation Quantization

*Base: Phase 4 | Thêm PoTActivation sau mỗi ReLU — fully multiply-free end-to-end*

| Config | CIFAR-10 | SVHN | Sparsity | Energy (GpJ) | vs P4 |
|---|---|---|---|---|---|
| sparseshift_fixed50_potbn_w30 + PoTAct | TBD | TBD | ~50% | TBD | — |
| sparseshift_learnable_potbn_w30 + PoTAct | TBD | TBD | TBD | TBD | — |

---

## Phase 7 — OmniShift v2 (EWGS + PoT Activations)

*Base: Phase 5 + Phase 6 | Final model*

| Config | CIFAR-10 | SVHN | Sparsity | Energy (GpJ) | vs P4 | vs ResNet-20 |
|---|---|---|---|---|---|---|
| omnishift_v2_fixed50 | TBD | TBD | ~50% | TBD | — | — |
| omnishift_v2_learnable | TBD | TBD | TBD | TBD | — | — |

---

## Energy Ladder (CIFAR-10, progressive)

| Phase | Model | Energy (GpJ) | vs. ResNet-20 |
|---|---|---|---|
| 1 | ResNet-20 (full mul) | 0.1887 | 1× |
| 1 | DeepShift | 0.0445 | 4.2× |
| 2 | DeepShift + PoT-BN | 0.0438 | 4.3× |
| 3 | Sparse fixed 50% + std BN | 0.0238 | 7.9× |
| 4 | Sparse fixed 50% + PoT-BN | 0.0230 | 8.2× |
| 4 | **Sparse learnable + PoT-BN** | **0.0169** | **11.2×** |
| 7 | **OmniShift v2 (learnable + EWGS + PoT act)** | **TBD** | **TBD** |
