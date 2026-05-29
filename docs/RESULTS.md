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

*Base: Phase 4 | Thay STE → EWGS (λ=0.02) trong tất cả quantizers | Forward pass giống hệt P4*

| Config | CIFAR-10 | SVHN | Sparsity (C10 / SVHN) | Energy (GpJ) | vs P4 (energy) |
|---|---|---|---|---|---|
| sparseshift_fixed50_potbn_w30_ewgs | **91.09%** | **96.55%** | 50.00% / 50.00% | **0.0230** | 0% (identical forward) |
| sparseshift_learnable_potbn_w30_ewgs | **91.09%** | **96.39%** | 76.24% / 87.33% | **0.0121 / 0.0075** | −28.4% / −24.2% |

**Key findings:**
- EWGS did **not** improve accuracy (−0.06 to −0.24 pp vs P4) — within seed variance
- EWGS **dramatically increased learnable sparsity**: +11.6 pp on CIFAR-10, +5.7 pp on SVHN
- Energy reduction via higher sparsity: **15.6× on CIFAR-10, 25.2× on SVHN** vs full-precision baseline (learnable)
- Primary EWGS benefit: sparsity amplification → energy reduction, not gradient quality

---

## Phase 6 — P4 + PoT Activation Quantization

*Base: Phase 4 | Thêm PoTActivation sau mỗi ReLU — fully multiply-free end-to-end*

| Config | CIFAR-10 | SVHN | Sparsity (C10 / SVHN) | Energy (GpJ) | vs P4 (acc) |
|---|---|---|---|---|---|
| sparseshift_fixed50_potbn_w30_act | **89.68%** | **96.16%** | 50.00% / 50.00% | **0.0231** | −1.65 / −0.49 pp |
| sparseshift_learnable_potbn_w30_act | **89.77%** | **96.02%** | 58.79% / 74.97% | **0.0194 / 0.0127** | −1.44 / −0.43 pp |

**Key findings:**
- PoT activations cost **~1.5 pp** on CIFAR-10, **~0.5 pp** on SVHN
- Energy *increases* vs P4 (activation quantization ops are additive in current model — pessimistic for real PoT-aware hardware that would exploit structured activations downstream)
- Learnable sparsity *decreases* vs P4 (58.79% vs 64.67% C10; 74.97% vs 81.66% SVHN) — dual quantization disrupts threshold learning
- Fixed50 beats learnable on SVHN accuracy (96.16% vs 96.02%) — only case in project history
- **Phase 5 dominates Phase 6 on every metric** (higher accuracy, lower energy) under current energy model

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
| 4 | Sparse learnable + PoT-BN | 0.0169 | 11.2× |
| 5 | **Sparse learnable + PoT-BN + EWGS** | **0.0121** | **15.6×** |
| 6 | Sparse learnable + PoT-BN + PoT-Act | 0.0194 | 9.7× (energy ↑, accuracy ↓1.5pp) |
| 7 | **OmniShift v2 (learnable + EWGS + PoT act)** | **TBD** | **TBD** |
