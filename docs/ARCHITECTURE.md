# OmniShift — Architecture Reference

## Overview

OmniShift is ResNet-20 (3 stages × 3 BasicBlocks, channels 16→32→64, 32×32 input)
with multiplier-less convolutions and batch normalization.

**Fixed convention (all phases)**:
- First conv (3→16): always `nn.Conv2d` — raw pixel input needs full precision
- Last FC (64→10): always `nn.Linear` — logits need full precision
- All interior convs: quantized (shift / sparse-shift)

---

## Class Hierarchy

```
src/quantize/
  shift.py
    RoundToPoT          — STE: w → sign(w) * 2^round(log2|w|)
    ShiftConv2d         — DeepShift conv, W ∈ {±2^p}

  sparse_shift.py
    FixedSparseShiftQuantize    — percentile threshold, exact sparsity ratio
    LearnableSparseShiftQuantize — learnable threshold (log-parameterized)
    SparseShiftConv2d           — W ∈ {0, ±2^p}, sparse_mode ∈ {fixed, learnable}

  pot_bn.py
    ScaleToPoT          — STE: γ/σ → sign * 2^round(log2|γ/σ|), p clamped [-15,15]
    PoTBatchNorm2d      — BN with PoT scale, warmup support
    set_bn_epoch()      — sets current_epoch on all PoTBatchNorm2d in model

src/models/
  resnet20.py         → ResNet20          (Phase 1: mul/shift/apot/denseshift)
  resnet20_potbn.py   → ResNet20PoTBN     (Phase 3: shift conv + PoT-BN)
  resnet20_sparse.py  → ResNet20Sparse    (Phase 4: sparse shift + std BN)
  resnet20_full.py    → ResNet20SparsePoTBN (Phase 5: sparse shift + PoT-BN)
```

---

## ResNet20SparsePoTBN (Phase 5 Final Model)

```
Input (B, 3, 32, 32)
  │
  ▼ nn.Conv2d(3→16, 3×3)          ← always mul
  ▼ PoTBatchNorm2d(16)            ← shift after warmup
  ▼ ReLU
  │
  ├─ Stage 1: 3 × BasicBlock(16, stride=1)
  │     conv1: SparseShiftConv2d  ← W ∈ {0, ±2^p}
  │     bn1:   PoTBatchNorm2d     ← scale → ±2^q
  │     conv2: SparseShiftConv2d
  │     bn2:   PoTBatchNorm2d
  │     shortcut: identity (no dim change)
  │
  ├─ Stage 2: 3 × BasicBlock(32, stride=2 for first)
  │     same as above; shortcut uses 1×1 SparseShiftConv2d + PoTBN
  │
  ├─ Stage 3: 3 × BasicBlock(64, stride=2 for first)
  │
  ▼ AdaptiveAvgPool2d(1)          ← 1 add per element
  ▼ Flatten
  ▼ nn.Linear(64→10)              ← always mul
Output (B, num_classes)
```

---

## SparseShiftConv2d

```python
SparseShiftConv2d(
    in_channels, out_channels, kernel_size, stride, padding,
    sparse_mode = "fixed" | "learnable",
    sparsity_ratio = 0.5,           # only for fixed mode
    init_threshold = 0.05,          # only for learnable mode
)
```

- `weight`: full-precision latent weight (trained with Adam/SGD)
- `log_threshold`: learnable scalar (learnable mode only); threshold = exp(log_threshold)
- Forward: quantize weight → {0, ±2^p}, apply conv
- `get_actual_sparsity()`: fraction of quantized weights == 0

---

## PoTBatchNorm2d

```python
PoTBatchNorm2d(
    num_features,
    use_pot_after_epoch = 30,   # warmup epochs; 0 = PoT from start
)
```

- `current_epoch`: set externally via `set_bn_epoch(model, epoch)` each epoch
- `_should_use_pot()`: `current_epoch >= use_pot_after_epoch`
- Forward (training): compute mean/var from batch, update running stats
- Forward (eval): use running_mean / running_var
- Scale: `scale = round_to_PoT(γ/σ)` when PoT active, else `γ/σ` (float)
- Output: `scale * (x - mean) + bias`

---

## Input/Output Shapes (CIFAR-10)

| Layer | Output shape |
|---|---|
| Input | (B, 3, 32, 32) |
| conv1 + bn1 + relu | (B, 16, 32, 32) |
| Stage 1 | (B, 16, 32, 32) |
| Stage 2 | (B, 32, 16, 16) |
| Stage 3 | (B, 64, 8, 8) |
| AvgPool | (B, 64, 1, 1) |
| Flatten | (B, 64) |
| FC | (B, 10) |
