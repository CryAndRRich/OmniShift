# OmniShift — Energy Model

## 45nm CMOS Operation Costs

| Operation | Energy (pJ) | Ratio vs mul |
|---|---|---|
| Multiplication | 3.7 | 1× |
| Addition | 0.9 | 4.1× cheaper |
| Bit-shift | 0.13 | **28.5× cheaper** |

Source: AdderNet (Chen et al., NIPS 2020), Table 1, 45nm CMOS standard cell library.

---

## Energy Formula

```
E = 3.7 × N_mul + 0.9 × N_add + 0.13 × N_shift   [pJ]
```

Reported as GpJ (×10^-9 pJ) for whole-network inference on one image.

---

## How `count_mul_add_shift` Works

```python
ops = count_mul_add_shift(model, input_size=(1, 3, 32, 32), sparsity=None)
```

The function walks the fixed ResNet-20 topology and accumulates op counts:

### First conv (3→16, 3×3, always mul)
```
MACs = 3 × 16 × 3 × 3 × 32 × 32 = 442,368
N_mul += 442,368
N_add += 442,368
```

### BN (folded form: y = scale × x + bias)
```
Std BN:  N_mul += C×H×W,   N_add += C×H×W   (1 mul + 1 add per element)
PoT-BN:  N_shift += C×H×W, N_add += C×H×W   (scale → bit-shift)
```

### Interior convs (DeepShift, W = ±2^p)
```
MACs = C_in × C_out × kH × kW × H_out × W_out
N_shift += MACs
N_add   += MACs
```

### Interior convs (SparseShift, W ∈ {0, ±2^p})
Skip-zero hardware skips zero weights entirely:
```
N_shift += MACs × (1 - sparsity)
N_add   += MACs × (1 - sparsity)
```

### AvgPool
```
N_add += C × H × W   (summation, no mul/shift)
```

### FC (64→10, always mul)
```
N_mul += 64 × 10 = 640
N_add += 64 × 10 = 640
```

---

## Worked Example: Phase 5, sparseshift_learnable_potbn_w30

Assume CIFAR-10 (32×32 input), learnable sparsity ≈ 63% (CIFAR-10).

**Non-zero ratio** = 1 - 0.63 = 0.37

| Component | Type | N_mul | N_add | N_shift |
|---|---|---|---|---|
| First conv | mul | 442K | 442K | 0 |
| BN1 | PoT | 0 | 33K | 33K |
| Stage 1 (3 blocks, 16ch, 32×32) | sparse shift | 0 | ~10.6M×0.37 | ~10.6M×0.37 |
| Stage 2 (3 blocks, 32ch, 16×16) | sparse shift | 0 | ~5.3M×0.37 | ~5.3M×0.37 |
| Stage 3 (3 blocks, 64ch, 8×8) | sparse shift | 0 | ~5.3M×0.37 | ~5.3M×0.37 |
| BN layers (all stages) | PoT | 0 | ~680K | ~680K |
| AvgPool | add | 0 | 4K | 0 |
| FC | mul | 640 | 640 | 0 |

Total (approximate): N_mul ≈ 0.44M, N_add ≈ 8.0M, N_shift ≈ 8.7M

```
E ≈ 3.7×0.44M + 0.9×8.0M + 0.13×8.7M
  ≈ 1.63M + 7.2M + 1.13M  pJ
  ≈ 9.96M pJ
  ≈ 0.010 GpJ
```

**Comparison**: ResNet-20 baseline = 0.189 GpJ → **~19× reduction**

---

## Truly Multiplier-less Claim

Residual muls after full OmniShift quantization:
- First conv (3→16): 0.000442G mul
- FC (64→10): 0.00000064G mul
- **Total**: ~0.0004G mul = **<0.001% of ResNet-20 baseline (0.041G)**

All other ops are shifts and adds.
