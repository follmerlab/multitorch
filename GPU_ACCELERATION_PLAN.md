# GPU Acceleration Plan for multitorch

**Date:** 2026-04-12
**Status:** Assessment complete, implementation not started
**Current state:** 398/398 tests passing on CPU; code is functionally correct

---

## Executive summary

multitorch can run on GPU today by passing `device='cuda'` to `calcXAS`,
but ~40 tensor-creation sites lack device propagation, causing implicit
CPU→GPU transfers. The fix is mechanical (add `device=` kwargs) but
touches many files. Whether it's *worth* doing depends on workload:

| Workload | Largest matrix | GPU speedup estimate | Worth it? |
|---|---|---|---|
| Single Ni d8 Oh XAS | 17×17 | ~0× (overhead dominates) | No |
| Single Cr d3 Oh XAS | 1074×1074 | ~5–10× on eigh alone | Maybe |
| Parameter sweep (100 spectra) | 1074×1074 × 100 | ~10–50× | Yes |
| RIXS 2D map (400×400 grid) | 192 MB broadcast | ~5–20× | Yes |
| Autograd optimization loop | varies | ~10–100× (batched) | Yes |

**Recommendation:** Implement Tier 1 (critical path) now. Defer Tier 2
and Tier 3 until profiling confirms they matter.

---

## Current device-awareness inventory

### Already GPU-ready (no changes needed)

| Module | Why it works |
|---|---|
| `hamiltonian/assemble.py` | All `torch.zeros`/`torch.eye` calls pass `device=device` |
| `hamiltonian/diagonalize.py` | Uses `torch.linalg.eigh` (auto-dispatches to GPU) |
| `spectrum/broaden.py` | Pure tensor ops, inherits device from inputs |
| `spectrum/sticks.py` | Pure tensor ops, inherits device from inputs |
| `spectrum/rixs.py` | Kramers-Heisenberg kernel is fully vectorized |

### Broken on GPU (needs fixes)

| Module | Issue | Tier |
|---|---|---|
| `io/read_rme.py:574` | `assemble_matrix_from_adds` creates `torch.zeros` on CPU | 1 |
| `hamiltonian/build_cowan.py:225` | `torch.zeros(h_parsed.shape)` on CPU | 1 |
| `hamiltonian/charge_transfer.py:87,100,139,179` | 4× `torch.zeros`/`torch.eye` on CPU | 1 |
| `_constants.py:12-24` | Global tensor constants always on CPU | 2 |
| `atomic/hfs.py:307-308,641,717-718` | `.cpu().numpy()` in SCF loop (scipy) | 3 (can't fix) |
| `io/read_oba.py:175-177` | `torch.tensor()` in parser on CPU | 2 |
| `io/read_rme.py:215,325` | `torch.tensor()` in fixture readers on CPU | 2 |
| `io/read_rcf.py:60,67` | `torch.tensor()` in parser on CPU | 2 |
| `hamiltonian/assemble.py:381,429,438` | `conf_labels` int tensors on CPU | 2 |
| `atomic/slater.py:207,215` | `torch.tensor(0.0)` defaults on CPU | 3 |

---

## Tier 1 — Critical path (high impact, ~2 hours)

These are the tensor-creation sites on the hot path between
`build_rac_in_memory` → `build_cowan_store_in_memory` →
`assemble_and_diagonalize_in_memory` → broadening. Fixing these gives
GPU acceleration for the eigenvalue solver and spectrum broadening —
the two operations where GPU actually helps.

### 1a. Add `device` parameter to `assemble_matrix_from_adds`

**File:** `multitorch/io/read_rme.py:536`

```python
# Before:
def assemble_matrix_from_adds(add_entries, cowan_section, n_bra, n_ket, scale=1.0):
    mat = torch.zeros(n_bra, n_ket, dtype=DTYPE)

# After:
def assemble_matrix_from_adds(add_entries, cowan_section, n_bra, n_ket, scale=1.0, device=None):
    mat = torch.zeros(n_bra, n_ket, dtype=DTYPE, device=device)
```

The `cowan_section` tensors and `add.coeff` scalars flow in from the
COWAN store — if those are on GPU, the slice-add operations will
auto-dispatch. The only thing that needs the explicit `device=` is the
initial `torch.zeros`.

**Callers to update:** `assemble.py:_process_triad` passes `device`
already — just thread it through to this function.

### 1b. Add `device` parameter to `build_cowan_store_in_memory`

**File:** `multitorch/hamiltonian/build_cowan.py:225`

```python
# Before:
h_new = torch.zeros(h_parsed.shape, dtype=DTYPE)

# After:
h_new = torch.zeros(h_parsed.shape, dtype=DTYPE, device=device)
```

Add `device='cpu'` to the function signature and plumb it through all
`torch.zeros` calls inside. The COWAN store is a `List[List[Tensor]]`
— all tensors in it should live on the target device.

### 1c. Add `device` to charge transfer helpers

**File:** `multitorch/hamiltonian/charge_transfer.py`

Four `torch.zeros` / `torch.eye` calls at lines 87, 100, 139, 179.
Add `device=device` to each. The `device` parameter should be threaded
from the caller in `assemble.py` (which already has it).

### 1d. Thread `device` through `_calcXAS_phase5` and `calcRIXS`

The public API already accepts `device='cpu'`. Verify it flows into:
- `build_cowan_store_in_memory(... device=device)`
- `assemble_matrix_from_adds(... device=device)` (via assemble.py)
- The broadening layer (already inherits from input tensors)

### Validation

After Tier 1, this test should pass:

```python
def test_calcXAS_device_cuda():
    """Full pipeline runs on CUDA without CPU fallback."""
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")
    x, y = calcXAS(
        element="Ni", valence="ii", sym="oh", edge="l",
        cf={}, slater=1.0, soc=1.0, device="cuda",
    )
    assert x.device.type == "cuda"
    assert y.device.type == "cuda"
```

---

## Tier 2 — Parser and constants cleanup (low impact, ~1 hour)

These are tensor-creation sites in file parsers and global constants.
They affect startup cost (one-time file→tensor transfer) but not
the hot loop. Fix only if profiling shows the transfer is a bottleneck.

### 2a. Make `_constants.py` tensors device-lazy

Replace global `torch.tensor(...)` constants with plain Python floats
and create device-aware tensors on demand:

```python
# Before:
k_B = torch.tensor(8.6173303e-05, dtype=DTYPE)

# After:
K_B_FLOAT = 8.6173303e-05
def k_B(device=None):
    return torch.tensor(K_B_FLOAT, dtype=DTYPE, device=device)
```

**Impact:** Every file that imports `k_B` or `RY_TO_EV` would need
updating. ~15 call sites. The alternative (and simpler) approach: keep
them as CPU tensors and let PyTorch auto-transfer. The cost is one
scalar CPU→GPU copy per use — negligible.

**Recommendation:** Skip this. Scalar constants auto-transfer for free.

### 2b. Add `device` to file parsers

`read_ban_output`, `read_cowan_store`, `read_rme_rac_full` all create
tensors on CPU. Adding `device=` would let them parse directly to GPU.

**Recommendation:** Skip this too. File I/O is inherently CPU-bound;
the tensors are small; and the one-time transfer after parsing is
negligible compared to the eigenvalue solve.

### 2c. Device-aware `conf_labels` in assembler

`assemble.py:381,429,438` creates `torch.int64` index tensors on CPU.
These are used for config-weight projection and never enter the
eigenvalue solver. Fix only if needed.

---

## Tier 3 — Structural limitations (can't fix easily)

### 3a. HFS SCF is CPU-only

`multitorch/atomic/hfs.py` calls `scipy.integrate.solve_ivp` for the
radial Schrödinger equation (lines 307-308). SciPy requires numpy
arrays on CPU. This is fundamental — fixing it would require replacing
SciPy's ODE solver with a pure-PyTorch Numerov integrator.

**Impact:** HFS SCF is only used in the "bootstrap-from-Z" path
(computing Slater integrals from scratch). The Phase 5 pipeline reads
pre-computed parameters from `.rcn31_out` and never calls HFS. So this
limitation does **not** affect the GPU-accelerated XAS/RIXS pipeline.

### 3b. Angular RME builders use numpy

`multitorch/angular/rme.py` computes Wigner symbols and CFPs in numpy,
then wraps results as `torch.as_tensor()` in the `torch_blocks.py`
wrappers. The angular coefficients are constants (no gradient flow), so
GPU acceleration here would only save the one-time numpy→torch transfer.

**Impact:** Negligible. Angular coefficients are computed once per
fixture and are small matrices (< 100×100).

---

## Implementation order and effort

| Step | Files changed | LOC changed | Test impact | Effort |
|---|---|---|---|---|
| **1a** assemble_matrix_from_adds device | read_rme.py, assemble.py | ~10 | None (default=None → CPU) | 30 min |
| **1b** build_cowan device | build_cowan.py | ~10 | None | 20 min |
| **1c** charge_transfer device | charge_transfer.py | ~8 | None | 15 min |
| **1d** Thread device through API | calc.py | ~15 | None | 30 min |
| **Test** Add GPU integration test | test_integration/ | ~20 | +1 test (skipped on CPU) | 15 min |
| **2a** Constants (skip) | — | — | — | — |
| **2b** Parsers (skip) | — | — | — | — |

**Total Tier 1:** ~2 hours, ~65 LOC, 0 test regressions (all changes
are additive `device=None` defaults).

---

## When to implement

**Now:** If you plan to run parameter sweeps or RIXS calculations on
exxa's GPUs. The eigenvalue solver (`torch.linalg.eigh`) and RIXS
kernel are the two operations that benefit most.

**Later:** If you're only running single-spectrum calculations. The
matrix sizes for most 3d ions (< 200×200) are too small for GPU
overhead to be worthwhile.

**Never needed:** Tier 2 and Tier 3. Parser tensors auto-transfer,
and HFS SCF will always be CPU-bound.

---

## Profiling command (run on exxa to decide)

```python
import torch, time
from multitorch.api.calc import calcXAS

# CPU baseline
t0 = time.time()
for _ in range(10):
    calcXAS(element="Ni", valence="ii", sym="oh", edge="l", cf={}, slater=1.0, soc=1.0)
cpu_time = (time.time() - t0) / 10

# GPU (after Tier 1 fixes)
t0 = time.time()
for _ in range(10):
    calcXAS(element="Ni", valence="ii", sym="oh", edge="l", cf={}, slater=1.0, soc=1.0, device="cuda")
    torch.cuda.synchronize()
gpu_time = (time.time() - t0) / 10

print(f"CPU: {cpu_time:.3f}s, GPU: {gpu_time:.3f}s, speedup: {cpu_time/gpu_time:.1f}×")
```

Run this for Ni Oh (small), Cr Oh (large), and a 100-spectrum sweep to
see where GPU starts winning.
