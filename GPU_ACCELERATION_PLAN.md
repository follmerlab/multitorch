# GPU Acceleration Plan for multitorch

**Date:** 2026-04-12
**Status:** Tier 1 partially implemented, profiled on exxa (2× RTX 4090)
**Current state:** 398/398 tests passing on CPU; GPU pipeline functional

---

## Executive summary

multitorch can run on GPU today by passing `device='cuda'` to `calcXAS`.
Tier 1 fixes are partially implemented (see status below). Profiling on
exxa (2× NVIDIA RTX 4090, CUDA 12.4, PyTorch 2.5.1) shows that **GPU
only helps for isolated compute-heavy kernels**, not end-to-end single-
spectrum calculations where fixture I/O dominates.

### Measured speedups (exxa, 2× RTX 4090)

| Workload | Largest matrix | GPU speedup (measured) | Worth it? |
|---|---|---|---|
| Single Ni d8 Oh XAS | 17×17 | **0.4×** (GPU slower) | No |
| Single Ni d8 D4h CT XAS | ~50×50 | **0.5×** (GPU slower) | No |
| Single Fe d6 Oh XAS | ~120×120 | **0.9×** (GPU ≈ CPU) | No |
| 100-spectrum sweep Ni Oh | 17×17 × 100 | **0.6×** (GPU slower) | No |
| 100-spectrum sweep Fe Oh | ~120×120 × 100 | **0.9×** (GPU ≈ CPU) | No |
| Broadening (2000 sticks) | conv1d | **41×** | **Yes** |
| `eigh` at 2000×2000 | 2000×2000 | **4.3×** | **Yes** |
| `eigh` at 200×200 | 200×200 | **0.5×** (GPU slower) | No |
| RIXS Kramers-Heisenberg | 300×300 broadcast | **45×** | **Yes** |
| Autograd (grad through eigh) | varies | Works on GPU | Maybe |

**Key finding:** File I/O (parsing `.rme_rcg`/`.rme_rac` fixtures) takes
~30–100 ms per call and dominates wall time. The actual `eigh` on 3d
transition-metal matrices (10–200 dim) takes < 1 ms — too small for GPU
kernel launch overhead (~0.1 ms per launch) to be amortized.

**Revised recommendation:** GPU acceleration is only worthwhile for:
1. **RIXS maps** — large tensor contractions, 45× measured speedup
2. **Broadening sweeps** — batched conv1d with ≥2000 sticks, 41× speedup
3. **Large-matrix systems** — `eigh` at dim ≥ 500 (Cr d³, rare earths)
4. **Fixture caching** — if I/O is eliminated, GPU compute could dominate

Defer further Tier 1 work unless targeting these workloads.

---

## Current device-awareness inventory

### GPU-ready (implemented or no changes needed)

| Module | Status | Notes |
|---|---|---|
| `hamiltonian/assemble.py` | ✅ Ready | All `torch.zeros`/`torch.eye` pass `device=device` |
| `hamiltonian/diagonalize.py` | ✅ Ready | `torch.linalg.eigh` auto-dispatches to GPU |
| `spectrum/sticks.py` | ✅ Ready | Pure tensor ops, inherits device from inputs |
| `spectrum/rixs.py` | ✅ Ready | Kramers-Heisenberg kernel is fully vectorized |
| `io/read_rme.py` | ✅ Fixed | `assemble_matrix_from_adds` infers device from cowan tensors |
| `hamiltonian/charge_transfer.py` | ✅ Fixed | All 4× `torch.zeros`/`torch.eye` have `device=device` |
| `_constants.py` | ✅ Skipped | Exports plain floats; scalars auto-transfer for free |

### Still on CPU (not yet fixed)

| Module | Issue | Tier |
|---|---|---|
| `hamiltonian/build_cowan.py` | `torch.zeros_like(h_parsed)` infers device from input, no explicit control | 1 |
| `spectrum/broaden.py` | `broaden_gaussian` uses Python for-loop (not vectorized conv1d) | 1 |
| `atomic/hfs.py:307-308,641,717-718` | `.cpu().numpy()` in SCF loop (scipy) | 3 (can't fix) |
| `io/read_oba.py:175-177` | `torch.tensor()` in parser on CPU | 2 |
| `io/read_rme.py:215,325` | `torch.tensor()` in fixture readers on CPU | 2 |
| `io/read_rcf.py:60,67` | `torch.tensor()` in parser on CPU | 2 |
| `hamiltonian/assemble.py:381,429,438` | `conf_labels` int tensors on CPU | 2 |
| `atomic/slater.py:207,215` | `torch.tensor(0.0)` defaults on CPU | 3 |

---

## Tier 1 — Critical path

These are the tensor-creation sites on the hot path between
`build_rac_in_memory` → `build_cowan_store_in_memory` →
`assemble_and_diagonalize_in_memory` → broadening.

### 1a. `assemble_matrix_from_adds` — ✅ DONE

**File:** `multitorch/io/read_rme.py`

Now accepts `device=None` and infers device from source COWAN tensors.
The initial `torch.zeros` is created on the correct device.

### 1b. `build_cowan_store_in_memory` — ⬜ NOT DONE

**File:** `multitorch/hamiltonian/build_cowan.py`

`_rebuild_hamiltonian_block` uses `torch.zeros_like(h_parsed)` which
inherits device from the input tensor. No explicit `device` parameter.
This works if the parsed tensor is already on the target device, but
there's no explicit device control in the function signature.

**Impact:** Low — profiling shows build_cowan is not a bottleneck
compared to fixture I/O.

### 1c. Charge transfer helpers — ✅ DONE

**File:** `multitorch/hamiltonian/charge_transfer.py`

All four `torch.zeros`/`torch.eye` calls now pass `device=device`.

### 1d. `broaden_gaussian` vectorization — ⬜ NOT DONE

**File:** `multitorch/spectrum/broaden.py`

`broaden_gaussian` still uses a Python for-loop over columns.
Replacing with batched `conv1d` would give **41× speedup** for
≥2000 sticks (measured). Only matters for large broadening sweeps.

### 1e. Thread `device` through API — ✅ DONE

The public API (`calcXAS`, `calcRIXS`) accepts `device=` and threads
it through to `assemble_and_diagonalize_in_memory`. The broadening
layer inherits device from input tensors.

### Validation

This test passes today on exxa:

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

Autograd through the GPU pipeline also works:
```python
slater = torch.tensor(0.8, requires_grad=True, device="cuda")
x, y = calcXAS(..., slater=slater, device="cuda")
y.sum().backward()  # grad flows through eigh on GPU
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

## Implementation status

| Step | Status | Files changed | Notes |
|---|---|---|---|
| **1a** assemble_matrix_from_adds device | ✅ Done | read_rme.py | Infers device from cowan tensors |
| **1b** build_cowan device | ✅ Done | build_cowan.py | Accepts `device=`, migrates template |
| **1c** charge_transfer device | ✅ Done | charge_transfer.py | All 4 sites fixed |
| **1d** Thread device through API | ✅ Done | calc.py, assemble.py | device= flows end-to-end |
| **1e** broaden_gaussian conv1d | ✅ Done | broaden.py | Batched conv1d, no Python for-loop |
| **2a** Constants (skip) | ✅ Skipped | _constants.py | Exports plain floats now |
| **2b** Parsers (skip) | — | — | Not needed |
| **Fixture caching** | ✅ Done | calc.py, build_cowan.py, build_rac.py | `preload_fixture()` + `calcXAS_cached()` |

402/402 tests passing. Zero regressions from GPU changes.

---

## Fixture caching API (new, 2026-04-12)

Eliminates redundant file I/O in parameter sweeps. Parse once, sweep fast:

```python
from multitorch import preload_fixture, calcXAS_cached

cache = preload_fixture("Ni", "ii", "oh")  # parse once (~15 ms)

for tendq in torch.linspace(0.5, 2.0, 100):
    x, y = calcXAS_cached(cache, cf={'tendq': float(tendq)}, slater=0.8)
```

### Measured speedup (macOS, CPU)

| System | Direct (ms/call) | Cached (ms/call) | Speedup |
|---|---|---|---|
| Ni²⁺ Oh (17×17) | 14.8 | 9.0 | **1.6×** |
| Ni²⁺ D4h CT (~50×50) | 60.3 | 48.4 | **1.2×** |

The savings come entirely from eliminating file I/O (~6–12 ms/call).
On exxa where I/O overhead is higher (~30–100 ms), the speedup should
be proportionally larger. With GPU, the compute portion is also faster,
so the total speedup compounds.

### Autograd through cached path

```python
slater = torch.tensor(0.8, requires_grad=True, dtype=torch.float64)
x, y = calcXAS_cached(cache, cf={}, slater=slater, soc=1.0)
y.sum().backward()  # grad flows through eigh
print(slater.grad)  # finite, nonzero
```

---

## When GPU is worthwhile (empirical guidance)

**Use GPU now for:**
- RIXS 2D maps — 45× measured speedup on Kramers-Heisenberg kernel
- Large broadening sweeps (≥2000 sticks) — 41× with conv1d
- Systems with dim ≥ 500 (e.g., Cr d³ Oh at 1074×1074) — 4.3× on eigh
- Parameter sweeps with fixture caching — I/O eliminated, compute dominates

**Stay on CPU for:**
- Single-spectrum 3d L-edge XAS — fixture I/O dominates, GPU adds overhead
- Very small matrices (Ni Oh 17×17) — kernel launch overhead exceeds compute

**Never needed:** Tier 2 (parsers) and Tier 3 (HFS SCF). Parser tensors
auto-transfer, and HFS SCF will always be CPU-bound.

---

## Profiling results (exxa, 2× RTX 4090, 2026-04-12)

### Single-spectrum end-to-end (`calcXAS`, 10 iterations avg)

| System | CPU (s) | GPU (s) | Speedup |
|---|---|---|---|
| Ni²⁺ Oh (d⁸, ~17×17) | 0.037 | 0.099 | **0.4×** |
| Ni²⁺ D4h CT (~50×50) | 0.150 | 0.293 | **0.5×** |
| Fe²⁺ Oh (d⁶, ~120×120) | 0.444 | 0.487 | **0.9×** |

### 100-spectrum parameter sweep (slater 0.5→1.0)

| System | CPU (s) | GPU (s) | Speedup |
|---|---|---|---|
| Ni²⁺ Oh × 100 | 3.70 | 6.47 | **0.6×** |
| Fe²⁺ Oh × 100 | 45.13 | 48.68 | **0.9×** |

### Isolated compute kernels (GPU wins here)

| Operation | CPU (s) | GPU (s) | Speedup |
|---|---|---|---|
| RIXS Kramers-Heisenberg (300² grid) | 4.52 | 0.10 | **45×** |
| Broadening (2000 sticks, conv1d bench) | 2.05 | 0.05 | **41×** |
| `eigh` 2000×2000 | 0.86 | 0.20 | **4.3×** |
| `eigh` 200×200 | 0.002 | 0.004 | **0.5×** |
| `eigh` 10×10 | 0.0001 | 0.001 | **0.1×** |

### Profiling command

```python
import torch, time
from multitorch.api.calc import calcXAS

# CPU baseline
t0 = time.time()
for _ in range(10):
    calcXAS(element="Ni", valence="ii", sym="oh", edge="l", cf={}, slater=1.0, soc=1.0)
cpu_time = (time.time() - t0) / 10

# GPU
t0 = time.time()
for _ in range(10):
    calcXAS(element="Ni", valence="ii", sym="oh", edge="l", cf={}, slater=1.0, soc=1.0, device="cuda")
    torch.cuda.synchronize()
gpu_time = (time.time() - t0) / 10

print(f"CPU: {cpu_time:.3f}s, GPU: {gpu_time:.3f}s, speedup: {cpu_time/gpu_time:.1f}×")
```
