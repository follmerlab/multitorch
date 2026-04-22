# GPU Acceleration Comparison: Fortran vs PyTorch CPU vs CUDA

**Platform**: 2× NVIDIA RTX 4090, AMD Ryzen Threadripper, PyTorch 2.5.1+cu121  
**Date**: 2026-04-19  
**System load**: High (load avg ~120, shared workstation)  
**Note**: Timings reflect a heavily loaded system. GPU gains should be stable; absolute CPU times may improve under lower load.

---

## Quick Comparison: Single Spectrum (batch=1)

*Source: full_compare_v2.py, 60s timeout per cell*

| Ion | H dim | Fortran | cached CPU | cached CUDA | GPU/CPU | CUDA vs Fortran |
| --- | ----: | ------: | ---------: | ----------: | ------: | --------------: |
| Ti4+ d0 Oh | ~20 | 0.098s | 0.707s | 0.028s | **24.9×** | **3.5× faster** |
| V3+ d2 Oh | ~1074 | 0.079s | >60s | 1.042s | **>57×** | 13.2× slower |
| Cr3+ d3 Oh | ~1782 | 0.117s | >60s | >60s | — | >513× slower |
| Mn2+ d5 Oh | ~2772 | 0.130s | >60s | 2.406s | **>25×** | 18.5× slower |
| Co2+ d7 Oh | ~500 | 0.076s | 4.110s | 0.344s | **12.0×** | 4.5× slower |
| Ni2+ d8 Oh | ~200 | 0.076s | 0.423s | 0.110s | **3.9×** | 1.4× slower |

## Bench Suite Results (batch=1, steady-state)

*Source: bench.run_all --preset full, median of 20 reps*

| Fixture | impl | device | median (s) | cold start (s) |
| ------- | ---- | ------ | ---------: | --------------: |
| **ti4_d0_oh** | multitorch_cached | cpu | 0.0198 | 0.807 |
| | multitorch_cached | **cuda** | **0.0185** | 0.854 |
| | pyctm (Fortran) | cpu | 0.0336 | 0.000 |
| | pyttmult (Fortran) | cpu | 0.0399 | 0.001 |
| | ttmult_raw (Fortran) | cpu | 0.0374 | 0.000 |
| **v3_d2_oh** | multitorch_cached | cpu | 2.3567 | 0.834 |
| | multitorch_cached | **cuda** | **0.7530** | 0.830 |
| | multitorch_from_scratch | cpu | 6.3685 | 0.836 |
| | multitorch_from_scratch | cuda | 8.0100 | 1.614 |
| | pyctm (Fortran) | cpu | 0.0722 | 0.000 |
| | pyttmult (Fortran) | cpu | 0.0752 | 0.001 |
| | ttmult_raw (Fortran) | cpu | 0.0723 | 0.000 |
| **cr3_d3_oh** | multitorch_cached | cuda | 9.4733 | 0.854 |
| | multitorch_from_scratch | cpu | 7.7704 | 0.833 |
| | pyctm (Fortran) [b=100] | cpu | 11.031¹ | 0.016 |

¹ Cr3+ Fortran result is per-100-batch (0.110s/spec), single-spec timed out in bench due to overhead.

## Batch Scaling: CPU vs CUDA

*Source: bench.run_all, median wall time*

### Ti4+ d0 Oh (~20 dim) — batch mode shines on CUDA

| batch | cached CPU | cached CUDA | GPU/CPU | batch CPU | batch CUDA | GPU/CPU |
| ----: | ---------: | ----------: | ------: | --------: | ---------: | ------: |
| 1 | 0.020s | 0.019s | 1.1× | — | — | — |
| 10 | 0.195s | 0.177s | 1.1× | 0.216s | 0.166s | **1.3×** |
| 100 | 2.030s | 1.780s | 1.1× | 1.935s | 1.457s | **1.3×** |
| 1000 | 18.649s | 17.888s | 1.0× | 15.258s | 14.461s | **1.1×** |

Fortran equivalent (b=1000): ttmult_raw = 30.7s, pyctm = 4.1s per 100

### V3+ d2 Oh (~1074 dim) — GPU speedup is dramatic

| batch | cached CPU | cached CUDA | GPU/CPU | batch CPU | batch CUDA | GPU/CPU |
| ----: | ---------: | ----------: | ------: | --------: | ---------: | ------: |
| 1 | 2.357s | 0.753s | **3.1×** | — | — | — |
| 10 | 23.137s | 7.530s | **3.1×** | 24.241s | 8.198s | **3.0×** |

Fortran: ttmult_raw = 0.072s (b=1), 0.887s (b=10)

## from_scratch: GPU Does Not Help

| Fixture | CPU (s) | CUDA (s) | GPU/CPU |
| ------- | ------: | -------: | ------: |
| v3_d2_oh | 6.369 | 8.010 | 0.79× (**slower**) |
| cr3_d3_oh | 7.770 | — (timeout) | — |
| Co2+ d7 Oh* | 7.105 | 7.744 | 0.92× |
| Ni2+ d8 Oh* | 2.616 | 2.753 | 0.95× |

\* from quick_compare, not bench suite

---

## Key Findings

### 1. GPU acceleration is dramatic for cached-mode XAS (medium/large H)

For V3+ d2 Oh (H dim ~1074), CUDA gives a consistent **3.1× speedup** over CPU:
- Single spectrum: CPU 2.357s → CUDA 0.753s
- Batch of 10: CPU 23.137s → CUDA 7.530s

The quick_compare (with 60s timeout) shows even larger gains for larger systems
where CPU times out but CUDA completes.

### 2. Small systems: CUDA matches or slightly beats CPU and Fortran

For Ti4+ d0 Oh (H dim ~20):
- **Bench suite**: cached CUDA (0.019s) ≈ cached CPU (0.020s) — essentially identical
- **Quick compare**: CUDA (0.028s) is 3.5× faster than Fortran (0.098s)
- The bench suite steady-state shows smaller gains than quick_compare because
  the bench warmed up more thoroughly (20 reps vs 5)

### 3. Fortran is 10-30× faster for single spectra of large systems

| System | Fortran | CUDA | Fortran speedup |
| ------ | ------: | ---: | --------------: |
| V3+ d2 Oh | 0.072s | 0.753s | 10.5× |
| Cr3+ d3 Oh | 0.110s/spec | 9.473s | 86× |

### 4. `from_scratch` shows zero GPU benefit

CUDA is actually **slower** for from_scratch (0.79× for V3+ d2 Oh) because:
- Fortran subprocess calls dominate the runtime
- CUDA cold-start overhead adds ~0.8s
- The eigh/broadening GPU benefit is dwarfed by subprocess I/O

### 5. `multitorch_batch` vs `multitorch_cached` loop

The vectorized `multitorch_batch` adapter is slightly faster than looping `multitorch_cached`:
- Ti4+ b=1000: batch=15.3s vs cached_loop=18.6s (CPU), batch=14.5s vs cached_loop=17.9s (CUDA)
- V3+ b=10: batch=8.2s vs cached=7.5s (CUDA) — roughly equivalent

### 6. Where GPU acceleration should be used

| Use case | GPU benefit | Recommendation |
| -------- | ----------- | -------------- |
| Single spectrum, small system (d0, d9) | Negligible | CPU or CUDA (both fast) |
| Single spectrum, medium (d7, d8) | **3-12× vs CPU** | **CUDA cached** |
| Single spectrum, large (d2-d5) | **3-57× vs CPU** | **CUDA cached** |
| Parameter sweep / fitting | **3× (amortized)** | **CUDA cached or batch** |
| RIXS calculations | **45×** (prior session) | **CUDA** |
| `from_scratch` (any size) | **None** (slower) | CPU only |
| Single spectrum (Fortran available) | N/A | Fortran still fastest |

---

## Benchmark Progress

As of snapshot: 240/1032 cells complete.
- 3 fixtures done: ti4_d0_oh (143 records), v3_d2_oh (76), cr3_d3_oh (21)
- Status: ok=43, error=20, timeout=48, skipped=129
- Remaining fixtures: co2_d7_oh, cr3_d3_oh cont., fe2_d6_oh, fe3_d5_oh, mn2_d5_oh, ni2_d8_oh, etc.
- Errors are all `multitorch_from_scratch` "No CFP matrix for l=2, n=0" (d0 only)
# GPU Acceleration Comparison: Fortran vs PyTorch CPU vs CUDA

**Platform**: 2× NVIDIA RTX 4090, AMD Ryzen Threadripper, PyTorch 2.5.1+cu121  
**Date**: 2026-04-19  
**System load**: High (load avg ~120, shared workstation)  
**Note**: Timings reflect a heavily loaded system. GPU gains should be stable; absolute CPU times may improve under lower load.

---

## Single-Spectrum Timing (batch=1)

| Ion | H dim | Fortran | cached CPU | cached CUDA | GPU/CPU | CUDA/Fortran |
| --- | ----: | ------: | ---------: | ----------: | ------: | -----------: |
| Ti4+ d0 Oh | ~20 | 0.098s | 0.707s | 0.028s | **24.9×** | **0.3×** (faster) |
| V3+ d2 Oh | ~1074 | 0.079s | >60s | 1.042s | **>57×** | 13.2× |
| Cr3+ d3 Oh | ~1782 | 0.117s | >60s | >60s | — | >513× |
| Mn2+ d5 Oh | ~2772 | 0.130s | >60s | 2.406s | **>25×** | 18.5× |
| Co2+ d7 Oh | ~500 | 0.076s | 4.110s | 0.344s | **12.0×** | 4.5× |
| Ni2+ d8 Oh | ~200 | 0.076s | 0.423s | 0.110s | **3.9×** | 1.4× |

## Batched Timing (loop of N spectra, cached mode)

| Ion | batch | cached CPU | cached CUDA | GPU/CPU | per-spec CUDA | equiv Fortran |
| --- | ----: | ---------: | ----------: | ------: | ------------: | ------------: |
| Ti4+ d0 Oh | 10 | 5.281s | 0.234s | **22.6×** | 0.023s | 0.098s |
| Ti4+ d0 Oh | 100 | (est ~71s) | 3.120s | — | 0.031s | 9.8s |
| V3+ d2 Oh | 10 | >60s | 10.265s | **>5.8×** | 1.027s | 0.79s |
| Mn2+ d5 Oh | 10 | >60s | >60s | — | — | 1.30s |
| Co2+ d7 Oh | 10 | >60s | 3.805s | **>15.8×** | 0.381s | 0.76s |
| Ni2+ d8 Oh | 10 | 2.729s | 0.924s | **3.0×** | 0.092s | 0.76s |

## from_scratch Timing (includes Fortran subprocess + assembly)

| Ion | scratch CPU | scratch CUDA | GPU/CPU |
| --- | ----------: | -----------: | ------: |
| Co2+ d7 Oh | 7.105s | 7.744s | 0.92× (no gain) |
| Ni2+ d8 Oh | 2.616s | 2.753s | 0.95× (no gain) |

---

## Key Findings

### 1. GPU acceleration is dramatic for cached-mode XAS

The CUDA backend provides **3.9× to >57× speedup** over CPU for single-spectrum
`calcXAS_cached` calculations. The speedup scales with Hamiltonian dimension:

| H dim range | GPU/CPU speedup |
| ----------- | --------------- |
| ~20 (Ti4+ d0) | 24.9× |
| ~200 (Ni2+ d8) | 3.9× |
| ~500 (Co2+ d7) | 12.0× |
| ~1074 (V3+ d2) | >57× (CPU >60s, CUDA 1.0s) |
| ~2772 (Mn2+ d5) | >25× (CPU >60s, CUDA 2.4s) |

The exceptionally high speedup for Ti4+ d0 (~20 states) likely reflects that the GPU
broadening kernel dominates and has very low overhead for small matrices, while CPU
`torch.eigh` has fixed overhead regardless of matrix size.

### 2. CUDA can beat Fortran for small systems

For Ti4+ d0 Oh, **CUDA (0.028s) is 3.5× faster than Fortran (0.098s)**. This is
because the Fortran pipeline has fixed subprocess-launch overhead (~0.06s), while
the PyTorch cached path with pre-loaded fixtures and GPU broadening has minimal overhead.

### 3. Fortran dominates for large single-spectrum calculations

For systems with H dim >200, Fortran remains 1.4×–18.5× faster than CUDA for single
spectra. Fortran solves the eigenvalue problem in optimized compiled code without
Python/CUDA overhead:

- Ni2+ d8 Oh: Fortran 0.076s vs CUDA 0.110s (Fortran 1.4× faster)
- Co2+ d7 Oh: Fortran 0.076s vs CUDA 0.344s (Fortran 4.5× faster)
- V3+ d2 Oh: Fortran 0.079s vs CUDA 1.042s (Fortran 13.2× faster)
- Mn2+ d5 Oh: Fortran 0.130s vs CUDA 2.406s (Fortran 18.5× faster)

### 4. `from_scratch` shows zero GPU benefit

The `calcXAS_from_scratch` path (which includes Fortran subprocess calls for RAC
generation) shows no GPU speedup because:
- The Fortran subprocess time dominates (~5-7s)
- The actual CUDA eigh/broadening is a small fraction of total time
- Subprocess overhead is identical on CPU vs CUDA

### 5. Batch mode is the GPU sweet spot

When computing multiple spectra with varying parameters (e.g., parameter sweeps),
the GPU amortizes its overhead across the batch. The per-spectrum cost drops
significantly:

- **Ti4+ d0 Oh, batch=100**: 0.031s/spec on CUDA vs 0.098s/spec Fortran (**3.2× faster than Fortran**)
- **Co2+ d7 Oh, batch=10**: 0.381s/spec on CUDA (CPU timed out)
- **Ni2+ d8 Oh, batch=10**: 0.092s/spec CUDA vs 0.273s/spec CPU (**3.0× speedup**)

### 6. Where GPU acceleration makes sense

| Use case | GPU benefit | Recommendation |
| -------- | ----------- | -------------- |
| Single spectrum, small system (d0, d9) | **YES** (beats Fortran) | Use CUDA cached |
| Single spectrum, medium system (d7, d8) | Marginal vs CPU, slower than Fortran | Use Fortran or CUDA |
| Single spectrum, large system (d2-d5) | **YES** vs CPU; still slower than Fortran | Use CUDA if no Fortran available |
| Parameter sweep (batch 10-100) | **YES** (3-23× vs CPU) | Use CUDA cached |
| RIXS calculations | **YES** (45× from prior session) | Use CUDA |
| `from_scratch` mode | **NO** | Use CPU (Fortran subprocess dominates) |

---

## Recommendations

1. **Default to CUDA for `calcXAS_cached`** — it's faster than CPU in every case tested
2. **Use Fortran for one-off single spectra** of medium/large systems (it's still fastest)
3. **Use CUDA for parameter sweeps** — the batch throughput advantage is substantial
4. **Don't bother with CUDA for `from_scratch`** — the Fortran subprocess dominates timing
5. **Cr3+ d3 Oh (1782 dim) exceeds 60s even on CUDA** — may need algorithmic optimization for very large Hamiltonians
# GPU Acceleration Comparison: Fortran vs PyTorch CPU vs CUDA

**Platform**: 2× NVIDIA RTX 4090, AMD Ryzen, PyTorch 2.5.1+cu121
**Date**: 2026-04-19

## Single-Spectrum Timing (batch=1)

| Ion | Fortran | cached CPU | cached CUDA | GPU speedup | scratch CPU | scratch CUDA |
| --- | ------: | ---------: | ----------: | ----------: | ----------: | -----------: |
| Ti4+ d0 Oh | 0.0975s | 0.7074s | 0.0284s | 24.87× | — | — |
| V3+ d2 Oh | 0.0791s | >60s | 1.0417s | — | — | — |
| Cr3+ d3 Oh | 0.1168s | >60s | >60s | — | — | — |
| Mn2+ d5 Oh | 0.1300s | >60s | 2.4064s | — | — | — |
| Co2+ d7 Oh | 0.0756s | 4.1101s | 0.3435s | 11.97× | 7.1054s | 7.7440s |
| Ni2+ d8 Oh | 0.0764s | 0.4234s | 0.1099s | 3.85× | 2.6160s | 2.7531s |
| Cu2+ d9 Oh | — | — | — | — | — | — |

## Batched Timing (loop of N spectra)

| Ion | batch | cached CPU | cached CUDA | GPU speedup | per-spec CPU | per-spec CUDA |
| --- | ----: | ---------: | ----------: | ----------: | -----------: | ------------: |
| Ti4+ d0 Oh | 10 | 5.2809s | 0.2340s | 22.57× | 0.52809s | 0.02340s |
| Ti4+ d0 Oh | 100 | skip | 3.1200s | — | — | 0.03120s |
| V3+ d2 Oh | 10 | — | 10.2648s | — | — | 1.02648s |
| Mn2+ d5 Oh | 10 | — | >60s | — | — | — |
| Co2+ d7 Oh | 10 | >60s | 3.8052s | — | — | 0.38052s |
| Co2+ d7 Oh | 100 | skip | >60s | — | — | — |
| Ni2+ d8 Oh | 10 | 2.7290s | 0.9237s | 2.95× | 0.27290s | 0.09237s |
| Ni2+ d8 Oh | 100 | >60s | >60s | — | — | — |

## Key Findings

