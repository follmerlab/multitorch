# Phase 3: Full Batch Diagonalization Pipeline

**Status:** Not Yet Implemented (Future Work)  
**Prerequisites:** Phase 2 Complete ✅  
**Expected Timeline:** 7-10 days  
**Expected Speedup:** Additional 2-3× on top of Phase 2 (total 5-10× vs sequential)

---

## Overview

Phase 3 extends batch processing through the **entire** `assemble_and_diagonalize_in_memory` workflow, not just the ground-state COWAN rebuild. While Phase 2 achieves 2-5× speedup via shared V(11) computation, Phase 3 would batch all triad processing for an additional 2-3× gain.

### Current State (Phase 2)

✅ **Implemented:**
- `batch_scale_atomic_params()` - (N,) parameter tensors
- `build_cowan_store_in_memory_batch()` - Shared V(11) residual, 2-3× faster
- `safe_eigh_batch()` - Vectorized eigendecomposition
- `calcXAS_batch()` - High-level API
- **Performance:** 2-5× speedup for 1000+ spectra

⬜ **Limitations:**
- `assemble_and_diagonalize_in_memory()` still runs N times sequentially
- Each parameter set triggers separate fixture parsing for final states
- N separate GPU kernel launches for transition matrix diagonalization
- Redundant transition operator assembly across similar systems

---

## Phase 3 Goals

### Primary Objectives
1. **Batch full triad processing** - Single pass through ground + final manifolds
2. **Eliminate redundant final-state parsing** - Fixture shared across N parameter sets
3. **GPU kernel consolidation** - 1 batched eigh call vs N sequential calls per triad
4. **Memory efficiency** - Smart batching to stay within GPU memory limits

### Performance Targets
- **Small systems** (Ni d8 Oh, 17×17): 5-8× total speedup vs sequential
- **Medium systems** (Fe d6 Oh, 120×120): 7-10× total speedup
- **GPU workloads** (dim ≥ 200): 10-15× total speedup
- **RIXS workflows**: 50-100× total speedup (45× from GPU + batching)

---

## Implementation Plan

### Phase 3A: Batch Hamiltonian Assembly (3-4 days)

**Goal:** Extend `assemble_matrix_from_adds()` to handle batched ADD operations

#### Task 1: Batch ADD Entry Processing
**File:** `multitorch/io/read_rme.py`

Current sequential pattern:
```python
for add_entry in add_list:
    mat[r0:r0+nr, c0:c0+nc] += scale * source[sr:sr+nr, sc:sc+nc]
```

Batched approach:
```python
def assemble_matrix_from_adds_batch(adds, sources_batch, device=None):
    """
    Assemble N matrices from the same ADD template with batched sources.
    
    Parameters
    ----------
    adds : List[AddEntry]
        Template ADD operations (shared across all N)
    sources_batch : Dict[str, Tensor]
        Batched source matrices, shape (N, dim, dim) per key
    
    Returns
    -------
    matrices : Tensor, shape (N, target_dim, target_dim)
        Assembled Hamiltonian blocks for all N parameter sets
    """
```

**Challenge:** ADD entries have different source/target dimensions per block. Need to:
- Group ADDs by target matrix dimensions
- Batch-process homogeneous groups
- Handle ragged batching when matrix sizes differ across triads

**Autograd Preservation:** Gradient must flow through `sources_batch` tensors independently per sample.

#### Task 2: Extend `build_cowan_store_in_memory_batch` for Final States
**File:** `multitorch/hamiltonian/build_cowan.py`

Current: Only rebuilds ground-state COWAN matrices in batch  
Target: Extend to final-state manifolds (e.g., 2p^5 3d^9 for L-edge XAS)

Pattern:
```python
def build_cowan_store_in_memory_batch(
    cowan_template, 
    scaled_params_batch,  # (N,) batched ScaledAtomicParams
    rebuild_keys=None,    # Which HAMILTONIAN blocks to batch rebuild
    device=None
):
    # Current: Only 'HAMILTONIAN' key (ground state)
    # Phase 3: Add 'XHAM_000', 'XHAM_001', etc. (final states)
```

**Impact:** Eliminates N×M fixture reads (N params × M final states)

---

### Phase 3B: Batch Diagonalization Across Triads (2-3 days)

**Goal:** Replace N sequential `assemble_and_diagonalize_in_memory` calls with single batched pass

#### Task 3: Create `assemble_and_diagonalize_in_memory_batch()`
**File:** `multitorch/hamiltonian/assemble.py`

```python
def assemble_and_diagonalize_in_memory_batch(
    ban_data,
    rac_data, 
    rcg_data,
    cowan_store_batch,  # Batched COWAN stores for N parameter sets
    device=None
) -> BanResultBatch:
    """
    Batch-process all triads for N parameter sets.
    
    Returns
    -------
    BanResultBatch
        .egvals : Dict[ConfigKey, Tensor]  # shape (N, n_levels)
        .egvecs : Dict[ConfigKey, Tensor]  # shape (N, n_levels, n_levels)
        .trans_matrices : Dict[TriadKey, Tensor]  # shape (N, n_rows, n_cols)
    """
```

**Key optimizations:**
1. **Ground-state diagonalization:** Already batched via `safe_eigh_batch()`
2. **Final-state diagonalization:** Batch across (N params × M final configs)
3. **Transition matrices:** Batch `tdm_raw @ egvecs_g.T @ egvecs_f.conj()` operations
4. **Memory management:** Process large batches in chunks to avoid OOM

#### Task 4: Batch Transition Operator Assembly
**File:** `multitorch/hamiltonian/assemble.py::_assemble_triad_batch()`

Current bottleneck: Each triad reads RME/RCG fixtures and assembles operators N times.

Batched approach:
```python
def _assemble_triad_batch(triad, rac_data, rcg_data, egvecs_g_batch, egvecs_f_batch):
    # Parse RME operators ONCE (not N times)
    tdm_raw = parse_rme_operators(triad, rac_data, rcg_data)  # on CPU
    
    # Batched rotation into eigenbasis
    # T_batch[i] = tdm_raw @ egvecs_g[i].T @ egvecs_f[i].conj()
    T_batch = torch.einsum('ij,nkj,nlj->nikl', tdm_raw, egvecs_g_batch, egvecs_f_batch.conj())
    return T_batch.reshape(N, -1, -1)  # (N, n_rows, n_cols)
```

**Speedup source:** Parsing RME/RCG once + batched einsum vs N sequential matrix mults

---

### Phase 3C: Batch Stick Spectrum Extraction (1-2 days)

**Goal:** Generate (N, n_sticks) stick spectra in single vectorized pass

#### Task 5: Extend `get_sticks_from_banresult()` for Batched BanResult
**File:** `multitorch/spectrum/sticks.py`

```python
def get_sticks_from_banresult_batch(
    banresult_batch: BanResultBatch,
    T: float = 300.0,
    thin: int = 1
) -> Tuple[Tensor, Tensor]:
    """
    Extract stick spectra for N BanResults simultaneously.
    
    Returns
    -------
    Etrans_batch : Tensor, shape (N, n_sticks_max)
        Transition energies (padded to max across batch)
    intensities_batch : Tensor, shape (N, n_sticks_max)
        Intensities (zero-padded where n_sticks < n_sticks_max)
    """
```

**Challenges:**
- Each parameter set may have different number of sticks (ragged output)
- Need smart padding or return list-of-tensors
- Autograd flow through Boltzmann weights still needed

**Optimization:** Vectorize Boltzmann weighting across batch dimension

---

### Phase 3D: Integration & API (1 day)

#### Task 6: Extend `calcXAS_batch()` to Use Full Pipeline
**File:** `multitorch/api/calc.py`

Current Phase 2 implementation:
```python
def calcXAS_batch(cache, slater_values, soc_values, ...):
    # Phase 2: Batches atomic scaling + COWAN rebuild only
    # Still calls assemble_and_diagonalize_in_memory() N times ← bottleneck
```

Phase 3 enhancement:
```python
def calcXAS_batch(cache, slater_values, soc_values, ...):
    # Phase 3: Full batch pipeline
    cowan_batch = build_cowan_store_in_memory_batch(...)
    banresult_batch = assemble_and_diagonalize_in_memory_batch(...)
    sticks_batch = get_sticks_from_banresult_batch(...)
    spectra_batch = broaden_batch(sticks_batch, ...)  # Already vectorized
    return spectra_batch  # (N, n_bins)
```

**Backward compatibility:** Phase 2 API remains unchanged; Phase 3 is internal optimization.

---

### Phase 3E: Testing & Validation (2-3 days)

#### Task 7: Comprehensive Phase 3 Test Suite
**File:** `tests/test_batch/test_batch_phase3.py`

Required tests:
1. **Numerical parity:** `calcXAS_batch()` Phase 3 matches Phase 2 output (< 1e-6)
2. **Autograd correctness:** Per-sample gradients independent and correct
3. **Memory profiling:** Track peak memory vs batch size (identify limits)
4. **Performance benchmarks:**
   - 100/1000/5000 spectra on Ni Oh, Fe Oh, Ni D4h CT
   - Measure time/spectrum vs batch size
   - GPU vs CPU crossover point
5. **Edge cases:**
   - Single spectrum (N=1) should match non-batch API
   - Large batches (N=5000) shouldn't OOM on 32GB RAM
   - Ragged stick counts handled correctly

#### Task 8: End-to-End Performance Validation
**File:** `examples/phase3_performance_demo.py`

Benchmark suite:
```python
# Measure Phase 2 vs Phase 3 speedup
for N in [10, 100, 1000, 5000]:
    # Phase 2 timing
    t2 = time_calcXAS_batch_phase2(N)
    
    # Phase 3 timing
    t3 = time_calcXAS_batch_phase3(N)
    
    print(f"N={N}: Phase 2={t2:.2f}s, Phase 3={t3:.2f}s, speedup={t2/t3:.1f}×")
```

Target outcomes:
- N=100: 1.2-1.5× speedup over Phase 2
- N=1000: 2-3× speedup over Phase 2
- N=5000: 2.5-3.5× speedup over Phase 2 (memory-limited on CPU)

---

## Expected Performance Gains

### Speedup Breakdown (1000 Spectra, Ni d8 Oh)

| Component | Phase 1 (s/spectrum) | Phase 2 (s/spectrum) | Phase 3 (s/spectrum) | Phase 3 vs Phase 1 |
|-----------|---------------------|---------------------|---------------------|-------------------|
| Fixture parsing | 0.012 | 0.001 (cached) | 0.001 (cached) | **12×** |
| COWAN rebuild | 0.003 | 0.001 (batched V11) | 0.001 (same) | **3×** |
| Final state assembly | 0.005 | 0.005 (sequential) | 0.002 (batched) | **2.5×** |
| Diagonalization | 0.006 | 0.006 (sequential) | 0.002 (batched eigh) | **3×** |
| Sticks extraction | 0.001 | 0.001 | 0.0008 (batched) | **1.2×** |
| Broadening | 0.003 | 0.003 | 0.003 (same) | **1×** |
| **Total** | **0.030** | **0.017** | **0.010** | **3× over Phase 2, 8× over Phase 1** |

### GPU Acceleration (Phase 3 + GPU, dim ≥ 200)

| System | Phase 2 CPU (s) | Phase 3 CPU (s) | Phase 3 GPU (s) | Total Speedup |
|--------|----------------|----------------|----------------|---------------|
| Ni Oh (17×17) × 1000 | 17 | 10 | 15 (slower) | **1.7×** (stay CPU) |
| Fe Oh (120×120) × 1000 | 260 | 150 | 100 | **2.6×** (GPU wins) |
| Cr Oh (1074×1074) × 100 | 1800 | 900 | 200 | **9×** (GPU big win) |

**Key insight:** Phase 3 + GPU only matters when matrix dim ≥ 200. Otherwise, Phase 3 on CPU is optimal.

---

## Implementation Risks & Mitigations

### Risk 1: Memory Explosion
**Problem:** Batching 5000 spectra with autograd could require 20-50 GB RAM  
**Mitigation:**
- Implement smart chunking: process in batches of 200-500, concatenate results
- Add `max_batch_size` parameter to auto-chunk large requests
- Provide memory estimation utility: `estimate_batch_memory(system, N)`

### Risk 2: Ragged Tensor Handling
**Problem:** Different parameter sets may yield different numbers of sticks  
**Mitigation:**
- Zero-pad to `max(n_sticks)` within batch
- Use masking for broadening to ignore padded zeros
- Alternative: Return list of tensors if padding overhead > 20%

### Risk 3: Autograd Complexity
**Problem:** Batched einsum for transition matrices may break gradient flow  
**Mitigation:**
- Extensive `torch.autograd.gradcheck()` validation
- Test per-sample gradient independence explicitly
- Fallback to sequential if autograd fails (with warning)

### Risk 4: Fixture Compatibility
**Problem:** Some COWAN blocks may not be rebuildable in batch (e.g., 2-shell CT)  
**Mitigation:**
- Detect non-rebuildable blocks, fall back to sequential for those triads only
- Hybrid approach: batch ground state, sequential for special final states

---

## Success Criteria

Phase 3 is complete when:
1. ✅ `calcXAS_batch()` achieves 5-10× total speedup vs sequential (1000+ spectra)
2. ✅ Numerical parity: max error < 1e-6 vs Phase 2 output
3. ✅ Autograd correctness: per-sample gradients independent and accurate
4. ✅ GPU crossover documented: speedup vs matrix dimension and batch size
5. ✅ Memory limits characterized: max batch size vs system RAM
6. ✅ All 402 existing tests still pass (zero regression)
7. ✅ Performance benchmarks in `examples/phase3_performance_demo.py`

---

## When to Implement Phase 3

**High Priority - Implement soon if:**
- Regularly generating 1000+ spectra (grid searches, MCMC sampling)
- Using GPU-capable hardware (RTX 4090, A100, etc.)
- Working with large systems (dim ≥ 200: rare earths, Cr d³, large CT)
- RIXS workflows (would get 50-100× total speedup)

**Medium Priority - Nice to have if:**
- Generating 100-1000 spectra occasionally
- Parameter refinement workflows with many iterations
- Want best-in-class performance

**Low Priority - Skip if:**
- Mostly single-shot calculations
- Small parameter sweeps (< 100 spectra)
- CPU-only hardware with limited RAM (< 32 GB)
- Phase 2 speedup (2-5×) already sufficient

---

## Alternative: Phase 3-Lite (Minimal Effort)

If full Phase 3 is too complex, a lightweight version could deliver 60-70% of the benefit:

**Phase 3-Lite Scope:**
1. ✅ Batch final-state fixture parsing (eliminate N × M file reads)
2. ✅ Batch transition matrix assembly (batched einsum)
3. ⬜ Skip: Batch ADD processing (complex, low gain)
4. ⬜ Skip: Full BanResultBatch refactor (use existing data structures)

**Implementation Time:** 3-5 days  
**Expected Speedup:** 1.5-2× over Phase 2 (total 3-6× vs sequential)

---

## Future Extensions (Beyond Phase 3)

### Phase 4: Dynamic Batching & Adaptive Chunking
- Auto-detect optimal batch size based on available GPU memory
- Dynamic recompilation for different batch sizes (TorchScript/torch.compile)
- Mixed-precision batching (FP16 forward, FP32 backward for stability)

### Phase 5: Distributed Batch Processing
- Multi-GPU batching via `torch.nn.DataParallel` or `DistributedDataParallel`
- Cluster support for 10,000+ spectra sweeps (MPI + PyTorch distributed)
- Fault tolerance for long-running jobs

### Phase 6: JIT Compilation & Kernel Fusion
- Use `torch.compile()` or TorchScript to fuse batch operations
- Custom CUDA kernels for Kramers-Heisenberg (potential 2-3× over native PyTorch)
- Operator fusion: eigh + transition matrix mult in single kernel

---

## References & Related Work

- **Phase 1 Plan:** `GPU_ACCELERATION_PLAN.md` (smart device selection, fixture caching)
- **Phase 2 Implementation:** Commit `2c20a98` (batch COWAN rebuild, `calcXAS_batch`)
- **Profiling Data:** `GPU_ACCELERATION_PLAN.md` (measured speedups on 2× RTX 4090)
- **Test Suite:** `tests/test_batch/` (11 tests validating Phase 2 correctness)

---

**Document Version:** 1.0  
**Last Updated:** April 18, 2026  
**Status:** Planning phase, awaiting implementation decision
