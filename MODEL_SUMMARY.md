# multitorch — Model & Code Review Summary

**Last updated:** 2026-04-18
**Head commit:** `fd63bdf` ("Fix indentation error in test_batch_api.py")
**Version:** 0.1.0 (MIT)
**Test status:** 508 passed / 0 failed / 1 skipped (509 collected) — Phase 2 batch parity regression fixed 2026-04-18
**Supersedes:** the 2026-04-11 `CODE_REVIEW.md` (still present at repo root for historical reference)

---

## What It Is

multitorch is a differentiable PyTorch port of the Cowan / Butler / ttmult
Fortran multiplet suite for L-edge X-ray absorption (XAS), emission (XES),
and resonant inelastic X-ray scattering (RIXS) of 3d transition-metal
complexes. The full pipeline — Wigner coupling coefficients, atomic Slater
integrals, Blume-Watson spin-orbit, Hamiltonian assembly, eigenvalue
decomposition, and pseudo-Voigt broadening — is float64 and fully
`torch.autograd`-traceable. Gradients flow from the physical parameters
(Slater reduction, SOC scaling, crystal field, charge transfer) through
`torch.linalg.eigh` to the output spectrum, which makes the library
suitable for gradient-based parameter refinement against experiment. The
project is a package rather than a trained model — there are no weights,
checkpoints, or training curves; "logs" below means git history, test
runs, and audit artifacts.

---

## Architecture

Module breakdown (post-pull). File counts confirmed against the 2026-04-11
CODE_REVIEW.md; new/modified files are flagged:

| Module | Files | Role |
|---|---|---|
| `io/` | 7 | Parsers for `.rme_rcg`, `.rme_rac`, `.ban`, `.ban_out`, `.rcn31_out` + paired RIXS reader |
| `angular/` | 8 | Wigner 3j/6j/9j, CFP binary parser, reduced matrix elements, symmetry branching, `torch_blocks.py` (Track C3a torch wrappers), `rac_generator.py` |
| `atomic/` | 8 | HFS SCF, Slater integrals, Blume-Watson SOC, radial mesh, atomic tables, **`scaled_params.py`** (+ batch scaling added 2026-04-18) |
| `hamiltonian/` | 9 | Assembly from files + in-memory, BAN / RAC / COWAN builders, **`diagonalize.py`** (+ `safe_eigh_batch`, `diagonalize_batch` added 2026-04-18), transitions, crystal field, charge transfer, deployment checks |
| `spectrum/` | 5 | Boltzmann sticks, pseudo-Voigt broadening, Kramers-Heisenberg RIXS, background/saturation |
| `api/` | 3 | `calc.py` (calcXAS/XES/RIXS/DOC + `calcXAS_batch`, `batch_parameter_sweep`, `CachedFixture`, `preload_fixture`, `calcXAS_cached`), `plot.py` |
| **`device_utils.py`** (top-level) | 1 | **NEW 2026-04-18** — operation-aware CPU/GPU selection |

### Canonical differentiable pipeline (Phase 5)

```
calcXAS(element, valence, sym, edge, cf, slater, soc)
  ├─ _find_fixture_dir() → parse .ban template
  ├─ modify_ban_params(ban, cf, delta, lmct, mlct)
  ├─ read_rcn31_out_params()
  ├─ scale_atomic_params(slater, soc)          ← autograd leaves
  ├─ build_rac_in_memory()  → SectionPlan
  ├─ build_cowan_store_in_memory(scaled, plan) ← gradient enters COWAN store
  ├─ assemble_and_diagonalize_in_memory(...)   ← eigh backward
  ├─ get_sticks_from_banresult(result, T, max_gs)
  └─ pseudo_voigt(x, E, I, fwhm_g, fwhm_l)     ← gradient exits spectrum
```

### NEW — Phase 2 batch pipeline (2026-04-18)

```
calcXAS_batch(cached_fixture, slater_values, soc_values, device=...)
  ├─ batch_scale_atomic_params(slater[N], soc[N])       multitorch/atomic/scaled_params.py
  ├─ build_cowan_store_in_memory_batch(...)             multitorch/hamiltonian/build_cowan.py
  ├─ safe_eigh_batch(H[N, dim, dim])                    multitorch/hamiltonian/diagonalize.py
  └─ pseudo_voigt per sample
```

Targets 2-5× speedup for parameter sweeps by sharing the V(11) residual
and collapsing N sequential `eigh` calls into one batched call.
`batch_parameter_sweep()` at `multitorch/api/calc.py:457` provides the
grid / custom sweep wrapper.

### NEW — operation-aware device selection (2026-04-18)

`get_optimal_device(operation, matrix_dim, n_sticks, force_device)` at
`multitorch/device_utils.py:11` routes by measured speedup:

| Operation | Threshold | Device |
|---|---|---|
| RIXS 2D map | any | GPU (measured 45×) |
| Broadening | `n_sticks > 1000` | GPU (measured 41×) |
| `eigh` | `matrix_dim ≥ 500` | GPU (4-10×) |
| Single 3d TM L-edge XAS | default | CPU (kernel overhead dominates at dim<200) |

Used internally by `calcXAS`, `calcRIXS`, and `calcXAS_batch` when
`device=None` (default).

---

## Training Setup

Not applicable — this is a deterministic simulator, not a trained model.
No optimizer, no scheduler, no epochs. Gradients exist for user-level
parameter fitting (external to the library).

---

## Current Test Results (2026-04-18, post-pull)

```
507 passed, 1 failed, 1 skipped in 214.50s (3:34 wall)
```

Pytest discovered 509 tests across 37 test files. Coverage table:

| Directory | Files | ≈ test functions | Notes |
|---|---|---|---|
| `test_angular/` | 8 | 76 | Wigner / CFP / RME / torch wrappers / L-edge RAC |
| `test_atomic/` | 5 | 90 | HFS, radial mesh, scaled params, parameter fixtures, RCF parser |
| `test_hamiltonian/` | 8 | 92 | Assembly, BAN, RAC, COWAN, diagonalize, deployment checks |
| `test_spectrum/` | 3 | 29 | Sticks, broaden, **background (added post 4-11)** |
| `test_integration/` | 9 | 56 | nid8ct XAS, Phase 5 parity/multi, RIXS pair, from-scratch, DOC, TM series, fixture cache, deployment checks |
| **`test_batch/`** | 3 | 19 | **NEW 2026-04-18** — batch scaling, batch COWAN, batch API |
| **`test_device_utils.py`** | 1 | 13 | **NEW 2026-04-18** — 4 test classes covering RIXS/XAS/eigh/broaden routing + force-override |

### The failing test

`tests/test_batch/test_batch_api.py::test_calcXAS_batch_vs_sequential_parity`

```
N = 3
slater_vals = [0.7, 0.8, 0.9]
soc_vals    = [0.9, 1.0, 1.1]
→ Sample 1 (slater=0.8, soc=1.0): max |y_batch - y_seq| = 0.2713
  tolerance: 1e-3
```

Samples 0 and 2 apparently pass; only sample 1 fails. The fact that the
middle sample fails while the boundary samples match is diagnostic — it
is consistent with a stride, offset, or broadcast bug in the batched
code path that happens to produce correct output at the extremes but
misrouts the middle sample. The batch path was introduced in commit
`2c20a98` and is not yet validated end-to-end. **This is the single
highest-priority issue in the codebase right now.**

---

## Development Log — commits since 2026-04-11

34 commits landed between 2026-04-11 and 2026-04-17 (documented in the
prior `CODE_REVIEW.md`) bringing the test count from 363 → 477. The
2026-04-18 pull adds 5 more:

| Date | Hash | Subject | Files | LOC |
|---|---|---|---|---|
| 2026-04-18 | `6ecffbb` | Phase 1: Add smart GPU/CPU device selection | 5 | +437 / −4 |
| 2026-04-18 | `2c20a98` | Phase 2: Batch parameter sweep infrastructure for 2-5× speedup | 10 | +1635 |
| 2026-04-18 | `dd15c82` | Merge Phase 2 | — | — |
| 2026-04-18 | `6a63dda` | Add Phase 3 implementation roadmap | 1 | +403 |
| 2026-04-18 | `fd63bdf` | Fix indentation error in test_batch_api.py | 1 | +1 / −1 |

The pull was a clean fast-forward from `4cb91d7` → `fd63bdf` with no
merge conflicts. Local uncommitted edits to `CLAUDE.md` and
`notebooks/02_pipeline_walkthrough.ipynb` were preserved via stash/pop
(neither file was touched upstream).

### Audit / orchestration log (from `.claude/orchestration/`)

- `DEPLOYMENT-AUDIT-FIXES.md` (2026-04-17) — 11 fixes + 21 new tests
  from the three-skill audit (scientific / code review / test builder).
  All critical findings resolved: safe `eigh` backward at degenerate
  eigenvalues, differentiable broadening FWHM, thread-safe CFP cache,
  edge-case KeyError guards, removal of double Ry→eV conversion.
- `TRACK-C-PHASE5.md` (2026-04-11) — C1 through C3d complete; C3e
  (from-scratch COWAN reconstruction) remains deferred. `build_rac_in_memory`
  is a loader, not a generator, by explicit scoping decision (see
  CLAUDE.md "Track C Phase 5 — scoping decision at C3d").
- `audit-exchange/` (2026-04-17) — three auditor JSON reports.

### Runtime warnings observed during test run

`rac_generator.py:762` emits `UserWarning: F2_pd is nonzero but direct
Coulomb contribution is not implemented` for Mn-ii, Cr-iii, Fe-iii,
Ni-iii, Cu-ii in `test_from_scratch_multi_element`. The warning text is
explicit that this is a known small systematic error in excited-state
absolute energies for the from-scratch path. Not flagged as a test
failure, but worth tracking for the manuscript.

---

## Known Issues & Observations

Ordered by severity:

1. **[FIXED 2026-04-18] Phase 2 batch parity failure.** Root cause was
   an x-grid scoping bug in `calcXAS_batch` at
   `multitorch/api/calc.py`: `xmin`/`xmax` were loop-scoped kwargs that
   got set from sample 0's stick range on the first iteration and then
   retained for every subsequent sample, so interior samples were
   broadened on sample 0's grid. Endpoint sample 0 matched sequential
   by coincidence. The bug was **not** in `batch_scale_atomic_params`,
   `build_cowan_store_in_memory_batch`, or `safe_eigh_batch` — those
   are correct. Fix replaces the per-iteration range mutation with a
   two-pass structure: pass 1 diagonalizes and extracts sticks for
   every sample; pass 2 derives a shared x-range as the union of all
   non-empty sample stick ranges and broadens each on the shared
   grid. The parity test was also updated to pass explicit
   `xmin`/`xmax` to both batch and sequential so they are guaranteed
   to share a grid. Full suite: 508 / 1 skipped, zero failing.

2. **[MEDIUM] No batch autograd parity test.** `test_batch_api.py`
   includes `test_calcXAS_batch_autograd`, but it only checks that
   gradients are finite and nonzero — not that batched gradients equal
   sequential gradients. Once issue #1 is fixed, add a gradcheck-style
   parity test.

3. **[MEDIUM] `broaden_mode` still defaults to `"legacy"`.** SCA-001
   from the scientific audit remains open. `multitorch/spectrum/broaden.py`
   continues to default to the older shape at `pseudo_voigt(...,
   mode="legacy")`. Breaking API change across six signatures, deferred
   pending a deprecation cycle.

4. **[MEDIUM] F2_pd Coulomb term not implemented.** Emits a UserWarning
   during `test_from_scratch_multi_element`. The affected elements
   (Mn-ii, Cr-iii, Fe-iii, Ni-iii, Cu-ii) show a small absolute-energy
   bias in from-scratch mode. Does not affect the validated Phase 5
   fixture path.

5. **[LOW] Cr³⁺ d³ autograd still excluded** from Phase 5 autograd
   tests because `torch.linalg.eigh` backward produces NaN at exact
   eigenvalue degeneracies. Deployment audit added `safe_eigh` with a
   linear perturbation, but the test fixture for Cr³⁺ d³ remains
   excluded from the gradient parity suite. Upstream PyTorch limitation.

6. **[LOW] `calcDOC()` is now implemented** for the bootstrap path
   (2026-04-17 work) but `_calcDOC_phase5` at `multitorch/api/calc.py:1653`
   still raises — Phase 5 DOC deferred.

7. **[LOW] `PHASE_3_PLAN.md` is a planning document**, not a shipping
   feature. It describes batch assembly + batch triad processing
   targeting 5-10× total speedup; marked "Not Yet Implemented".

8. **[LOW] Two uncommitted local files.** `CLAUDE.md` has status-block
   additions; `notebooks/02_pipeline_walkthrough.ipynb` has stripped
   output cells. Not touched by the upstream GPU commits. Need to be
   either committed or discarded before the next push.

9. **[LOW] 1 test skipped.** Identity of the skipped test not captured
   in the run summary; likely one of the CFP-binary-conditional tests
   or a CUDA-conditional GPU test (CI machine is CPU-only).

### Test integrity notes (Phase 4b audit)

- Batch parity tests compare batched output to sequential output on the
  same inputs — independent oracle, good pattern.
- `test_device_utils.py` exercises only routing logic, not actual GPU
  kernels; GPU-specific code paths are CPU-tautological on this machine
  (`test_rixs_prefers_gpu_if_available` returns `"cpu"` when CUDA
  unavailable). Not a defect — CUDA validation requires a GPU host.
- `test_batch_autograd_preservation` checks gradient flow but not
  numerical magnitude against finite difference — a known weak pattern
  from the audit framework; should be upgraded once issue #1 is fixed.

---

## What It Can Do Right Now

**Working and validated:**
- L-edge XAS for all 3d TM (Ti⁴⁺ through Ni²⁺) in Oh and D4h symmetry,
  numerical agreement with Cowan/ttmult (cosine ≥ 0.97 across 9 Ti-Ni
  fixtures).
- Full autograd through `slater`, `soc` in Phase 5; through `cf`,
  `delta`, `lmct`, `mlct` in bootstrap path only (Phase 5 still
  converts these to floats in `modify_ban_params()`).
- Bootstrap mode (Fortran `.ban_out` inputs): byte-exact against pyctm.
- 2D RIXS maps via Kramers-Heisenberg kernel (bootstrap; synthetic
  fixture).
- Blume-Watson SOC matching Fortran to ≤ 3% reduction ratio.
- `CachedFixture` / `preload_fixture` / `calcXAS_cached` API for
  parameter sweeps without per-call file I/O.
- Smart CPU/GPU selection via `device_utils.get_optimal_device()`.

**Not yet working / not yet validated:**
- `calcXAS_batch` / `batch_parameter_sweep` — **fails interior-sample
  parity test** (issue #1).
- Phase 5 from-scratch RIXS and XES (`NotImplementedError`).
- Phase 5 DOC.
- Autograd through crystal-field / charge-transfer parameters in
  Phase 5 (needs tensor plumbing through `modify_ban_params`).
- Cr³⁺ d³ autograd (PyTorch `eigh` backward NaN).
- GPU acceleration beyond device routing — Phase 3 plan describes full
  batched assembly; not yet implemented.

---

## Capability gaps flagged by alpha tester (2026-04-18)

Two items raised in alpha-tester feedback for the manuscript
positioning. Neither is a bug; both are scope questions.

### Trigonal symmetry (D3, C3v, D3h, D3d)

**Status: not supported in multitorch.** The public API accepts only
`sym="oh"`, `"d4h"`, `"c4h"` (`multitorch/api/calc.py:628`). Every
trigonal-dependent piece of the stack is Oh/D4h-hardcoded:

- Subduction / character tables at `multitorch/angular/symmetry.py:15`
  and `multitorch/angular/point_group.py:30-76` implement only Oh and
  D4h irreps.
- Operator orderings in `multitorch/hamiltonian/assemble.py:42-49`
  (`OPERATOR_ORDER_OH`, `OPERATOR_ORDER_D4H`, `HYBR_ORDER_*`) have no
  trigonal equivalents.
- All 9 bundled fixtures (ni2_d8_oh, fe3_d5_oh, mn2_d5_oh, co2_d7_oh,
  v3_d2_oh, cr3_d3_oh, ti4_d0_oh, nid8/nid8ct) are Oh/D4h.
- CFP tables (`multitorch/data/cfp/rcg_cfp72`, `rcg_cfp73`) are
  O-coupling only.

multitorch is standalone at runtime — no Fortran dependency — so
"what the Fortran could do" is not the relevant question. The
question is what multitorch can do. The answer today is Oh/D4h/C4h
only; trigonal is simply not implemented.

**Port path (all inside multitorch):**

1. `multitorch/angular/point_group.py` — add D3, C3v, D3h, D3d irrep
   definitions and basis functions; extend the existing Oh/D4h table.
2. `multitorch/angular/symmetry.py` — add O3→D3 and O3→C3v subduction
   coefficients.
3. `multitorch/angular/rac_generator.py` — extend the from-scratch
   pipeline so trigonal operator blocks can be generated in pure
   Python without an external fixture. multitorch already has a
   from-scratch path (`test_from_scratch_multi_element`,
   `calcXAS_from_scratch` at `api/calc.py:1071`); trigonal is a new
   branch off that existing machinery, not a new stack.
4. `multitorch/hamiltonian/assemble.py:42-49` — add
   `OPERATOR_ORDER_D3H`, `OPERATOR_ORDER_C3V`, etc., with the correct
   trigonal crystal-field operator set (e.g., `Dσ`, `Dτ` in the
   Ballhausen convention for C3v / D3).
5. `multitorch/api/calc.py` — extend the `sym=` dispatcher and
   `_find_fixture_dir` to recognize trigonal values.
6. Either (a) pre-computed CFP tables bundled under
   `multitorch/data/cfp/` the way `rcg_cfp72` / `rcg_cfp73` already are
   for Oh, or (b) a Python CFP routine — the former matches the
   current design, which ships precomputed angular data as package
   resources.
7. At least one reference fixture per point group under
   `multitorch/data/fixtures/` for regression testing. These can be
   generated by whichever trusted off-the-shelf multiplet code the
   group prefers; the bundled fixture is what multitorch consumes at
   runtime.

Medium-effort greenfield feature. The architecture does not block it;
the lift is in items (1)–(4), and the autograd and GPU paths
downstream do not care about symmetry.

### Arbitrary number of charge-transfer configurations

**Status: hardcoded to two configurations (ground d^N + one CT d^(N+1)L
or d^(N-1)L).** The `.ban` file format itself (`NCONF 2 2` in every
fixture) is flexible and the parser (`multitorch/io/read_ban.py:90-111`)
does not rely on the number being 2; the lockdown is in the Python
consumers:

- `AtomicParams.ground` / `.excited` in
  `multitorch/atomic/parameter_fixtures.py:119-143` — hardcoded
  properties addressing only NCONF=1 and NCONF=2.
- `build_ct_energy_offsets` in `multitorch/hamiltonian/charge_transfer.py:189-231`
  — takes scalar `delta`, `lmct`, `mlct`, not lists. Offset recipe
  bakes in the 2-config LMCT/MLCT topology.
- `modify_ban_params` in `multitorch/hamiltonian/build_ban.py:38-160`
  — writes to hardcoded `eg[2]`, `xmix[0]` slots.
- `assemble_and_diagonalize_in_memory` in
  `multitorch/hamiltonian/assemble.py:236-242,272-273` — uses the
  literal mapping `gs_cowan_sec = 2 if nconf >= 2 else 0`,
  `fs_cowan_sec = 3`. Any 3+ config arrangement has no section
  allocation.
- COWAN store rebuild in `multitorch/hamiltonian/build_cowan.py`
  decomposes only section-2 `HAMILTONIAN` / config-1 blocks — the
  current decomposition story is single-shell d^N. Multi-CT would
  need a two-shell Coulomb + SOC decomposition at a minimum.

**Port path:** an N-configuration generalization is a ~100-150 LOC
refactor concentrated in three files (`charge_transfer.py`,
`build_ban.py`, `assemble.py`) plus one extension of the scaling /
COWAN rebuild for multi-shell configs. It also requires at least one
real 3-CT fixture (`.ban` with `NCONF 3 3` plus matching
`.rme_rac`/`.rme_rcg`) to validate against — none exist in the repo
today. The tester's framing that "faster code → more CT states" is
consistent with this: once Phase 2 / Phase 3 batch paths land, the
cost of 3+ CT states comes primarily from the enlarged Hilbert space
per spectrum, not from wall-clock iteration count.

---

## Immediate Improvement Priorities

1. **Add batch autograd parity test** — Phase 2's
   `test_calcXAS_batch_autograd` checks only that gradients are
   nonzero. With the 2026-04-18 fix landed, add a gradcheck-style
   parity test that verifies batched gradients equal sequential
   gradients sample-by-sample.
2. **Commit or discard the two uncommitted local files**
   (`CLAUDE.md`, `notebooks/02_pipeline_walkthrough.ipynb`) before the
   next push.
3. **Implement F2_pd direct Coulomb term** in `rac_generator.py:762`
   to eliminate the from-scratch excited-state energy bias.
4. **Update CLAUDE.md "Test status"** — stale text says "398/398" /
   "477/477"; actual is 509 collected / 508 passing / 1 skipped.
5. **Flip `broaden_mode` default from `"legacy"` to `"correct"`** with
   a deprecation warning — SCA-001 has been open since the scientific
   audit.
6. **Implement Phase 3** per `PHASE_3_PLAN.md`. Plan targets 5-10×
   total speedup via batched Hamiltonian assembly and consolidated GPU
   kernel launches; unblocked now that the Phase 2 regression is
   fixed.
7. **Scope decisions for manuscript (see "Capability gaps" above):**
   trigonal symmetry port and arbitrary-N CT state generalization.
   Both are medium-effort; neither is currently in the roadmap.

---

## Version History

| Date | Milestone | Tests |
|---|---|---|
| 2026-04-05 | Initial port: parsers, angular, HFS, assembly, broadening | 150 |
| 2026-04-05 | Track A (RIXS bootstrap) complete | 150 |
| 2026-04-06 | Track B (Blume-Watson SOC) complete | 163 |
| 2026-04-11 | Track C (Phase 5 autograd pipeline) complete | 363 |
| 2026-04-17 | Deployment audit gate passed (11 fixes, 21 tests) | 477 |
| **2026-04-18** | **Phase 1 (GPU device selection) + Phase 2 (batch sweep); parity regression fixed** | **509 (508 passing, 1 skipped)** |

---

## Artifacts

- Full source: `multitorch/multitorch/` (43 modules)
- Tests: `tests/` (37 test files, 509 tests)
- Examples: `examples/gpu_device_selection.py`, `examples/batch_processing_demo.py` (new 2026-04-18)
- Docs: `docs/GPU_ACCELERATION_PLAN.md` (Tier 1 & 2 implemented; Tier 3 pending), `PHASE_3_PLAN.md` (new 2026-04-18; not yet implemented)
- Audits: `CODE_REVIEW.md` (2026-04-11), `SCIENTIFIC_AUDIT.md` (2026-04-09), this `MODEL_SUMMARY.md` (2026-04-18)
- Orchestration: `.claude/orchestration/INDEX.md` + track-level logs
- Related literature review (upstream): `../manuscript/ctm-l-edge-rixs-differentiable-xray-lit-review-guide.docx`
