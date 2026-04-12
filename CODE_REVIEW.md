# multitorch — Senior Code Review
**Last updated:** 2026-04-11
**Status:** 363/363 tests passing. Tracks A (RIXS), B (Blume-Watson), C (Phase 5 autograd) complete.

---

## What It Is

A PyTorch port of the Fortran Cowan/ttmult multiplet X-ray spectroscopy suite
for L-edge XAS/XES/RIXS of 3d transition metal complexes. Float64 throughout,
GPU-agnostic. The Phase 5 pipeline (`calcXAS(element, valence, sym, edge, cf,
slater, soc)`) is autograd-traceable: gradients flow from `slater` and `soc`
through eigenvalue decomposition and broadening to the output spectrum.

41 Python modules, 24 test files, 263 test functions, 11 reference fixtures.

---

## Architecture Overview

| Module | Files | Role |
|--------|-------|------|
| `io/` | 7 | File parsers for Fortran outputs (.rme_rcg, .rme_rac, .ban, .ban_out, .rcn31_out) + paired RIXS reader |
| `angular/` | 6 | Wigner 3j/6j/9j, CFP (Fortran binary parser), reduced matrix elements, symmetry branching, torch wrappers |
| `atomic/` | 7 | HFS SCF solver, Slater integrals, Blume-Watson SOC, radial mesh, atomic tables, scaled params, fixture loader |
| `hamiltonian/` | 8 | Hamiltonian assembly (from files + in-memory), BAN builder, RAC builder, COWAN builder, diagonalize, transitions, crystal field, charge transfer |
| `spectrum/` | 4 | Boltzmann sticks, pseudo-Voigt broadening, Kramers-Heisenberg RIXS, background/saturation |
| `api/` | 2 | High-level entry points (calcXAS, calcXES, calcRIXS, calcDOC) + pyctm-compatible wrappers |

### Data flow — Phase 5 pipeline

```
calcXAS(element, valence, sym, edge, cf, slater, soc)
  → _find_fixture_dir() → parse .ban template
  → modify_ban_params(ban, cf, delta, lmct, mlct)
  → read_rcn31_out_params() → scale_atomic_params(slater, soc)  ← autograd leaves
  → build_rac_in_memory()  → SectionPlan
  → build_cowan_store_in_memory(scaled_params, plan)             ← gradient enters COWAN store
  → assemble_and_diagonalize_in_memory(cowan, rac, ban)          ← eigh backward
  → get_sticks_from_banresult(result, T, max_gs)
  → pseudo_voigt(x, E, I, fwhm_g, fwhm_l)                      ← gradient exits spectrum
```

---

## Autograd Safety Assessment

**Verdict: Safe.** The differentiable pipeline (Phase 5) has no gradient-severing
operations. All `.detach()` / `.numpy()` calls are isolated inside the HFS SCF
solver (`atomic/hfs.py`), which is explicitly documented as non-differentiable.
The autograd path routes through `slater_scale` and `soc_scale` scalars that
multiply into the COWAN store *after* HFS completes.

| Pattern | Count | Classification |
|---------|-------|----------------|
| `.detach()` | 4 | All in `hfs.py` (EXPECTED — numpy Numerov ODE solver) |
| `.numpy()` | 2 | All in `hfs.py` (EXPECTED — same) |
| `float(tensor)` | 42+ | All SAFE — scalar extraction for grid bounds, loop indices, constants |
| `.item()` | 1 | SAFE — in test validation function (`rme.py:881`) |
| `scatter_add_()` | 1 | SAFE — on freshly created zero tensor in `sticks.py:144` (legacy path only; Phase 5 path skips deduplication) |
| `torch.no_grad()` | 0 | None in codebase |

**One subtle finding:** `sticks.py:144` uses `scatter_add_()` in the `get_sticks()`
function (bootstrap path), which operates on a `torch.zeros_like()` output from
`torch.unique()`. This tensor has no `grad_fn`, so the in-place op is safe but
means `get_sticks()` is not differentiable. The Phase 5 path uses
`get_sticks_from_banresult()` instead, which avoids `scatter_add_` entirely.

---

## Remaining NotImplementedError Stubs

These are the only `NotImplementedError` raises in production code:

| Location | Function | What's missing | Impact |
|----------|----------|----------------|--------|
| `api/calc.py:400` | `calcXES()` | Full XES pipeline (bootstrap works) | Low — XES is a diagonal RIXS cut; `calcRIXS` subsumes it |
| `api/calc.py:463` | `calcRIXS()` | Phase 5 from-scratch RIXS (bootstrap works) | Medium — would need Phase 5 to produce paired abs+ems results |
| `api/calc.py:543` | `calcDOC()` | Degree of covalency calculation | Low — orbital character fractions from eigenvectors |
| `hamiltonian/transitions.py:93` | `build_ct_transition_matrix()` | CT transition matrix assembly | **Not blocking** — `assemble.py` handles CT transitions via the file-based/in-memory paths already |

---

## Placeholder / Partially-Implemented Modules

These modules have real code but contain incomplete sections:

| File | What exists | What's incomplete | Used by production? |
|------|-------------|-------------------|---------------------|
| `hamiltonian/diagonalize.py:103` | `diagonalize_block()` works | `get_diagonalized_matrices()` returns `{}` with TODO | **No** — `assemble.py` calls `diagonalize_block()` directly |
| `hamiltonian/crystal_field.py:136` | `build_crystal_field_matrix()` has structure | Loop body is empty (TODO: operator→branch coefficient mapping) | **No** — CF is handled through the COWAN store path |
| `hamiltonian/charge_transfer.py` | `assemble_hamiltonian_block()`, `assemble_mixing_block()`, `assemble_transition_block()`, `build_ct_energy_offsets()` all complete | None — this module is complete | **Indirectly** — `assemble.py` reimplements the same logic inline |
| `io/write_inputs.py` | RCG/RAC/BAN writers | Wraps pyctm (external dependency) | **No** — only for generating new Fortran inputs, not for the Phase 5 path |

**Assessment:** The incomplete modules (`diagonalize.get_diagonalized_matrices`,
`crystal_field.build_crystal_field_matrix`) are dead code — they were early
prototypes superseded by the production `assemble.py` + `build_cowan.py` path.
They could be removed or documented as stubs.

---

## Test Coverage Analysis

| Module | Test files | Test functions | Coverage notes |
|--------|-----------|----------------|----------------|
| angular/ | 6 | 73 | Comprehensive: Wigner (13), CFP (6, conditional on binary), RME (12+13), torch wrappers (9) |
| atomic/ | 5 | 90 | Strong: HFS (18), radial mesh (12), scaled params (28), parameter fixtures (23), RCF parsers (9) |
| hamiltonian/ | 7 | 75 | Strong: assembly (10+8), build_ban (16), build_rac (21), build_cowan (19), BAN parser (9) |
| spectrum/ | 2 | 14 | Adequate: sticks (8), broaden (6). No tests for `background.py` or `rixs.py` unit tests |
| integration/ | 4 | 11 | Good: nid8ct (10), Phase 5 parity (9), Phase 5 multi (3 parametrized → 15 cases), RIXS pair (16) |

### Notable gaps

1. **`spectrum/background.py`** — no unit tests. Contains `arctan_step()`,
   `gaussian_peak()`, `linear_bg()`, `xas_background()`, `saturation_correction()`.
2. **`spectrum/rixs.py`** — no dedicated unit tests. Tested indirectly via
   `test_rixs_pair.py::test_kramers_heisenberg_is_autograd_safe` (1 test).
3. **`hamiltonian/charge_transfer.py`** — no dedicated tests. The functions
   are tested indirectly through `assemble.py` integration tests.
4. **`hamiltonian/crystal_field.py`** — no tests (placeholder code).
5. **`io/write_inputs.py`** — no tests (pyctm wrapper, external dependency).

---

## External Dependencies

| Dependency | Where | Runtime? | Purpose |
|------------|-------|----------|---------|
| pyctm (write_RCG/RAC/BAN) | `io/write_inputs.py` | Only if generating new Fortran inputs | String I/O for .rcg/.rac/.ban files |
| rcg_cfp72/73 Fortran binaries | `angular/cfp.py` | Only if parsing CFP from binary | Coefficients of fractional parentage tables |
| scipy/sympy | `tests/test_angular/test_wigner.py` | Test-time only, optional | Cross-validation of Wigner symbols |

**No subprocess calls.** All Fortran reference data is pre-computed and shipped as
fixtures under `tests/reference_data/`.

---

## Known Limitations (documented in README)

1. HFS SCF orbital energies differ from Fortran by 1–8% (2nd-order FD vs O(h⁴) Numerov)
2. HFS spin-orbit ζ: BW reduction ratio matches Fortran to ≤3%; absolute 3d ζ ~25% high (radial bias)
3. Ti⁴⁺ d⁰ spectrum cosine similarity 0.978 (not 0.99); eigenvalues correct to 3.7e-7 Ry
4. Cr³⁺ d³ autograd NaN from degenerate eigenvalues in `eigh` backward
5. `get_sticks(max_gs=1)` is T-independent by design

---

## Issues and Observations

### Code quality

1. **Dead code in early-prototype modules.** `hamiltonian/diagonalize.py::get_diagonalized_matrices()`
   and `hamiltonian/crystal_field.py::build_crystal_field_matrix()` are incomplete stubs
   that are never called. They could be removed or clearly marked as deprecated.

2. **Duplicate assembly logic.** `hamiltonian/charge_transfer.py` reimplements
   `assemble_hamiltonian_block()` / `assemble_mixing_block()` with the same
   `assemble_matrix_from_adds` calls that `assemble.py` uses directly. The
   charge_transfer module is complete but unused — `assemble.py` does the work
   inline. Consider either routing `assemble.py` through `charge_transfer.py`
   or documenting the duplication.

3. **`io/write_inputs.py` has a fragile `sys.path` hack** (dynamically adds
   `../pyctm` to import path). This only matters for generating new Fortran
   inputs, not for the Phase 5 pipeline, but it's a maintenance risk.

### Completeness

4. **No Phase 5 path for RIXS.** `calcRIXS()` works in bootstrap mode (paired
   `.ban_out` files) but raises `NotImplementedError` for the from-scratch path.
   This would require running Phase 5 twice (absorption + emission edges) and
   pairing the results.

5. **No Phase 5 path for XES.** `calcXES()` similarly only works in bootstrap mode.
   Since XES is a diagonal cut of RIXS, this naturally follows from #4.

6. **`calcDOC()` is a complete stub.** Degree of covalency requires decomposing
   eigenvectors by configuration character — straightforward given the existing
   `BanResult` structure but not yet implemented.

7. **No emission `.ban_out` fixture.** The RIXS notebook and tests use synthetic
   data because no paired absorption+emission fixture is committed. Generating one
   requires running pyttmult with emission-edge parameters.

8. **Autograd does not flow through `cf`, `delta`, `lmct`, `mlct`.** These
   parameters are applied via `modify_ban_params()` which converts them to Python
   floats (`float(cf['tendq'])` at `build_ban.py:111`). Making them differentiable
   would require carrying them as tensors through the BAN data structure into the
   assembler's energy-offset and XHAM-scaling logic.

### Testing

9. **`spectrum/background.py` has no tests.** Five functions (arctan step, Gaussian
   peak, linear background, composite XAS background, saturation correction) are
   untested.

10. **`spectrum/rixs.py` has minimal direct testing.** Only one autograd test via
    `test_rixs_pair.py`. The kernel's numerical correctness against a reference
    implementation (pyctm) is not tested because no real fixture exists (#7).

11. **6 CFP tests are conditional** on the Fortran `rcg_cfp72` binary being present.
    In a CI environment without the binary, these skip silently.

### Numerical

12. **Cr³⁺ d³ autograd is excluded** from all autograd tests because
    `torch.linalg.eigh` backward produces NaN at exact eigenvalue degeneracies.
    This is a PyTorch limitation, not a multitorch bug. A custom `eigh` backward
    with regularization would fix it but is out of scope.

13. **Ti⁴⁺ d⁰ is a degenerate edge case** — no d-d Slater integrals, so the Phase 5
    path passes HAMILTONIAN blocks through unchanged. The 0.978 cosine similarity
    (vs 0.99 for others) is in the broadening layer, not eigenvalues.

---

## Immediate Priorities (ordered by expected impact)

1. **Add `spectrum/background.py` tests.** Five untested public functions. Low effort,
   high value for regression safety.

2. **Add a real RIXS emission fixture** (`nid8ct.ems.ban_out` or similar). Unblocks
   RIXS numerical validation and makes the RIXS notebook self-contained.

3. **Clean up dead prototype code.** Remove or mark as deprecated:
   `diagonalize.get_diagonalized_matrices()`, `crystal_field.build_crystal_field_matrix()`.

4. **Make `cf`/`delta`/`lmct`/`mlct` autograd-traceable.** Carry these as tensors
   through `modify_ban_params()` → assembler energy offsets / XHAM scaling.
   This completes the autograd story for all physical parameters.

5. **Implement `calcDOC()`.** Straightforward: decompose eigenvectors by configuration
   block indices already stored in `BanResult`.

6. **Wire Phase 5 into `calcRIXS()`.** Requires running the Phase 5 pipeline twice
   (absorption edge + emission edge) and pairing the results.

---

## Version History

| Date | Milestone | Tests |
|------|-----------|-------|
| 2026-04-05 | Initial port: parsers, angular, HFS, assembly, broadening | 150 |
| 2026-04-05 | Track A (RIXS bootstrap) complete | 150 |
| 2026-04-06 | Track B (Blume-Watson SOC) complete | 163 |
| 2026-04-11 | Track C (Phase 5 autograd pipeline) complete | 363 |

---

## What It Can Do (Current State)

**Working and validated:**
- L-edge XAS for all 3d transition metals (Ti⁴⁺ through Ni²⁺) in Oh and D4h symmetry
- Autograd gradients through `slater` and `soc` parameters (Phase 5)
- Bootstrap from pre-computed Fortran files (byte-exact with pyctm)
- RIXS 2D maps via Kramers-Heisenberg kernel (bootstrap mode, synthetic data)
- Blume-Watson multi-orbital spin-orbit coupling (matches Fortran ≤3% reduction ratio)
- HFS SCF from atomic number (adequate for starting parameters, not production ζ)
- Full pseudo-Voigt broadening with Thompson 1987 correct mode

**Not yet working:**
- Phase 5 RIXS (from-scratch, no Fortran files)
- Phase 5 XES
- Degree of covalency (DOC)
- Autograd through crystal-field tensor components (`cf` dict values)
- Autograd through charge-transfer parameters (`delta`, `lmct`, `mlct`)
- Cr³⁺ d³ autograd (PyTorch `eigh` backward limitation at degenerate eigenvalues)
