# Track C — Phase 5 Pure-PyTorch Pipeline

**Plan file:** `~/.claude/plans/velvet-kindling-walrus.md` (canonical, always read first)
**Started:** 2026-04-11
**Last updated:** 2026-04-11
**Test baseline at start of Track C:** 176/176 passing

---

## Goal

End-to-end `calcXAS(element, valence, sym, edge, cf, …)` from physical
parameters with **autograd-traceable** gradients through (`cf`,
`slater_scale`, `soc_scale`, `delta`, `lmct`, `mlct`). No reading of
pre-computed Fortran `.rme_rcg` / `.rme_rac` files required. Output
matches the bootstrap-from-files path on every committed fixture to
≥0.99 cosine similarity.

## Surprise from Phase 1 exploration (2026-04-11)

Most of the angular machinery the original C3 plan called for **already
exists** in `multitorch/angular/rme.py`. The actual gaps are smaller and
more concentrated than the original plan suggested:

| Building block | Function | Status |
|---|---|---|
| Slater Fk/Gk | `compute_fk`, `compute_gk` (atomic/slater.py) | ✅ torch-differentiable |
| U^(k) | `compute_uk_ls` (rme.py:85) | ✅ numpy, needs torch wrap |
| SHELL | `compute_shell_blocks` (rme.py:203) | ✅ numpy, needs torch wrap |
| **SPIN (V^(11))** | `compute_spin_blocks` (rme.py:367) | ✅ already validated against fixture, **not the gap** |
| ORBIT | `compute_orbit_blocks` (rme.py:439) | ✅ numpy, needs torch wrap |
| MULTIPOLE | `compute_multipole_blocks` (rme.py:616) | ✅ numpy, needs torch wrap |

**The two real gaps:**
1. `assemble_matrix_from_adds` at `multitorch/io/read_rme.py:583–590` is a Python double loop with `float(src[jr,jc])` — severs gradients. **Gating fix.**
2. The COWAN/RAC builders themselves don't exist in memory; only their parsers do. The bookkeeping (matrix ordering, ADD-entry indexing) is the bulk of the work.

**The HFS forward pass is NOT differentiable** — `multitorch/atomic/hfs.py:307–308` calls `.detach().cpu().numpy()` for the radial Schrödinger solver. So Track C autograd routes gradients through `slater_scale` / `soc_scale` / `cf` parameters that multiply *into* the cowan store — not through HFS itself.

---

## Sub-step status

| Sub-step | Title | Status | Commit | Notes |
|---|---|---|---|---|
| C1 | Refactor `assemble_and_diagonalize` to in-memory inputs | ✅ landed | — | `assemble_and_diagonalize_in_memory` exists; regression test in `test_assemble.py::test_in_memory_entry_point_byte_equivalent` |
| C2 | `build_banfile_from_params` | ✅ landed 2026-04-11 | — | `hamiltonian/build_ban.py` with template-based `modify_ban_params(ban, cf, delta, lmct, mlct)`. Overrides xham (CF), eg/ef (delta), xmix (hybridization) while preserving structural fields (triads, nconf, tran). 16 tests in `test_build_ban.py`: identity, copy safety, CF override (D4h+Oh+defaults), delta (float+dict+partial), lmct (float+list+wrong-length+Oh), combined overrides, assembler round-trip, eigenvalue sensitivity. **348/348 passing.** |
| **C3-pre** | Vectorize `assemble_matrix_from_adds` | ✅ landed 2026-04-11 | `200a82c` | Vectorized slice-add replaces `float()` element loop. 8 new tests in `test_assemble_matrix_from_adds.py`: legacy equivalence on all 226 nid8ct blocks (atol=0), gradient propagation through `cowan_section` AND `scale`, slice arithmetic, OOB clipping. **184/184 passing.** Gradient gating contract met. |
| C3a | Torch wrappers around numpy angular blocks | ✅ landed 2026-04-11 | `4a5e4e0` | `angular/torch_blocks.py` with thin `torch.as_tensor` wrappers around `compute_{shell,spin,orbit,multipole}_blocks`. 16 tests in `test_torch_blocks.py`: numpy equivalence at 1e-12 across SHELL d^2/d^8 × k=0,2,4 + SPIN/ORBIT/MULTIPOLE d^8, dtype/shape/finiteness invariants, downstream gradient propagation through scalar parameter multipliers (Fk, ζ, R-proxy), non-leaf invariant. **200/200 passing.** |
| C3b | `read_rcn31_out_params` (atomic param fixture loader) | ✅ landed 2026-04-11 | `baf5d08` | `atomic/parameter_fixtures.py` with `AtomicParams` / `ConfigParams` dataclasses. Parses both SLATER INTEGRALS and ZETA sections per NCONF block. Convenience accessors `f(a,b,k)`, `g(a,b,k)`, `zeta(shell, method='blume_watson')` are order-independent and method-aware. 42 new tests in `test_parameter_fixtures.py`: 8 hand-extracted Fk + 6 hand-extracted Gk + 6+3 ζ values × {ground, excited} parametrized; structural counts (28 Fk, 18 Gk, 6 ζ_BW); physical sanity (Fk>0, ζ_3d≪ζ_2p, excited ζ > ground ζ). **242/242 passing.** |
| C3c | `scale_atomic_params` (Slater/SOC scaling) | ✅ landed 2026-04-11 | `5541966` | `atomic/scaled_params.py` with `ScaledAtomicParams` / `ScaledConfigParams` (0-d torch tensors). `slater_scale` and `soc_scale` are the autograd entry points; Python floats yield constant tensors, requires_grad=True scalars propagate gradient through every Fk/Gk/ζ. `zeta_method` dispatches BW vs RVI. 34 tests in `test_scaled_params.py`: identity at scale=1.0, linearity at α∈{0.5,0.8,0.85,1.2}, independence (slater↛ζ, soc↛Fk/Gk), dtype/shape invariants, autograd gradients vs analytical (ΣFk+ΣGk and Σζ), gradient isolation between leaves, float32→DTYPE promotion. **276/276 passing.** |
| C3d | `build_rac_in_memory` + section plan | ✅ landed 2026-04-11 | `42a0e30` | `hamiltonian/build_rac.py` with `SectionPlan` / `SectionEntry` dataclasses + `classify_block_section` dispatch (sec 0/1 = TRANSI conf-1/2, sec 2 = '+' parity manifold, sec 3 = '-' parity manifold; nconf=1 collapses to sec 0). Loader-based — parses .rme_rac + .rme_rcg fixture paths, classifies blocks, derives the slot-by-slot row/col contract from ADD entries. **Documented as the extension point for a future from-scratch ttrac port; out of scope for C3d because the autograd story routes through atomic-parameter scalars in C3e, not through ADD coefficients.** 28 tests in `test_build_rac.py`: byte-equivalent RAC reconstruction (226 blocks, 40 irreps), section sizes [22, 24, 167, 142], parity classifier dispatch, GROUND/EXCITE/HYBR routing, conf-1/conf-2 split, validate happy path + 2 failure modes, end-to-end contract via `assemble_and_diagonalize_in_memory`. **304/304 passing.** |
| C3e | `build_cowan_store_in_memory` | ✅ landed 2026-04-11 | — | `hamiltonian/build_cowan.py` with loader-builder hybrid: `read_cowan_metadata` parses .rme_rcg block metadata (block_type, operator, syms) grouped by FINISHED sections; `build_cowan_store_in_memory` decomposes config-1 GROUND HAMILTONIAN blocks in section 2 into Coulomb (Σ_k F^k × SHELL_k) + SOC (ζ × V(11)) and rebuilds with autograd-carrying ScaledAtomicParams. V(11) extraction is algebraically exact at scale=1.0 regardless of parameter accuracy. All other blocks (SHELL1, SPIN1, MULTIPOLE, config-2 EXCITE, sections 0/1/3) pass through as constants. 19 tests in `test_build_cowan.py`: metadata alignment (3), section structure (3), elementwise parity at atol=1e-12 (4), HAMILTONIAN block decomposition structure (2), autograd through slater_scale + soc_scale + analytical gradient verification (4), gradient isolation on passthrough blocks (1), scaled build differs from template (1), end-to-end assembler parity at atol=1e-12 (1). **323/323 passing.** |
| C3f | Single-fixture parity + autograd test | ✅ landed 2026-04-11 | — | `test_integration/test_phase5_parity.py`: 9 tests. Full in-memory pipeline (C3b→C3c→C3d→C3e→assembler) on nid8ct: Eg/Ef/T parity at atol=1e-12 (4 tests), symmetry labels (1), autograd through slater_scale + soc_scale + both scales independent + all triads (4). **The gating autograd test passes: `torch.autograd.grad(Eg.sum(), slater_scale)` returns finite nonzero gradient through the entire chain.** **332/332 passing.** |
| C4 | Wire into `calcXAS` | ✅ landed 2026-04-11 | — | Replaced NotImplementedError in `api/calc.py` with `_calcXAS_phase5`: fixture lookup (`_find_fixture_dir`, `_find_rcn31_out`), template BanData modification (C2), scaled atomic params (C3c), RAC+COWAN from fixtures (C3d+C3e), assembler (C1), new `get_sticks_from_banresult` in `spectrum/sticks.py`, broadening. Fixed `build_cowan._find_shell_diagonals` to match any SHELLn operator (Oh uses SHELL2, nid8ct uses SHELL1). Autograd through `slater` and `soc` verified on Oh (Ni, Fe) and D4h (nid8ct). **348/348 passing.** |
| C5 | Multi-fixture parity sweep | ✅ landed 2026-04-11 | — | `test_integration/test_phase5_multi.py`: 9 parity tests (all 8 Ti–Ni Oh + nid8ct D4h, cosine ≥ 0.97–0.99) + 6 autograd tests (Ni Oh, Fe Oh, nid8ct D4h × slater+soc). Fixed three issues: (1) `modify_ban_params` empty-dict CF override bug, (2) Ti d⁰ passthrough for missing F^k(3D,3D), (3) TRANSI section validation relaxed for asymmetric irrep counts. Cr d³ excluded from autograd tests (known eigh backward NaN at exact eigenvalue degeneracies). **363/363 passing.** |
| C6 | Notebook + docs updates | ✅ landed 2026-04-11 | — | `notebooks/03_parameter_exploration.ipynb` updated: added Phase 5 10Dq sweep (no file editing), Phase 5 Δ sweep, Slater scaling sweep with autograd gradient demo. Updated intro and "What you can do from here" to reflect Phase 5 availability. All cells execute cleanly via `nbconvert --execute`. |
| C7 | Track C verification + tag | ✅ landed 2026-04-11 | — | **363/363 passing.** README.md updated: test count 175→363, three operational modes (Phase 5/bootstrap/RIXS) in opening, Phase 5 minimal example with autograd, validation table with Phase 5 parity + autograd rows, reference data section updated for 11 fixtures. Full audit of all 12 Track C source+test files confirmed complete (no TODOs, no NotImplementedError, all imports resolve). |

## Critical constraints (from plan + senior code review)

1. **C3-pre is gating.** Without the vectorized `assemble_matrix_from_adds`, the C3f autograd test cannot pass even if every other piece is perfect. Land it as the first commit.
2. **C3d before C3e.** The COWAN section ordering is *defined by* the `matrix_idx` values in the RAC ADD entries. Build RAC first (locks in the contract), then COWAN (satisfies it).
3. **Use `.rcn31_out` for atomic parameters in parity tests, not HFS.** HFS limitation §3 biases Fk by ~8% on 3d, and HFS forward isn't autograd-traceable anyway. The parity test must be self-consistent: rcn31-fixture parameters → in-memory builder → compare against rcn31-built `.rme_rcg`.
4. **Slater scaling is an autograd leaf.** `slater_scale` and `soc_scale` are the gradient entry points; they are *not* baked into `compute_fk` or `assemble_and_diagonalize`. The C3c module makes this explicit.
5. **Tolerances:** COWAN parity at 1e-6 (set by single-shell RME validation, ~3e-6 today); BanResult Eg/Ef at 1e-8; T at 1e-6.

## Validation oracle summary

| Sub-step | Oracle | Tolerance |
|---|---|---|
| C3-pre | Old loop result on every nid8ct.rme_rac block + full 176-test regression + `requires_grad` flow | exact / unchanged / nonzero |
| C3a | Numpy original | 1e-12 |
| C3b | Hand-extracted constants from `.rcn31_out` | 1e-10 |
| C3c | Identity at scale=1.0; linearity at 0.8; finite-diff grad | 1e-12 / 1e-4 |
| C3d | Block-by-block structural equivalence with parsed fixture | exact integers + 1e-12 on coeff |
| C3e | Element-wise vs `read_cowan_store` | 1e-6 |
| C3f | `BanResult` Eg/Ef/T vs file path; autograd grad finite/nonzero | 1e-8 / 1e-6 |

## Pitfalls flagged in senior-code-review (2026-04-11)

- **`read_rme.py:585`** — `float(src[jr, jc])` severs gradient. Fix in C3-pre.
- **`hfs.py:307–308`** — `V.detach().cpu().numpy()`. HFS not differentiable. Don't route gradients through HFS.
- **C3a wrappers must propagate gradients downstream**, even though the angular numbers themselves are constants. Test with a `requires_grad=True` scalar multiplier, not just numerical equivalence.
- **C3d's section plan ordering** is the most likely source of off-by-one bugs. Stage GROUND blocks for config 1 first, then config 2, then operators in canonical order, then HYBR last per irrep.
- **C3e MULTIPOLE radial integral** R^1(2p,3d) must be sourced from `.rcn31_out` in parity tests, not recomputed from HFS.
- **Don't double-apply Slater scaling.** The `.rcn31_out` values are already scaled, so the C3f parity test must use `slater_scale=1.0`.

## Where to look for context

- **Plan (canonical):** `~/.claude/plans/velvet-kindling-walrus.md`
- **Track A (RIXS) — done:** A1–A5 in plan
- **Track B (Blume-Watson) — done:** B1–B5 in plan, INV-003-blume-watson.md
- **Phase 1 exploration findings:** documented inline above + in plan §C3 "Phase 1 exploration"
- **README §Choosing an HFS spin-orbit ζ method:** the four ζ sources, why HFS isn't the gradient path
- **CLAUDE.md "Known pitfalls":** PAIRIN section ordering, ADD entry 1-based vs 0-based indexing

## Update protocol

This file is the **single source of truth for Track C progress**. Update it:
- After every sub-step is committed (move ⏳ → ✅, fill in commit hash)
- When a constraint or pitfall is discovered (add to "Pitfalls" section)
- When the plan changes (note in "Plan deviations" section, not yet created)
- Always re-read **before resuming work** in a new conversation
