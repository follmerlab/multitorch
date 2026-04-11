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
| C2 | `build_banfile_from_params` | ⏳ pending | — | New file `hamiltonian/build_ban.py` |
| **C3-pre** | Vectorize `assemble_matrix_from_adds` | ✅ landed 2026-04-11 | (uncommitted) | Vectorized slice-add replaces `float()` element loop. 8 new tests in `test_assemble_matrix_from_adds.py`: legacy equivalence on all 226 nid8ct blocks (atol=0), gradient propagation through `cowan_section` AND `scale`, slice arithmetic, OOB clipping. **184/184 passing.** Gradient gating contract met. |
| C3a | Torch wrappers around numpy angular blocks | ⏳ pending | — | `angular/torch_blocks.py` |
| C3b | `read_rcn31_out_params` (atomic param fixture loader) | ⏳ pending | — | `atomic/parameter_fixtures.py` |
| C3c | `scale_atomic_params` (Slater/SOC scaling) | ⏳ pending | — | `atomic/scaled_params.py` |
| C3d | `build_rac_in_memory` + section plan | ⏳ pending | — | `hamiltonian/build_rac.py`. **Defines the contract C3e must satisfy.** |
| C3e | `build_cowan_store_in_memory` | ⏳ pending | — | `hamiltonian/build_cowan.py`. Must match parsed fixture element-wise to 1e-6 |
| C3f | Single-fixture parity + autograd test | ⏳ pending | — | `test_phase5_parity.py` (nid8ct) |
| C4 | Wire into `calcXAS` | ⏳ pending | — | `api/calc.py:131–147` |
| C5 | Multi-fixture parity sweep | ⏳ pending | — | All 8 Ti–Ni cases + als1ni2 |
| C6 | Notebook + docs updates | ⏳ pending | — | `notebooks/03_*.ipynb`, optional `05_autograd_optimization.ipynb` |
| C7 | Track C verification + tag | ⏳ pending | — | `pytest tests/ -q`, audit sweep |

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
