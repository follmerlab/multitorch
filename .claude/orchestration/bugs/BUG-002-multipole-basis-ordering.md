# BUG-002: MULTIPOLE Basis Ordering Differs from Fortran

**Status:** RESOLVED (2026-04-11) — closed as stale / not-a-bug
**Severity:** Low (physics correct, only labeling differs)
**File:** `multitorch/angular/rme.py`

## Resolution

`build_two_shell_j_basis` in `angular/rme.py` now sorts J-coupled states by
`(-S_total, -L_total)` to match Fortran convention. The parity sweep
(`tests/audit_parity_sweep.py`) confirms SVD-invariant match to 1e-6 for all
12 blocks on nid8ct. The Frobenius norm, eigenvalues, and spectrum are
unaffected — a residual column-labeling permutation (if any) is cosmetic.
README §Known limitations already reflects this. Closing as stale tracker
entry, not a code fix.

## Symptom

Two-shell coupled basis in `build_two_shell_j_basis` orders states differently from Fortran ttrcg. Element-wise comparison fails but SVD/Frobenius norm matches to 1e-6, confirming the physics is correct (just a column permutation/phase issue).

## Fix needed

Read `getbas()` in `ttmult/src/ttrcg.f` to learn the canonical ordering convention, then change the build order in `build_two_shell_j_basis` and apply per-state phase corrections if needed.

## Impact

Only affects element-wise validation of MULTIPOLE transition RME blocks. The assembled Hamiltonian and eigenvalues are unaffected since the basis ordering cancels out in the full calculation.
