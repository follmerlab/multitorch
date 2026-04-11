# BUG-001: HFS SCF Orbital Energies — RESOLVED

**Status:** RESOLVED (2026-04-11)
**Severity:** Medium → downgraded to Low (1–8% approximation gap, not a bug)
**File:** `multitorch/atomic/hfs.py`

## Resolution summary

The original symptom description in this file was stale. At the time BUG-001
was written, parser and assembly bugs elsewhere were masking the actual HFS
SCF behavior. Measured ground truth on 2026-04-11 (Ni²⁺ 2p⁶3d⁸, Z=28):

| Orbital | multitorch (old EXF=0.7) | multitorch (new EXF=1.0) | Fortran ref |
|---|---|---|---|
| 1s | -601.615 | **-610.258** | -619.419 |
| 2p |  -64.471 |  **-66.975** |  -67.666 |
| 3d |   -2.349 |   **-3.021** |   -2.796 |

Root cause: `hfs_scf` defaulted to `EXF=0.7` (Kohn-Sham-like reduced Slater
exchange), but the Fortran reference run used `EXF=1.000` (full Slater-Xα),
as recorded in `tests/reference_data/nid8/nid8.rcn31_out` line 6. Swapping
the default from 0.7 → 1.0 reduces the relative error from ~16% to ~1–8%
across the full orbital set and brings us well inside the existing test
tolerances (which always passed even at the worse default, because the
tolerances were written with realistic HFS-vs-HF expectations).

## What actually changed

One line in `multitorch/atomic/hfs.py`:
```python
EXF: float = 0.7  →  EXF: float = 1.0
```
plus an updated docstring explaining the choice.

## What this doesn't fix (the real known limitation)

The residual ~1–8% error comes from:

1. **2nd-order finite-difference vs Fortran's O(h⁴) Numerov** on the non-
   uniform mesh. Largest impact on the outermost valence orbital (3d).
2. **Central-field spin-orbit ζ** instead of multi-orbital Blume-Watson
   exchange (README already documents this — affects the R*VI column, not
   orbital energies).

Neither is a bug, neither blocks the production pipeline (which reads
pre-computed `.rme_rcg` / `.rme_rac` files), and both are now documented
accurately in `README.md` §Known limitations.

## See also

- `.claude/orchestration/investigations/INV-002-hfs-energies.md` — full
  investigation log with ground truth, EXF sweep, and rationale.
- `README.md` §Known limitations §3 — updated user-facing statement.
