# INV-002: HFS SCF Orbital-Energy Discrepancy

**Date:** 2026-04-11
**Status:** RESOLVED → fix merged, BUG-001 closed
**Files touched:** `multitorch/atomic/hfs.py` (1-line default change)

## Motivation

Three sources gave conflicting accounts of the HFS SCF accuracy:

- `.claude/orchestration/bugs/BUG-001-hfs-scf-energies.md` — claimed
  Ni²⁺ 3d was -43.6 Ry (vs -2.8 Ry expected), Ni²⁺ 2p was -98.0 Ry
  (vs -67.7), He 1s was -2.0 Ry (vs -0.9). Framed as a ~30-40 Ry bug.
- `README.md` "Known limitations" §3 — "~10%" error claim.
- `README.md` validation table — "HFS SCF orbital energies | 1 Ry |
  < 1 Ry | OK".
- `tests/test_atomic/test_hfs.py` — passing at baseline (141/141) with
  assertions `|E_3d - (-2.796)| < 1.0 Ry` and `|E_2p - (-67.7)| < 10.0 Ry`.

The stale bug tracker plus the passing tests pointed to a
documentation-vs-reality gap rather than a new regression.

## Ground truth (measured 2026-04-11)

Ran `hfs_scf` at the old default `EXF=0.7` on Z=28 Ni²⁺ and compared to
`tests/reference_data/nid8/nid8.rcn31_out`:

| Orbital | multitorch (EXF=0.7) | Fortran rcn31 | Abs err | Rel err |
|---|---|---|---|---|
| Ni²⁺ 1s | -601.615 Ry | -619.41904 Ry | 17.80 | 2.87% |
| Ni²⁺ 2p | -64.471 Ry | -67.66589 Ry | 3.19 | 4.72% |
| Ni²⁺ 3d | -2.349 Ry | -2.79595 Ry | 0.45 | 15.98% |
| He 1s (from SCF) | -1.2465 Ry | ~-1.836 Ry (HF) | — | — |

None of these match the 30-40 Ry figures in BUG-001. The BUG-001 file is
stale and was written before unrelated parser/assembly fixes had stabilized
the SCF loop.

## Root cause

The header of `tests/reference_data/nid8/nid8.rcn31_out` (line 6) records
the Fortran run parameters:

    TOLSTB=1.000   ...   EXF=1.000   CORRF=0.000      CA0= 0.500

**Fortran rcn31 used `EXF=1.000` for this reference run**, but the
`hfs_scf` Python signature defaulted to `EXF=0.7`. The `EXF` parameter is
the multiplier on the Slater-Xα statistical exchange potential:
`V_ex(r) = -3·EXF·(3ρ/8π)^(1/3)` (Rydberg). A smaller EXF means weaker
exchange binding, which systematically shifts all orbital energies up
(less bound) — the discrepancy grows with orbital radius because
valence states sample the exchange tail more strongly.

## EXF sensitivity sweep

Re-ran Ni²⁺ at three EXF values, everything else fixed:

| EXF  | 1s         | 2p        | 3d       | max rel err |
|------|------------|-----------|----------|-------------|
| 0.7  | -601.615   | -64.471   | -2.349   | 16.0% (3d)  |
| 0.8  | -604.483   | -65.294   | -2.563   | 8.3% (3d)   |
| 1.0  | -610.258   | -66.975   | -3.021   | 8.0% (3d)   |
| **Fortran ref** | **-619.419** | **-67.666** | **-2.796** | — |

At EXF=1.0, errors drop to:
- 1s: 1.48%
- 2p: 1.02%
- 3d: 8.04% (and the sign actually flips — we slightly *over*-bind 3d
  relative to Fortran, consistent with the remaining gap being
  mesh/Numerov-order effects rather than exchange).

Cross-check on He 1s at EXF=1.0: -1.7416 Ry (HF reference: -1.836 Ry;
4.5% high). Cross-check on Fe²⁺ 3d at EXF=1.0: -2.7372 Ry, converges in
~30 iterations.

## Remaining (bona fide) approximation gap

Even at the corrected EXF=1.0, we retain a systematic ~1–8% discrepancy
vs Fortran. This is the *real* "known limitation" and has two independent
contributions, neither of which is a bug:

1. **Numerov order.** hfs.py uses a 2nd-order finite-difference
   recurrence (`scheq`, lines 360-400, 447-461) on the non-uniform mesh.
   Cowan's rcn31.f uses 4th-order Numerov on the same mesh. O(h²) vs
   O(h⁴) shows up primarily in the outermost bound orbital because its
   effective kinetic energy is smallest and the integrator error per
   step is largest in absolute terms near the classical turning point.
2. **Spin-orbit / exchange integrals.** `hfs.py` line 687 computes SOC ζ
   from a central-field formula, not the full Blume-Watson multi-orbital
   exchange. README already documents this. Does not affect orbital
   energies directly but does mean the Slater integral column "R*VI"
   in Fortran's rcn31 output won't match to better than ~5%.

Neither is a 1-line fix, and neither matters for the production pipeline
(which reads pre-computed `.rme_rcg` / `.rme_rac` from Fortran). Deferred.

## Decision

Applied the one change that *is* a 1-line fix:

- `multitorch/atomic/hfs.py` line 555 — `EXF: float = 0.7` → `EXF: float = 1.0`
  plus an updated docstring explaining why.

Full test suite re-run after the change: **141/141 passing**, same as
baseline. No reference-data regeneration needed.

Documentation follow-ups (this commit):
- `.claude/orchestration/bugs/BUG-001-hfs-scf-energies.md` → closed,
  updated with real numbers and points here.
- `.claude/orchestration/INDEX.md` → BUG-001 → Resolved.
- `README.md` §Known limitations §3 → rewritten with the corrected
  EXF=1.0 numbers, reframed as a 1–8% approximation rather than a bug.

## Files touched

- `multitorch/atomic/hfs.py` (1 line + docstring)
- `README.md` (Known limitations §3)
- `.claude/orchestration/INDEX.md`
- `.claude/orchestration/bugs/BUG-001-hfs-scf-energies.md`
- `.claude/orchestration/investigations/INV-002-hfs-energies.md` (new)

---

## 2026-04-11 — Track B closure note (Blume-Watson now landed)

The "BW residual gap" identified at the end of this investigation is no
longer the main story. Track B (commits Track B1..B5) implemented full
multi-orbital Blume-Watson via `multitorch/atomic/blume_watson.py` and
wired it into `hfs_scf` behind `zeta_method="blume_watson"`. See
[`INV-003-blume-watson.md`](INV-003-blume-watson.md) for the line-by-line
mapping of `rcn31.f::ZETABW` to the PyTorch port.

What's left: the **absolute** 3d ζ is still ~25% above Fortran, because
the underlying HFS solution binds 3d ~8% too tightly (this investigation's
core finding — see §3 of README "Known limitations"). The BW correction
itself is correct: the *reduction ratio* (BW vs central-field) matches
Fortran on every row tested:

| orb | PyTorch BW/CF | Fortran BW/RVI | Δ |
|---|---|---|---|
| 2P | 0.939 | 0.951 | 1.3% |
| 3P | 0.943 | 0.956 | 1.4% |
| 3D | 0.834 | 0.860 | 2.6% |

So the remaining gap is the O(h²) → O(h⁴) Numerov upgrade tracked here,
not more BW work. Track B is closed; no further action under this
investigation until someone takes on the Numerov port.
