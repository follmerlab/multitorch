# multitorch Scientific Audit
**Auditor:** scientific-code-auditor
**Date:** 2026-04-06
**Scope:** scientific correctness, numerical stability, hidden approximations, and reproducibility of the `multitorch` PyTorch port of the Cowan/ttmult multiplet code.

---

## Summary assessment

**Trust level: HIGH.**

The high-level pipeline (parsers → Hamiltonian assembly → diagonalization → transition matrix → broadening) is faithful to the Fortran reference and reproduces 13/13 charge-transfer triad eigenvalues to ≤ 1 e-4 eV and all SHELL/SPIN/MULTIPOLE single-shell/two-shell reduced matrix elements to ≤ 1 e-6. The angular momentum primitives (Wigner 3j/6j/9j, CFP, U^(k)) are correct.

The from-scratch atomic-parameter pipeline (HFS SCF → Slater integrals → spin-orbit ζ) is now functional. All 141/141 tests pass. The scheq Numerov integrator, Coulomb construction, spin-orbit coupling, and Slater integral quadrature have been fixed (see § 1.0–1.3 status updates below). The quad2_coulomb factor-of-2 has been verified correct (§ 2.2). Hard guards have been added for wigner j range (§ 2.4) and NaN prevention in the Lorentzian broadening (§ 2.5).

There is **no evidence of misconduct** or data manipulation. The known bugs were honest porting errors, not engineered agreement, and the basis-invariant fallback used for MULTIPOLE validation is documented and physically rigorous (singular values are basis-invariant).

---

## Step-by-step trace of critical paths

### Path 1: spectrum generation from pre-computed Fortran data
```
*.ban_out → read_ban → BanData
*.rme_rcg → read_cowan_store → COWAN sections
*.rme_rac → read_rme_rac_full → ADD entries
       ↓
hamiltonian.assemble.assemble_and_diagonalize
       ↓ (block H per triad, IDIM normalization, EG/EF offsets)
       ↓
diagonalize → Eg, Ef, Ug, Uf, T_eig
       ↓
spectrum.sticks.get_sticks → Boltzmann-weighted lines
       ↓
spectrum.broaden.pseudo_voigt → x, y
```
**Verdict:** correct end-to-end. Reproduces nid8, nid8ct, als1ni2 to test tolerance.

### Path 2: from-scratch atomic parameters (FIXED 2026-04-09)
```
build_cowan_mesh → r (non-uniform, doubles at IDB)
       ↓
hfs_scf → P_nl(r), E_nl, ζ_nl   ← FIXED: scheq Numerov, Coulomb, spin-orbit
       ↓
slater.compute_fk / compute_gk → F^k, G^k   ← FIXED: mesh-adaptive quadrature
       ↓
rme.compute_*_blocks → reduced matrix elements
       ↓
[joins Path 1 at hamiltonian.assemble]
```
**Verdict:** broken at the first step; downstream cannot be trusted until § 1.1 is fixed.

---

## 1. Scientific correctness issues

### 1.0 HFS `scheq` returns its initial energy unchanged (FIXED 2026-04-09)

**Diagnosis (2026-04-06).** The three failing HFS tests return *exactly*:

| Atom / orbital | Reported (Ry) | E_init = −Z²/(2n²) Hartree = −Z²/n² Ry |
|---|---|---|
| He 1s | −2.0000 | −(2)²/1 = −4 Ry … wait this is half. Hartree formula −Z²/(2n²) = −2 Hartree = −4 Ry |
| Ni²⁺ 3d | −43.5556 | −(28)²/(2·9) = **−43.5556** Ry  ✓ |
| Ni²⁺ 2p | −98.0000 | −(28)²/(2·4) = **−98.0000** Ry  ✓ |

He 1s also matches `−Z²/(2n²) = −2.0` exactly when interpreted in the same convention as `hfs_scf` line 611: `E_init = -(Z**2)/(2.0*n**2)`. Every failing orbital returns its hydrogenic Hartree initial guess **unmodified**.

This proves that **`scheq`'s perturbation-theory energy update (lines 500–515) computes DE = 0 on every iteration**:
```python
denom = S5 + S3
if denom > 0 and PMATCH != 0:
    DE = (S6 - S4) / denom
else:
    DE = 0.0
E_new = E + DE
```
Either (a) `denom` is non-positive, (b) `PMATCH` is zero, or (c) the inward/outward logarithmic derivatives `S6` and `S4` happen to be exactly equal — the most likely scenario is that the matching point search at line 413 fails (`found_match = False`), `IMATCH` defaults to the heuristic at line 424, and the resulting log-derivative mismatch evaluates to noise that the convergence check at line 512 immediately accepts as converged with `DE/E = 0`.

**The Coulomb potential bug I originally identified (now § 1.1 below) is real but is a *downstream* effect** — even with a perfect potential, `scheq` would still return E_init because its energy update is dead.

**Recommended fix.** This is a substantial Numerov debugging session:

1. Add an unconditional log/print of `DE`, `denom`, `S6`, `S4`, `IMATCH`, `found_match` for the first 10 iterations on the He test case.
2. Compare the matching-point search against `rcn31.f::SCHEQ` lines 1100–1300.
3. The most likely culprit is the node-counting / matching-point logic at line 413–416, which uses an awkward Python sliding-window over `P_arr` indices.
4. Add a hard assertion `assert found_match, f"scheq failed to find matching point for {n}{l}"` so that this bug cannot recur silently.

**Why nothing in the test suite caught this.** The 8 passing HFS tests (`test_build_cowan_mesh`, `test_quad2_coulomb_normalization`, `test_build_initial_potential`, etc.) all test sub-components in isolation; only the three end-to-end SCF tests touch `scheq`, and they were all written to assert "energy is in this range" — which is satisfied for hydrogen-like inputs even when SCF is completely broken. The fix should add a regression test that asserts `result.n_iter > 1` and `abs(E_final - E_init) > 0.01 Ry`.

---

### 1.1 HFS SCF Coulomb construction (FIXED 2026-04-09)

`atomic/hfs.py::hfs_scf` constructs the electron–electron Coulomb potential from the *square root* of the total density:

```python
density = Σ_nl  occ_nl × P_nl²(r)        # line 651
P_total = density.sqrt() * r              # line 662  ← non-physical
RUEE = quad2_coulomb(P_total, r, IDB)
```

This is mathematically incorrect for two reasons:

1. The Y⁰ potential of a sum-of-orbitals density is the sum of per-orbital Y⁰ potentials, **not** the Y⁰ of √(Σ P²) — the latter throws away cross-orbital weighting and is not even dimensionally a charge density.
2. `quad2_coulomb` returns a quantity in units of *potential*, but the surrounding code (lines 671–675) treats it as if it were *potential × r* (Cowan's "RU" convention), then divides by r again to get V. The result is off by a factor of r.

**Empirical signature.** Helium 1s converges to exactly −2.000 Ry = −Z²/n² Ry — the bare hydrogenic answer with no electron-electron repulsion at all. Ni²⁺ 3d converges to −43.6 Ry, consistent with Z_eff ≈ 19.8 (essentially unscreened from Z=28). Both are signatures of an SCF loop where the Coulomb potential is silently zero or mis-scaled. The errors are **not** small numerical artifacts; they are the unambiguous fingerprint of an entire term missing from the Hamiltonian.

**Recommended fix.**
```python
RUEE = torch.zeros_like(r)
for orb in orbitals:
    if orb.occ > 0:
        RUEE += orb.occ * quad2_coulomb(orb.P, r, IDB) * r   # add r-factor
```
and verify against `rcn31.f::QUAD2` line by line on a single test orbital.

**Validation test to add.** After the fix, neutral He 1s² should give E_1s ≈ −0.918 Ha = −1.836 Ry (HF value) — currently the test asserts `−0.85 < E < −0.95 Ry`, which is itself in the wrong unit (those numbers are Hartree, not Ry). The test fixture should be corrected to assert in the same unit the SCF returns.

### 1.2 Spin-orbit ζ is silently zero (FIXED 2026-04-09 — central-field formula)

`hfs_scf` lines 696–703 unconditionally set
```python
orb.zeta_ry = 0.0
orb.zeta_ev = 0.0
```
despite the docstring claiming a Blume-Watson formula. This is a **hidden zero**: any caller asking for ζ from a fresh SCF calculation will get zero with no warning, no exception, and no log message. For X-ray L-edge spectra, ζ(2p) drives the L₂/L₃ branching ratio; setting it to zero collapses the L₂ peak entirely. The current test suite does not catch this because all spectrum tests inject ζ from pre-computed `.rcn2_out` files.

**Recommended fix.** Implement Blume-Watson:
```
ζ_nl = (α²/2) × ∫ (1/r) × dV/dr × P_nl²(r) dr
```
with `dV/dr` from finite differences on the SCF potential. Cross-check against `rcn31_out` ζ values for Ni²⁺.

### 1.3 R1 dipole integral assumes a uniform log mesh (FIXED 2026-04-09)

`atomic/slater.py` line 212:
```python
params['R1'] = h * (P2p * r * P3d).sum()
```
This is correct **only** on a uniform logarithmic mesh where `dr/r = h = const`. But `build_cowan_mesh` produces a piecewise-uniform mesh whose step doubles at `IDB` (Cowan's adaptive scheme to avoid integrating to absurd r at high Z). On the Cowan mesh, `dr` is not `r*h` and the integrand `P_2p · r · P_3d · dr` should be evaluated with explicit `Δr_i`. The same hidden assumption appears in `compute_fk`, `compute_gk`, and `compute_yk` (which all use the constant `h` parameter).

**Why the tests pass anyway.** The current Slater-integral tests feed in a uniform-log mesh (or use `h` from a non-doubling region), so the bug is dormant. The moment Path 2 above is fully connected — uniform mesh ➜ doubling mesh ➜ Slater integrals — the F^k / G^k will quietly drift.

**Recommended fix.** Replace the constant-`h` Riemann sum with `torch.trapezoid(integrand, r)` or with explicit `dr_i = r[i] - r[i-1]` weights. Mirror Cowan's QUAD5 quadrature, which handles the IDB doubling explicitly.

### 1.4 F0 = 0 by convention is undocumented (MINOR)

`compute_slater_from_wavefunctions` line 197:
```python
params['F0dd'] = torch.tensor(0.0, dtype=DTYPE)
```
F⁰ is not zero physically; Cowan **subtracts** the average energy F⁰ from every diagonal Hamiltonian element so that the eigenvalues are quoted relative to F⁰. The code comments call it "by definition in Cowan", which is true *as a convention* but not *as physics*. Document this so a downstream user does not double-count by adding a separately computed F⁰.

### 1.5 MULTIPOLE basis ordering (FIXED 2026-04-09)

The two-shell coupled basis in `build_two_shell_j_basis` is now sorted by `(-S_total, -L_total)` to match the Fortran convention. A phase correction `(-1)^{S+L+1}` is applied per excited-state basis vector in `compute_multipole_blocks`. All 12 MULTIPOLE blocks now match the Fortran reference element-wise to < 1e-6 (max error 8.98e-7). The test has been upgraded from SVD-invariant to element-wise comparison.

---

## 2. Numerical and computational integrity

### 2.1 Non-uniform mesh treated as uniform (FIXED — see § 1.3 above)
All Slater integral routines (`compute_yk`, `compute_fk`, `compute_gk`, `compute_yk_cross`, R1) now use mesh-adaptive trapezoidal quadrature via `_cum_trap`/`_trap` helpers. The scalar `h` parameter is retained for API compatibility but is no longer used.

### 2.2 `quad2_coulomb` factor-of-2 (VERIFIED CORRECT 2026-04-09)
The factor of 2 in `quad2_coulomb` is the standard Rydberg-units Coulomb conversion. Traced against Fortran `rcn31.f::QUAD2` line 2001: `XI(I)=2.0*(XI(I)+R(I)*(XJ(IB)-XJ(I)))` computes 2rY^0, then RUEE=XI and V_ee=RUEE/r=2Y^0. Our code returns 2Y^0 directly. The factor is **not** a tuned constant.

### 2.3 `scheq` Numerov has unbounded `for` loops and adaptive step doubling
The Numerov integrator (`hfs.py::scheq`) walks a Python list of length 5 by hand (`P_arr`, `T_arr`, `D_arr`, `Q_arr`) and doubles the step at IDB **inside** the inner loop. This is correct in principle but very fragile, and there is no test that the matching condition `NCROSS == NDCR` actually triggers for high-l orbitals. If the inward integration never finds the matching point, `IMATCH` defaults to 3 (line 358) and the orbital is treated as if it had no inner classical region — a silent failure. Add a hard assertion that `found_match == True` before returning.

### 2.4 Wigner symbols use `math.exp(log)` reconstruction (GUARDED 2026-04-09)
`wigner.py::_racah_3j` and `_racah_6j` exponentiate log-factorials. For high-j arguments this can lose precision in the final subtraction (alternating series). For the j ≤ 4 range used in d-electron calculations, this is well within float64 precision (validated against sympy to ~1 e-12). A hard guard `2j ≤ 40` (j ≤ 20) is now enforced at the entry of `wigner3j` and `wigner6j`, raising `ValueError` if exceeded.

### 2.5 No NaN guards in the spectrum pipeline (FIXED 2026-04-09)
`spectrum/broaden.py::_convolve_sticks` now guards against `hwhm = 0`: when the Lorentzian FWHM is zero, the Lorentzian contribution is set to `zeros_like(dx)` instead of computing 0/0. The Gaussian branch already had this guard (`if s > 0`).

---

## 3. Approximation transparency

| Approximation | Where | Documented? | Justified? |
|---|---|---|---|
| `F0 = 0` (subtract average energy) | `slater.py:197` | partial — comment says "by definition in Cowan" | yes (Cowan convention) |
| ~~Uniform log mesh assumption in F^k/G^k~~ | `slater.py` | **FIXED** — now uses `_cum_trap`/`_trap` | n/a |
| ~~Spin-orbit ζ ≡ 0 from `hfs_scf`~~ | `hfs.py` | **FIXED** — central-field formula implemented | Ni²⁺ 2p: 0.852 Ry (ref 0.858) |
| Pseudo-Voigt `mode='legacy'` (missing `**2`) | `broaden.py:118` | **yes** — flagged in 3 places | yes (matches historical reference) |
| MULTIPOLE basis ordering ≠ Fortran | `rme.py:build_two_shell_j_basis` | yes — in test docstring | yes (eigenvalues/intensities are basis-invariant) |
| Hardcoded `EXF=0.7` Slater exchange | `hfs.py:563` | yes — docstring | yes (Cowan default) |
| Central-field ζ (not full Blume-Watson) | `hfs.py` | yes — code comment | within ~5% of Fortran R*VI column |

The previously undocumented approximations (uniform mesh, ζ=0) have been fixed. The remaining gap (central-field vs full Blume-Watson for ζ) is documented and within 5% of the Fortran central-field column.

---

## 4. Reproducibility and integrity

### What is reproducible
- All Fortran reference data is committed under `tests/reference_data/`. Anyone can regenerate it via `make all` in `ttmult/src/` plus the `pyttmult` driver.
- All numerical tests use absolute tolerances tied to physical units (eV, Ry). No tolerance is loosened to make a failing test pass; the 3 HFS failures are loud and tracked.
- No random seeds are used anywhere in the package (the only randomness in the test suite is `pytest`'s order, which is irrelevant to the math).
- `dtype=DTYPE=float64` is enforced via a single constant; no silent float32 fallbacks.

### Integrity check: no red flags
I specifically searched for:
- Cached/hardcoded outputs presented as computed: **none found**
- Post-hoc parameter tuning to match references: **none found** (the `EXF=0.7` is Cowan's published default, not a fitted value)
- Selective data filtering: **none found**
- Overwritten intermediate results without traceability: **none found**
- "Magic" tolerances that conveniently match the worst-case error: the SHELL test uses `< 1e-5` and the actual max error is `3e-6` — comfortable margin, not engineered

### One borderline finding
`hfs.py::quad2_coulomb` returns `2.0 * XI` (line 207) with the comment "factor 2 for Ry units". I cannot verify this factor against `rcn31.f::QUAD2` without re-deriving the Cowan units convention; it **may** be a tuned constant rather than a derived one. Worth a careful trace during the Phase B1 fix.

---

## 5. Critical red flags

**None.** The known failures and known limitations are openly documented in the test suite, the docstrings, and (now) `CODE_REVIEW.md`. The MULTIPOLE basis-invariant validation is a *physically rigorous fallback*, not an attempt to mask wrong physics — singular values and Frobenius norms are conjugation-invariant, so agreement to 1 e-6 there is genuine evidence that the matrix elements differ only by a unitary basis change.

---

## 6. Recommendations (ordered by priority)

### Completed (2026-04-09)
1. ~~Fix `hfs_scf` scheq Numerov integrator (§ 1.0)~~ — Rewrote with 2nd-order non-uniform FD scheme. H 1s = -1.0005 Ry, Ni²⁺ 3d/2p within tolerance.
2. ~~Fix `hfs_scf` Coulomb construction (§ 1.1)~~ — Per-orbital Y⁰ summation instead of sqrt-density.
3. ~~Implement spin-orbit ζ (§ 1.2)~~ — Central-field formula: ζ=(α²/2)∫(1/r)(dV/dr)P²dr. Ni²⁺ 2p: 0.852 Ry (ref 0.858).
4. ~~Fix uniform-mesh assumption in slater.py (§ 1.3, 2.1)~~ — All integrals use `_cum_trap`/`_trap`.
5. ~~Verify quad2_coulomb factor-of-2 (§ 2.2)~~ — Confirmed: standard Ry-units Coulomb conversion.
6. ~~Add j-range guard in wigner3j/6j (§ 2.4)~~ — `ValueError` if 2j > 40.
7. ~~Add NaN guard in broaden.py (§ 2.5)~~ — Zero-width Lorentzian returns zeros.

### Remaining
1. **Upgrade ζ from central-field to full Blume-Watson.** The current central-field formula agrees with the Fortran "R*VI" column to ~5%, but the production Fortran code uses the multi-orbital Blume-Watson method which includes exchange corrections. For Ni²⁺ 2p the difference is 0.852 vs 0.816 Ry (~4%).
2. ~~Trace the MULTIPOLE basis ordering~~ — FIXED: sort by (-S,-L), phase (-1)^{S+L+1}. Element-wise test passes.
3. **Add validation tests for the from-scratch pipeline end-to-end.** The Phase A3 parity sweep driver exists but the HFS layer still reports FAIL (it uses a hardcoded status); update it to run the actual SCF and compare.

---

## Closing remark

The code is **honest and well-tested**. All 141 tests pass. The HFS SCF pipeline (scheq Numerov, Coulomb, exchange, spin-orbit) is now functional and produces orbital energies and ζ values within a few percent of the Fortran reference. The pre-computed-parameters pipeline reproduces 13/13 triads to ≤ 1e-4 eV. The package is on a credible trajectory to a publication-quality, Fortran-free, differentiable multiplet calculator. The remaining gap (full Blume-Watson ζ, MULTIPOLE basis ordering) affects edge cases, not the primary use case of L-edge spectrum calculation from pre-computed atomic parameters.
