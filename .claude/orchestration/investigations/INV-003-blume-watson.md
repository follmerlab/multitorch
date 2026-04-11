# INV-003 — Blume-Watson spin-orbit ζ in HFS SCF

## Why

`README.md` § Known limitations §1 currently warns that
`multitorch/atomic/hfs.py:691-726` uses the **central-field** spin-orbit
formula (`ζ_nl = (α²/2) ∫ (1/r)(dV/dr) P²(r) dr`) and matches the Fortran
**R*VI** column to ~5% on Ni²⁺. The production Cowan code uses the full
**Blume-Watson** treatment (Proc. Roy. Soc. London A270, 127 (1962); A271,
565), which adds multi-orbital exchange corrections — the BW vs R*VI gap is
~5% on the 2p orbital and ~16% on the 3d orbital of `nid8`. This
investigation maps the Fortran reference and prepares the PyTorch port.

Tracks: [`INV-002-hfs-energies.md`](INV-002-hfs-energies.md) noted BW as
the residual gap.

## Fortran reference

`../ttmult/src/rcn31.f`, `FUNCTION ZETABW(M, RM3)` — lines 3302–3421
(~120 lines). Helpers `SM`, `SN`, `VK`, `ZK`, `DYK` follow at lines
3424–3624.

### Inputs from COMMON blocks

| Symbol | Source | Meaning |
|---|---|---|
| `M` | argument | index of the orbital whose ζ we want |
| `RM3` | argument | precomputed `⟨r⁻³⟩` for orbital `M` |
| `Z` | `/C2/` | nuclear charge |
| `L(MP)` | `COMMON` | orbital angular momentum of orbital `MP` |
| `WWNL(MP)` | `COMMON` | occupation of orbital `MP` |
| `EE(MP)` | `COMMON` | orbital energy (skipped if > 0, i.e. continuum) |
| `NCSPVS` | `/C2/` | total number of orbitals (cores+valence) |
| `PNL(:,MP)` | `COMMON PNL(1801,20)` | bound radial wavefunction |
| `R(:)` | `COMMON` | radial mesh |

### Step-by-step structure

#### Step 1 — Bare central-field seed (line 3326)

```
SP = Z * RM3 * 5.325135E-5
```

`5.325135E-5 ≈ α²` (fine-structure constant squared, in atomic units;
gives ζ in **Ry**).

#### Step 2 — Loop over other occupied orbitals MP ≠ M (lines 3330–3391)

For each occupied orbital `MP` (`EE(MP) ≤ 0`, `MP ≠ M`):

  - `WP = WWNL(MP)`, `LP = L(MP)`, `FLM = L(M)`, `FLP = LP`.
  - **Direct piece (line 3337):**
    ```
    SP -= 2.0 * WP * SM(M, MP)
    ```
  - **Build the per-K Slater integrals** (lines 3339–3353):

    K-loop bounds: `KN = |L(M) - LP| - 2`, `KX = L(M) + LP`.
    For each K, alternate parity with `INV`:
    - if `INV < 0` (`L+LP+K` parity): `SNKK[K+3] = SN(M, MP)`
    - if `INV > 0`: `VKK[K+3] = VK(M, MP)`

    Note that `SN` and `VK` use the K from `COMMON/C4` set inside `ZK` /
    `DYK`, not the local K — this is part of the Fortran-only call
    convention via `ZK(M1, M2, M3, M4)`.

  - **SUM2: even-parity (SN) contribution** (lines 3357–3372).
    Bounds: K runs from `KN+2` to `KX-2`. For each K:
    ```
    ERAS  = S6J(K, 1, K+1; L_M, LP, L_M)²
    ERAS *= S3J0SQ(L_M, K, LP)            ← (3j 000)²
    ERAS *= (2K+1)(2K+3)/(K+2)
    SUM2 += ERAS * SNKK[K+3]
    ```
    Then `SP -= E2 * WP * SUM2`, where `E2 = -12·L_M − 6`.

  - **SUM13: odd-parity (VK + cross-SN) contribution** (lines 3374–3389).
    Bounds: K from `max(KN+2, 2)` to `KX`. For each K:
    ```
    ERAS  = √(K(K+1)(2K+1))
    ERAS *= S3J0SQ(L_M, K, LP)
    ERAS *= S6J(K, 1, K; L_M, LP, L_M)
    ERAS1 = E13 × (SNKK[K+1]/K  −  SNKK[K+3]/(K+1))     ← cross-SN coupling
    SUM13 += ERAS × (VKK[K+2] + ERAS1)
    ```
    where `E13 = LP(LP+1) − L_M(L_M+1)`. Then `SP -= E3 * WP * SUM13`,
    `E3 = 3 √((2L_M+1) / (L_M(L_M+1)))`.

#### Step 3 — Self-interaction correction for w_M ≥ 2 (lines 3393–3417)

Only triggered when the orbital itself has ≥ 2 electrons:

  - `SM0 = SM(M, M)`
  - `SP -= (2W − 3) × SM0`
  - If `L_M ≥ 2`, additional sum over even K from 2 to `2L_M − 2`:
    ```
    For each K, with FK=K, FKP1=K+1:
      SUM = 0
      For KP in {K, K+2}:
        ERAS = S6J(K+1, 1, KP; L_M, L_M, L_M)²  × S3J0SQ(L_M, KP, L_M)
        SUM += (FKP1 − KP) × (2 KP + 1) × ERAS
      SUM1 = SUM × (2K + 3) × SM(M, M)
      SP  += E × SUM1                  with  E = 6 (2L_M+1)²
    ```

### Helper integrals (lines 3424–3624)

All three helpers carry a `2.6625666E-5 ≈ α²/2` prefactor and produce
results in Rydbergs. They build on the radial mesh `R(:)` and bound
wavefunctions `PNL(:,M)`.

| Helper | What it computes | Fortran routine |
|---|---|---|
| `SM(M, MP)` | Direct radial Slater integral with the `1/r^(K+3)` form factor used by the BW direct term | `ZK(M, MP, M, MP)` then `QUAD5` × α²/2 |
| `SN(M, MP)` | Same routine, but with M and MP swapped in the M3,M4 slots — exchange-density variant | `ZK(M, MP, MP, M)` then `QUAD5` × α²/2 |
| `VK(M, MP)` | Difference of two derivative-Yk integrals, `DYK(M,MP) − DYK(MP,M)`, capturing the Blume-Watson exchange-derivative term | `(DYK(M,MP) − DYK(MP,M)) × α²/2` |
| `ZK(M1,M2,M3,M4)` | Builds `XJ[i] = P_M1(r_i) P_M3(r_i) Y^k(r_i) / r^(K+3)` then prepares for `QUAD5` | rcn31.f:3478–3526 |
| `DYK(M, MP)` | Builds `XJ[i] = P_M(r_i) × (THM1·(P_MP[i+1]−P_MP[i−1]) − P_MP[i]/r[i]) ⊗ Y^(k+1)`-style | rcn31.f:3558–3624 |

`K` (the multipole rank consumed by ZK/DYK) is **not** the local K from
`ZETABW`'s loop — it is set inside `QUADK`/`DYK` from `COMMON/C4`. This
matches Cowan's stateful Fortran convention and means a direct
PyTorch port has to thread `K` through the call interface explicitly.

### Mapping the angular factors to existing PyTorch helpers

| Fortran | PyTorch |
|---|---|
| `S6J(j1, j2, j3, j4, j5, j6)` | `multitorch.angular.wigner.wigner6j(j1, j2, j3, j4, j5, j6)` |
| `S3J0SQ(j1, j2, j3)` | `wigner3j(j1, j2, j3, 0, 0, 0) ** 2` |
| `Z * RM3 * 5.325135E-5` | seed in `multitorch/atomic/hfs.py:691–726` (already correct in Ry) |

### Mapping the radial integrals to existing PyTorch helpers

| Fortran | What it does | Existing PyTorch building block |
|---|---|---|
| `QUADK` (forward Yk Poisson) | Fortran's two-pass Y^k accumulator | `multitorch.atomic.slater.compute_yk` (already mesh-adaptive) |
| `QUAD5` | Simpson-like quadrature on log mesh | `torch.trapezoid` already used by `slater._trap` |
| `ZK` (4-orbital form factor) | Builds `P_M1·P_M3·Yk(P_M2,P_M4)/r^(k+3)` | **NEW**: needs a `compute_yk_general(Pa,Pb)` (use `compute_yk_cross`) followed by a `1/r^(k+3)` weighted integral |
| `DYK` (derivative form) | Builds the derivative variant `P_M·(dP_MP/dr − P_MP/r)`-style integrand | **NEW**: derivative not yet exposed; can be done with finite differences on the existing log mesh OR via the Y^(k+1) recurrence |

The `compute_yk` and `compute_yk_cross` routines in
`multitorch/atomic/slater.py` already implement the two-pass cumulative
trapezoidal Y^k accumulator on a non-uniform mesh, including for cross
densities. This is the heart of `ZK`. What's missing for B2 is:

1. The `1/r^(K+3)` weighting (i.e., the SO-like radial moment that turns
   a bare F^k into the BW direct integral `SM`). This is a one-line
   wrapper on `compute_yk_cross` plus a `_trap` of the result divided by
   `r^(K+3)`.
2. The derivative integrand for `VK`. The cleanest path is a centered
   finite difference of `P_MP` on the log mesh (we already use the same
   for `dV/dr` in the central-field ζ at `hfs.py:691–726`).
3. Wiring it all into a `zeta_blume_watson(orbitals, mesh, Z, m_idx)`
   driver that mirrors `ZETABW` line-for-line.

## Reference data for validation

`tests/reference_data/nid8/nid8.rcn31_out:31–39` ships both columns
side-by-side. Spot values for **Ni²⁺ 2p⁶3d⁸** at `EXF=1.0` (Slater-Xα):

| Orbital | R*VI (Ry) | BLUME-WATSON (Ry) | Δ |
|---|---|---|---|
| 2P | 0.85796 | 0.81583 | 5.1% |
| 3D | 0.00706 | 0.00607 | 16.3% |

The PyTorch central-field implementation already matches the **R*VI**
column to ≤ 6% (existing test in `tests/test_atomic/test_hfs.py`). The
Track B test target is matching the **BLUME-WATSON** column to ≤ 1% on
the 3d row.

## Next steps (B2 → B5)

1. **B2** — implement `multitorch/atomic/blume_watson.py` with three
   building blocks (`compute_sm`, `compute_sn`, `compute_vk`) on top of
   `compute_yk_cross`, plus `zeta_blume_watson(...)` that mirrors
   `ZETABW` line-for-line.
2. **B3** — wire into `hfs_scf` behind a `zeta_method` kwarg
   (default unchanged).
3. **B4** — extend `tests/test_atomic/test_hfs.py` with a BW-vs-Fortran
   test on Ni²⁺ 2p⁶3d⁸.
4. **B5** — verify the 166-test baseline still passes, remove README
   §1, append a closing entry to `INV-002-hfs-energies.md`, tag
   `track-b-done`.

## Open risks

- **Derivative term in VK.** Cowan's `DYK` is a hand-tuned 4th-order
  finite difference around `R(I)`. A naive 2nd-order centered difference
  on the log mesh may introduce larger errors than the 1% target. If the
  validation in B4 misses, fall back to a 4th-order scheme matching `DYK`
  literally.
- **Stateful K.** `ZK` and `DYK` read `K` from `COMMON/C4`. The PyTorch
  versions must take `K` as an explicit argument, which means the
  `ZETABW` port can't be a one-to-one line transcription — the per-K
  inner loop has to compute the four Slater quantities up front and
  store them indexed by K, exactly as the Fortran does in the
  `SNKK[K+3]` / `VKK[K+3]` arrays.
- **Self-interaction branch (Step 3) is only exercised when `L_M ≥ 2`
  AND `w_M ≥ 2`.** For Ni²⁺ 2p⁶3d⁸, both the 3d row (`L=2`, `w=8`) and
  the 2p row (`L=1`, `w=6`) hit different branches: 2p hits the
  `(2W−3)·SM0` term but skips the K-loop (`L<2`), while 3d hits both.
  Tests need to cover both code paths.
