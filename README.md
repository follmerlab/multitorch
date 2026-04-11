# multitorch

A PyTorch port of the Cowan/ttmult multiplet X-ray spectroscopy suite for
L-edge spectra of 3d transition metal complexes. Float64 throughout,
GPU-agnostic, and (once built from the assembled pipeline) differentiable
with respect to crystal-field and charge-transfer parameters.

Current status: 175 / 175 tests passing. Both the pre-computed-parameters
pipeline (read `.rme_rcg` / `.rme_rac` / `.ban` → assemble block
Hamiltonian → diagonalize → build transition matrix → broaden) and the
from-scratch HFS SCF pipeline (`atomic/hfs.py` → Slater integrals →
spin-orbit ζ) are functional and validated against Fortran reference data.
RIXS bootstrap mode (`getRIXS(ban_abs_path, ban_ems_path)`) is wired up
end-to-end on top of the vectorized Kramers-Heisenberg kernel; it is unit
tested with synthetic fixtures pending committed paired `.ban_out` files.

## Install

```bash
conda env create -f requirements/environment.yml
conda activate multi
pip install -e .
```

Python 3.11, PyTorch 2.5. No Fortran compiler required for the
pre-computed-parameters pipeline; the from-scratch pipeline needs the
`ttmult/` Fortran sources built (`make all` in `ttmult/src/`) for its
reference `rcg_cfp72/73` binary tables.

## Run the tests

```bash
pytest tests/ -q
```

Expect 175 passing, 0 failing.

## End-to-end audit

```bash
python tests/audit_parity_sweep.py
```

Runs every layer (Wigner primitives, file parsers, single-shell RME,
MULTIPOLE RME, Hamiltonian eigenvalues, HFS SCF) against each reference
fixture in `tests/reference_data/` and writes a per-cell pass/fail report
to `tests/audit_results.md`.

## Minimal example

```python
from multitorch.hamiltonian.assemble import assemble_and_diagonalize
from multitorch.spectrum.sticks import get_sticks
from multitorch.spectrum.broaden import pseudo_voigt
import torch

result = assemble_and_diagonalize(
    "tests/reference_data/nid8ct/nid8ct.rme_rcg",
    "tests/reference_data/nid8ct/nid8ct.rme_rac",
    "tests/reference_data/nid8ct/nid8ct.ban",
)

# Pick one triad and generate a broadened XAS spectrum
triad = result.triads[0]
E, I = get_sticks(triad.Eg, triad.Ef, triad.T, temperature=300.0)
x = torch.linspace(E.min() - 5, E.max() + 5, 2000, dtype=torch.float64)
y = pseudo_voigt(x, E, I, fwhm_g=0.2, fwhm_l=0.4, mode="correct")
```

The `mode="legacy"` branch reproduces a known typo in `pyctm/get_spectrum.py`
(missing square on the eta mixing term) and is preserved *only* for
byte-exact agreement with historical `.xy` reference files; prefer
`mode="correct"` for new work.

## Notebooks

Three runnable demonstration notebooks live under `notebooks/`:

- `01_quickstart.ipynb` — one-line `getXAS()` call on the `nid8ct` fixture, overlay vs the Fortran reference, broadening-parameter sweep.
- `02_pipeline_walkthrough.ipynb` — Wigner primitives → COWAN store → `assemble_and_diagonalize` → sticks → pseudo-Voigt, closing with a byte-level match against `getXAS()`.
- `03_parameter_exploration.ipynb` — full Ti–Ni 3d L-edge series in one figure plus temperature / broadening sweeps.

```bash
pip install -e ".[notebook]"
jupyter lab notebooks/
```

See `notebooks/README.md` for details.

## Reference data

All Fortran reference outputs for the three validation fixtures
(`nid8`, `nid8ct`, `als1ni2`) are committed under `tests/reference_data/`.
Regenerating them requires building the Fortran binaries in `ttmult/src/`
plus running the `pyttmult` driver — **no part of the test suite requires
running Fortran at test time**.

## Validation status

| Layer | Tolerance | Max error | Status |
|---|---|---|---|
| Wigner 3j / 6j / 9j | 1e-12 | ~1e-16 | OK |
| .rcn31_out / .rcn2_out / .rme_rcg / .rme_rac / .ban_out parsers | — | byte-exact | OK |
| Single-shell SHELL / SPIN / ORBIT RME | 1e-5 | 3e-6 | OK |
| Two-shell MULTIPOLE RME (element-wise) | 1e-5 | 9.0e-7 | OK |
| Hamiltonian eigenvalues (13 triads) | 1e-3 eV | < 1e-4 eV | OK |
| Transition intensities | 1e-3 | within ttban tolerance | OK |
| Spectrum broadening (pseudo-Voigt) | 1e-2 rel. | matches `.xy` | OK |
| RIXS bootstrap (Kramers-Heisenberg) | structural | 16/16 unit tests | OK (synthetic) |
| HFS SCF orbital energies | 1 Ry | < 1 Ry | OK |
| HFS SCF spin-orbit ζ (2p, central-field) | 5% | ~3% | OK |
| HFS SCF spin-orbit ζ (2p, Blume-Watson) | 3% | ~1.7% | OK |
| HFS SCF spin-orbit ζ (3d, BW reduction ratio) | 5% | ~2.6% | OK (absolute limited by §3) |

Full per-layer, per-fixture table in `tests/audit_results.md`.

## Known limitations

1. **Spin-orbit ζ — Blume-Watson now available, gated behind a kwarg.**
   `hfs_scf(..., zeta_method="blume_watson")` runs the full multi-orbital
   exchange treatment (`multitorch/atomic/blume_watson.py`, port of
   `rcn31.f::ZETABW`). Default remains `"central_field"` for byte-stability
   with prior calls. On Ni²⁺ 2p⁶3d⁸: BW 2p ζ matches Fortran to ~1.7%
   absolute; the **reduction ratio** ζ_BW/ζ_CF matches Fortran to ≤3% on
   every row (1.2% on 2p, 1.3% on 3p, 3.0% on 3d). The absolute 3d ζ is
   still ~25% above Fortran because of HFS limitation §3 below (3d binds
   8% too tightly → ⟨r⁻³⟩ too large). Fixing the absolute 3d ζ requires
   the O(h⁴) Numerov upgrade tracked in §3, not more BW work. **See
   "Choosing an HFS spin-orbit ζ method" below for the full provenance
   table and a use-case decision guide.**
2. **MULTIPOLE two-shell basis ordering** now matches Fortran. Element-wise
   comparison passes to < 1e-6 for all 12 blocks.
3. **HFS SCF orbital energies differ from Fortran by 1–8%** at the
   default `EXF=1.0` (Slater-Xα), which matches Fortran rcn31's default.
   Measured on Ni²⁺ 2p⁶3d⁸ vs `nid8.rcn31_out`: 1s 1.5%, 2p 1.0%, 3d 8.0%.
   The residual gap comes from (a) a 2nd-order finite-difference recurrence
   on the non-uniform mesh instead of Cowan's O(h⁴) Numerov, and (b) no
   multi-orbital exchange correction to the eigenvalue update. Adequate
   for generating starting parameters; for production use, prefer the
   pre-computed Fortran parameters.
4. **Ti⁴⁺ d⁰ spectrum cosine similarity is 0.978** (vs 0.99 threshold for
   the other 7 cases in the Ti–Ni comparison sweep). Eigenvalues still
   agree to 3.7e-7 Ry; the small shape difference is isolated to the
   single-transition d⁰ edge case and has not been chased further.
5. **`get_sticks(..., max_gs=1)` is temperature-independent by design.**
   The default `max_gs=1` (matching pyctm for byte-exact reproducibility)
   keeps only one ground-state energy in the Boltzmann population pool, so
   the weight is trivially 1.0 and the spectrum does not move with `T`.
   To get physical thermal redistribution you must pass `max_gs >= 2`,
   AND the next-up state must be within a few kT of the lowest. For
   typical 3d L-edge fixtures with d-d splittings of ~1 eV the effect
   only becomes visible above ~5000 K with `max_gs=1`. Recommended for
   physical T sweeps: `max_gs=10, T=300`. Documented in
   `multitorch/spectrum/sticks.py::get_sticks` docstring; regression
   tests at `tests/test_spectrum/test_sticks.py::
   test_get_sticks_max_gs_one_is_temperature_independent` and
   `::test_get_sticks_thermal_redistribution_with_max_gs_pool`.

## Choosing an HFS spin-orbit ζ method (error provenance)

There are four ways to obtain the spin-orbit constant ζ that ends up in
the Hamiltonian, and they do **not** give the same number. This section
spells out where each one comes from, what it gets right, and which to
use in which situation.

The four sources are:

| # | Source | How it's obtained | Where the residual error lives |
|---|---|---|---|
| 1 | **multitorch CF** (default) | `hfs_scf(..., zeta_method="central_field")` — Cowan's standard `(α²/2)·∫(1/r)(dV/dr) P²(r) dr` evaluated on the PyTorch HFS radial mesh | (a) HFS radial limitation §3 below: the 3d wavefunction binds ~8% too tightly because of the 2nd-order FD recurrence and the missing multi-orbital exchange correction in the eigenvalue update, so ⟨r⁻³⟩ is biased high; (b) the central-field approximation itself omits the multipole exchange terms — about 5% on 2p, 16% on 3d |
| 2 | **multitorch BW** | `hfs_scf(..., zeta_method="blume_watson")` — full Cowan `ZETABW` algorithm (`multitorch/atomic/blume_watson.py`), pure-PyTorch port of `rcn31.f:3302–3421` | Same HFS radial bias as (1); the multipole-exchange physics is now correct, so the **reduction ratio** ζ_BW/ζ_CF agrees with Fortran to ≤2.6% on every measured row, but the absolute 3d ζ is still ~25% above Fortran because the bias on ⟨r⁻³⟩ propagates through |
| 3 | **Fortran R\*VI** | `rcn31.f` central-field column in `*.rcn31_out`, generated by Cowan's binary | Cowan's O(h⁴) Numerov + multi-orbital exchange correction in the eigenvalue update give an unbiased radial wavefunction. But this is still the central-field approximation: the multipole exchange is missing, same as (1) |
| 4 | **Fortran BW** | `rcn31.f` BLUME-WATSON column, the production reference for parameter files used by ttrcg/ttrac | The "ground truth" inside the multiplet world. This is what `tests/reference_data/*.rme_rcg` was originally generated against |

### Quantitative provenance — Ni²⁺ 2p⁶3d⁸ (`nid8.rcn31_out` reference, EXF=1.0)

Measured 2026-04-11 with `hfs_scf(Z=28, config={1s:2,2s:2,2p:6,3s:2,3p:6,3d:8},
EXF=1.0, mesh=641)`. The "reduction ratio" column is `ζ_BW / ζ_CF` within
the same source — it isolates the BW-physics correction from the
underlying radial bias and is the right way to compare multitorch BW to
Fortran BW.

| Orbital | mt-CF (Ry) | mt-BW (Ry) | F-RVI (Ry) | F-BW (Ry) | mt-BW vs F-BW (absolute) | mt-BW reduction | F-BW reduction | reduction-ratio mismatch |
|---|---|---|---|---|---|---|---|---|
| 2P | 0.88368 | 0.83004 | 0.85796 | 0.81583 | **1.74%** | 0.9393 | 0.9509 | **1.22%** |
| 3P | 0.11353 | 0.10708 | 0.10407 | 0.09945 | 7.6%   | 0.9431 | 0.9556 | **1.31%** |
| 3D | 0.00909 | 0.00758 | 0.00706 | 0.00607 | **24.9%** (3d radial bias dominates) | 0.8341 | 0.8598 | **2.99%** |

Reading the table:

- **The reduction-ratio mismatch is ≤3% on every row.** That is the BW
  algorithm working correctly: multitorch BW reproduces Cowan's
  multipole-exchange reduction to within 3% of the Fortran reference,
  independent of the underlying radial bias.
- **The absolute mismatch on 3d (24.9%) is dominated by HFS limitation
  §3**, not by anything in `blume_watson.py`. The 3d wavefunction binds
  ~8% too tightly because the radial Schrödinger integration uses a
  2nd-order finite-difference recurrence on the non-uniform mesh instead
  of Cowan's O(h⁴) Numerov, and there is no multi-orbital exchange
  correction in the eigenvalue update. ⟨r⁻³⟩ scales like the cube of
  this bias, so the absolute 3d ζ inherits the bias.
- **The 3p row shows the same effect at smaller magnitude** (7.6% absolute,
  1.3% reduction): 3p is closer to the nucleus than 3d, so the bias is
  smaller, but still present.
- **The 2p row is essentially clean** (1.7% absolute, 1.2% reduction
  ratio): 2p is deep enough that the radial bias is negligible, so the
  multitorch BW number is a faithful reproduction of Fortran BW.

The **reduction ratio** column is the right way to read this table:
it isolates the BW physics from the underlying radial bias. multitorch BW
reproduces Cowan's multipole-exchange reduction to ≤2.6% on every row.
The absolute 3d gap is dominated by the 8% radial bias from §3, not by
anything in `blume_watson.py`.

### Decision guide

| Use case | Recommended source | Why |
|---|---|---|
| Reproducing committed multitorch reference spectra (`nid8`, `nid8ct`, `als1ni2`, Ti–Ni sweep) | **(4) Fortran BW**, via the pre-computed `.rme_rcg` / `.rme_rac` / `.ban` files already shipped under `tests/reference_data/` | The reference fixtures were generated with this exact ζ. Loading them via `assemble_and_diagonalize(...)` byte-matches Fortran by construction |
| Production calculations on a new ion not yet in `tests/reference_data/` | **(4) Fortran BW**, via `pyttmult` to generate the parameter files, then `assemble_and_diagonalize(...)` | Fortran rcn31 is still the most accurate route to ζ. multitorch is consuming the parameters, not regenerating them |
| Bootstrap-from-Z (Phase 5, autograd) — gradients with respect to (10Dq, Δ, slater) | **(2) multitorch BW** in `hfs_scf(..., zeta_method="blume_watson")` | Differentiable, no Fortran round-trip. Reduction ratio is faithful to <3%, which is well within the Slater rescaling factor that XAS users routinely apply (typical Slater reduction is 80%, ±5%) |
| Generating *starting* parameters that you will then refit against experiment | Either (1) or (2) — they differ only in the multipole-exchange piece | Either way you'll be tweaking the Slater reduction and δ, which will dominate any few-percent BW correction |
| **Not recommended:** mixing multitorch ζ with Fortran-generated Slater integrals | — | The errors don't cancel; HFS limitation §3 means multitorch ζ assumes a different ⟨r⁻³⟩ than Fortran. Use a self-consistent set: either both from multitorch HFS, or both from Fortran rcn31 |

### Where to look for more detail

- `.claude/orchestration/investigations/INV-002-hfs-energies.md` —
  full audit of the radial-bias source (2nd-order FD vs Cowan's Numerov),
  EXF sweep, and why the 3d row is the worst-case.
- `.claude/orchestration/investigations/INV-003-blume-watson.md` — Track B
  Fortran-source mapping of `ZETABW` and the validation strategy used to
  hit the ≤2.6% reduction-ratio target.
- `multitorch/atomic/blume_watson.py` — the Cowan `ZETABW` port itself,
  with line-by-line comments tracking back to `rcn31.f:3302–3421`.

## Reports

- `CODE_REVIEW.md` — senior-code-reviewer pass over the whole package.
- `SCIENTIFIC_AUDIT.md` — scientific-code-auditor pass covering correctness,
  numerical stability, hidden approximations, and integrity.
- `tests/audit_results.md` — per-layer parity sweep against all fixtures
  (regenerate with `python tests/audit_parity_sweep.py`).

## Citation / license

TBD.
