# multitorch

A PyTorch port of the Cowan/ttmult multiplet X-ray spectroscopy suite for
L-edge spectra of 3d transition metal complexes. Float64 throughout,
GPU-agnostic, and (once built from the assembled pipeline) differentiable
with respect to crystal-field and charge-transfer parameters.

Current status: 173 / 173 tests passing. Both the pre-computed-parameters
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

Expect 173 passing, 0 failing.

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
   with prior calls. On Ni²⁺ 2p⁶3d⁸: BW 2p ζ matches Fortran to ~1.7%; the
   3d BW *reduction ratio* matches to ~2.6%, but the absolute 3d ζ is still
   ~25% above Fortran because of HFS limitation §3 below (3d binds 8% too
   tightly → ⟨r⁻³⟩ too large). Fixing the absolute 3d ζ requires the
   O(h⁴) Numerov upgrade tracked in §3, not more BW work.
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

## Reports

- `CODE_REVIEW.md` — senior-code-reviewer pass over the whole package.
- `SCIENTIFIC_AUDIT.md` — scientific-code-auditor pass covering correctness,
  numerical stability, hidden approximations, and integrity.
- `tests/audit_results.md` — per-layer parity sweep against all fixtures
  (regenerate with `python tests/audit_parity_sweep.py`).

## Citation / license

TBD.
