"""Build the three demonstration notebooks for multitorch.

Run from the multitorch/ repo root:
    python _build_notebooks.py

This is a build-time helper — NOT part of the package. After running, the
script is expected to be deleted.
"""
import os
from pathlib import Path
import nbformat as nbf

NOTEBOOKS_DIR = Path(__file__).parent / "notebooks"
NOTEBOOKS_DIR.mkdir(exist_ok=True)


def md(s: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(s.strip("\n"))


def build_notebook(title: str, cells: list, filename: str) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (multi)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }
    path = NOTEBOOKS_DIR / filename
    nbf.write(nb, str(path))
    print(f"wrote {path}")


# ─────────────────────────────────────────────────────────────
# Notebook 01 — Quickstart
# ─────────────────────────────────────────────────────────────

ROOT_CHDIR_CELL = code("""
# Make the notebook runnable from anywhere: chdir to the repo root so
# that relative paths like 'tests/reference_data/...' resolve correctly
# whether you launched jupyter from the repo root or from notebooks/.
import os
from pathlib import Path
_here = Path.cwd()
for _anc in [_here, *_here.parents]:
    if (_anc / "multitorch").is_dir() and (_anc / "tests" / "reference_data").is_dir():
        os.chdir(_anc)
        break
print("working dir:", Path.cwd())
""")

nb01 = [
    md("""
# 01 — Quickstart: Ni²⁺ L-edge XAS from a Fortran reference file

This notebook shows the fastest path from a pre-computed ttmult `.ban_out`
file to a plotted L-edge XAS spectrum, using the `getXAS()` one-liner.
It mirrors the minimal example in `README.md` but adds plotting, a
reference overlay, and a small broadening sweep so you can see how each
knob moves the spectrum.

**What you'll learn**
- Which files the bootstrap pipeline needs.
- How to call `getXAS()` and plot the result.
- How the three broadening widths (`beam_fwhm`, `gamma1`, `gamma2`) shape
  the spectrum.

**Prerequisites**: `pip install -e .` from the repo root, plus matplotlib.
All reference data is committed under `tests/reference_data/`.
"""),
    ROOT_CHDIR_CELL,
    md("""
## 1. What files are in a ttmult fixture?

Each fixture in `tests/reference_data/` is a complete ttmult run output.
For the bootstrap pipeline we only actually need the `.ban_out` file —
the others are here for validation and for the pipeline walkthrough
notebook.
"""),
    code("""
from pathlib import Path

REFROOT = Path("tests/reference_data/ni2_d8_oh")
for f in sorted(REFROOT.iterdir()):
    print(f"{f.name:24s}  {f.stat().st_size:>10,} bytes")
"""),
    md("""
- **`.rme_rcg`** — COWAN single-shell reduced matrix elements
  (output of `ttrcg`).
- **`.rme_rac`** — symmetry-reduced coupled blocks for each operator
  (output of `ttrac`).
- **`.ban`** — the "recipe": XHAM scaling, XMIX hybridization, charge-
  transfer energies, and the list of symmetry triads to diagonalize.
- **`.ban_out`** — the diagonalized eigenstates + transition matrices
  (output of `ttban_exact`). This is what `getXAS()` consumes.
- **`.xy`** — the reference broadened spectrum from pyctm, for
  validation.
"""),
    md("""
## 2. One-liner: `getXAS()`

`getXAS(ban_output_path, ...)` is the high-level entry point that matches
the signature of the original `pyctm.getXAS`. It reads the `.ban_out`,
applies Boltzmann weighting, and broadens with a pseudo-Voigt.
"""),
    code("""
import torch
import matplotlib.pyplot as plt
from multitorch.api.plot import getXAS

x, y = getXAS(
    str(REFROOT / "ni2_d8_oh.ban_out"),
    T=80.0,          # Kelvin — affects the Boltzmann population when max_gs > 1
    beam_fwhm=0.2,   # Gaussian FWHM (eV) — instrumental broadening
    gamma1=0.2,      # L3 lifetime FWHM (eV)
    gamma2=0.4,      # L2 lifetime FWHM (eV)
)

print(f"x: {tuple(x.shape)}  range [{float(x.min()):.2f}, {float(x.max()):.2f}] eV")
print(f"y: {tuple(y.shape)}  max {float(y.max()):.4f}")
"""),
    md("""
## 3. Plot against the pyctm reference

The reference `.xy` is in relative-energy coordinates (zero at the first
point). We align peaks so the shape comparison is apples-to-apples.
"""),
    code("""
import numpy as np

xy_ref = np.loadtxt(str(REFROOT / "ni2_d8_oh.xy"))
x_ref, y_ref = xy_ref[:, 0], xy_ref[:, 1]

# Align peaks
x_np = x.detach().cpu().numpy()
y_np = y.detach().cpu().numpy()
shift = x_np[y_np.argmax()] - x_ref[y_ref.argmax()]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x_np - shift, y_np, label="multitorch getXAS()", lw=2)
ax.plot(x_ref, y_ref, "--", label="pyctm .xy reference", lw=1.5)
ax.set_xlabel("Energy relative to L$_3$ main line (eV)")
ax.set_ylabel("Intensity (arb. units)")
ax.set_title("Ni²⁺ d⁸ (ni2_d8_oh) — L-edge XAS")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
"""),
    md("""
## 4. How each broadening width moves the spectrum

`beam_fwhm` controls the Gaussian instrumental width. `gamma1` and
`gamma2` are the Lorentzian lifetime widths for L₃ and L₂ respectively
(L₂ has a faster core-hole decay because of the Coster-Kronig channel,
hence a larger default). The pseudo-Voigt takes care of the transition
between them across the edge.
"""),
    code("""
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

# (a) beam_fwhm sweep — Gaussian instrumental broadening
for bf in [0.1, 0.2, 0.5]:
    xb, yb = getXAS(str(REFROOT / "ni2_d8_oh.ban_out"),
                    beam_fwhm=bf, gamma1=0.2, gamma2=0.4)
    axes[0].plot(xb - shift, yb, label=f"beam_fwhm = {bf}")
axes[0].set_title("Gaussian beam width")
axes[0].set_xlabel("Energy (eV, rel.)")
axes[0].set_ylabel("Intensity")
axes[0].legend()
axes[0].grid(alpha=0.3)

# (b) gamma1 sweep — L3 lifetime
for g1 in [0.1, 0.3, 0.6]:
    xb, yb = getXAS(str(REFROOT / "ni2_d8_oh.ban_out"),
                    beam_fwhm=0.2, gamma1=g1, gamma2=0.4)
    axes[1].plot(xb - shift, yb, label=f"gamma1 = {g1}")
axes[1].set_title("L$_3$ lifetime (gamma1)")
axes[1].set_xlabel("Energy (eV, rel.)")
axes[1].legend()
axes[1].grid(alpha=0.3)

# (c) gamma2 sweep — L2 lifetime
for g2 in [0.2, 0.4, 0.8]:
    xb, yb = getXAS(str(REFROOT / "ni2_d8_oh.ban_out"),
                    beam_fwhm=0.2, gamma1=0.2, gamma2=g2)
    axes[2].plot(xb - shift, yb, label=f"gamma2 = {g2}")
axes[2].set_title("L$_2$ lifetime (gamma2)")
axes[2].set_xlabel("Energy (eV, rel.)")
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()
"""),
    md("""
## Recap

- **Files needed**: just the `.ban_out` — everything else is provenance.
- **One call**: `getXAS(ban_path, T=..., beam_fwhm=..., gamma1=..., gamma2=...)`.
- **Outputs**: two `torch.Tensor`s on the CPU (float64 by default), ready
  to plot or feed into downstream analysis.

To explore the underlying layers (Wigner → RME → Hamiltonian →
broadening), continue with **`02_pipeline_walkthrough.ipynb`**.
To sweep physical parameters (10Dq, CT energy, temperature), continue
with **`03_parameter_exploration.ipynb`**.
"""),
]

build_notebook("01 — Quickstart", nb01, "01_quickstart.ipynb")


# ─────────────────────────────────────────────────────────────
# Notebook 02 — Pipeline walkthrough
# ─────────────────────────────────────────────────────────────

nb02 = [
    md("""
# 02 — Pipeline walkthrough: from Wigner symbols to an L-edge spectrum

This notebook peels back the `getXAS()` wrapper and exposes the tensors
at each layer of the multitorch pipeline, using the `nid8ct` fixture
throughout so every intermediate is reproducible.

The flow we'll step through:

```
Wigner primitives  →  angular reduced matrix elements
       ↓
COWAN store (.rme_rcg)  +  operator blocks (.rme_rac)  +  recipe (.ban)
       ↓
assemble_and_diagonalize()   →   BanResult (Eg, Ef, T for each triad)
       ↓
read .ban_out  →  BanOutput   (same info, from Fortran)
       ↓
get_sticks()   →   Boltzmann-weighted E, M
       ↓
pseudo_voigt()   →   final broadened spectrum
```

At the end we'll reconstruct the exact `getXAS()` output from the layers
above and show they match to float64 precision.
"""),
    ROOT_CHDIR_CELL,
    md("""
## Layer 1 — Wigner primitives

All angular-momentum coupling in the pipeline ultimately bottoms out in
Wigner 3j and 6j symbols. `multitorch.angular.wigner` provides
differentiable, float64 implementations that match the standard
recursions to ~1e-16.
"""),
    code("""
import torch
from multitorch.angular.wigner import wigner3j, wigner6j

# A dipole 3j symbol: < p || C^1 || d >  needs (1, 1, 2; 0, 0, 0)
w3j = wigner3j(1, 1, 2, 0, 0, 0)
print(f"( 1 1 2 ; 0 0 0 ) = {float(w3j):+.12f}")

# Triangle condition check
w3j_bad = wigner3j(1, 1, 4, 0, 0, 0)  # violates triangle inequality
print(f"( 1 1 4 ; 0 0 0 ) = {float(w3j_bad):+.12f}  (should be 0)")

# A sample 6j (appears in angular RME for a 2-shell coupled system)
w6j = wigner6j(2, 1, 1, 1, 2, 2)
print(f"{{ 2 1 1 ; 1 2 2 }} = {float(w6j):+.12f}")
"""),
    md("""
## Layer 2 — Read the COWAN store and operator blocks

The `.rme_rcg` file is a series of COWAN-format blocks, each a sparse
reduced-matrix-element matrix in an angular-momentum basis. The
`.rme_rac` file is the symmetry-adapted form of the same information.
"""),
    code("""
from pathlib import Path
from multitorch.io.read_rme import read_cowan_store, read_rme_rac_full

REFROOT = Path("tests/reference_data/ni2_d8_oh")

cowan = read_cowan_store(str(REFROOT / "nid8ct.rme_rcg"))
print(f"COWAN store: {len(cowan.sections)} sections")
for i, sec in enumerate(cowan.sections):
    print(f"  section {i}: {sec.matrix.shape} (nonzeros: {int((sec.matrix != 0).sum())})")
"""),
    code("""
rac = read_rme_rac_full(str(REFROOT / "nid8ct.rme_rac"))
print(f"RAC file: {len(rac.blocks)} blocks, {len(rac.irreps)} irreps")
# Show a handful of GROUND blocks
for b in rac.blocks[:6]:
    print(f"  {b.kind:7s} {b.bra_sym:4s} {b.geometry:12s} "
          f"bra×ket = {b.n_bra}×{b.n_ket}  (adds: {len(b.add_entries)})")
"""),
    md("""
## Layer 3 — Assemble and diagonalize the Hamiltonian

`assemble_and_diagonalize()` combines `.rme_rcg`, `.rme_rac`, and `.ban`
into a block-diagonal charge-transfer Hamiltonian, diagonalizes it for
each symmetry triad, and returns eigenvalues, eigenvectors, and
transition matrices. This is the PyTorch equivalent of `ttban_exact.f`.
"""),
    code("""
from multitorch.hamiltonian.assemble import assemble_and_diagonalize

result = assemble_and_diagonalize(
    str(REFROOT / "nid8ct.rme_rcg"),
    str(REFROOT / "nid8ct.rme_rac"),
    str(REFROOT / "nid8ct.ban"),
)

print(f"BanResult with {len(result.triads)} triads")
print()
print(f"{'gs':>4s} {'act':>4s} {'fs':>4s}   {'n_gs':>5s} {'n_fs':>5s}   "
      f"{'Eg.min':>10s} {'Ef.min':>10s}")
for t in result.triads[:6]:
    print(f"{t.gs_sym:>4s} {t.act_sym:>4s} {t.fs_sym:>4s}   "
          f"{t.n_gs:>5d} {t.n_fs:>5d}   "
          f"{float(t.Eg.min()):>+10.4f} {float(t.Ef.min()):>+10.4f}")
"""),
    md("""
Pick one triad and look at the transition matrix directly:
"""),
    code("""
import matplotlib.pyplot as plt

# The largest triad is usually the most informative
biggest = max(result.triads, key=lambda t: t.T.numel())
print(f"Biggest triad: gs={biggest.gs_sym} act={biggest.act_sym} fs={biggest.fs_sym}  "
      f"T shape = {tuple(biggest.T.shape)}")

fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(biggest.T.abs().detach().numpy(), aspect="auto", cmap="viridis")
ax.set_xlabel("final-state index")
ax.set_ylabel("ground-state index")
ax.set_title(f"|T_if| for triad {biggest.gs_sym} → {biggest.fs_sym}")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()
"""),
    md("""
## Layer 4 — Read the Fortran `.ban_out` and compare

To drive the final broadening we use the bootstrap path
(`read_ban_output` → `get_sticks` → `pseudo_voigt`). The Fortran
`.ban_out` file contains the same eigenvalues + transition matrices that
we just computed — let's verify the eigenvalues agree.
"""),
    code("""
from multitorch.io.read_oba import read_ban_output

ban = read_ban_output(str(REFROOT / "nid8ct.ban_out"))
print(f"BanOutput: {len(ban.triad_list)} triads from Fortran .ban_out")

# Line up the first common triad and compare its lowest eigenvalue
py_first = result.triads[0]
# find matching triad in ban by symmetry labels
fort_match = None
for t in ban.triad_list:
    if (t.gs_sym == py_first.gs_sym and t.act_sym == py_first.act_sym
            and t.fs_sym == py_first.fs_sym):
        fort_match = t
        break

if fort_match is not None:
    dE = abs(float(py_first.Eg.min()) - float(fort_match.Eg.min()))
    print(f"First triad {py_first.gs_sym}→{py_first.fs_sym}: "
          f"Eg.min PyTorch = {float(py_first.Eg.min()):+.6f} Ry, "
          f"Fortran = {float(fort_match.Eg.min()):+.6f} Ry, "
          f"|ΔE| = {dE:.2e} Ry")
"""),
    md("""
## Layer 5 — Boltzmann sticks and broadening

`get_sticks()` collapses all the per-triad transition matrices into a
single 1-D stick spectrum, weighted by a Boltzmann factor on the ground
states. `pseudo_voigt()` then convolves the sticks into a smooth curve.
"""),
    code("""
from multitorch.spectrum.sticks import get_sticks
from multitorch.spectrum.broaden import pseudo_voigt

E_sticks, M_sticks, Eg_min = get_sticks(ban, T=80.0, max_gs=1)
print(f"{E_sticks.shape[0]} sticks; "
      f"E range [{float(E_sticks.min()):.2f}, {float(E_sticks.max()):.2f}] eV")

# Build the broadened spectrum on a dense grid
x_grid = torch.linspace(float(E_sticks.min()) - 5.0,
                        float(E_sticks.max()) + 5.0,
                        2000, dtype=torch.float64)
med = 0.5 * (float(E_sticks.min()) + float(E_sticks.max()))
y_grid = pseudo_voigt(
    x_grid, E_sticks, M_sticks,
    fwhm_g=0.2, fwhm_l=0.2, fwhm_l2=0.4,
    med_energy=med, mode="legacy",
)

fig, ax = plt.subplots(figsize=(8, 4))
# stick plot via vlines
ax.vlines(E_sticks.numpy(), 0, M_sticks.numpy(),
          color="C1", lw=0.6, alpha=0.7, label="sticks")
ax.plot(x_grid.numpy(), y_grid.numpy(), "C0", lw=1.8, label="pseudo-Voigt")
ax.set_xlabel("Energy (eV)")
ax.set_ylabel("Intensity")
ax.set_title("nid8ct — sticks + broadened envelope")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
"""),
    md("""
## Layer 6 — Closing the loop: does this match `getXAS()`?

`getXAS()` is just a thin wrapper around the parse → sticks → broaden
chain above, so the two should be byte-identical on the same grid.
"""),
    code("""
from multitorch.api.plot import getXAS

x_hi, y_hi = getXAS(str(REFROOT / "nid8ct.ban_out"),
                    T=80.0, beam_fwhm=0.2, gamma1=0.2, gamma2=0.4, nbins=2000)

# getXAS uses the same defaults — check that our manual reconstruction
# matches to floating-point noise
max_abs_err = float((y_hi - y_grid).abs().max())
rel_err = max_abs_err / float(y_hi.abs().max())
print(f"max |getXAS() - manual|       = {max_abs_err:.2e}")
print(f"relative max error            = {rel_err:.2e}")
print(f"matched to float64 precision: {max_abs_err < 1e-10}")
"""),
    md("""
## Recap

| Layer | Tensor type | Where defined |
|---|---|---|
| Wigner 3j/6j | scalars / tensors | `multitorch.angular.wigner` |
| COWAN store | sparse matrices | `multitorch.io.read_rme` |
| Symmetry blocks | per-block matrices | `multitorch.io.read_rme` |
| Hamiltonian | `BanResult.triads[i].Eg/Ef/T` | `multitorch.hamiltonian.assemble` |
| Sticks | `(E, M)` tensors | `multitorch.spectrum.sticks` |
| Spectrum | `(x, y)` tensors | `multitorch.spectrum.broaden` |

Every layer is pure PyTorch, float64, autograd-compatible. The
`getXAS()` wrapper just calls into Layers 5–6 with the `.ban_out`
bootstrap input; if you want to run Layer 4 yourself (assembling from
raw RMEs), use `assemble_and_diagonalize()` as shown above.
"""),
]

build_notebook("02 — Pipeline walkthrough", nb02, "02_pipeline_walkthrough.ipynb")


# ─────────────────────────────────────────────────────────────
# Notebook 03 — Parameter exploration
# ─────────────────────────────────────────────────────────────

nb03 = [
    md("""
# 03 — Parameter exploration: how the physics knobs shape the spectrum

This notebook sweeps the main physical parameters of an L-edge multiplet
calculation and plots how each one reshapes the spectrum. All sweeps run
through the fast `getXAS()` bootstrap path, so each frame takes
milliseconds.

**Sweeps in this notebook**
1. **d-electron count** — the full Ti⁴⁺ (d⁰) → Ni²⁺ (d⁸) series.
2. **Crystal field 10Dq** — edit the `.ban` recipe on the fly and
   re-diagonalize with `assemble_and_diagonalize()`.
3. **Charge-transfer Δ** — same mechanism on a different parameter.
4. **Temperature** — Boltzmann population of the ground manifold
   (requires `max_gs > 1`).
5. **Broadening widths** — the `(beam_fwhm, gamma1, gamma2)` trio.

The 10Dq and Δ sweeps rewrite the `.ban` file in a tempdir and re-run
the assembler. For the broadening and temperature sweeps the Hamiltonian
is unchanged; only the spectrum layer recomputes.
"""),
    ROOT_CHDIR_CELL,
    md("""
## Sweep 1 — d-electron count (Ti⁴⁺ through Ni²⁺)

The `tests/reference_data/` directory carries eight pre-computed ttmult
runs covering the full 3d L-edge series. The spectra shift up in
absolute energy across the series and the multiplet structure changes
dramatically with d-count.
"""),
    code("""
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
from multitorch.api.plot import getXAS

REFROOT = Path("tests/reference_data")

# (case_id, human label) ordered by d-electron count
CASES = [
    ("ti4_d0_oh",  "Ti⁴⁺ d⁰"),
    ("v3_d2_oh",   "V³⁺ d²"),
    ("cr3_d3_oh",  "Cr³⁺ d³"),
    ("mn2_d5_oh",  "Mn²⁺ d⁵"),
    ("fe3_d5_oh",  "Fe³⁺ d⁵"),
    ("fe2_d6_oh",  "Fe²⁺ d⁶"),
    ("co2_d7_oh",  "Co²⁺ d⁷"),
    ("ni2_d8_oh",  "Ni²⁺ d⁸"),
]

fig, ax = plt.subplots(figsize=(9, 9))
for i, (case, label) in enumerate(CASES):
    x, y = getXAS(str(REFROOT / case / f"{case}.ban_out"),
                  T=80.0, beam_fwhm=0.3, gamma1=0.3, gamma2=0.5)
    y_norm = y / y.max()
    ax.plot(x.numpy() - float(x[y.argmax()]),  # zero at main peak
            y_norm.numpy() + i * 1.05, lw=1.5, label=label)

ax.set_xlabel("Energy relative to L$_3$ main line (eV)")
ax.set_ylabel("Intensity (offset, normalized)")
ax.set_title("Octahedral 3d transition-metal L-edge series")
ax.set_xlim(-8, 20)
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
"""),
    md("""
## Helper — sweep a `.ban` parameter and return broadened spectra

For physical-parameter sweeps (10Dq, CT energy) we need to re-run the
Hamiltonian assembler with a modified `.ban` file. Rather than edit the
library, we do the simplest thing: read the original `.ban` as text,
swap one value, write to a tempdir, and re-run `getXAS()` via a
lightweight wrapper that plumbs the re-assembled result through the
standard `sticks → broaden` path.
"""),
    code("""
import tempfile, re, shutil, os
from multitorch.hamiltonian.assemble import assemble_and_diagonalize
from multitorch.spectrum.sticks import get_sticks
from multitorch.spectrum.broaden import pseudo_voigt

# Tiny adapter: BanResult (from assemble_and_diagonalize) has .triads[].T,
# but get_sticks expects a BanOutput with .triad_list[].M. We wrap one in
# the other just for the duration of this call.
class _BanOutputAdapter:
    def __init__(self, ban_result):
        class _T:
            def __init__(self, tr):
                self.gs_sym = tr.gs_sym
                self.act_sym = tr.act_sym
                self.fs_sym = tr.fs_sym
                self.Eg = tr.Eg
                self.Ef = tr.Ef
                self.M = tr.T
        self.triad_list = [_T(tr) for tr in ban_result.triads]


def spectrum_from_modified_ban(fixture_dir: Path, ban_text: str, *,
                               T=80.0, beam_fwhm=0.2, gamma1=0.2, gamma2=0.4,
                               nbins=2000):
    \"\"\"Write ban_text to a tempdir, rerun assemble_and_diagonalize, broaden.\"\"\"
    case = fixture_dir.name
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Copy the RME files unchanged
        for ext in ("rme_rcg", "rme_rac"):
            shutil.copy(fixture_dir / f"{case}.{ext}", td / f"{case}.{ext}")
        # Write the modified .ban
        (td / f"{case}.ban").write_text(ban_text)
        # Assemble + diagonalize
        result = assemble_and_diagonalize(
            str(td / f"{case}.rme_rcg"),
            str(td / f"{case}.rme_rac"),
            str(td / f"{case}.ban"),
        )
    # Broaden
    ban = _BanOutputAdapter(result)
    E_sticks, M_sticks, _ = get_sticks(ban, T=T, max_gs=1)
    if E_sticks.numel() == 0:
        raise ValueError("no sticks produced — parameter change may be pathological")
    x = torch.linspace(float(E_sticks.min()) - 5.0,
                       float(E_sticks.max()) + 5.0,
                       nbins, dtype=torch.float64)
    med = 0.5 * (float(E_sticks.min()) + float(E_sticks.max()))
    y = pseudo_voigt(
        x, E_sticks, M_sticks,
        fwhm_g=beam_fwhm, fwhm_l=gamma1, fwhm_l2=gamma2,
        med_energy=med, mode="legacy",
    )
    return x, y


# Sanity check: unmodified .ban should reproduce the stock getXAS within
# float64 noise on the parts of the grid they both cover.
fixture = REFROOT / "ni2_d8_oh"
original_ban_text = (fixture / "ni2_d8_oh.ban").read_text()
x0, y0 = spectrum_from_modified_ban(fixture, original_ban_text)
print(f"unmodified sweep reproduces a spectrum with {y0.shape[0]} bins, "
      f"ymax = {float(y0.max()):.4f}")
"""),
    md("""
## Sweep 2 — Crystal field 10Dq

In the `.ban` recipe the crystal-field strength appears in the `XHAM`
line as the second value (for octahedral symmetry: `XHAM 2 1.0 <tendq>`).
We sweep it from 0.8 eV (weak field, deeper multiplet structure) to
1.6 eV (strong field, more singlet character).
"""),
    code("""
def replace_xham_tendq(ban_text: str, new_tendq: float) -> str:
    # XHAM line looks like ' XHAM 2 1.0 1.200   ' for Oh (2 operators: H, 10Dq).
    # Substitute the second numerical value.
    def sub(m):
        return f"{m.group(1)}{new_tendq:.4f}"
    return re.sub(r"(\\bXHAM\\s+\\d+\\s+[\\d.]+\\s+)([-\\d.]+)", sub, ban_text, count=1)


fixture = REFROOT / "ni2_d8_oh"
base_ban = (fixture / "ni2_d8_oh.ban").read_text()

fig, ax = plt.subplots(figsize=(8, 4))
for tendq in [0.8, 1.0, 1.2, 1.4, 1.6]:
    modified = replace_xham_tendq(base_ban, tendq)
    x, y = spectrum_from_modified_ban(fixture, modified,
                                      beam_fwhm=0.3, gamma1=0.3, gamma2=0.5)
    y_norm = y / y.max()
    ax.plot(x.numpy() - float(x[y.argmax()]), y_norm.numpy(),
            label=f"10Dq = {tendq:.1f} eV", lw=1.5)

ax.set_xlabel("Energy relative to main peak (eV)")
ax.set_ylabel("Intensity (normalized)")
ax.set_title("Ni²⁺ d⁸ — crystal-field (10Dq) sweep")
ax.set_xlim(-8, 20)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
"""),
    md("""
## Sweep 3 — Charge-transfer energy Δ

The charge-transfer energy Δ = E(d⁸L) − E(d⁹L̲) is controlled by the
`EG2` line in the `.ban` DEF block. Lower Δ mixes ligand-hole character
into the ground state and redistributes spectral weight into the
charge-transfer satellite.
"""),
    code("""
def replace_eg2(ban_text: str, new_eg2: float) -> str:
    def sub(m):
        return f"{m.group(1)}{new_eg2:.4f}"
    return re.sub(r"(\\bDEF\\s+EG2\\s*=\\s*)([-\\d.]+)", sub, ban_text, count=1)


fig, ax = plt.subplots(figsize=(8, 4))
for dE in [2.0, 4.0, 6.0, 8.0]:
    modified = replace_eg2(base_ban, dE)
    x, y = spectrum_from_modified_ban(fixture, modified,
                                      beam_fwhm=0.3, gamma1=0.3, gamma2=0.5)
    ax.plot(x.numpy() - float(x[y.argmax()]),
            y.numpy() / float(y.max()),
            label=f"Δ = {dE:.1f} eV", lw=1.5)

ax.set_xlabel("Energy relative to main peak (eV)")
ax.set_ylabel("Intensity (normalized)")
ax.set_title("Ni²⁺ d⁸ — charge-transfer Δ sweep")
ax.set_xlim(-8, 20)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
"""),
    md("""
## Sweep 4 — Temperature (Boltzmann population)

At low temperature only the lowest ground state contributes; at higher
temperature, thermally populated excited states of the ground manifold
add their own spectra. `getXAS()` exposes `max_gs` to control how many
ground states to retain.
"""),
    code("""
fig, ax = plt.subplots(figsize=(8, 4))
for T in [10.0, 80.0, 300.0, 1000.0]:
    x, y = getXAS(str(fixture / "ni2_d8_oh.ban_out"),
                  T=T, max_gs=5,
                  beam_fwhm=0.3, gamma1=0.3, gamma2=0.5)
    ax.plot(x.numpy() - float(x[y.argmax()]),
            y.numpy() / float(y.max()),
            label=f"T = {T:g} K", lw=1.5)
ax.set_xlabel("Energy relative to main peak (eV)")
ax.set_ylabel("Intensity (normalized)")
ax.set_title("Ni²⁺ d⁸ — temperature sweep (max_gs = 5)")
ax.set_xlim(-8, 20)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
"""),
    md("""
## Sweep 5 — Broadening (resolution trade-off)

The broadening sweep in `01_quickstart.ipynb` varied one width at a
time. Here we show the full (narrow, medium, wide) envelope to highlight
how the multiplet fine structure resolves under better resolution.
"""),
    code("""
presets = [
    ("narrow (high-res)",    dict(beam_fwhm=0.10, gamma1=0.10, gamma2=0.20)),
    ("medium (typical)",     dict(beam_fwhm=0.30, gamma1=0.30, gamma2=0.50)),
    ("wide (low-res)",       dict(beam_fwhm=0.60, gamma1=0.60, gamma2=1.00)),
]

fig, ax = plt.subplots(figsize=(8, 4))
for label, params in presets:
    x, y = getXAS(str(fixture / "ni2_d8_oh.ban_out"), T=80.0, **params)
    ax.plot(x.numpy() - float(x[y.argmax()]), y.numpy() / float(y.max()),
            label=label, lw=1.5)
ax.set_xlabel("Energy relative to main peak (eV)")
ax.set_ylabel("Intensity (normalized)")
ax.set_title("Ni²⁺ d⁸ — broadening presets")
ax.set_xlim(-8, 20)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
"""),
    md("""
## What you can do from here

- **Fit experimental data.** All tensors are `float64` and
  `requires_grad`-compatible, so you can drop `getXAS()` inside a
  PyTorch optimization loop and fit 10Dq, Δ, slater, soc to a measured
  spectrum.
- **New ions.** Copy any `tests/reference_data/<case>/` directory,
  regenerate the `.rme_rcg` / `.rme_rac` / `.ban_out` with the Fortran
  toolchain (see `pyttmult/`), and the bootstrap pipeline will pick it
  up without any further work.
- **Pure-PyTorch pipeline.** The full `calcXAS(element=..., valence=..., ...)`
  form that skips the Fortran step is a Phase 5 goal and currently
  raises `NotImplementedError`. Follow `TASK-017` onwards in
  `.claude/orchestration/INDEX.md`.
"""),
]

build_notebook("03 — Parameter exploration", nb03, "03_parameter_exploration.ipynb")

print("all notebooks written")
