# multitorch

**Differentiable multiplet simulations for X-ray spectroscopy in PyTorch.**

multitorch computes L-edge X-ray absorption (XAS), emission (XES), and
resonant inelastic X-ray scattering (RIXS) spectra for 3d transition
metal complexes using crystal-field and charge-transfer multiplet theory.
The entire pipeline — from Wigner coupling coefficients through
Hamiltonian diagonalization to broadened spectra — runs in pure PyTorch
with float64 precision and full `torch.autograd` support.

## Features

- **Differentiable end-to-end** — gradients flow from physical parameters
  (Slater reduction, spin-orbit coupling, crystal field, charge transfer)
  through eigenvalues and transition matrices to the final spectrum
- **No Fortran runtime** — all physics computed natively in Python/PyTorch;
  pre-computed angular coefficients are bundled as package data
- **XAS, XES, RIXS** — L-edge absorption, emission, and 2D resonant
  inelastic scattering maps via the Kramers-Heisenberg kernel
- **Validated** — 477 tests; numerical agreement with the established
  Cowan/ttmult Fortran suite across 9 Ti-Ni fixtures (cosine similarity
  >= 0.97)
- **GPU-ready** — tensors move to any PyTorch device

## Installation

```bash
git clone https://github.com/follmerlab/multitorch.git
cd multitorch
pip install -e .
```

**Requirements:** Python >= 3.10, PyTorch >= 2.2, NumPy, SciPy.

Optional extras:

```bash
pip install -e ".[plot]"      # adds matplotlib
pip install -e ".[notebook]"  # adds matplotlib + jupyter
pip install -e ".[dev]"       # adds pytest, pytest-cov, sympy
```

Or with conda:

```bash
conda env create -f requirements/environment.yml
conda activate multi
pip install -e .
```

## Quick start

```python
import torch
from multitorch import calcXAS

# Ni2+ L-edge XAS in D4h symmetry
x, y = calcXAS(
    element="Ni", valence="ii", sym="d4h", edge="l",
    cf={"tendq": 1.0, "ds": 0.0, "dt": 0.01},
    slater=0.8, soc=1.0, T=300,
)

# Plot
import matplotlib.pyplot as plt
plt.plot(x.numpy(), y.detach().numpy())
plt.xlabel("Energy (eV)")
plt.ylabel("Intensity")
plt.show()
```

### Autograd example

```python
slater = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
x, y = calcXAS(
    element="Ni", valence="ii", sym="d4h", edge="l",
    cf={}, slater=slater, soc=1.0,
)

# Gradient of the spectrum w.r.t. Slater reduction
loss = y.sum()
loss.backward()
print(slater.grad)  # d(spectrum)/d(slater)
```

### RIXS

```python
from multitorch import calcRIXS

rixs_map = calcRIXS(
    "multitorch/data/fixtures/nid8ct/nid8ct.ban_out",  # absorption
    "multitorch/data/fixtures/nid8ct/nid8ct.ban_out",  # emission
)
```

## API reference

| Function | Description |
|---|---|
| `calcXAS(element, valence, sym, edge, cf, ...)` | Compute XAS from physical parameters (differentiable) |
| `calcXAS(..., ban_output_path=...)` | Compute XAS from a pre-computed `.ban_out` file |
| `calcRIXS(ban_abs_path, ban_ems_path, ...)` | 2D RIXS map via Kramers-Heisenberg |
| `calcXES(...)` | X-ray emission spectrum |
| `calcDOC(...)` | Ground-state configuration weights |
| `getXAS(ban_output_path, ...)` | Convenience wrapper for file-based XAS |
| `preload_fixture(fixture_dir)` | Cache parsed fixture for fast parameter sweeps |
| `calcXAS_cached(fixture, ...)` | XAS from a cached fixture (no file I/O per call) |

All functions are importable from `multitorch` directly:

```python
from multitorch import calcXAS, calcRIXS, calcDOC, getXAS
```

## Notebooks

Four tutorial notebooks in `notebooks/`:

| Notebook | Description |
|---|---|
| `01_quickstart` | One-call spectrum with broadening parameter sweep |
| `02_pipeline_walkthrough` | Layer-by-layer: Wigner symbols -> Hamiltonian -> sticks -> spectrum |
| `03_parameter_exploration` | Ti-Ni L-edge series, 10Dq/Slater sweeps with autograd |
| `04_rixs_quickstart` | 2D RIXS maps, line cuts, autograd through the kernel |

```bash
pip install -e ".[notebook]"
jupyter lab notebooks/
```

## Architecture

```
Physical parameters (slater, soc, cf, delta)
    |
    v
Atomic scaling (multitorch.atomic)
    |
    v
Angular coefficients (multitorch.angular) -- Wigner 3j/6j, CFP, RME
    |
    v
Hamiltonian assembly + diagonalization (multitorch.hamiltonian)
    |
    v
Boltzmann stick spectrum (multitorch.spectrum.sticks)
    |
    v
Pseudo-Voigt broadening (multitorch.spectrum.broaden)
    |
    v
(x, y) spectrum tensors -- autograd-compatible
```

## Testing

```bash
pytest tests/ -q
```

477 tests covering Wigner primitives, file parsers, angular RME
construction, Hamiltonian eigenvalues, spectral broadening, autograd
gradients, and end-to-end parity with Fortran reference outputs.

## Background

multitorch implements the multiplet theory described in:

- F.M.F. de Groot, "Multiplet effects in X-ray spectroscopy,"
  *Coordination Chemistry Reviews* 249 (2005) 31-63
- R.D. Cowan, *The Theory of Atomic Structure and Spectra*
  (University of California Press, 1981)

The computational approach follows the Cowan/Butler/ttmult framework,
replacing the Fortran numerical core with differentiable PyTorch tensors
while preserving numerical agreement.

## License

MIT -- see [LICENSE](LICENSE).
