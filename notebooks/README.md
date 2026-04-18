# multitorch demonstration notebooks

Four runnable Jupyter notebooks that introduce the multitorch package
end-to-end. They run from a fresh `pip install -e .` plus the optional
notebook dependencies — no Fortran at runtime, no HFS SCF.

## Setup

```bash
conda activate multi
pip install -e ".[notebook]"
jupyter lab notebooks/
```

Each notebook chdirs to the repo root on its first cell, so you can launch
Jupyter from anywhere inside the repository.

## Contents

| Notebook | What it shows |
|---|---|
| `01_quickstart.ipynb` | One-line `getXAS()` call on the `nid8ct` fixture, overlay vs the Fortran reference `.xy`, and a sweep over the broadening parameters (`beam_fwhm`, `gamma1`, `gamma2`). |
| `02_pipeline_walkthrough.ipynb` | Peels back the `getXAS()` wrapper layer by layer: Wigner primitives → COWAN store → `assemble_and_diagonalize` → Boltzmann sticks → pseudo-Voigt. Closes with a byte-level match against `getXAS()`. |
| `03_parameter_exploration.ipynb` | Full 3d L-edge series, 10Dq/Δ/Slater sweeps (both file-editing and Phase 5 `calcXAS`), temperature and broadening sweeps, autograd demo. |
| `04_rixs_quickstart.ipynb` | Synthetic RIXS 2D map via the Kramers-Heisenberg kernel, CIE/CEE line cuts, autograd through the kernel, `calcRIXS` API reference. |

## Re-executing in CI / from the command line

```bash
for nb in notebooks/*.ipynb; do
    jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```

All four notebooks should run to completion against the fixtures bundled
under `multitorch/data/fixtures/`.
