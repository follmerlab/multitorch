# multitorch demonstration notebooks

Three runnable Jupyter notebooks that introduce the multitorch package
end-to-end. All three use only the `.ban_out` bootstrap pipeline (no
Fortran at runtime, no HFS SCF), so they run from a fresh `pip install -e .`
plus the optional notebook dependencies.

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
| `03_parameter_exploration.ipynb` | Uses the eight committed Ti–Ni fixtures to plot the full 3d L-edge series in a single figure, then explores temperature and broadening sweeps on `nid8ct`. |

## Re-executing in CI / from the command line

```bash
for nb in notebooks/*.ipynb; do
    jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```

All three notebooks should run to completion against the fixtures committed
under `tests/reference_data/`.
