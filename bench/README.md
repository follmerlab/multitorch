# multiplet-bench

Internal performance + output-parity harness for the multitorch
PyTorch port. Two modes:

- **multitorch-only** (default for public users): time the three
  multitorch entry points (`calcXAS_cached`, `calcXAS_from_scratch`,
  `calcXAS_batch`) on CPU and CUDA across the bundled Ti–Ni L-edge
  fixtures.
- **Fortran-comparison** (development only): also runs the legacy
  pyctm / pyttmult / ttmult Fortran stack and reports parity. Requires
  the private dev repos described under *Optional: Fortran comparison*.

## What you get

- **Timings**: median + IQR wall-clock, CPU-time, and CUDA-event time per cell
- **Stage breakdown**: setup / compute / broadening, separated
- **Parity metrics** (Fortran mode): cosine similarity, max-abs-diff,
  peak-position error, L3/L2 ratio — vs `ttmult_raw` on a common grid
  after peak alignment
- **Plots**: single-spectrum bars, scaling vs batch size, parity
  heatmap, CPU vs CUDA speedup

## Layout

`bench/` ships inside the multitorch repo:

```
multitorch/              # the public repo
  bench/                 # ← this package
  multitorch/            # PyTorch package source
  tests/
```

## Environment

Uses the `multi` conda env (Python 3.11 + PyTorch 2.5). Plus:

```
pip install pandas matplotlib pyyaml psutil gitpython tabulate
```

Public users running the multitorch-only mode are done at this point.

## Optional: Fortran comparison (dev only)

The Fortran stack (pyctm, pyttmult, ttmult) is the legacy
implementation that multitorch ports. Comparing against it is useful
during development but requires the private repos checked out as
peers of multitorch:

```
parent_dir/
  multitorch/        # this repo, with bench/ inside
  pyctm/             # legacy Python orchestrator
  pyttmult/          # thin wrapper over Fortran binaries
  ttmult/            # Fortran source + binaries in ttmult/bin/
```

bench resolves these via `WORKSPACE_ROOT = BENCH_ROOT.parent.parent`.
Override with `PYCTM_ROOT`, `PYTTMULT_ROOT`, or `ttmult` env vars if
your peers live elsewhere. Set `$ttmult` so pyttmult finds the
binaries:

```
export ttmult=/abs/path/to/ttmult
```

Without these peers the Fortran adapters skip cleanly with
`NotSupported`; the multitorch cells still run.

## Quick start — MVP smoke (~8 min on CPU)

```bash
cd bench
PYTHONPATH=. python -m bench.run_all \
    --preset mvp \
    --output results/mvp.jsonl \
    --timeout-s 300

PYTHONPATH=. python -m bench.postprocess.report \
    --input results/mvp.jsonl \
    --out results/report_mvp
```

Produces `results/report_mvp/{p1,p2,p3}_*.png` and `report.md`.

## Full overnight run

```bash
PYTHONPATH=. python -m bench.run_all \
    --preset full \
    --output results/full.jsonl \
    --timeout-s 300 \
    --resume
```

The harness is **resume-safe**: if you kill it or it crashes, re-run
with the same `--output` and it skips any cell already marked
`status=ok`. Each cell runs in its own subprocess with a timeout, so
a Fortran hang or CUDA OOM cannot take the driver down.

## CLI flags

- `--preset {mvp, full}` — matrix size
- `--output PATH` — append-only JSONL (sidecar `*.hardware.json` written once)
- `--timeout-s SECONDS` — per-cell timeout (default 300)
- `--resume` — skip cells already marked ok in `--output`
- `--dry-run` — print the cell plan, don't execute
- `--only-impl SUBSTR` / `--only-fixture SUBSTR` — filter cells
- `--max-cells N` — cap the number of cells (debugging)

## Implementations benchmarked

| Name | Description |
|---|---|
| `multitorch_cached` | `preload_fixture` + `calcXAS_cached` (realistic usage) |
| `multitorch_from_scratch` | `calcXAS_from_scratch` (Oh only, regenerates every call) |
| `multitorch_batch` | `calcXAS_batch` (Phase 2 parameter-sweep mode) |
| `pyctm` | `pyctm.calc.calcXAS` (writes files, runs Fortran via pyttmult) |
| `pyttmult` | `pyttmult.ttmult.ttmult` (Fortran pipeline orchestrator) |
| `ttmult_raw` | direct subprocess invocation of ttrcg / ttrac / ttban_exact |

The Fortran-side three are in a strict hierarchy — subtracting timings
pairwise (pyctm − pyttmult − ttmult_raw) decomposes the overhead into
orchestration / Python-wrapper / pure-Fortran cost. See `p5` in the
full-matrix plots.

## Parity

Every non-multitorch impl is compared against `ttmult_raw` (the pure
Fortran reference). The parity metrics live in `bench/parity.py` and
apply peak alignment before computing cosine / line-shape metrics —
different CTM codes use different energy zeros and the shift is
reported explicitly as `peak_shift_ev`.

## Development

```bash
PYTHONPATH=. python -m pytest tests/
```

Runs 14 unit tests for the parity metrics on synthetic spectra.

## First MVP data (CPU, Apple Silicon)

Median wall time per spectrum in ms:

```
                    batch=1   batch=10   batch=100
multitorch_cached      10         92       946
multitorch_from_scratch 1216    12054       (timeout >300s)
pyctm                  35        342      3489
pyttmult               34        338      3493
```

Takeaway: **multitorch cached is ~3.5× faster than the Fortran stack
for parameter-sweep workflows. From-scratch is ~35× slower** — the
speed story only holds when fixtures are reused.
