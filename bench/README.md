# multiplet-bench

Performance and output-parity harness comparing the PyTorch port
(**multitorch**) against the legacy Fortran stack (**pyctm →
pyttmult → ttmult**) across the bundled Ti–Ni L-edge fixture set.

## What you get

- **Timings**: median + IQR wall-clock, CPU-time, and CUDA-event time per cell
- **Stage breakdown**: setup / Fortran compute / broadening, separated
- **Parity metrics**: cosine similarity, max-abs-diff, peak-position
  error, L3/L2 ratio — computed against `ttmult_raw` on a common grid
  after peak alignment
- **Plots**: single-spectrum bars, scaling vs batch size, parity
  heatmap, CPU vs CUDA speedup

## Layout assumption

The four packages must be checked out side by side:

```
multiplets/
  multitorch/        # PyTorch port
  pyctm/             # legacy Python orchestrator
  pyttmult/          # thin wrapper over Fortran binaries
  ttmult/            # Fortran source + binaries in ttmult/bin/
  bench/             # this package (benchmark harness)
```

## Environment

Uses the existing `multi` conda env (Python 3.11 + PyTorch 2.5). Plus:

```
pip install pandas matplotlib pyyaml psutil gitpython tabulate
```

Set the `$ttmult` environment variable to point at the ttmult root:

```
export ttmult=/Users/afollmer/Follmer_UCD/Follmer_Lab/Code/multiplets/ttmult
```

(On first run the harness defaults this to `<workspace>/ttmult` if unset.)

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
