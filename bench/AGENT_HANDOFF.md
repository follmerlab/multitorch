# Bench Agent Handoff

**Purpose:** you are the agent picking this up on the user's NVIDIA GPU workstation. This file is everything you need to continue the multiplet-benchmark project from where the Mac session left off.

**Author of this handoff:** Claude, 2026-04-19, Apple Silicon laptop. The `bench/` directory you are looking at was built and validated on an M-series CPU; the full matrix and all CUDA cells are waiting for your GPU host.

---

## 1. What this project is

A reproducible performance + output-parity comparison between four multiplet-spectroscopy implementations:

| Impl | What it is |
|---|---|
| `multitorch_cached` | PyTorch port, preload-fixture + reuse (realistic modern usage) |
| `multitorch_from_scratch` | PyTorch port, regenerate angular blocks every call (fair to Fortran) |
| `multitorch_batch` | PyTorch port, Phase 2 batched parameter sweep (N spectra at once) |
| `pyctm` | Legacy Python orchestrator that drives the Fortran stack |
| `pyttmult` | Thin Python wrapper around Fortran binaries |
| `ttmult_raw` | Direct subprocess invocation of `ttrcg` / `ttrac` / `ttban_exact` |

The deliverable is a set of plots (single-spectrum bars, scaling vs batch size, parity heatmap, CPU-vs-CUDA speedup) + a markdown report, all generated from a JSONL record of every benchmark cell. This goes into the multitorch manuscript.

**Related upstream artifacts (context, not required to execute):**

- Literature review: `<workspace>/manuscript/ctm-l-edge-rixs-differentiable-xray-lit-review-guide.docx`
- Model summary: `<workspace>/multitorch/MODEL_SUMMARY.md` — has Phase 1/2 GPU batch context you should skim before running
- Plan file: `~/.claude/plans/lexical-toasting-kahan.md`
- Prior handoff cards: `<workspace>/.research_workspace/cards/*.json` + `manifest.json`

---

## 2. Workstation layout the suite assumes

```
<workspace>/
  multitorch/        # PyTorch port (public: follmerlab/multitorch)
  pyctm/             # legacy Python orchestrator
  pyttmult/          # thin wrapper over Fortran binaries
  ttmult/            # Fortran source + binaries in ttmult/bin/
  bench/             # <-- this directory; what you care about
```

The user confirmed all four upstream repos are already cloned and in place on the workstation. Your first job is to verify that (step 3 below).

Paths inside `bench/bench/config.py` are derived from `BENCH_ROOT.parent`, so they'll work regardless of where `<workspace>` lives on the workstation — as long as the four peer directories are siblings.

---

## 3. First-run checklist (do these in order, takes ~5 min)

### 3.1 Verify the four-package layout

```bash
cd <workspace>/bench
ls ../multitorch/multitorch/api/calc.py  || echo "MISSING multitorch"
ls ../pyctm/pyctm/calc.py                || echo "MISSING pyctm"
ls ../pyttmult/pyttmult/ttmult.py        || echo "MISSING pyttmult"
ls ../ttmult/bin/ttban_exact             || echo "MISSING ttmult binaries"
```

All four must exist. If any is missing, stop and ask the user.

### 3.2 Check the Python environment

The Mac session used conda env `multi` (Python 3.11, PyTorch 2.5). On a workstation with CUDA you'll need:

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

If `cuda=False`, stop and ask the user which env to activate. The workstation-side matrix depends on CUDA being available for the multitorch CUDA cells.

### 3.3 Install bench's own deps if not present

```bash
pip install pandas matplotlib pyyaml psutil gitpython tabulate
```

`tabulate` is optional (report falls back to `to_string`), but everything else is required.

### 3.4 Export `$ttmult`

```bash
export ttmult=<absolute path to workspace>/ttmult
```

`pyttmult.ttmult.getBin()` reads this env var. The bench adapters default it to `<workspace>/ttmult` if unset (see `bench/adapters/pyctm_adapter.py:cold_start`), but setting it explicitly is safer.

### 3.5 Run the parity unit tests

```bash
cd <workspace>/bench
PYTHONPATH=. python -m pytest tests/test_parity_metrics.py -q
```

Expected: **14 passed**. If any fail, stop and read the failure — it's either a NumPy version issue (`np.trapezoid` requires 1.26+) or a bug the Mac session didn't catch.

### 3.6 Run a 3-cell smoke to verify all adapters work on CPU

```bash
PYTHONPATH=. python -m bench.run_all \
    --preset mvp \
    --output results/workstation_smoke.jsonl \
    --only-fixture ni2_d8_oh \
    --max-cells 3 \
    --timeout-s 120
```

Expected output: `3 ok, 0 skipped, 0 failed / timed-out` in roughly 10 seconds. If pyttmult or pyctm errors, it's almost certainly the `$ttmult` env var.

### 3.7 Run a 1-cell CUDA smoke

```bash
PYTHONPATH=. python -m bench.run_all \
    --preset mvp \
    --output results/cuda_smoke.jsonl \
    --only-impl multitorch_cached \
    --only-fixture ni2_d8_oh \
    --max-cells 1 \
    --timeout-s 120
```

The MVP preset is CPU-only. To force a CUDA cell you'll need to use the full preset with a filter, or run a one-off:

```bash
PYTHONPATH=. python -c "
from bench.adapters.multitorch_adapter import MultitorchAdapter
from bench.config import BenchCell
cell = BenchCell(impl='multitorch_cached', fixture='ni2_d8_oh', calctype='xas', batch_size=1, device='cuda', nbins=500, reps=3)
a = MultitorchAdapter(mode='cached'); a.cold_start(); a.warm(cell)
out = a.run(cell)
print('CUDA output:', out.x.shape, out.y[:5])
"
```

Expected: prints shape `(500,)` + five intensity values. If CUDA kernel launch fails, something is wrong with the env.

---

## 4. The real runs

### 4.1 Full matrix — the manuscript run

```bash
cd <workspace>/bench
PYTHONPATH=. python -m bench.run_all \
    --preset full \
    --output results/full.jsonl \
    --timeout-s 300 \
    --resume
```

**Expected cells:** 1032 (confirmed via `--dry-run` on the Mac). Breakdown:
- 11 fixtures × 3 calctypes (many RIXS/XES will be `NotSupported` and marked `skipped`)
- 5 batch sizes: 1, 10, 100, 1000, 10000
- 6 impls
- 2 devices (cpu, cuda) — cuda only for multitorch impls, so ~1/3 of impl × fixture cells

**Expected runtime:** Hard to predict without running one impl first. Conservative estimate:
- CPU Fortran cells at batch=10000 could take tens of minutes each (300s timeout will cut some off)
- CUDA multitorch cells at batch=10000 should be seconds
- Overall: 4–12 hours on a good workstation. Overnight is safe.

**Resume is mandatory:** kill + restart at will. Completed cells are skipped; errored / timed-out cells are retried.

**Monitoring:** `tail -f results/full.jsonl | jq -c '{t: .timestamp, s: .status, i: .config.impl, f: .config.fixture, b: .config.batch_size, d: .config.device}'` gives a live feed.

### 4.2 Generate plots + report

```bash
PYTHONPATH=. python -m bench.postprocess.report \
    --input results/full.jsonl \
    --out results/report_full
```

Outputs:
- `results/report_full/report.md` — summary markdown with median-time pivot table
- `results/report_full/p1_single_spectrum_time.png`
- `results/report_full/p2_scaling_vs_batch.png`
- `results/report_full/p3_parity_cosine.png`
- `results/report_full/p4_cpu_vs_cuda.png` — **this one only populates from the workstation run** since the Mac had no CUDA

Copy these four PNGs + the markdown back to the Mac (or wherever the manuscript is being edited).

---

## 5. MVP baseline from the Mac (for sanity comparison)

The Mac already ran a 24-cell MVP. Your workstation numbers for CPU-only cells should be **in the same order of magnitude** (workstations are usually faster than laptops on single-thread CPU but this depends on the box).

Median wall time per spectrum (ms), Mac M-series, 14 cores:

```
                    batch=1   batch=10   batch=100
multitorch_cached      10         92       946
multitorch_from_scratch 1216    12054       (timeout >300s)
pyctm                  35        342      3489
pyttmult               34        338      3493
```

If your workstation CPU numbers are wildly different (say >3× slower or faster), note that in the handoff card you'll emit at the end — it's useful context for the manuscript's "methods" section.

Expected CUDA numbers (uncharted — this is what you'll contribute):
- `multitorch_cached` batch=1 on CUDA: likely **slower** than CPU for small dim (kernel overhead). `device_utils.suggest_device_for_xas` routes 3d TM L-edge to CPU by default; the full preset forces CUDA anyway so we capture the data.
- `multitorch_cached` batch=100+ on CUDA: should win noticeably.
- `multitorch_batch` batch=100+ on CUDA: the headline number. Expected 10–50× speedup vs CPU sequential on RIXS-sized fixtures.

---

## 6. Known footnotes (not blockers)

1. **CT fixtures skipped by Fortran adapters.** `pyctm`, `pyttmult`, and `ttmult_raw` raise `NotSupported` on `nid8ct` and `als1ni2` because the adapter code doesn't pass `delta` / `lmct` / `mlct` kwargs through. Only `multitorch_cached` produces CT fixture data. Extending the Fortran adapters is a ~20-line change in each — low priority unless the manuscript needs those cells.

2. **Parity cosine of 0.82 on `multitorch vs pyctm/pyttmult/ttmult_raw`** after peak-alignment and rigid shift. This is a broadening-convention difference (pyctm's `getXAS` uses a different Gaussian/Lorentzian kernel than multitorch's `pseudo_voigt`). The underlying sticks should agree at ≥0.97 (previously documented in multitorch's CODE_REVIEW). A cleaner parity story requires either (a) disabling multitorch's broadening and broadening its sticks with pyctm's kernel, or (b) vice versa. Out of scope for v1; document in the manuscript as "broadening conventions differ; physics agrees at the stick level".

3. **`multitorch_from_scratch` at batch=100 times out >300s** on Mac. Probably similar on the workstation. This is not a bug — it's the expected cost of regenerating angular blocks 100× in pure Python. The full preset will mark these cells as `status=timeout` and postprocessing handles that gracefully (dropped from P2 with a footnote).

4. **`multitorch_batch` requires `batch_size >= 2`.** The preset iterator already skips batch=1 cells for this impl (see `bench/config.py:iter_preset`). No action needed.

5. **Fortran pseudo-batching:** for `pyctm`/`pyttmult`/`ttmult_raw` at batch_size=N, the adapter just loops N times (each is a full Fortran invocation). Flagged `pseudo_batch=true` in the record meta. Scaling plots should label these lines distinctly.

---

## 7. Where things live

| File | Role |
|---|---|
| `bench/bench/config.py` | Fixture registry, preset definitions, `BenchCell`, `iter_preset` |
| `bench/bench/harness.py` | `wall_timer`, `cuda_event_timer`, `RepStats` |
| `bench/bench/parity.py` | Peak-aligned cosine / max-abs-diff / peak-error / L3/L2 metrics |
| `bench/bench/artifacts.py` | Append-only JSONL writer + resume-aware reader |
| `bench/bench/hardware.py` | CPU/GPU/torch/cuda/git-SHA/binary-mtime capture |
| `bench/bench/adapters/base.py` | `Adapter` ABC + `NotSupported` sentinel |
| `bench/bench/adapters/multitorch_adapter.py` | 3-mode multitorch wrapper |
| `bench/bench/adapters/pyctm_adapter.py` | pyctm wrapper (tempdir per call) |
| `bench/bench/adapters/pyttmult_adapter.py` | pyttmult wrapper (stage-timed) |
| `bench/bench/adapters/ttmult_raw_adapter.py` | Direct subprocess Fortran invocation |
| `bench/bench/run_all.py` | CLI driver (`--preset`, `--resume`, `--timeout-s`, etc.) |
| `bench/bench/postprocess/plots.py` | P1/P2/P3/P4 plot builders |
| `bench/bench/postprocess/report.py` | Plot driver + report.md assembly |
| `bench/tests/test_parity_metrics.py` | 14 unit tests for the parity module |
| `bench/results/mvp.jsonl` | Mac MVP results (for comparison) |
| `bench/results/report/` | Mac MVP plots (p1/p2/p3) |

---

## 8. Done criteria

You know you're done when:

1. `results/full.jsonl` has ≥900 records with `status=ok` (some cells will legitimately `skip` or `timeout`; that's fine).
2. `results/full.hardware.json` sidecar exists with the workstation's CPU/GPU/torch/cuda fingerprint.
3. `results/report_full/` has P1, P2, P3, **and P4** PNGs (the latter proves CUDA data populated).
4. `report.md` shows the median-time pivot table with CUDA columns populated for multitorch impls.
5. A one-paragraph summary of surprising findings (especially P4 — the CPU-vs-CUDA speedup story is where the manuscript figure lives).

Emit a handoff card at `<workspace>/.research_workspace/cards/bench-<TIMESTAMP>-<ID>.json` following the existing card schema you'll see in that directory. Include:
- `skill: "bench-runner"`
- `artifact_path`: relative path to `results/report_full/report.md`
- `summary`: the one-paragraph finding
- `skill_specific`:
  - `total_cells_run`, `ok_cells`, `skipped_cells`, `timeout_cells`, `error_cells`
  - `hardware_hash` from `results/full.hardware.json`
  - `key_finding` — one line, e.g. "multitorch_batch on CUDA is 28× faster than pyttmult at batch=1000"

Then rebuild `<workspace>/.research_workspace/manifest.json` atomically (see existing scripts for the pattern).

---

## 9. If something goes sideways

Common failures and their causes:

| Symptom | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError: Fortran binaries not found at .../ttmult/bin` | `$ttmult` unset or pointing somewhere else | `export ttmult=<workspace>/ttmult` |
| `ImportError: No module named pyttmult` | bench didn't find the peer dir | Verify `<workspace>/pyttmult/pyttmult/__init__.py` exists |
| `np.trapezoid AttributeError` | NumPy < 1.26 | Upgrade NumPy or swap in `scipy.integrate.trapezoid` in `parity.py:184` |
| `CUDA out of memory` on a batch=10000 cell | Expected for large Hilbert spaces on limited VRAM | Let the cell timeout/error and continue; it'll be recorded `status=error` |
| `ttban_exact: exited with code 139` | Segfault — usually input file corruption | Check the tempdir contents before deletion; probably a writeBAN issue in pyctm |
| Multiple duplicate records in JSONL | Ran without `--resume` or with a different hardware_hash | Dedupe by `record_id` + `timestamp` in postprocess |

If you hit something not in this table, **stop and ask the user** rather than guessing — the Fortran stack is fragile and easy to damage.

---

## 10. What comes next (for scoping, not for you to do)

After the full matrix run lands, the next manuscript steps are (not part of your task):

1. Plot polishing + figure captions
2. Extending Fortran adapters to CT fixtures
3. Tuning the parity comparison so cosine lands ≥0.97 (matched broadening)
4. P5 — stacked breakdown bar showing `fortran_compute | subprocess_launch | file_io | python_orchestration` for pyctm

Your job ends at step 4 of section 8. Hand off cleanly and let the user decide which of the above they want next.
