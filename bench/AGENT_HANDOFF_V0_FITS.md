# exxa v0 fitter sweep handoff (α track)

**Author:** Claude, 2026-05-03
**Goal:** Fit all 5 Fe L-edge XAS spectra on workstation CPU using the
v0 Oh-approximation differentiable fitter. Save figures + JSON results
for the manuscript.

**UPDATE 2026-05-03 (β track):** the "CUDA-autograd bug" reported in
the GPU eigh benchmark was a **false alarm**. Re-verified on exxa via
`bench/diag_cuda_autograd.py` — all 4 configs (CPU, CUDA-explicit,
CUDA-with-cpu-cache, mismatch) preserve the autograd graph correctly.
slater.grad on cuda matches CPU to 1e-9.

**GPU is the recommended device.** Smoke test on exxa (RTX 4090):
- Fe(III) Adam step: ~3.3 sec
- 200-step Fe(III) fit: ~11 min
- 5-spectrum sweep (200 steps each): ~30-50 min total

CPU fallback (--device cpu): ~7.4 sec/step, ~25 min/spectrum,
~80 min total — usable but ~3× slower.

---

## 1. Sync the fits/ scratchpad onto exxa

The `fits/` directory lives **outside** the multitorch repo (peer to
multitorch/, pyctm/, pyttmult/, ttmult/). It's not under git
because it's a research scratchpad. To get it onto exxa:

From your **local Mac**, run:

```bash
rsync -av --exclude __pycache__ --exclude .pytest_cache \
    /Users/afollmer/Follmer_UCD/Follmer_Lab/Code/multiplets/fits/ \
    exxa:<workspace_path>/multiplets/fits/
```

Replace `<workspace_path>` with the path on exxa where multiplets/
lives (likely `~/multiplets` or similar — wherever multitorch/,
pyctm/, etc. are checked out).

Confirm the rsync landed:

```bash
ssh exxa 'ls <workspace_path>/multiplets/fits/'
```

Should show: `fe_xas_fit.py`, `fit_all.py`, `fit_and_plot_all.py`,
`plot_fit.py`, `diagnose.py`, `README.md`, `results/`.

## 2. Pull latest multitorch on exxa

```bash
ssh exxa
cd <workspace_path>/multitorch
git pull origin main   # should land you at a300524 or later
```

(`a300524` includes your GPU benchmark results pushed earlier.)

## 3. Run the sweep

```bash
# from <workspace_path>/multiplets on exxa
/home/afollmer/miniconda3/envs/multi/bin/python -u fits/fit_and_plot_all.py
```

(Adjust the python path if your conda env is elsewhere.) The script
auto-detects CUDA and defaults to GPU. Force CPU with `--device cpu`.

Output streams to terminal AND saves to `fits/results/`. The script:

- Fits all 5 spectra at LR=0.01, 200 Adam steps each
- Saves a PNG figure per spectrum (data + fit overlay + residual)
- Writes `v0_summary.json`, `v0_summary.txt`, `v0_runtime.txt`
- Prints per-spectrum wall time + loss

Expected per-spectrum walls:
- FeIIPc (d6): ~2-3 min
- FeIIIPc-Cl (d5): ~25 min
- FeIIIPor-Cl (d5): ~25 min
- FeTPC-Cl (d5): ~25 min
- Fe[TPC]-NO (d5 or d6, ambiguous): ~5-25 min

Total: ~80-90 min on Threadripper CPU.

## 4. Sync results back to local Mac

When done, from your Mac:

```bash
rsync -av exxa:<workspace_path>/multiplets/fits/results/ \
    /Users/afollmer/Follmer_UCD/Follmer_Lab/Code/multiplets/fits/results/
```

Then I can read them locally and assemble manuscript-ready figures.

## 5. (Optional) Spawn Claude on exxa

If you'd rather have Claude run + monitor:

```
On exxa, cd <workspace_path>/multiplets, run claude, paste:

Run /multiplets/fits/fit_and_plot_all.py via 
/opt/anaconda3/envs/multi/bin/python -u and monitor progress.
This is the v0 Fe XAS fitter sweep — fits all 5 spectra
(1 Fe(II), 4 Fe(III)) at LR=0.01, 200 Adam steps each.
Each Fe(III) fit takes ~25 min on Threadripper CPU; total
expected wall ~80 min.

Save outputs to fits/results/. Confirm v0_summary.json,
v0_summary.txt, v0_runtime.txt, and 5 PNG files exist after.

Also, after the sweep completes:
- Inspect each PNG residual panel for systematic deviation
- Note whether Fe[TPC]-NO converges to Fe(II)-like or Fe(III)-like
  parameters (resolves the oxidation-state ambiguity)
- Report total wall + summary table to the user

Don't push anything to git — fits/ is intentionally outside the
multitorch repo.

Background context in:
- multitorch/CODE_REVIEW.md (Addendum on Perf-001 + Addendum 2)
- multitorch/docs/PORT_NOTES.md (BUGs, perf, Threads 2/3 unification)
- fits/results/v0_diagnostics.md (the report that motivated this run)
```

## What "good" looks like

Each PNG should show:
- Black experimental curve, red multitorch fit overlay
- L3 peak at ~707-708 eV well-positioned (peak alignment)
- L2 peak at ~720 eV captured
- Residual panel ≤ 0.2 amplitude
- Final loss in 0.005-0.05 range (Oh approximation ceiling — D4h
  port would tighten further, see PORT_NOTES Threads 2/3
  unification)

Fitted parameters in `v0_summary.json` should fall in physically
reasonable ranges:
- slater: 0.7-0.9
- soc: 0.9-1.1
- tendq: 1.0-2.5 eV
- gamma1 (L3 broadening): 0.2-0.6 eV
- gamma2 (L2 broadening): 0.4-1.0 eV
- dE_shift: ~580-620 eV (multitorch-vs-experimental energy zero offset)

If anything is at a hard bound (slater=0.5 or 1.0, tendq=3.0, etc.)
the fitter is fighting the bounds — either bound was too tight, or
Oh approximation is genuinely insufficient for that spectrum.

## What I'm working on in parallel (β)

While the workstation runs, I'm tracing the CUDA-autograd bug in
`calcXAS_cached`. On Mac CPU the graph is intact (verified — `y.grad_fn
= AddBackward0`, finite gradients on slater/soc/tendq). Bug is
specific to the cuda path with cpu-cached fixtures. Will commit a
fix when I have one; that unlocks the GPU path for future runs.
