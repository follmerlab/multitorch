# exxa GPU eigh handoff (P0 confirmation)

**Author:** Claude, 2026-05-03, Apple Silicon laptop
**Target agent:** workstation agent on exxa (NVIDIA GPUs available)
**Bottleneck being investigated:** Fe(III) `calcXAS_cached` is 25.4 s
on Mac CPU vs Fe(II)'s 0.32 s — 80× slowdown. cProfile shows 89 % of
time in `torch.linalg.eigh` itself (not Python overhead). On GPU,
this should drop to <1 s per forward.

If P0 confirms (10×+ speedup), the v0 differentiable fitter at
`multiplets/fits/` becomes tractable for Fe(III) spectra and the
whole 5-spectrum sweep can finish in tens of minutes instead of
~15 hours.

Detailed per-priority refactor plan in `multitorch/CODE_REVIEW.md`
(addendum starting at line 269) and `multitorch/docs/PORT_NOTES.md`
(Perf-001 section).

---

## 1. Setup (verify before timing)

```bash
cd <workspace>/multitorch
git pull origin main   # should put you at 8ded04b or later
python -c "import torch; print('cuda:', torch.cuda.is_available(),
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

Expect `cuda: True` + a GPU name. If False, stop and ask the user
which env.

## 2. CPU vs GPU eigh microbenchmark

This is the smallest possible test — just `torch.linalg.eigh` on a
representative Hermitian matrix size.

```python
import torch, time
N = 190    # matches the largest Fe(III) Hamiltonian block

torch.manual_seed(0)
A_cpu = torch.randn(N, N, dtype=torch.float64)
H_cpu = (A_cpu + A_cpu.T) / 2

# Warmup CPU
_ = torch.linalg.eigh(H_cpu)
t0 = time.perf_counter()
for _ in range(5):
    _ = torch.linalg.eigh(H_cpu)
cpu_avg = (time.perf_counter() - t0) / 5
print(f"CPU eigh ({N}x{N}, fp64) avg: {cpu_avg*1000:.1f} ms")

# GPU
H_gpu = H_cpu.cuda()
torch.cuda.synchronize()
_ = torch.linalg.eigh(H_gpu)  # warmup (compiles cuSOLVER kernel)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(5):
    _ = torch.linalg.eigh(H_gpu)
torch.cuda.synchronize()
gpu_avg = (time.perf_counter() - t0) / 5
print(f"GPU eigh ({N}x{N}, fp64) avg: {gpu_avg*1000:.1f} ms")
print(f"Speedup: {cpu_avg / gpu_avg:.1f}x")
```

**Expected**: CPU ~2000–3000 ms, GPU ~50–200 ms, speedup 10–50×.

If speedup < 5×, something is wrong (driver, CUDA compute capability,
or float64 emulation on consumer GPU).

## 3. Full Fe(III) calcXAS_cached on GPU

```python
from multitorch.api.calc import calcXAS_cached, preload_fixture
import torch, time

cache = preload_fixture('Fe', 'iii', 'oh')
print(f"Fixture preloaded; ban xham[0].values={cache.ban.xham[0].values}")

# CPU baseline
with torch.no_grad():
    _ = calcXAS_cached(cache, cf={'tendq': 1.5}, slater=0.85,
                       soc=1.0, beam_fwhm=0.2, gamma1=0.3, gamma2=0.5)
    t0 = time.perf_counter()
    _ = calcXAS_cached(cache, cf={'tendq': 1.5}, slater=0.85,
                       soc=1.0, beam_fwhm=0.2, gamma1=0.3, gamma2=0.5)
    cpu_t = time.perf_counter() - t0
print(f"Fe(III) calcXAS_cached on CPU: {cpu_t:.2f} s")

# GPU
with torch.no_grad():
    _ = calcXAS_cached(cache, cf={'tendq': 1.5}, slater=0.85,
                       soc=1.0, beam_fwhm=0.2, gamma1=0.3, gamma2=0.5,
                       device='cuda')   # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = calcXAS_cached(cache, cf={'tendq': 1.5}, slater=0.85,
                       soc=1.0, beam_fwhm=0.2, gamma1=0.3, gamma2=0.5,
                       device='cuda')
    torch.cuda.synchronize()
    gpu_t = time.perf_counter() - t0
print(f"Fe(III) calcXAS_cached on GPU: {gpu_t:.2f} s")
print(f"Speedup: {cpu_t / gpu_t:.1f}x")
```

**Expected**: CPU ~25 sec (matches Mac measurements), GPU ~0.5–2.5 sec,
speedup 10–50×.

If CPU on exxa is much faster than 25 sec (say <5 sec), MKL or some
build difference is doing the work for us — note that and skip
the GPU push for now.

## 4. Autograd-on-GPU sanity check

The fitting machinery needs gradients to flow through the CUDA path.
Confirm with a finite-difference parity check:

```python
import torch
from multitorch.api.calc import calcXAS_cached, preload_fixture
cache = preload_fixture('Fe', 'iii', 'oh')

slater = torch.tensor(0.85, requires_grad=True, dtype=torch.float64,
                       device='cuda')
soc    = torch.tensor(1.0,  requires_grad=True, dtype=torch.float64,
                       device='cuda')
x, y = calcXAS_cached(cache, cf={'tendq': 1.5}, slater=slater, soc=soc,
                       beam_fwhm=0.2, gamma1=0.3, gamma2=0.5,
                       device='cuda')
loss = (y ** 2).sum()
loss.backward()
print(f"slater.grad on cuda: {slater.grad.item():.4e}")
print(f"soc.grad    on cuda: {soc.grad.item():.4e}")
assert torch.isfinite(slater.grad).all() and torch.isfinite(soc.grad).all()
```

Both gradients must be finite and non-zero. If NaN/Inf, the cuSOLVER
eigh is degenerating somewhere — open an issue, don't proceed.

## 5. End-to-end GPU fit

If 2/3/4 all pass, run the v0 fitter for FeIIIPc on GPU. Edit
`multiplets/fits/fe_xas_fit.py:FeXASModel` to pass `device='cuda'`
to `calcXAS_cached`, then:

```bash
cd <workspace>
PYTHONPATH=multitorch /opt/anaconda3/envs/multi/bin/python \
    fits/fe_xas_fit.py FeIIIPc-Cl-SR-SC-bg-remov.txt --n-iter 200 --lr 0.01
```

**Expected wall time**:
- CPU baseline: ~3 hours (untested — reasoned from the per-call timing)
- GPU: ~10–20 minutes if 10× speedup, ~3–5 minutes if 30×

Final loss should land in the 0.05–0.10 range (Oh approximation
ceiling — D4h needed for tighter fit, see PORT_NOTES BUG-002).

## 6. Reporting

Write a brief report including:

- Workstation specs (GPU model + CUDA version + driver)
- CPU timings from steps 2 and 3
- GPU timings from steps 2 and 3
- Speedup ratios
- Whether autograd flowed correctly (step 4)
- Final loss + parameters from step 5
- Total wall time for the FeIIIPc-Cl Adam fit on GPU

If you have access to the research_workspace card system, emit a
`bench-gpu-eigh-{timestamp}` card with the metrics. Otherwise, just
paste the numbers into a comment on the next user check-in.

## 7. If P0 confirms (≥ 10× speedup)

**Land the change as a multitorch commit** that defaults to GPU when
available. Suggested edit at `multitorch/api/calc.py` calcXAS_cached
signature:

```python
device: str = "cpu",  # change to ...
device: str = None,   # auto-select via device_utils.suggest_device_for_xas()
```

`multitorch/device_utils.py:suggest_device_for_xas` already exists
and does the right thing (use CUDA if matrix dim ≥ some threshold).

Then the v0 fitter's edits in step 5 become unnecessary — the cached
path defaults to GPU automatically.

## 8. If P0 does NOT confirm (< 5× speedup)

Don't refactor. The bottleneck is elsewhere (LAPACK MKL on the
workstation might already be near-optimal for these matrix sizes).
Move to P3 (Lanczos partial eigh) which has different scaling
characteristics and might still help.

## Local-side context (read once)

- v0 fitter: `multiplets/fits/fe_xas_fit.py` — works on Fe(II)Pc,
  not Fe(III) due to the perf issue we're investigating
- Diagnostics: `multiplets/fits/results/v0_diagnostics.md`
- Code review with full P0–P5 plan: `multitorch/CODE_REVIEW.md`
  (gitignored, so check it out from your local Mac if you don't see
  it on exxa — or read the addendum in PORT_NOTES.md instead)
- D4h dispatcher (Phase 1c interim): committed at `e3c39c4`. Cosine
  0.978 vs nid8; DS block silently empty (BUG follow-up in PORT_NOTES)
- Test data: `multiplets/test-data/Fe*.txt` (5 spectra, 240 pts each)

---

## RESULTS — exxa run, 2026-05-03

**Workstation:** AMD Ryzen Threadripper, 2× NVIDIA GeForce RTX 4090 (24 GB each)
**Torch:** 2.5.1+cu121 | CUDA: 12.1 | Driver: 535.288.01

### Step 2 — eigh microbenchmark (190×190 fp64, 10 runs)

| Backend | Avg time |
|---------|----------|
| CPU (MKL) | **5.25 ms** |
| GPU (RTX 4090) | **4.79 ms** |
| **Speedup** | **1.10×** |

Takeaway: MKL on Threadripper solves a 190×190 fp64 symmetric
eigenproblem in ~5 ms — ~400× faster than the Mac CPU baseline that
motivated the handoff. No advantage from GPU at this matrix size.

### Step 3 — Fe(III) calcXAS_cached end-to-end (warmup + 1 timed call)

| Backend | Wall time |
|---------|-----------|
| CPU (MKL) | **7.366 s** |
| GPU (RTX 4090) | **1.927 s** |
| **Speedup** | **3.82×** |

Note: Mac baseline was 25.4 s (89% in eigh). On this workstation the
full forward is only 7.4 s on CPU — MKL already dominates.

### Step 4 — Autograd GPU sanity check

**CORRECTION (2026-05-03, follow-up SSH session):** The original report
that "calcXAS_cached does not preserve the autograd graph" was a
**false alarm**. Re-running the focused diagnostic
`bench/diag_cuda_autograd.py` (committed b6377c6, results saved at
`bench/results_cuda_autograd_diag.txt`) confirms autograd works
correctly on all four tested configurations:

- CPU cache + CPU inputs + no device kwarg: y.grad_fn=AddBackward0,
  slater.grad=6.1212 (finite)
- CPU cache + CPU inputs + device='cpu': identical to above
- **CPU cache + CUDA inputs + device='cuda'**: y.grad_fn=AddBackward0,
  slater.grad=6.1212 (matches CPU to 1e-9)
- CPU cache + CPU inputs + device='cuda' (mismatch): handled
  gracefully, autograd intact

The GPU autograd path is FULLY FUNCTIONAL.

GPU configs ran ~3× faster than CPU at the full-fit level (2.79s vs
8.0s) — consistent with the 3.82× full-pipeline speedup measured
earlier. So GPU IS useful for fitting workloads even if the eigh
microbench was only 1.10×.

### Step 5 — End-to-end fit

NOT RUN. Speedup is 3.82× (<5× threshold) so step 5 is not warranted.

### Decision (revised 2026-05-03)

**P0 partially confirms. GPU is worth using EXPLICITLY for Fe(III)
fits (3.8× full-pipeline speedup), but DON'T change the global
`device='cpu'` default** — small fixtures (Ni d8, Co d7) are slower
on GPU than CPU.

Per `device_utils.suggest_device_for_xas()` already exists in
multitorch — it picks GPU only when matrix dim is large enough to
benefit. The right move is to wire `device=None` → suggest_device_for_xas
default in `calcXAS_cached`, NOT to hard-code device='cuda'.

For the v0 fitter sweep (`bench/AGENT_HANDOFF_V0_FITS.md`): the
runbook now passes `device='cuda'` explicitly because we know all
5 spectra in the test set are large-fixture cases. ~30 min total
sweep wall on workstation GPU vs ~80 min CPU.

Relevant context for next agent:
- At batch=1, the full bench suite shows GPU helps only for larger-dim
  fixtures: fe3_d5_oh 3.64×, mn2_d5_oh 3.62×, v3_d2_oh 3.13×.
  GPU hurts small-dim fixtures (ni2, co2, fe2, ti4).
- `suggest_device_for_xas()` already exists in `device_utils.py` and
  could be used for auto-selection once P3 or another optimization
  makes the GPU path universally faster.
- autograd broken in calcXAS_cached (see step 4) — track as separate bug.
