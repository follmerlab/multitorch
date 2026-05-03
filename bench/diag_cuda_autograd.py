"""Focused diagnostic for the CUDA-autograd issue in calcXAS_cached.

Tests four configurations and reports y.grad_fn + propagated gradient
at each stage, so we can pinpoint where (if anywhere) the graph severs.

Local Mac CPU result (verified 2026-05-03):
  config 1 (cache=cpu, inputs=cpu, no device kwarg): y.grad_fn=AddBackward0, grads finite

Run on exxa with:
    cd <workspace>/multitorch
    git pull
    /opt/anaconda3/envs/multi/bin/python -u bench/diag_cuda_autograd.py \\
        > /tmp/cuda_autograd_diag.log 2>&1

Then push the log back via:
    git add bench/results_cuda_autograd_diag.txt
    git commit -m "exxa: cuda autograd diagnostic results"
    git push
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import torch

from multitorch.api.calc import calcXAS_cached, preload_fixture


def report(name: str, fn):
    print(f"\n=== {name} ===", flush=True)
    try:
        t0 = time.perf_counter()
        fn()
        print(f"  [PASS] wall = {time.perf_counter() - t0:.2f} s", flush=True)
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()


def cfg1_cpu_cpu_no_device():
    """Baseline: cache CPU, inputs CPU, no device kwarg."""
    cache = preload_fixture("Fe", "iii", "oh")
    slater = torch.tensor(0.85, requires_grad=True, dtype=torch.float64)
    soc = torch.tensor(1.0, requires_grad=True, dtype=torch.float64)
    x, y = calcXAS_cached(cache, cf={"tendq": 1.5}, slater=slater, soc=soc,
                          beam_fwhm=0.2, gamma1=0.3, gamma2=0.5)
    print(f"  x.device={x.device}  y.device={y.device}")
    print(f"  y.grad_fn={y.grad_fn}")
    print(f"  y.requires_grad={y.requires_grad}")
    loss = (y ** 2).sum()
    loss.backward()
    print(f"  slater.grad={slater.grad}")
    print(f"  soc.grad={soc.grad}")
    assert torch.isfinite(slater.grad).all() and slater.grad.item() != 0, \
        "slater.grad should be finite and nonzero"


def cfg2_cpu_cpu_device_cpu():
    """cache CPU, inputs CPU, device='cpu' explicit."""
    cache = preload_fixture("Fe", "iii", "oh")
    slater = torch.tensor(0.85, requires_grad=True, dtype=torch.float64)
    soc = torch.tensor(1.0, requires_grad=True, dtype=torch.float64)
    x, y = calcXAS_cached(cache, cf={"tendq": 1.5}, slater=slater, soc=soc,
                          beam_fwhm=0.2, gamma1=0.3, gamma2=0.5, device="cpu")
    print(f"  x.device={x.device}  y.device={y.device}")
    print(f"  y.grad_fn={y.grad_fn}")
    loss = (y ** 2).sum()
    loss.backward()
    print(f"  slater.grad={slater.grad}")
    assert torch.isfinite(slater.grad).all() and slater.grad.item() != 0


def cfg3_cpu_cuda_cuda():
    """cache CPU, inputs CUDA, device='cuda' — the SUSPECT case."""
    if not torch.cuda.is_available():
        print("  SKIP: no CUDA"); return
    cache = preload_fixture("Fe", "iii", "oh")  # CPU cache
    slater = torch.tensor(0.85, requires_grad=True, dtype=torch.float64,
                          device="cuda")
    soc = torch.tensor(1.0, requires_grad=True, dtype=torch.float64,
                       device="cuda")
    x, y = calcXAS_cached(cache, cf={"tendq": 1.5}, slater=slater, soc=soc,
                          beam_fwhm=0.2, gamma1=0.3, gamma2=0.5, device="cuda")
    print(f"  x.device={x.device}  y.device={y.device}")
    print(f"  y.grad_fn={y.grad_fn}")
    print(f"  y.requires_grad={y.requires_grad}")
    if y.grad_fn is None:
        print("  *** y.grad_fn IS NONE — graph severed somewhere ***")
        print("  Confirming severance — backward should error or do nothing")
    loss = (y ** 2).sum()
    print(f"  loss.grad_fn={loss.grad_fn}")
    try:
        loss.backward()
        print(f"  slater.grad={slater.grad}  (cuda path backward succeeded)")
    except RuntimeError as e:
        print(f"  loss.backward() raised: {e}")
        return
    if slater.grad is None:
        print(f"  slater.grad is None — graph definitely severed")
    else:
        print(f"  slater.grad={slater.grad}  finite={torch.isfinite(slater.grad).all().item()}")


def cfg4_cpu_only_with_device_cuda():
    """cache CPU, inputs CPU, device='cuda' — what happens with mismatched inputs?"""
    if not torch.cuda.is_available():
        print("  SKIP: no CUDA"); return
    cache = preload_fixture("Fe", "iii", "oh")
    slater = torch.tensor(0.85, requires_grad=True, dtype=torch.float64)  # CPU
    soc = torch.tensor(1.0, requires_grad=True, dtype=torch.float64)  # CPU
    try:
        x, y = calcXAS_cached(cache, cf={"tendq": 1.5}, slater=slater, soc=soc,
                              beam_fwhm=0.2, gamma1=0.3, gamma2=0.5,
                              device="cuda")
        print(f"  x.device={x.device}  y.device={y.device}")
        print(f"  y.grad_fn={y.grad_fn}")
    except RuntimeError as e:
        print(f"  RuntimeError as expected (device mismatch): {e}")


def main():
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    print()

    report("config 1: cache=cpu, inputs=cpu, no device kwarg", cfg1_cpu_cpu_no_device)
    report("config 2: cache=cpu, inputs=cpu, device='cpu'", cfg2_cpu_cpu_device_cpu)
    report("config 3: cache=cpu, inputs=cuda, device='cuda' (SUSPECT)", cfg3_cpu_cuda_cuda)
    report("config 4: cache=cpu, inputs=cpu, device='cuda' (mismatch)", cfg4_cpu_only_with_device_cuda)

    print("\n=== SUMMARY ===")
    print("If config 3 shows y.grad_fn=None or slater.grad=None,")
    print("the CUDA-autograd bug is real. Trace the cached path for")
    print("a `.detach()` or non-leaf-preserving `.to()` — likely in")
    print("build_cowan_store_in_memory or assemble_and_diagonalize_in_memory.")


if __name__ == "__main__":
    sys.exit(main())
