#!/usr/bin/env python3
"""Comprehensive timing comparison: Fortran vs PyTorch CPU vs CUDA.

Tests single-spectrum AND batched scenarios to find where GPU actually helps.
Outputs JSON results for markdown table generation.
"""
import time, sys, os, json, tempfile, shutil
import numpy as np

os.environ['ttmult'] = '/home/afollmer/code/ttmult'
sys.path.insert(0, '/home/afollmer/code/pyctm')
sys.path.insert(0, '/home/afollmer/code/pyttmult')

import torch
from multitorch import calcXAS_from_scratch, preload_fixture, calcXAS_cached

# Try importing batch API
try:
    from multitorch import calcXAS_batch
    HAS_BATCH = True
except ImportError:
    HAS_BATCH = False

# Fortran imports
from pyctm.calc import calcXAS as fortran_calcXAS

DISK0 = '/home/afollmer/code/ttmult/inputs/disk0'

def time_fn(fn, n=5, warmup=1):
    """Time a function, return median of n runs after warmup."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def time_fortran(elem, val, sym, tendq, n=3):
    """Time Fortran pipeline via pyctm.calc.calcXAS."""
    edge = 'l'
    def run_once():
        with tempfile.TemporaryDirectory(prefix="bench_") as tmp:
            shutil.copy(DISK0, os.path.join(tmp, 'disk0'))
            prev = os.getcwd()
            os.chdir(tmp)
            try:
                fortran_calcXAS('run', elem, val, sym, edge, cf={'tendq': tendq},
                                slater=0.8, soc=1.0, save=False, get=True, run=True, verbose=False)
            finally:
                os.chdir(prev)
    # Warmup
    run_once()
    # Timed
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        run_once()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


# ── Test cases ──
test_cases = [
    # (element, valence, sym, tendq, label, approx_hdim)
    ('Ti', 'iv', 'oh', 1.0, 'Ti4+ d0 Oh', 20),
    ('V', 'iii', 'oh', 1.0, 'V3+ d2 Oh', 1074),
    ('Cr', 'iii', 'oh', 1.0, 'Cr3+ d3 Oh', 1782),
    ('Mn', 'ii', 'oh', 1.0, 'Mn2+ d5 Oh', 2772),
    ('Fe', 'iii', 'oh', 1.0, 'Fe3+ d5 Oh', 2772),
    ('Co', 'ii', 'oh', 1.0, 'Co2+ d7 Oh', 500),
    ('Ni', 'ii', 'oh', 1.0, 'Ni2+ d8 Oh', 200),
    ('Cu', 'ii', 'oh', 1.0, 'Cu2+ d9 Oh', 50),
]

batch_sizes = [1, 10, 100]

results = []

for elem, val, sym, tendq, label, hdim in test_cases:
    print(f"\n{'='*60}", flush=True)
    print(f"  {label} (H_dim ≈ {hdim})", flush=True)
    print(f"{'='*60}", flush=True)
    
    row = {'label': label, 'element': elem, 'valence': val, 'hdim': hdim}
    
    # ── Fortran ──
    try:
        t = time_fortran(elem, val, sym, tendq, n=3)
        row['fortran'] = round(t, 5)
        print(f"  Fortran:           {t:.4f}s", flush=True)
    except Exception as e:
        row['fortran'] = None
        print(f"  Fortran: ERR {e}", flush=True)
    
    # ── PyTorch cached (preload once, then re-run) ──
    cache = None
    try:
        cache = preload_fixture(elem, val, sym)
    except Exception as e:
        print(f"  preload_fixture: ERR {e}", flush=True)
    
    for batch in batch_sizes:
        for device in ['cpu', 'cuda']:
            key = f'cached_b{batch}_{device}'
            if cache is None:
                row[key] = None
                continue
            try:
                if batch == 1:
                    fn = lambda: calcXAS_cached(cache, slater=0.8, soc=1.0, nbins=2000, device=device)
                else:
                    fn = lambda b=batch: [calcXAS_cached(cache, slater=0.8, soc=1.0, nbins=2000, device=device) for _ in range(b)]
                
                t = time_fn(fn, n=3 if batch <= 10 else 1, warmup=1)
                row[key] = round(t, 5)
                per_spec = t / batch
                print(f"  cached b={batch:>4d} {device:4s}: {t:.4f}s  ({per_spec:.5f}s/spec)", flush=True)
            except Exception as e:
                row[key] = None
                print(f"  cached b={batch:>4d} {device:4s}: ERR {e}", flush=True)
    
    # ── PyTorch batch API (true vectorized batching) ──
    if HAS_BATCH and cache is not None:
        for batch in [10, 100]:
            for device in ['cpu', 'cuda']:
                key = f'batch_b{batch}_{device}'
                try:
                    slater_v = torch.full((batch,), 0.8, dtype=torch.float64)
                    soc_v = torch.full((batch,), 1.0, dtype=torch.float64)
                    fn = lambda: calcXAS_batch(cache, slater_values=slater_v, soc_values=soc_v, nbins=2000, device=device)
                    t = time_fn(fn, n=3, warmup=1)
                    row[key] = round(t, 5)
                    per_spec = t / batch
                    print(f"  batch  b={batch:>4d} {device:4s}: {t:.4f}s  ({per_spec:.5f}s/spec)", flush=True)
                except Exception as e:
                    row[key] = None
                    print(f"  batch  b={batch:>4d} {device:4s}: ERR {e}", flush=True)
    
    # ── from_scratch (only batch=1, skip huge matrices) ──
    if hdim <= 1100:  # Skip Cr3+, Mn2+, Fe3+ which are >300s
        for device in ['cpu', 'cuda']:
            key = f'scratch_{device}'
            try:
                fn = lambda: calcXAS_from_scratch(elem, val, cf={'tendq': tendq}, device=device, nbins=2000)
                t = time_fn(fn, n=3, warmup=1)
                row[key] = round(t, 5)
                print(f"  scratch     {device:4s}: {t:.4f}s", flush=True)
            except Exception as e:
                row[key] = None
                print(f"  scratch     {device:4s}: ERR {e}", flush=True)
    
    results.append(row)
    
    # Skip remaining if single-spectrum cached CPU already >120s
    if row.get('cached_b1_cpu') and row['cached_b1_cpu'] > 120:
        print(f"  SKIPPING larger batches (too slow)", flush=True)

# Save results
with open('/home/afollmer/code/bench/results/comparison.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n\nResults saved to /home/afollmer/code/bench/results/comparison.json")
print(f"Tested {len(results)} ions")
