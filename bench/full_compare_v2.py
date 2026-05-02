#!/usr/bin/env python3
"""Focused timing comparison: Fortran vs PyTorch CPU vs CUDA.

Strategy: measure single-spectrum first, then extrapolate which batch sizes are feasible.
Timeout per cell = 60s.
"""
import time, sys, os, json, tempfile, shutil, signal
import numpy as np

os.environ['ttmult'] = '/home/afollmer/code/ttmult'
sys.path.insert(0, '/home/afollmer/code/pyctm')
sys.path.insert(0, '/home/afollmer/code/pyttmult')

import torch
from multitorch import calcXAS_from_scratch, preload_fixture, calcXAS_cached
from pyctm.calc import calcXAS as fortran_calcXAS

DISK0 = '/home/afollmer/code/ttmult/inputs/disk0'
CELL_TIMEOUT = 60  # seconds per benchmark cell

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Cell timed out")

def time_fn(fn, n=3, warmup=1, timeout=CELL_TIMEOUT):
    """Time a function, return median. Raises TimeoutError if too slow."""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
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
    finally:
        signal.alarm(0)

def time_fortran(elem, val, sym, tendq, n=3):
    """Time Fortran pipeline."""
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
    return time_fn(run_once, n=n)

# ── Test cases ──
test_cases = [
    # (element, valence, sym, tendq, label)
    ('Ti', 'iv', 'oh', 1.0, 'Ti4+ d0 Oh'),
    ('V', 'iii', 'oh', 1.0, 'V3+ d2 Oh'),
    ('Cr', 'iii', 'oh', 1.0, 'Cr3+ d3 Oh'),
    ('Mn', 'ii', 'oh', 1.0, 'Mn2+ d5 Oh'),
    ('Co', 'ii', 'oh', 1.0, 'Co2+ d7 Oh'),
    ('Ni', 'ii', 'oh', 1.0, 'Ni2+ d8 Oh'),
    ('Cu', 'ii', 'oh', 1.0, 'Cu2+ d9 Oh'),
]

results = []

for elem, val, sym, tendq, label in test_cases:
    print(f"\n{'='*60}", flush=True)
    print(f"  {label}", flush=True)
    print(f"{'='*60}", flush=True)
    
    row = {'label': label, 'element': elem, 'valence': val}
    
    # ── Fortran (single spectrum) ──
    try:
        t = time_fortran(elem, val, sym, tendq, n=3)
        row['fortran_1'] = round(t, 5)
        print(f"  Fortran     b=1 :  {t:.4f}s", flush=True)
    except TimeoutError:
        row['fortran_1'] = 'timeout'
        print(f"  Fortran     b=1 :  TIMEOUT", flush=True)
    except Exception as e:
        row['fortran_1'] = None
        print(f"  Fortran     b=1 :  ERR {e}", flush=True)
    
    # ── Preload fixture ──
    cache = None
    try:
        cache = preload_fixture(elem, val, sym)
    except Exception as e:
        print(f"  preload_fixture: ERR {e}", flush=True)
    
    # ── Single spectrum: cached CPU vs CUDA ──
    for device in ['cpu', 'cuda']:
        key = f'cached_1_{device}'
        if cache is None:
            row[key] = None
            continue
        try:
            fn = lambda d=device: calcXAS_cached(cache, slater=0.8, soc=1.0, nbins=2000, device=d)
            t = time_fn(fn, n=5, warmup=1)
            row[key] = round(t, 5)
            print(f"  cached      b=1  {device:4s}:  {t:.4f}s", flush=True)
        except TimeoutError:
            row[key] = 'timeout'
            print(f"  cached      b=1  {device:4s}:  TIMEOUT", flush=True)
        except Exception as e:
            row[key] = None
            print(f"  cached      b=1  {device:4s}:  ERR {e}", flush=True)
    
    # ── Batched: only if single-spectrum is fast enough ──
    for batch in [10, 100]:
        for device in ['cpu', 'cuda']:
            key = f'cached_{batch}_{device}'
            single_key = f'cached_1_{device}'
            single_t = row.get(single_key)
            
            # Skip if single already timed out or errored, or estimated batch > timeout
            if single_t is None or single_t == 'timeout':
                row[key] = None
                continue
            estimated = single_t * batch
            if estimated > CELL_TIMEOUT * 0.8:
                row[key] = 'skipped'
                print(f"  cached      b={batch:<3d} {device:4s}:  SKIP (est {estimated:.0f}s)", flush=True)
                continue

            if cache is None:
                row[key] = None
                continue
            try:
                fn = lambda b=batch, d=device: [calcXAS_cached(cache, slater=0.8, soc=1.0, nbins=2000, device=d) for _ in range(b)]
                t = time_fn(fn, n=3, warmup=1)
                row[key] = round(t, 5)
                per = t / batch
                speedup_vs_cpu = ''
                cpu_key = f'cached_{batch}_cpu'
                if cpu_key in row and isinstance(row[cpu_key], (int, float)):
                    speedup_vs_cpu = f'  (CPU/GPU={row[cpu_key]/t:.2f}x)'
                print(f"  cached      b={batch:<3d} {device:4s}:  {t:.4f}s  ({per:.5f}s/spec){speedup_vs_cpu}", flush=True)
            except TimeoutError:
                row[key] = 'timeout'
                print(f"  cached      b={batch:<3d} {device:4s}:  TIMEOUT", flush=True)
            except Exception as e:
                row[key] = None
                print(f"  cached      b={batch:<3d} {device:4s}:  ERR {e}", flush=True)
    
    # ── from_scratch: only for small matrices ──
    cached_cpu = row.get('cached_1_cpu')
    if isinstance(cached_cpu, (int, float)) and cached_cpu < 5.0:
        for device in ['cpu', 'cuda']:
            key = f'scratch_{device}'
            try:
                fn = lambda d=device: calcXAS_from_scratch(elem, val, cf={'tendq': tendq}, device=d, nbins=2000)
                t = time_fn(fn, n=3, warmup=1)
                row[key] = round(t, 5)
                print(f"  scratch          {device:4s}:  {t:.4f}s", flush=True)
            except TimeoutError:
                row[key] = 'timeout'
                print(f"  scratch          {device:4s}:  TIMEOUT", flush=True)
            except Exception as e:
                row[key] = None
                print(f"  scratch          {device:4s}:  ERR {e}", flush=True)
    
    results.append(row)

# Save raw results
outpath = '/home/afollmer/code/bench/results/comparison.json'
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n\nResults saved to {outpath}", flush=True)

# ── Generate markdown table ──
md_lines = [
    "# GPU Acceleration Comparison: Fortran vs PyTorch CPU vs CUDA",
    "",
    f"**Platform**: 2× NVIDIA RTX 4090, AMD Ryzen, PyTorch 2.5.1+cu121",
    f"**Date**: {time.strftime('%Y-%m-%d')}",
    "",
    "## Single-Spectrum Timing (batch=1)",
    "",
    "| Ion | Fortran | cached CPU | cached CUDA | GPU speedup | scratch CPU | scratch CUDA |",
    "| --- | ------: | ---------: | ----------: | ----------: | ----------: | -----------: |",
]

for r in results:
    def fmt(v):
        if v is None: return '—'
        if v == 'timeout': return '>60s'
        if v == 'skipped': return 'skip'
        return f'{v:.4f}s'
    
    cpu = r.get('cached_1_cpu')
    gpu = r.get('cached_1_cuda')
    if isinstance(cpu, (int, float)) and isinstance(gpu, (int, float)) and gpu > 0:
        speedup = f'{cpu/gpu:.2f}×'
    else:
        speedup = '—'
    
    md_lines.append(
        f"| {r['label']} | {fmt(r.get('fortran_1'))} | {fmt(r.get('cached_1_cpu'))} "
        f"| {fmt(r.get('cached_1_cuda'))} | {speedup} "
        f"| {fmt(r.get('scratch_cpu'))} | {fmt(r.get('scratch_cuda'))} |"
    )

md_lines.extend([
    "",
    "## Batched Timing (loop of N spectra)",
    "",
    "| Ion | batch | cached CPU | cached CUDA | GPU speedup | per-spec CPU | per-spec CUDA |",
    "| --- | ----: | ---------: | ----------: | ----------: | -----------: | ------------: |",
])

for r in results:
    for batch in [10, 100]:
        cpu = r.get(f'cached_{batch}_cpu')
        gpu = r.get(f'cached_{batch}_cuda')
        
        # Skip if both are None/skipped
        if cpu in (None, 'skipped') and gpu in (None, 'skipped'):
            continue
        
        def fmt(v):
            if v is None: return '—'
            if v == 'timeout': return '>60s'
            if v == 'skipped': return 'skip'
            return f'{v:.4f}s'
        
        if isinstance(cpu, (int, float)) and isinstance(gpu, (int, float)) and gpu > 0:
            speedup = f'{cpu/gpu:.2f}×'
        else:
            speedup = '—'
        
        per_cpu = f'{cpu/batch:.5f}s' if isinstance(cpu, (int, float)) else '—'
        per_gpu = f'{gpu/batch:.5f}s' if isinstance(gpu, (int, float)) else '—'
        
        md_lines.append(
            f"| {r['label']} | {batch} | {fmt(cpu)} | {fmt(gpu)} | {speedup} | {per_cpu} | {per_gpu} |"
        )

md_lines.extend([
    "",
    "## Key Findings",
    "",
])

md_path = '/home/afollmer/code/bench/results/gpu_comparison.md'
with open(md_path, 'w') as f:
    f.write('\n'.join(md_lines) + '\n')
print(f"Markdown saved to {md_path}", flush=True)
print("Done!", flush=True)
