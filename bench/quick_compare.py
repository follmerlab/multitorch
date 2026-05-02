#!/usr/bin/env python3
"""Quick timing comparison: Fortran vs PyTorch CPU vs CUDA."""
import time, sys, os, json
import numpy as np

os.environ['ttmult'] = '/home/afollmer/code/ttmult'
sys.path.insert(0, '/home/afollmer/code/pyctm')
sys.path.insert(0, '/home/afollmer/code/pyttmult')

test_cases = [
    ('Ti', 'iv', 'oh', 1.0, 'Ti4+ d0 Oh'),
    ('V', 'iii', 'oh', 1.0, 'V3+ d2 Oh'),
    ('Cr', 'iii', 'oh', 1.0, 'Cr3+ d3 Oh'),
    ('Mn', 'ii', 'oh', 1.0, 'Mn2+ d5 Oh'),
    ('Fe', 'iii', 'oh', 1.0, 'Fe3+ d5 Oh'),
    ('Co', 'ii', 'oh', 1.0, 'Co2+ d7 Oh'),
    ('Ni', 'ii', 'oh', 1.0, 'Ni2+ d8 Oh'),
    ('Cu', 'ii', 'oh', 1.0, 'Cu2+ d9 Oh'),
]

results = []

for elem, val, sym, tendq, label in test_cases:
    print(f"\n=== {label} ===", flush=True)
    row = {'label': label}

    # PyTorch from_scratch CPU
    try:
        import torch
        from multitorch import calcXAS_from_scratch
        calcXAS_from_scratch(elem, val, cf={'tendq': tendq}, device='cpu', nbins=2000)
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            calcXAS_from_scratch(elem, val, cf={'tendq': tendq}, device='cpu', nbins=2000)
            times.append(time.perf_counter() - t0)
        row['scratch_cpu'] = round(sorted(times)[1], 4)
        print(f"  scratch CPU: {row['scratch_cpu']}s", flush=True)
    except Exception as e:
        row['scratch_cpu'] = f"ERR: {e}"
        print(f"  scratch CPU: ERR {e}", flush=True)

    # PyTorch from_scratch CUDA
    try:
        calcXAS_from_scratch(elem, val, cf={'tendq': tendq}, device='cuda', nbins=2000)
        torch.cuda.synchronize()
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            calcXAS_from_scratch(elem, val, cf={'tendq': tendq}, device='cuda', nbins=2000)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        row['scratch_gpu'] = round(sorted(times)[1], 4)
        print(f"  scratch GPU: {row['scratch_gpu']}s", flush=True)
    except Exception as e:
        row['scratch_gpu'] = f"ERR: {e}"
        print(f"  scratch GPU: ERR {e}", flush=True)

    # PyTorch cached CPU
    try:
        from multitorch import preload_fixture, calcXAS_cached
        cache = preload_fixture(elem, val, sym)
        calcXAS_cached(cache, slater=0.8, soc=1.0, nbins=2000, device='cpu')
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            calcXAS_cached(cache, slater=0.8, soc=1.0, nbins=2000, device='cpu')
            times.append(time.perf_counter() - t0)
        row['cached_cpu'] = round(sorted(times)[2], 4)
        print(f"  cached CPU:  {row['cached_cpu']}s", flush=True)
    except Exception as e:
        row['cached_cpu'] = f"ERR: {e}"
        print(f"  cached CPU: ERR {e}", flush=True)

    # PyTorch cached CUDA
    try:
        calcXAS_cached(cache, slater=0.8, soc=1.0, nbins=2000, device='cuda')
        torch.cuda.synchronize()
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            calcXAS_cached(cache, slater=0.8, soc=1.0, nbins=2000, device='cuda')
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        row['cached_gpu'] = round(sorted(times)[2], 4)
        print(f"  cached GPU:  {row['cached_gpu']}s", flush=True)
    except Exception as e:
        row['cached_gpu'] = f"ERR: {e}"
        print(f"  cached GPU: ERR {e}", flush=True)

    # Fortran (pyctm.calc.calcXAS — full pipeline)
    try:
        import tempfile, shutil
        from pathlib import Path
        from pyctm.calc import calcXAS
        # Edge map
        edge = 'l'  # lowercase 'l' for L-edge
        disk0_src = '/home/afollmer/code/ttmult/inputs/disk0'
        # Warm
        with tempfile.TemporaryDirectory(prefix="bench_") as tmp:
            shutil.copy(disk0_src, os.path.join(tmp, 'disk0'))
            cwd_prev = os.getcwd()
            os.chdir(tmp)
            try:
                calcXAS('warm', elem, val, sym, edge, cf={'tendq': tendq},
                        slater=0.8, soc=1.0, save=False, get=True, run=True, verbose=False)
            finally:
                os.chdir(cwd_prev)
        # Timed
        times = []
        for i in range(3):
            with tempfile.TemporaryDirectory(prefix="bench_") as tmp:
                shutil.copy(disk0_src, os.path.join(tmp, 'disk0'))
                cwd_prev = os.getcwd()
                os.chdir(tmp)
                try:
                    t0 = time.perf_counter()
                    calcXAS(f'run{i}', elem, val, sym, edge, cf={'tendq': tendq},
                            slater=0.8, soc=1.0, save=False, get=True, run=True, verbose=False)
                    times.append(time.perf_counter() - t0)
                finally:
                    os.chdir(cwd_prev)
        row['fortran'] = round(sorted(times)[1], 4)
        print(f"  Fortran:     {row['fortran']}s", flush=True)
    except Exception as e:
        import traceback
        row['fortran'] = f"ERR: {e}"
        print(f"  Fortran: ERR {e}", flush=True)
        traceback.print_exc()

    results.append(row)

print("\n\nJSON:")
print(json.dumps(results, indent=2))
