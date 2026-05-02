"""4-way smoke: multitorch_cached / pyctm / pyttmult / ttmult_raw on one Oh cell.

Runs each adapter through cold_start → warm → N reps and reports:
  - median / IQR timing for each
  - stage breakdown where the adapter exposes it
  - parity of each Fortran-side impl against ttmult_raw (the reference)
"""
from __future__ import annotations

import sys

import numpy as np

from bench.adapters.multitorch_adapter import MultitorchAdapter
from bench.adapters.pyctm_adapter import PyctmAdapter
from bench.adapters.pyttmult_adapter import PyttmultAdapter
from bench.adapters.ttmult_raw_adapter import TtmultRawAdapter
from bench.config import BenchCell
from bench.harness import RepStats, wall_timer
from bench.parity import compare


REPS = 3


def run_one(adapter, cell):
    print(f"\n[{adapter.name}]", flush=True)
    with wall_timer() as t_cold:
        adapter.cold_start()
    with wall_timer() as t_warm:
        adapter.warm(cell)
    wall = []
    last = None
    for _ in range(cell.reps):
        with wall_timer() as t:
            last = adapter.run(cell)
        wall.append(t.elapsed)
    stats = RepStats(wall)
    print(f"  cold={t_cold.elapsed:.3f}s  warm={t_warm.elapsed:.3f}s")
    print(f"  median={stats.median*1000:.2f}ms  IQR={stats.iqr*1000:.2f}ms")
    if last.meta:
        breakdown = []
        for k, v in last.meta.items():
            if isinstance(v, float) and k.endswith("_s"):
                breakdown.append(f"{k}={v*1000/cell.reps:.2f}ms")
        if breakdown:
            print(f"  per-rep breakdown: {' '.join(breakdown)}")
    return last, stats


def main():
    cell = BenchCell(
        impl="4way",
        fixture="ni2_d8_oh",
        calctype="xas",
        batch_size=1,
        nbins=2000,
        device="cpu",
        reps=REPS,
    )
    print(f"4-way smoke on {cell.fixture} XAS reps={cell.reps}")

    adapters = [
        MultitorchAdapter(mode="cached"),
        TtmultRawAdapter(),
        PyttmultAdapter(),
        PyctmAdapter(),
    ]
    results = {}
    for a in adapters:
        try:
            out, stats = run_one(a, cell)
            results[a.name] = (out, stats)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    # Parity vs ttmult_raw
    if "ttmult_raw" not in results:
        print("\nNo ttmult_raw reference — skipping parity.")
        return 1
    x_ref, y_ref = results["ttmult_raw"][0].x, results["ttmult_raw"][0].y
    print("\n=== Parity vs ttmult_raw ===")
    for name, (out, _) in results.items():
        if name == "ttmult_raw":
            continue
        p = compare(out.x, out.y, x_ref, y_ref, calctype="xas")
        print(
            f"  {name:30s} cosine={p.cosine:.4f}  peak_shift={p.peak_shift_ev:+6.2f}eV  "
            f"peak_err_max={p.peak_err_max_ev:.3f}eV  max|Δ|={p.max_abs_diff:.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
