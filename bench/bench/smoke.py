"""One-cell end-to-end smoke: confirm adapters produce (x, y) and timings land in JSONL.

Usage:
    PYTHONPATH=/path/to/bench python -m bench.smoke

This is step 1 of the implementation plan — it is NOT a full matrix
run. It validates that the adapters import, cold-start, warm, and run
a single cell (ni2_d8_oh, XAS, batch=1, CPU) against multitorch_cached
and pyctm, writing a JSONL record to bench/results/smoke.jsonl.

After this passes, we move on to parity.py + a real matrix.
"""
from __future__ import annotations

import datetime
import sys
import traceback
from pathlib import Path

from bench.adapters.multitorch_adapter import MultitorchAdapter
from bench.adapters.pyctm_adapter import PyctmAdapter
from bench.artifacts import append_record
from bench.config import BENCH_ROOT, BenchCell
from bench.harness import RepStats, wall_timer


REPS = 3
OUT = BENCH_ROOT / "results" / "smoke.jsonl"


def run_adapter(adapter, cell: BenchCell) -> dict:
    """Return a structured record covering cold-start, warm, reps, output shape."""
    record = {
        "record_id": cell.record_id(),
        "impl": adapter.name,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config": {
            "impl": adapter.name,
            "fixture": cell.fixture,
            "calctype": cell.calctype,
            "batch_size": cell.batch_size,
            "nbins": cell.nbins,
            "device": cell.device,
            "reps": cell.reps,
        },
        "status": "ok",
        "error": None,
        "timings": {},
        "output": {},
    }

    try:
        print(f"  [{adapter.name}] cold_start ...", flush=True)
        with wall_timer() as t_cold:
            adapter.cold_start()
        record["timings"]["cold_start_s"] = t_cold.elapsed

        print(f"  [{adapter.name}] warm ...", flush=True)
        with wall_timer() as t_warm:
            adapter.warm(cell)
        record["timings"]["warm_s"] = t_warm.elapsed

        print(f"  [{adapter.name}] reps ({cell.reps}) ...", flush=True)
        wall_samples = []
        cpu_samples = []
        last_out = None
        for _ in range(cell.reps):
            with wall_timer() as t:
                last_out = adapter.run(cell)
            wall_samples.append(t.elapsed)
            cpu_samples.append(t.cpu_elapsed)

        record["timings"]["wall_s"] = RepStats(wall_samples).as_dict()
        record["timings"]["cpu_s"] = RepStats(cpu_samples).as_dict()
        record["output"] = {
            "x_min": float(last_out.x.min()),
            "x_max": float(last_out.x.max()),
            "n_points": int(last_out.x.shape[0]),
            "y_sample": [float(v) for v in last_out.y[:8]],
        }
        record["adapter_meta"] = last_out.meta

    except Exception as e:
        record["status"] = "error"
        record["error"] = f"{type(e).__name__}: {e}"
        record["traceback"] = traceback.format_exc()

    return record


def main():
    print(f"Smoke test — writing to {OUT}")
    cell = BenchCell(
        impl="smoke",
        fixture="ni2_d8_oh",
        calctype="xas",
        batch_size=1,
        nbins=500,
        device="cpu",
        reps=REPS,
    )
    print(f"Cell: {cell.fixture} XAS batch={cell.batch_size} device={cell.device} reps={cell.reps}")

    for factory in [
        lambda: MultitorchAdapter(mode="cached"),
        lambda: PyctmAdapter(),
    ]:
        try:
            adapter = factory()
        except Exception as e:
            print(f"  FAILED to construct adapter: {e}")
            continue
        rec = run_adapter(adapter, cell)
        append_record(OUT, rec)
        status = rec["status"]
        if status == "ok":
            wall = rec["timings"]["wall_s"]
            print(
                f"  [{rec['impl']}] OK  cold={rec['timings']['cold_start_s']:.3f}s  "
                f"warm={rec['timings']['warm_s']:.3f}s  "
                f"median={wall['median']*1000:.2f}ms  "
                f"IQR={wall['iqr']*1000:.2f}ms  "
                f"n_points={rec['output']['n_points']}"
            )
        else:
            print(f"  [{rec['impl']}] ERROR {rec['error']}")

    print(f"\nDone. Records in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
