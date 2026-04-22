"""CLI harness: expand preset → BenchCell list → run each cell → append JSONL.

Usage:
    python -m bench.run_all --preset mvp --output results/mvp.jsonl --resume

Each cell runs in a dedicated multiprocessing.Process with a timeout
so a Fortran hang or CUDA OOM cannot take the driver down. Records
are appended atomically to JSONL. With --resume the driver reads the
existing JSONL and skips any record_id already marked status=ok.
"""
from __future__ import annotations

import argparse
import datetime
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

from bench.artifacts import append_record, read_completed_record_ids
from bench.config import (
    BENCH_ROOT, BenchCell, FIXTURE_BY_NAME, PRESETS, iter_preset,
)
from bench.hardware import collect_hardware, hardware_hash
from bench.harness import RepStats, wall_timer


# ─────────────────────────────────────────────────────────────
# Adapter factory — keeps imports lazy so run_all --help is fast
# ─────────────────────────────────────────────────────────────


def _build_adapter(impl: str):
    if impl == "multitorch_cached":
        from bench.adapters.multitorch_adapter import MultitorchAdapter
        return MultitorchAdapter(mode="cached")
    if impl == "multitorch_from_scratch":
        from bench.adapters.multitorch_adapter import MultitorchAdapter
        return MultitorchAdapter(mode="from_scratch")
    if impl == "multitorch_batch":
        from bench.adapters.multitorch_adapter import MultitorchAdapter
        return MultitorchAdapter(mode="batch")
    if impl == "pyctm":
        from bench.adapters.pyctm_adapter import PyctmAdapter
        return PyctmAdapter()
    if impl == "pyttmult":
        from bench.adapters.pyttmult_adapter import PyttmultAdapter
        return PyttmultAdapter()
    if impl == "ttmult_raw":
        from bench.adapters.ttmult_raw_adapter import TtmultRawAdapter
        return TtmultRawAdapter()
    raise ValueError(f"unknown impl {impl!r}")


# ─────────────────────────────────────────────────────────────
# Worker: runs one cell end-to-end and returns a record dict
# ─────────────────────────────────────────────────────────────


def _run_cell_in_child(cell: BenchCell, result_queue: "mp.Queue",
                        hardware_hash: str = ""):
    """Child-process entry: build adapter, run N reps, queue a record."""
    from bench.adapters.base import NotSupported

    record = {
        "record_id": cell.record_id(),
        "hardware_hash": hardware_hash,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config": {
            "impl": cell.impl,
            "fixture": cell.fixture,
            "calctype": cell.calctype,
            "batch_size": cell.batch_size,
            "nbins": cell.nbins,
            "device": cell.device,
            "reps": cell.reps,
            "T": cell.T,
            "slater": cell.slater,
            "soc": cell.soc,
            "force_device": cell.force_device,
        },
        "status": "ok",
        "error": None,
        "timings": {},
        "output": {},
        "adapter_meta": {},
    }
    try:
        adapter = _build_adapter(cell.impl)

        with wall_timer() as t_cold:
            adapter.cold_start()
        record["timings"]["cold_start_s"] = t_cold.elapsed

        try:
            with wall_timer() as t_warm:
                adapter.warm(cell)
            record["timings"]["warm_s"] = t_warm.elapsed
        except NotSupported as e:
            record["status"] = "skipped"
            record["error"] = f"NotSupported: {e}"
            result_queue.put(record)
            return

        wall_samples, cpu_samples = [], []
        per_rep_meta: List[dict] = []
        last_out = None
        for _ in range(cell.reps):
            with wall_timer() as t:
                last_out = adapter.run(cell)
            wall_samples.append(t.elapsed)
            cpu_samples.append(t.cpu_elapsed)
            per_rep_meta.append(dict(last_out.meta))

        record["timings"]["wall_s"] = RepStats(wall_samples).as_dict()
        record["timings"]["cpu_s"] = RepStats(cpu_samples).as_dict()
        # Aggregate the per-rep adapter meta (stage timings etc.) as medians.
        agg_meta = _aggregate_per_rep_meta(per_rep_meta)
        record["adapter_meta"] = agg_meta

        record["output"] = {
            "x_min": float(last_out.x.min()),
            "x_max": float(last_out.x.max()),
            "n_points": int(last_out.x.shape[0]),
            # Keep x and y as lists so a post-processor can recompute parity
            # offline against any reference. ~16 KB per 2000-point float64.
            "x": last_out.x.tolist(),
            "y": last_out.y.tolist(),
        }
    except NotSupported as e:
        record["status"] = "skipped"
        record["error"] = str(e)
    except Exception as e:
        record["status"] = "error"
        record["error"] = f"{type(e).__name__}: {e}"
        record["traceback"] = traceback.format_exc()

    result_queue.put(record)


def _aggregate_per_rep_meta(rep_metas: List[dict]) -> dict:
    """Take per-rep meta dicts and median-collapse float fields."""
    if not rep_metas:
        return {}
    keys = set()
    for m in rep_metas:
        keys.update(m.keys())
    agg = {}
    for k in keys:
        vals = [m.get(k) for m in rep_metas if k in m]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            import statistics
            agg[k] = {
                "median": float(statistics.median(vals)),
                "min": float(min(vals)),
                "max": float(max(vals)),
            }
        else:
            # Non-numeric: take the first
            agg[k] = vals[0]
    return agg


# ─────────────────────────────────────────────────────────────
# Driver: iterate cells, spawn workers, collect records
# ─────────────────────────────────────────────────────────────


def _run_cells(
    cells: List[BenchCell],
    output: Path,
    timeout_s: float,
    resume: bool,
    dry_run: bool,
) -> int:
    """Dispatch cells, return exit code."""
    if dry_run:
        print(f"Dry run — {len(cells)} cells to execute:")
        for c in cells:
            print(f"  {c.record_id()}  {c.impl:28s} {c.fixture:12s} "
                  f"{c.calctype:5s} batch={c.batch_size:>6d} device={c.device} reps={c.reps}")
        return 0

    # Session-level hardware/version capture. Written once per session to a
    # sidecar file; referenced from every JSONL record via hardware_hash().
    import json
    hw = collect_hardware()
    hw["hardware_hash"] = hardware_hash(hw)
    hw_path = output.parent / (output.stem + ".hardware.json")
    hw_path.parent.mkdir(parents=True, exist_ok=True)
    hw_path.write_text(json.dumps(hw, indent=2))
    print(f"Hardware fingerprint → {hw_path}  hash={hw['hardware_hash']}")
    print(f"  {hw.get('platform')} | torch={hw.get('torch')} | "
          f"cuda={hw.get('cuda_available')} | gpu={hw.get('gpu_name')}")

    completed: Dict[str, dict] = {}
    if resume and output.exists():
        completed = read_completed_record_ids(output)
        print(f"Resume: {len(completed)} cells already complete in {output}")

    ctx = mp.get_context("spawn")  # clean isolation; avoids fork issues w/ torch

    total = len(cells)
    n_run = 0
    n_skip = 0
    n_err = 0
    t_start = time.perf_counter()

    for idx, cell in enumerate(cells):
        if cell.record_id() in completed:
            continue
        tag = (f"[{idx+1:>4}/{total}] {cell.impl:28s} {cell.fixture:12s} "
               f"{cell.calctype:5s} batch={cell.batch_size:>6d} device={cell.device}")
        print(tag + " ...", end=" ", flush=True)

        q: "mp.Queue" = ctx.Queue()
        p = ctx.Process(target=_run_cell_in_child,
                        args=(cell, q, hw["hardware_hash"]))
        p.start()
        p.join(timeout=timeout_s)

        if p.is_alive():
            p.terminate()
            p.join(5)
            if p.is_alive():
                p.kill()
            record = {
                "record_id": cell.record_id(),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "config": {
                    "impl": cell.impl, "fixture": cell.fixture,
                    "calctype": cell.calctype, "batch_size": cell.batch_size,
                    "nbins": cell.nbins, "device": cell.device, "reps": cell.reps,
                },
                "status": "timeout",
                "error": f"Cell did not complete within {timeout_s}s",
            }
            append_record(output, record)
            n_err += 1
            print(f"TIMEOUT after {timeout_s}s")
            continue

        # Drain queue.
        if q.empty():
            record = {
                "record_id": cell.record_id(),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "config": {
                    "impl": cell.impl, "fixture": cell.fixture,
                    "calctype": cell.calctype, "batch_size": cell.batch_size,
                    "nbins": cell.nbins, "device": cell.device, "reps": cell.reps,
                },
                "status": "error",
                "error": "Worker exited without producing a record",
            }
            append_record(output, record)
            n_err += 1
            print("NO RESULT")
            continue

        record = q.get()
        append_record(output, record)
        status = record.get("status", "?")
        if status == "ok":
            wall = record["timings"].get("wall_s", {})
            med = wall.get("median", 0.0)
            print(f"OK  median={med*1000:.2f}ms")
            n_run += 1
        elif status == "skipped":
            print(f"SKIP  {record.get('error', '')}")
            n_skip += 1
        else:
            print(f"{status.upper()}  {record.get('error', '')[:80]}")
            n_err += 1

    elapsed = time.perf_counter() - t_start
    print(f"\nDone. {n_run} ok, {n_skip} skipped, {n_err} failed / timed-out. "
          f"Wall: {elapsed:.1f}s. Results: {output}")
    return 0 if n_err == 0 else 1


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="multiplet benchmark harness")
    ap.add_argument("--preset", choices=list(PRESETS.keys()), default="mvp",
                    help="mvp (small smoke) or full (overnight run)")
    ap.add_argument("--output", type=Path,
                    default=BENCH_ROOT / "results" / "results.jsonl",
                    help="Path to append-only JSONL")
    ap.add_argument("--timeout-s", type=float, default=300.0,
                    help="Per-cell timeout in seconds (default 300)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip cells already marked status=ok in --output")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the cell plan, don't execute")
    ap.add_argument("--only-impl", default=None,
                    help="Run only cells whose impl matches this substring")
    ap.add_argument("--only-fixture", default=None,
                    help="Run only cells whose fixture matches this substring")
    ap.add_argument("--max-cells", type=int, default=None,
                    help="Cap the number of cells run (handy for debugging)")
    args = ap.parse_args(argv)

    preset = PRESETS[args.preset]
    cells = iter_preset(preset)

    if args.only_impl is not None:
        cells = [c for c in cells if args.only_impl in c.impl]
    if args.only_fixture is not None:
        cells = [c for c in cells if args.only_fixture in c.fixture]
    if args.max_cells is not None:
        cells = cells[: args.max_cells]

    if not cells:
        print("No cells to run after filtering.")
        return 0

    return _run_cells(
        cells=cells,
        output=args.output,
        timeout_s=args.timeout_s,
        resume=args.resume,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
