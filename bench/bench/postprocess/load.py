"""Load a results JSONL into a pandas DataFrame.

Flattens the nested config / timings.wall_s / output structure into
top-level columns so downstream plotting code can do ``df.query(...)``
without nested indexing.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from bench.artifacts import iter_records


def load(path: Path) -> pd.DataFrame:
    rows: List[dict] = []
    for rec in iter_records(Path(path)):
        row = {
            "record_id": rec.get("record_id"),
            "timestamp": rec.get("timestamp"),
            "status": rec.get("status"),
            "error": rec.get("error"),
        }
        cfg = rec.get("config", {}) or {}
        for k, v in cfg.items():
            row[f"cfg_{k}"] = v
        timings = rec.get("timings", {}) or {}
        row["cold_start_s"] = timings.get("cold_start_s")
        row["warm_s"] = timings.get("warm_s")
        wall = timings.get("wall_s", {}) or {}
        for k in ("median", "mean", "std", "iqr", "q25", "q75", "min", "max", "n"):
            row[f"wall_{k}"] = wall.get(k)
        row["wall_samples"] = wall.get("samples")
        cpu = timings.get("cpu_s", {}) or {}
        row["cpu_median"] = cpu.get("median")
        output = rec.get("output", {}) or {}
        row["output_n_points"] = output.get("n_points")
        row["output_x_min"] = output.get("x_min")
        row["output_x_max"] = output.get("x_max")
        # Full x/y arrays (optional — for parity postprocessing).
        row["output_x"] = output.get("x")
        row["output_y"] = output.get("y")
        meta = rec.get("adapter_meta", {}) or {}
        for k, v in meta.items():
            if isinstance(v, dict) and "median" in v:
                row[f"meta_{k}_median"] = v["median"]
            elif not isinstance(v, (dict, list)):
                row[f"meta_{k}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def ok_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"] == "ok"].copy()
