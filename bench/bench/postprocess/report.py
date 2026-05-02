"""Assemble all plots + a summary markdown into one report directory.

Usage:
    python -m bench.postprocess.report --input results/mvp.jsonl --out results/report
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bench.postprocess.load import load, ok_only
from bench.postprocess.plots import generate_all


def _write_summary(df, out_dir: Path) -> None:
    ok = ok_only(df)
    lines = [
        "# Benchmark report",
        "",
        f"Total records: {len(df)}",
        f"  ok:      {(df['status'] == 'ok').sum()}",
        f"  skipped: {(df['status'] == 'skipped').sum()}",
        f"  error:   {(df['status'] == 'error').sum()}",
        f"  timeout: {(df['status'] == 'timeout').sum()}",
        "",
        "## Median wall time (ms) per (impl × fixture × batch) at CPU",
        "",
    ]
    if not ok.empty:
        pivot = ok[ok["cfg_device"] == "cpu"].pivot_table(
            index=["cfg_fixture", "cfg_batch_size"],
            columns="cfg_impl", values="wall_median",
        )
        if not pivot.empty:
            pivot_ms = (pivot * 1000).round(2)
            # Plain monospace; tabulate is optional. Users can re-render in Excel.
            lines.append("```")
            lines.append(pivot_ms.to_string())
            lines.append("```")
        else:
            lines.append("_no CPU data_")
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    for name, desc in [
        ("p1_single_spectrum_time.png", "Single-spectrum time per impl × fixture (batch=1, CPU)"),
        ("p2_scaling_vs_batch.png", "Scaling vs batch size, faceted by fixture"),
        ("p3_parity_cosine.png", "Parity cosine similarity heatmap"),
        ("p4_cpu_vs_cuda.png", "multitorch CPU-vs-CUDA speedup (if CUDA data present)"),
    ]:
        lines.append(f"- `{name}` — {desc}")
    (out_dir / "report.md").write_text("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="build benchmark plots + report")
    ap.add_argument("--input", type=Path, required=True, help="results JSONL")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    args = ap.parse_args(argv)

    df = load(args.input)
    args.out.mkdir(parents=True, exist_ok=True)
    generate_all(df, args.out)
    _write_summary(df, args.out)
    print(f"Report → {args.out}/report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
