"""Benchmark plot builders.

P1 — bars:    single-spectrum time per (impl × fixture), batch=1, CPU
P2 — scaling: time vs batch size, one line per impl, faceted by fixture
P3 — parity:  cosine-similarity heatmap per fixture, rows=impls, cols=impls
P4 — CUDA:    multitorch CPU-vs-CUDA speedup (only when CUDA data present)

All plots take the DataFrame produced by bench.postprocess.load.load.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bench.parity import compare


IMPL_ORDER = [
    "ttmult_raw",
    "pyttmult",
    "pyctm",
    "multitorch_from_scratch",
    "multitorch_cached",
    "multitorch_batch",
]


def _impl_palette() -> Dict[str, str]:
    # Cohesive coloring: Fortran stack shades of blue, multitorch shades of orange.
    return {
        "ttmult_raw":             "#08519c",
        "pyttmult":               "#3182bd",
        "pyctm":                  "#6baed6",
        "multitorch_from_scratch":"#fdae6b",
        "multitorch_cached":      "#e6550d",
        "multitorch_batch":       "#a63603",
    }


def plot_p1_single_spectrum_bars(df: pd.DataFrame, out: Path) -> None:
    """Grouped bar: median time per (impl × fixture) at batch=1, CPU."""
    sub = df[(df["status"] == "ok")
             & (df["cfg_batch_size"] == 1)
             & (df["cfg_device"] == "cpu")]
    if sub.empty:
        print(f"P1: no data to plot")
        return

    impls = [i for i in IMPL_ORDER if i in sub["cfg_impl"].unique()]
    fixtures = sorted(sub["cfg_fixture"].unique())
    palette = _impl_palette()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / max(len(impls), 1)
    x_pos = np.arange(len(fixtures))
    for i, impl in enumerate(impls):
        vals, errs = [], []
        for fx in fixtures:
            row = sub[(sub["cfg_impl"] == impl) & (sub["cfg_fixture"] == fx)]
            if row.empty:
                vals.append(np.nan)
                errs.append(0.0)
            else:
                vals.append(float(row["wall_median"].iloc[0]) * 1000.0)
                errs.append(float(row["wall_iqr"].iloc[0]) * 1000.0)
        ax.bar(x_pos + i * width - 0.4 + width / 2,
               vals, width,
               yerr=errs, capsize=2,
               label=impl, color=palette.get(impl, "#888"))
    ax.set_yscale("log")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(fixtures, rotation=15)
    ax.set_ylabel("median wall time per spectrum (ms)")
    ax.set_title("P1 — single-spectrum XAS time (batch=1, CPU)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, which="both", axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"P1 → {out}")


def plot_p2_scaling(df: pd.DataFrame, out: Path) -> None:
    """Time vs batch size, log-log, one line per impl, faceted by fixture."""
    sub = df[(df["status"] == "ok") & (df["cfg_device"] == "cpu")]
    if sub.empty:
        print(f"P2: no data")
        return
    fixtures = sorted(sub["cfg_fixture"].unique())
    impls = [i for i in IMPL_ORDER if i in sub["cfg_impl"].unique()]
    palette = _impl_palette()

    n = len(fixtures)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, fx in zip(axes, fixtures):
        for impl in impls:
            fsub = sub[(sub["cfg_impl"] == impl) & (sub["cfg_fixture"] == fx)]
            if fsub.empty:
                continue
            fsub = fsub.sort_values("cfg_batch_size")
            ax.plot(
                fsub["cfg_batch_size"],
                fsub["wall_median"] * 1000,
                marker="o", color=palette.get(impl, "#888"), label=impl,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("batch size")
        ax.set_title(fx)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("median total time (ms)")
    axes[-1].legend(loc="upper left", fontsize=8)
    fig.suptitle("P2 — scaling vs batch size (CPU)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"P2 → {out}")


def plot_p3_parity_heatmap(df: pd.DataFrame, out: Path) -> None:
    """Cosine-similarity heatmap per fixture at batch=1, rows=impls, cols=impls."""
    sub = df[(df["status"] == "ok") & (df["cfg_batch_size"] == 1)
             & (df["output_x"].notna())].copy()
    if sub.empty:
        print(f"P3: no spectra with (x, y) data")
        return

    fixtures = sorted(sub["cfg_fixture"].unique())
    # Pre-build the (fixture, impl) -> (x, y) lookup.
    spectra: Dict[tuple, tuple] = {}
    for _, row in sub.iterrows():
        key = (row["cfg_fixture"], row["cfg_impl"])
        spectra[key] = (np.asarray(row["output_x"]), np.asarray(row["output_y"]))

    impls_per_fixture = {
        fx: [i for i in IMPL_ORDER if (fx, i) in spectra]
        for fx in fixtures
    }

    n = len(fixtures)
    fig, axes = plt.subplots(1, n, figsize=(4 * n + 0.5, 4))
    if n == 1:
        axes = [axes]
    for ax, fx in zip(axes, fixtures):
        impls = impls_per_fixture[fx]
        k = len(impls)
        if k == 0:
            ax.set_title(f"{fx} (no spectra)")
            continue
        M = np.ones((k, k))
        for i, ia in enumerate(impls):
            for j, ib in enumerate(impls):
                if i == j:
                    continue
                try:
                    xa, ya = spectra[(fx, ia)]
                    xb, yb = spectra[(fx, ib)]
                    p = compare(xa, ya, xb, yb, calctype="xas")
                    M[i, j] = p.cosine
                except Exception as e:
                    M[i, j] = np.nan
        im = ax.imshow(M, vmin=0.5, vmax=1.0, cmap="viridis")
        ax.set_xticks(range(k))
        ax.set_xticklabels(impls, rotation=30, ha="right", fontsize=7)
        ax.set_yticks(range(k))
        ax.set_yticklabels(impls, fontsize=7)
        ax.set_title(fx)
        for i in range(k):
            for j in range(k):
                v = M[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            color="white" if v < 0.82 else "black", fontsize=7)
    fig.suptitle("P3 — parity (cosine similarity after peak alignment)")
    fig.tight_layout()
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"P3 → {out}")


def plot_p4_cpu_vs_cuda(df: pd.DataFrame, out: Path) -> None:
    """multitorch CPU vs CUDA speedup bar. Skipped if no CUDA data."""
    mt = df[(df["status"] == "ok")
            & df["cfg_impl"].str.startswith("multitorch", na=False)]
    if mt.empty or "cuda" not in mt["cfg_device"].unique():
        print(f"P4: no CUDA data — skipping")
        return

    wide = mt.pivot_table(
        index=["cfg_fixture", "cfg_impl", "cfg_batch_size"],
        columns="cfg_device", values="wall_median",
    ).dropna().reset_index()
    wide["speedup"] = wide["cpu"] / wide["cuda"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = wide.apply(
        lambda r: f"{r.cfg_fixture}/{r.cfg_impl}/b={int(r.cfg_batch_size)}",
        axis=1,
    )
    colors = ["#2ca02c" if s >= 1 else "#d62728" for s in wide["speedup"]]
    ax.barh(labels, wide["speedup"], color=colors)
    ax.axvline(1.0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("t_cpu / t_cuda (>1 means CUDA faster)")
    ax.set_title("P4 — multitorch CPU vs CUDA speedup")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"P4 → {out}")


def generate_all(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_p1_single_spectrum_bars(df, out_dir / "p1_single_spectrum_time.png")
    plot_p2_scaling(df, out_dir / "p2_scaling_vs_batch.png")
    plot_p3_parity_heatmap(df, out_dir / "p3_parity_cosine.png")
    plot_p4_cpu_vs_cuda(df, out_dir / "p4_cpu_vs_cuda.png")
