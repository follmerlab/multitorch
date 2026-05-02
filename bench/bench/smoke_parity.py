"""Quick parity check on the real Ni²⁺ L-edge spectrum pair.

Runs one multitorch_cached spectrum + one pyctm spectrum for the same
(Ni, ii, d4h, l) cell and reports the 4 parity metrics. This is the
first real numerical-agreement data point and catches any gross
wiring error before we pour time into the full matrix.
"""
from __future__ import annotations

import sys

from bench.adapters.multitorch_adapter import MultitorchAdapter
from bench.adapters.pyctm_adapter import PyctmAdapter
from bench.config import BenchCell
from bench.parity import compare


def main() -> int:
    cell = BenchCell(
        impl="parity_check",
        fixture="ni2_d8_oh",
        calctype="xas",
        batch_size=1,
        nbins=2000,
        device="cpu",
        reps=1,
    )

    print("Running multitorch_cached ...", flush=True)
    mt = MultitorchAdapter(mode="cached")
    mt.cold_start()
    mt_result = mt.run(cell)
    print(f"  x range: [{mt_result.x.min():.3f}, {mt_result.x.max():.3f}] n={mt_result.x.shape[0]}")

    print("Running pyctm ...", flush=True)
    py = PyctmAdapter()
    py.cold_start()
    py_result = py.run(cell)
    print(f"  x range: [{py_result.x.min():.3f}, {py_result.x.max():.3f}] n={py_result.x.shape[0]}")

    print("\nParity (a=multitorch_cached vs b=pyctm):")
    parity = compare(mt_result.x, mt_result.y, py_result.x, py_result.y, calctype="xas")
    d = parity.as_dict()
    for key, val in d.items():
        if isinstance(val, float):
            print(f"  {key:28s} = {val:.6f}")
        else:
            print(f"  {key:28s} = {val}")

    # Rough thresholds; the real thresholds lock in after the full run.
    ok = (
        parity.cosine > 0.95
        and parity.peak_err_max_ev < 0.5
    )
    print(f"\nPass (cosine>0.95 and peak_err<0.5eV): {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
