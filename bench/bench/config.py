"""Benchmark matrix definitions, fixture registry, and parity tolerances."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# Resolve paths relative to the multiplets workspace so the suite is portable
# between machines: the user just needs the four repos checked out side by side.
BENCH_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = BENCH_ROOT.parent
MULTITORCH_ROOT = WORKSPACE_ROOT / "multitorch"
PYCTM_ROOT = WORKSPACE_ROOT / "pyctm"
PYTTMULT_ROOT = WORKSPACE_ROOT / "pyttmult"
TTMULT_ROOT = WORKSPACE_ROOT / "ttmult"
FIXTURES_ROOT = MULTITORCH_ROOT / "multitorch" / "data" / "fixtures"


@dataclass(frozen=True)
class FixtureSpec:
    """A single reference system the benchmark iterates over.

    The tuple (element, valence, sym, edge) mirrors the multitorch /
    pyctm calcXAS signature. `input_files_present` tells the Fortran
    raw adapter whether it can regenerate the pipeline from disk or
    must rely on pyctm's writers.
    """

    name: str
    element: str
    valence: str
    sym: str
    edge: str = "l"
    has_charge_transfer: bool = False
    # What the multitorch bootstrap path can consume without any Fortran
    # binary invocation; always present for the 11 bundled fixtures.
    bootstrap_ban_out: str = ""
    # Does the fixture directory carry the raw Fortran inputs .rcg/.rac?
    # Only the charge-transfer fixtures (nid8ct/nid8) do; everyone else
    # would have to regenerate them via pyctm writers.
    has_raw_inputs: bool = False


FIXTURES: List[FixtureSpec] = [
    FixtureSpec("ti4_d0_oh", "Ti", "iv", "oh", bootstrap_ban_out="ti4_d0_oh.ban_out"),
    FixtureSpec("v3_d2_oh", "V", "iii", "oh", bootstrap_ban_out="v3_d2_oh.ban_out"),
    FixtureSpec("cr3_d3_oh", "Cr", "iii", "oh", bootstrap_ban_out="cr3_d3_oh.ban_out"),
    FixtureSpec("mn2_d5_oh", "Mn", "ii", "oh", bootstrap_ban_out="mn2_d5_oh.ban_out"),
    FixtureSpec("fe2_d6_oh", "Fe", "ii", "oh", bootstrap_ban_out="fe2_d6_oh.ban_out"),
    FixtureSpec("fe3_d5_oh", "Fe", "iii", "oh", bootstrap_ban_out="fe3_d5_oh.ban_out"),
    FixtureSpec("co2_d7_oh", "Co", "ii", "oh", bootstrap_ban_out="co2_d7_oh.ban_out"),
    FixtureSpec("ni2_d8_oh", "Ni", "ii", "oh", bootstrap_ban_out="ni2_d8_oh.ban_out"),
    FixtureSpec("nid8",    "Ni", "ii", "d4h", bootstrap_ban_out="nid8.ban_out"),
    FixtureSpec("nid8ct",  "Ni", "ii", "d4h", has_charge_transfer=True,
                bootstrap_ban_out="nid8ct.ban_out", has_raw_inputs=True),
    FixtureSpec("als1ni2", "Ni", "ii", "d4h", has_charge_transfer=True,
                bootstrap_ban_out="als1ni2.ban_out"),
]


FIXTURE_BY_NAME = {f.name: f for f in FIXTURES}


# ─────────────────────────────────────────────────────────────
# Benchmark matrix presets
# ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MatrixPreset:
    """A named subset of the full benchmark axes, per the plan."""

    name: str
    fixtures: Tuple[str, ...]
    calctypes: Tuple[str, ...]
    batch_sizes: Tuple[int, ...]
    impls: Tuple[str, ...]
    devices: Tuple[str, ...]
    reps_by_batch: dict           # mapping batch_size → reps


PRESET_MVP = MatrixPreset(
    name="mvp",
    fixtures=("ni2_d8_oh", "nid8ct"),
    calctypes=("xas",),
    batch_sizes=(1, 10, 100),
    impls=("multitorch_cached", "multitorch_from_scratch", "pyttmult", "pyctm"),
    devices=("cpu",),
    reps_by_batch={1: 5, 10: 5, 100: 5},
)


PRESET_FULL = MatrixPreset(
    name="full",
    fixtures=tuple(f.name for f in FIXTURES),
    calctypes=("xas", "rixs", "xes"),
    batch_sizes=(1, 10, 100, 1000, 10000),
    impls=(
        "multitorch_cached",
        "multitorch_from_scratch",
        "multitorch_batch",
        "pyctm",
        "pyttmult",
        "ttmult_raw",
    ),
    devices=("cpu", "cuda"),
    reps_by_batch={1: 20, 10: 10, 100: 10, 1000: 5, 10000: 5},
)


PRESETS = {"mvp": PRESET_MVP, "full": PRESET_FULL}


# ─────────────────────────────────────────────────────────────
# Parity tolerances (recorded, not hard gates)
# ─────────────────────────────────────────────────────────────

COMMON_GRID_SPACING_EV = 0.01         # 0.01 eV resolution for interpolation
COSINE_TOLERANCE = 0.9999
MAX_ABS_DIFF_TOLERANCE = 0.05         # on unit-max normalized intensities
PEAK_POS_TOLERANCE_EV = 0.05          # max |ΔeV| on top-K peak matches
L3L2_RATIO_TOLERANCE = 0.01           # fractional (1 %)
PEAK_TOP_K_LEDGE = 3
PEAK_TOP_K_RIXS = 5


# ─────────────────────────────────────────────────────────────
# Benchmark cell descriptor
# ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BenchCell:
    """A single benchmark cell — one row in the full matrix."""

    impl: str                 # e.g. "multitorch_cached", "ttmult_raw"
    fixture: str              # FixtureSpec.name
    calctype: str             # "xas" | "rixs" | "xes"
    batch_size: int
    nbins: int = 2000
    device: str = "cpu"
    reps: int = 5
    T: float = 80.0
    # Physics overrides (None = adapter uses defaults)
    slater: Optional[float] = None
    soc: Optional[float] = None
    # For multi-mode variants of multitorch
    force_device: bool = False

    def record_id(self) -> str:
        """Stable short identifier for this cell; used for resume."""
        import hashlib
        s = (
            f"{self.impl}|{self.fixture}|{self.calctype}|{self.batch_size}|"
            f"{self.nbins}|{self.device}|{self.reps}|{self.T}|"
            f"{self.slater}|{self.soc}|{self.force_device}"
        )
        return hashlib.sha1(s.encode()).hexdigest()[:16]


def iter_preset(preset: MatrixPreset) -> List[BenchCell]:
    """Expand a preset into concrete BenchCell list.

    Respects the 'RIXS only on CT fixtures' rule and skips combinations
    the Fortran stack can't realistically do (batch > 10 is marked
    pseudo_batch; still enumerated so users can opt in).
    """
    cells: List[BenchCell] = []
    for fixture_name in preset.fixtures:
        fx = FIXTURE_BY_NAME[fixture_name]
        for calctype in preset.calctypes:
            if calctype == "rixs" and not fx.has_charge_transfer:
                continue
            for batch_size in preset.batch_sizes:
                reps = preset.reps_by_batch.get(batch_size, 5)
                for impl in preset.impls:
                    for device in preset.devices:
                        # Non-multitorch impls do not use CUDA; skip.
                        if device == "cuda" and not impl.startswith("multitorch"):
                            continue
                        # The batch-specific multitorch mode only makes
                        # sense for batch_size > 1.
                        if impl == "multitorch_batch" and batch_size == 1:
                            continue
                        cells.append(BenchCell(
                            impl=impl,
                            fixture=fixture_name,
                            calctype=calctype,
                            batch_size=batch_size,
                            device=device,
                            reps=reps,
                        ))
    return cells
