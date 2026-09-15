"""
In-memory COWAN store builder for the fixture (Phase 5) pipeline.

Scope and approach
------------------
A fixture ``.rme_rcg`` store is Fortran ttrcg output for one fixed set of
atomic parameters. Every HAMILTONIAN block in it is linear in those
parameters (all energies in eV)::

    H(J) = E_av·sqrt(2J+1)·I + Σ F^k·O_F^k(J) + Σ G^k·O_G^k(J) + Σ ζ_i·O_ζi(J)

with parameter-free operators O in the Fortran basis
(:func:`~multitorch.angular.cowan_operators.configuration_operators`). This
module decomposes every HAMILTONIAN block of the store onto those operators by
a joint least-squares fit over all J of a configuration, asserts the fit is
exact (elementwise residual ≤ 1e-5·max(1, |H|)), and rebuilds the blocks with

    F^k, G^k  →  F^k · slater / slater_reduction
    ζ_i       →  ζ_i · soc / soc_reduction

anchored on the fixture block itself::

    H(J) = H_fixture(J) + (a − 1)·S(J) + (b − 1)·Z(J),
    a = slater / slater_reduction,  b = soc / soc_reduction,

with S = Σ F^k O_F + Σ G^k O_G and Z = Σ ζ_i O_ζi from the fit. The rebuild
itself is the shared parameter-linear contraction of
:mod:`multitorch.hamiltonian.parametric`, which the from-scratch generator uses
with a zero anchor. At the
fixture's own reduction the store is returned unchanged (bit-exact parity with
the Fortran chain); elsewhere it differs from E_av + a·S + b·Z only by the fit
residual, i.e. Fortran print noise. ``slater_reduction`` is the reduction the
fixture was generated at (0.8 for every bundled fixture except
``nid8ct_ems``), so ``slater`` and ``soc`` are absolute: fractions of the
Hartree-Fock values, as in pyctm and CTM4XAS. E_av is a constant of the fixture (pyctm fixes it; Δ enters via the
BAN EG/EF offsets). All non-HAMILTONIAN blocks pass through unchanged.

The configurations of each section are read from the ``%P06  D08  D10`` header
lines (GROUND first, EXCITE second); shell order is whatever the fixture used
(core-first for pyctm, valence-first for parts of ``nid8ct``). No
``.rcn31_out`` is needed.

Precision: pyctm rounds every RCG input parameter to 3 decimals, so a fixture
rebuilt at a different ``slater`` agrees with a Fortran run at that reduction
only to the propagated rounding (≈1e-3 eV per parameter), not to 1e-6.
"""
from __future__ import annotations

from collections import OrderedDict

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from multitorch._constants import DTYPE
from multitorch.angular.cowan_operators import Shell, configuration_operators
from multitorch.hamiltonian.build_rac import SectionPlan
from multitorch.hamiltonian.parametric import (
    ConfigDecomposition,
    HamiltonianDecomposition,
    as_scale,
    rebuild_hamiltonian_store,
)
from multitorch.io.read_rme import read_cowan_store


# ─────────────────────────────────────────────────────────────
# Block metadata (lightweight, just for internal routing)
# ─────────────────────────────────────────────────────────────


@dataclass
class CowanBlockMeta:
    """Metadata for one RME block in the COWAN store.

    Extracted from the RME header line in the ``.rme_rcg`` file::

        RME <block_type> <bra_sym> <op_sym> <ket_sym> <operator> ...
    """

    block_type: str  # GROUND, EXCITE, TRANSITION
    operator: str    # HAMILTONIAN, SHELL1, SPIN1, MULTIPOLE
    bra_sym: str     # e.g. '0+', '1-', 's0+' (J = 1/2)
    op_sym: str      # e.g. '0+', '1+', '2+', '4+'
    ket_sym: str     # e.g. '0+', '1-'


def read_cowan_metadata(path: str | Path) -> List[List[CowanBlockMeta]]:
    """Read ``.rme_rcg`` block metadata grouped by FINISHED sections.

    Returns one list of :class:`CowanBlockMeta` per FINISHED-delimited
    section, in the same order as :func:`read_cowan_store`. That is,
    ``metadata[s][j]`` describes the matrix at ``cowan_store[s][j]``.
    """
    path = Path(path)
    with open(path) as f:
        lines = f.readlines()

    sections: List[List[CowanBlockMeta]] = []
    current: List[CowanBlockMeta] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == "FINISHED":
            sections.append(current)
            current = []
            i += 1
            continue

        if line.startswith("RME "):
            parts = line.split()
            # RME <block_type> <bra_sym> <op_sym> <ket_sym> <operator> ...
            current.append(CowanBlockMeta(
                block_type=parts[1],
                bra_sym=parts[2],
                op_sym=parts[3],
                ket_sym=parts[4],
                operator=parts[5],
            ))
            # Skip body of this RME block — advance until next header-level
            # line. Mirror the termination condition from _parse_rme_block.
            i += 1
            while i < len(lines):
                body = lines[i].strip()
                if (not body
                        or body.startswith("%")
                        or body.startswith("RME")
                        or body.startswith("IRREP")
                        or body.startswith("FINISHED")):
                    break
                i += 1
        else:
            i += 1

    if current:
        sections.append(current)

    return sections


_CONFIG_LINE = re.compile(r"^%\s*(?:[SPDFGH]\s?\d{1,2}\s*)+$")
_SHELL_TOKEN = re.compile(r"([SPDFGH])\s?(\d{1,2})")
_L_OF = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4, "H": 5}


def read_cowan_configurations(path: str | Path) -> List[Dict[str, Tuple[Shell, ...]]]:
    """Open shells of the GROUND and EXCITE configuration of every section.

    Each FINISHED-delimited section of a ttrcg store starts with two
    configuration lines such as ``%P06  D08  D10`` (ground) and
    ``%P05  D09  D10`` (excited). Returns ``[{'GROUND': shells, 'EXCITE': shells}, ...]``
    with ``shells`` the open ``(l, n)`` in file order; closed and empty shells
    are dropped because they do not enter the coupling.
    """
    sections: List[Dict[str, Tuple[Shell, ...]]] = []
    current: List[Tuple[Shell, ...]] = []

    def close():
        if len(current) != 2:
            raise ValueError(
                f"{path}: section {len(sections)} has {len(current)} configuration "
                f"lines, expected 2"
            )
        sections.append({"GROUND": current[0], "EXCITE": current[1]})

    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if line == "FINISHED":
                close()
                current = []
            elif _CONFIG_LINE.match(line):
                shells = []
                for letter, n in _SHELL_TOKEN.findall(line[1:]):
                    l, n = _L_OF[letter], int(n)
                    if 0 < n < 4 * l + 2:
                        shells.append((l, n))
                current.append(tuple(shells))
    if current:
        close()
    return sections


def j_value(sym: str) -> float:
    """J of a store symmetry label: ``'2+'`` → 2, ``'s0+'`` → 1/2, ``'s3-'`` → 7/2."""
    s = sym.lstrip("^").rstrip("+-")
    return int(s[1:]) + 0.5 if s.startswith("s") else float(int(s))


# ─────────────────────────────────────────────────────────────
# Fixture metadata
# ─────────────────────────────────────────────────────────────

# Slater reduction each bundled fixture was generated at (store stem → factor).
# pyctm Oh fixtures: verified against Fortran reruns at 1.0 and 0.8
# (/data/ahf/multitorch/fixtures/oracle_oh8). ttmult examples: inferred from the
# fitted F²(3d,3d) = 9.7872 eV against the HF value 12.2341 eV of
# nid8.rcn31_out (ttrcg applies the reduction internally; nid8ct's G³ sits at
# 64%, so only the relative rescale is uniform there). nid8ct_ems fits the
# unreduced HF values.
FIXTURE_SLATER_REDUCTION: Dict[str, float] = {
    "ti4_d0_oh": 0.8, "v3_d2_oh": 0.8, "cr3_d3_oh": 0.8, "mn2_d5_oh": 0.8,
    "fe3_d5_oh": 0.8, "fe2_d6_oh": 0.8, "co2_d7_oh": 0.8, "ni2_d8_oh": 0.8,
    "nid8": 0.8, "nid8ct": 0.8, "als1ni2": 0.8, "nid8ct_ems": 1.0,
}
FIXTURE_SOC_REDUCTION_DEFAULT = 1.0


def fixture_slater_reduction(rcg_path: str | Path) -> float:
    stem = Path(rcg_path).stem
    try:
        return FIXTURE_SLATER_REDUCTION[stem]
    except KeyError:
        raise KeyError(
            f"No slater_reduction recorded for fixture '{stem}'. Pass "
            f"slater_reduction explicitly (the reduction the store was "
            f"generated at)."
        ) from None


# ─────────────────────────────────────────────────────────────
# Exact decomposition of HAMILTONIAN blocks
# ─────────────────────────────────────────────────────────────

RESIDUAL_REL_TOL = 1e-5


def _decompose_config(
    section: int,
    block_type: str,
    shells: Tuple[Shell, ...],
    blocks: Dict[float, Tuple[int, np.ndarray]],
    rel_tol: float,
    slater_reduction: float,
    soc_reduction: float,
) -> ConfigDecomposition:
    ops = configuration_operators(shells)
    for J, (_, M) in blocks.items():
        if ops.dims.get(J) != M.shape[0]:
            raise ValueError(
                f"section {section} {block_type} {shells}: J={J} block has dim "
                f"{M.shape[0]}, operators give {ops.dims.get(J)}"
            )
    names = [n for n in ops.blocks
             if any(np.abs(ops.blocks[n][J]).max() > 1e-12 for J in blocks)]

    cols: Dict[str, List[np.ndarray]] = {n: [] for n in ["E_av"] + names}
    y: List[np.ndarray] = []
    for J, (_, M) in blocks.items():
        iu = np.triu_indices(M.shape[0])
        y.append(M[iu])
        cols["E_av"].append(np.full(len(iu[0]), math.sqrt(2 * J + 1)) * (iu[0] == iu[1]))
        for n in names:
            cols[n].append(ops.blocks[n][J][iu])
    A = np.stack([np.concatenate(c) for c in cols.values()], axis=1)
    rhs = np.concatenate(y)
    coef, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    params = dict(zip(cols, (float(c) for c in coef)))

    fixture = {}
    worst = 0.0
    for J, (_, M) in blocks.items():
        d = M.shape[0]
        b = params["E_av"] * math.sqrt(2 * J + 1) * np.eye(d)
        sz = sum((params[n] * ops.blocks[n][J] for n in names), np.zeros((d, d)))
        rel = np.abs(b + sz - M) / np.maximum(1.0, np.abs(M))
        worst = max(worst, float(rel.max()))
        fixture[J] = torch.as_tensor(M, dtype=DTYPE)
    if worst > rel_tol:
        raise ValueError(
            f"HAMILTONIAN blocks of section {section} {block_type} {shells} are "
            f"not a combination of the configuration operators: max relative "
            f"residual {worst:.2e} > {rel_tol:.0e} (params {params})"
        )
    e_av = params.pop("E_av")
    return ConfigDecomposition(
        section=section, block_type=block_type, shells=shells,
        block_index={J: idx for J, (idx, _) in blocks.items()},
        e_av=e_av, anchor_params=params, reference=dict(params),
        operators={n: {J: torch.as_tensor(ops.blocks[n][J], dtype=DTYPE) for J in blocks} for n in names},
        anchor=fixture, reference_slater=slater_reduction, reference_soc=soc_reduction,
        max_residual=worst, label=f"{section}.{block_type}",
        states={J: list(ops.states[J]) for J in blocks},
    )


def decompose_cowan_hamiltonians(
    cowan_template: List[List[torch.Tensor]],
    cowan_metadata: List[List[CowanBlockMeta]],
    configurations: List[Dict[str, Tuple[Shell, ...]]],
    *,
    slater_reduction: float,
    soc_reduction: float = FIXTURE_SOC_REDUCTION_DEFAULT,
    rel_tol: float = RESIDUAL_REL_TOL,
) -> HamiltonianDecomposition:
    """Decompose every HAMILTONIAN block of a store onto the configuration operators.

    Raises ``ValueError`` if any configuration's blocks are not reproduced to
    ``rel_tol`` (elementwise, relative to max(1, |H|)): the floor observed on
    the bundled fixtures is ≤ 3e-6, set by Fortran's print precision on
    diagonals up to ~100 eV (E_av·sqrt(2J+1)).
    """
    if len(configurations) != len(cowan_template):
        raise ValueError(
            f"{len(configurations)} configuration headers for "
            f"{len(cowan_template)} store sections"
        )
    configs: List[ConfigDecomposition] = []
    for s, (mats, meta) in enumerate(zip(cowan_template, cowan_metadata)):
        for kind in ("GROUND", "EXCITE"):
            blocks = {
                j_value(m.bra_sym): (j, mats[j].detach().cpu().numpy().astype(np.float64))
                for j, m in enumerate(meta)
                if m.operator == "HAMILTONIAN" and m.block_type == kind
            }
            if blocks:
                configs.append(_decompose_config(s, kind, configurations[s][kind], blocks, rel_tol,
                                                 float(slater_reduction), float(soc_reduction)))
    return HamiltonianDecomposition(configs, float(slater_reduction), float(soc_reduction))


# LRU of fixture decompositions: each holds dense per-parameter operators
# (0.2-1.2 GB for the half-integer fixtures), so the cache is bounded.
DECOMPOSITION_CACHE_SIZE = 4
_DECOMPOSITION_CACHE: "OrderedDict[Tuple[str, int, float, float], HamiltonianDecomposition]" = OrderedDict()


def clear_decomposition_cache() -> None:
    """Drop every cached fixture decomposition (frees their operator tensors)."""
    _DECOMPOSITION_CACHE.clear()


def load_hamiltonian_decomposition(
    rcg_path: str | Path,
    *,
    cowan_template: Optional[List[List[torch.Tensor]]] = None,
    cowan_metadata: Optional[List[List[CowanBlockMeta]]] = None,
    slater_reduction: Optional[float] = None,
    soc_reduction: float = FIXTURE_SOC_REDUCTION_DEFAULT,
) -> HamiltonianDecomposition:
    """Cached :func:`decompose_cowan_hamiltonians` for a ``.rme_rcg`` file."""
    rcg_path = Path(rcg_path).resolve()
    if slater_reduction is None:
        slater_reduction = fixture_slater_reduction(rcg_path)
    key = (str(rcg_path), rcg_path.stat().st_mtime_ns, float(slater_reduction), float(soc_reduction))
    if key in _DECOMPOSITION_CACHE:
        _DECOMPOSITION_CACHE.move_to_end(key)
        return _DECOMPOSITION_CACHE[key]
    template = cowan_template if cowan_template is not None else read_cowan_store(rcg_path)
    meta = cowan_metadata if cowan_metadata is not None else read_cowan_metadata(rcg_path)
    dec = decompose_cowan_hamiltonians(
        template, meta, read_cowan_configurations(rcg_path),
        slater_reduction=slater_reduction, soc_reduction=soc_reduction,
    )
    _DECOMPOSITION_CACHE[key] = dec
    while len(_DECOMPOSITION_CACHE) > DECOMPOSITION_CACHE_SIZE:
        _DECOMPOSITION_CACHE.popitem(last=False)
    return dec


# ─────────────────────────────────────────────────────────────
# Public entry points
# ─────────────────────────────────────────────────────────────


def _resolve_inputs(plan, source_rcg_path, cowan_template, cowan_metadata, decomposition):
    if cowan_template is not None and cowan_metadata is not None:
        template, meta = cowan_template, cowan_metadata
    elif source_rcg_path is not None:
        template = read_cowan_store(source_rcg_path)
        meta = read_cowan_metadata(source_rcg_path)
    else:
        raise ValueError(
            "Either source_rcg_path or (cowan_template, cowan_metadata) must be provided"
        )
    if decomposition is None:
        if source_rcg_path is None:
            raise ValueError(
                "decomposition is required when the store is given without source_rcg_path"
            )
        decomposition = load_hamiltonian_decomposition(
            source_rcg_path, cowan_template=template, cowan_metadata=meta,
        )

    if len(template) != len(meta):
        raise ValueError(
            f"Template has {len(template)} sections but metadata has "
            f"{len(meta)} — the .rme_rcg file may be malformed"
        )
    for s in range(len(template)):
        if len(template[s]) != len(meta[s]):
            raise ValueError(
                f"Section {s}: {len(template[s])} matrices vs "
                f"{len(meta[s])} metadata entries"
            )
    if len(template) != plan.n_sections:
        raise ValueError(
            f"Template has {len(template)} sections but plan expects "
            f"{plan.n_sections}"
        )
    for s in range(len(template)):
        if len(template[s]) != plan.section_size(s):
            raise ValueError(
                f"Section {s}: {len(template[s])} matrices but plan "
                f"expects {plan.section_size(s)}"
            )
    return template, decomposition


def build_cowan_store_in_memory(
    plan: SectionPlan,
    *,
    slater=0.8,
    soc=1.0,
    source_rcg_path: Optional[str | Path] = None,
    cowan_template: Optional[List[List[torch.Tensor]]] = None,
    cowan_metadata: Optional[List[List[CowanBlockMeta]]] = None,
    decomposition: Optional[HamiltonianDecomposition] = None,
    atomic=None,
    device=None,
) -> List[List[torch.Tensor]]:
    """Build a COWAN store whose HAMILTONIAN blocks carry ``slater`` and ``soc``.

    Parameters
    ----------
    plan : SectionPlan
        Section plan from :func:`~multitorch.hamiltonian.build_rac.build_rac_in_memory`
        (cross-check on section sizes).
    slater, soc : float or torch.Tensor
        Absolute reductions of the Hartree-Fock Slater integrals (all F^k and
        G^k) and spin-orbit parameters. ``slater == decomposition.slater_reduction``
        and ``soc == decomposition.soc_reduction`` reproduce the fixture. Tensors
        with ``requires_grad=True`` carry gradients into every HAMILTONIAN block.
    atomic : dict, optional
        Per-configuration overrides of individual parameters (absolute eV),
        keyed by configuration label ``'<section>.<GROUND|EXCITE>'`` and then
        operator name (``F2_11``) or unambiguous alias (``F2dd``); see
        :func:`~multitorch.hamiltonian.parametric.rebuild_hamiltonian_store`.
    source_rcg_path : path-like, optional
        ``.rme_rcg`` fixture; parsed for the template, metadata and (cached)
        decomposition when those are not supplied.
    cowan_template, cowan_metadata : optional
        Pre-parsed store and metadata (skip file I/O).
    decomposition : HamiltonianDecomposition, optional
        Pre-computed decomposition (e.g. from :func:`load_hamiltonian_decomposition`).

    Returns
    -------
    List[List[torch.Tensor]]
        Store with the template's layout; every HAMILTONIAN block rebuilt as
        ``H_fixture + Σ (p_i − p_i^fit)·O_i`` with ``p_i = p_i^fit·slater/slater_reduction``
        (F^k, G^k) or ``p_i^fit·soc/soc_reduction`` (ζ), i.e.
        ``H_fixture + (slater/slater_reduction − 1)·S + (soc/soc_reduction − 1)·Z``
        without overrides (equal to the template at the fixture's reductions);
        every other block the template tensor itself.
    """
    template, dec = _resolve_inputs(plan, source_rcg_path, cowan_template, cowan_metadata, decomposition)
    return rebuild_hamiltonian_store(template, dec, slater=slater, soc=soc, atomic=atomic, device=device)


def build_cowan_store_in_memory_batch(
    plan: SectionPlan,
    *,
    slater_values: torch.Tensor,
    soc_values: torch.Tensor,
    source_rcg_path: Optional[str | Path] = None,
    cowan_template: Optional[List[List[torch.Tensor]]] = None,
    cowan_metadata: Optional[List[List[CowanBlockMeta]]] = None,
    decomposition: Optional[HamiltonianDecomposition] = None,
    device=None,
) -> List[List[torch.Tensor]]:
    """Batch version: N stores from (N,) ``slater_values`` and ``soc_values``.

    Every HAMILTONIAN block has shape (N, dim, dim); all other blocks are the
    (dim, dim) template tensors (broadcast-compatible). The decomposition is
    computed once and shared, so the per-sample cost is one scalar-times-matrix
    addition per (block, parameter).
    """
    slater_values = as_scale(slater_values, device, "slater")
    soc_values = as_scale(soc_values, device, "soc")
    if slater_values.ndim != 1 or soc_values.ndim != 1:
        raise ValueError(
            f"slater_values and soc_values must be 1D, got {tuple(slater_values.shape)} "
            f"and {tuple(soc_values.shape)}"
        )
    if slater_values.shape[0] != soc_values.shape[0]:
        raise ValueError(
            f"Batch size mismatch: {slater_values.shape[0]} slater vs "
            f"{soc_values.shape[0]} soc values"
        )
    template, dec = _resolve_inputs(plan, source_rcg_path, cowan_template, cowan_metadata, decomposition)
    return rebuild_hamiltonian_store(template, dec, slater=slater_values, soc=soc_values, device=device)
