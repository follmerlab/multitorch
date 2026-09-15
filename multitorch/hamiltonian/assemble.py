"""
Full Hamiltonian assembly from .rme_rcg, .rme_rac, and .ban files.

This module implements the PyTorch equivalent of the Fortran subroutines
MASTER, PAIRIN, COWAN, ETR, and make_hamiltonian from ttban_exact.f.

The assembly pipeline:
  1. Read COWAN store from .rme_rcg (indexed sparse matrices)
  2. Parse .rme_rac for block assembly instructions (ADD entries)
  3. Parse .ban for physical parameters (XHAM, XMIX, energy offsets)
  4. For each symmetry triad (gs_irrep, actor_irrep, fs_irrep):
     a. Build block-diagonal Hamiltonian H_gs from conf 1 + conf 2 blocks
     b. Add off-diagonal charge transfer mixing blocks (HYBR × XMIX)
     c. Apply 1/sqrt(IDIM) normalization and energy offsets
     d. Diagonalize to get eigenvalues and eigenvectors
     e. Build transition matrix similarly for excited state
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import torch

from multitorch._constants import DTYPE
from multitorch.hamiltonian.diagonalize import safe_eigh
from multitorch.io.read_rme import (
    read_cowan_store, read_rme_rac_full,
    RACFileFull, RACBlockFull, ADDEntry,
    assemble_matrix_from_adds,
)
from multitorch.io.read_ban import read_ban, BanData


# ─────────────────────────────────────────────────────────────
# Block mapping: find the right .rme_rac blocks for each irrep/config
# ─────────────────────────────────────────────────────────────

# Operator names in the order they appear in XHAM values
# For D4h: [HAMILTONIAN, 10DQ, DT, DS]  (XHAM has 4 values: 1.0, tendq, dt, ds)
# For Oh: [HAMILTONIAN, 10DQ]  (XHAM has 2 values: 1.0, tendq)
OPERATOR_ORDER_D4H = ['HAMILTONIAN', '10DQ', 'DT', 'DS']
OPERATOR_ORDER_OH = ['HAMILTONIAN', '10DQ']

# Hybridization channel names for XMIX
# For D4h: [B1HYBR, A1HYBR, B2HYBR, EHYBR]
# For Oh: [EGHYBR, T2GHYBR]  (or similar)
HYBR_ORDER_D4H = ['B1HYBR', 'A1HYBR', 'B2HYBR', 'EHYBR']
HYBR_ORDER_OH = ['EGHYBR', 'T2GHYBR']


def _is_constant_zero(xv) -> bool:
    """A plain-number zero coefficient: its block can be skipped.

    Tensors are never skipped, even at exactly 0, because a parameter sitting
    at 0 (e.g. ``dt=ds=0`` as a fit start) still has a gradient through its
    block.
    """
    return not isinstance(xv, torch.Tensor) and xv == 0.0


def _find_operator_blocks(
    rac: RACFileFull,
    kind: str,            # 'GROUND' or 'EXCITE'
    irrep: str,           # e.g. '0+', '1+'
    operators: List[str],  # e.g. ['HAMILTONIAN', '10DQ', 'DT', 'DS']
    n_dim: int,           # expected dimension
    occurrence: int = 0,  # which occurrence (0=first, 1=second, etc.)
) -> List[Optional[RACBlockFull]]:
    """Find the N-th occurrence of each operator block for a given kind+irrep."""
    result = []
    for op in operators:
        matches = []
        for b in rac.blocks:
            if (b.kind == kind and b.bra_sym == irrep
                    and b.geometry == op and b.n_bra == n_dim and b.n_ket == n_dim
                    and b.add_entries):
                matches.append(b)
        if occurrence < len(matches):
            result.append(matches[occurrence])
        else:
            result.append(None)
    return result


def _find_hybr_blocks(
    rac: RACFileFull,
    irrep: str,
    channels: List[str],
    n_bra: int,
    n_ket: int,
) -> List[Optional[RACBlockFull]]:
    """Find hybridization blocks for a given irrep."""
    result = []
    for ch in channels:
        found = None
        for b in rac.blocks:
            if (b.kind == 'TRANSI' and b.bra_sym == irrep
                    and b.geometry == ch and b.n_bra == n_bra and b.n_ket == n_ket):
                found = b
                break
        result.append(found)
    return result


def _find_transi_block(
    rac: RACFileFull,
    gs_sym: str,
    act_sym: str,
    fs_sym: str,
    geometry: str,
    n_bra: int,
    n_ket: int,
    copy: int = 0,
) -> Optional[RACBlockFull]:
    """Find a TRANSI block for a specific triad, geometry and multiplicity copy.

    Tries exact geometry match first, then falls back to any geometry
    matching the symmetry labels and dimensions (handles format differences
    between ttrac versions that use PERP/PARA vs MULTIPOLE).

    ``copy`` selects among repeated couplings of the same triad (PRMULT: a
    triad whose actor occurs twice in Γ_gs ⊗ Γ_fs, listed in the .ban as
    ``S1+ 1- S1- 0`` and ``S1+ 1- S1- 1``), which ttrac writes as consecutive
    blocks of equal dimensions.
    """
    def matches(check_geometry):
        return [b for b in rac.blocks
                if b.kind == 'TRANSI' and b.bra_sym == gs_sym
                and b.op_sym == act_sym and b.ket_sym == fs_sym
                and (not check_geometry or b.geometry == geometry)
                and b.n_bra == n_bra and b.n_ket == n_ket]

    found = matches(True) or matches(False)
    if copy >= len(found):
        if copy and found:
            raise ValueError(f"triad ({gs_sym}, {act_sym}, {fs_sym}) copy {copy}: the RAC has only "
                             f"{len(found)} TRANSI block(s) of dimensions {n_bra}x{n_ket}")
        return None
    return found[copy]


# ─────────────────────────────────────────────────────────────
# Main assembly
# ─────────────────────────────────────────────────────────────

@dataclass
class TriadResult:
    """Results for one symmetry triad after Hamiltonian diagonalization."""
    gs_sym: str
    act_sym: str
    fs_sym: str
    # Ground state
    Eg: torch.Tensor            # eigenvalues (n_gs,) in eV
    Ug: torch.Tensor            # eigenvectors (n_gs, n_gs)
    gs_conf_labels: torch.Tensor  # config label (1-based) for each state
    gs_conf_sizes: List[int]
    n_gs: int
    # Final state
    Ef: torch.Tensor            # eigenvalues (n_fs,) in eV
    Uf: torch.Tensor            # eigenvectors (n_fs, n_fs)
    fs_conf_labels: torch.Tensor
    fs_conf_sizes: List[int]
    n_fs: int
    # Transition matrix (in eigenbasis)
    T: torch.Tensor             # (n_gs, n_fs) transition matrix elements


@dataclass
class BanResult:
    """Full results from ttban-style Hamiltonian assembly and diagonalization."""
    triads: List[TriadResult]
    ban: BanData

    def get_triad(self, gs_sym: str, act_sym: str, fs_sym: str) -> Optional[TriadResult]:
        for t in self.triads:
            if t.gs_sym == gs_sym and t.act_sym == act_sym and t.fs_sym == fs_sym:
                return t
        return None


def assemble_and_diagonalize(
    rme_rcg_path: str | Path,
    rme_rac_path: str | Path,
    ban_path: str | Path,
    device: str = 'cpu',
) -> BanResult:
    """
    Full Hamiltonian assembly and diagonalization for a charge transfer
    calculation, **file-based entry point**.

    This is the PyTorch equivalent of the Fortran ttban_exact.f program.
    It reads the .rme_rcg, .rme_rac, and .ban files, then forwards to the
    in-memory core :func:`assemble_and_diagonalize_in_memory`.

    For autograd / Phase 5 (parameters → spectrum) you should call the
    in-memory variant directly with tensors that carry ``requires_grad``;
    going through this wrapper round-trips through plain Python objects
    and parsers, which breaks gradient flow.

    Parameters
    ----------
    rme_rcg_path : path to .rme_rcg file (ttrcg output)
    rme_rac_path : path to .rme_rac file (ttrac output)
    ban_path : path to .ban file
    device : PyTorch device

    Returns
    -------
    BanResult with eigenvalues, eigenvectors, and transition matrices per triad.
    """
    cowan = read_cowan_store(rme_rcg_path)
    rac = read_rme_rac_full(rme_rac_path)
    ban = read_ban(ban_path)
    return assemble_and_diagonalize_in_memory(cowan, rac, ban, device=device)


def assemble_and_diagonalize_in_memory(
    cowan,
    rac: RACFileFull,
    ban: BanData,
    device: str = 'cpu',
) -> BanResult:
    """
    In-memory core of the Hamiltonian assembly + diagonalization pipeline.

    This is the gradient-friendly entry point: ``cowan``, ``rac``, and
    ``ban`` may contain torch.Tensor leaves with ``requires_grad=True``
    (e.g. crystal-field coefficients or Slater reductions built from
    physical parameters in Phase 5), and the resulting transition
    matrices in the returned ``BanResult`` will participate in autograd.

    The signature accepts whatever objects the existing parsers
    (:func:`read_cowan_store`, :func:`read_rme_rac_full`, :func:`read_ban`)
    produce, so the file-based wrapper is byte-equivalent.

    Parameters
    ----------
    cowan : nested COWAN store as returned by ``read_cowan_store``
        (currently ``List[List[Tensor]]``: per-section, per-add-entry).
    rac : :class:`RACFileFull`
        Parsed .rme_rac block + ADD-entry structure.
    ban : :class:`BanData`
        Parsed .ban specification (XHAM/XMIX coefficients, EG/EF offsets,
        triads).
    device : PyTorch device.

    Returns
    -------
    BanResult with eigenvalues, eigenvectors, and transition matrices per triad.
    """
    c = _assembly_context(rac, ban)

    # 2. Process each triad. Triads share ground and final irreps (Ni d8 D4h CT:
    # 13 triads, 5 + 5 irreps); each irrep's Hamiltonian is built and
    # diagonalised once per call and reused, which also shares the autograd graph.
    results = []
    memo: dict = {}
    copies: Dict[Tuple[str, str, str], int] = {}
    for gs_sym, act_sym, fs_sym in ban.triads:
        # A triad listed k times in the .ban (PRMULT copies 0..k-1) uses the k-th
        # TRANSI block of that triad; the Hamiltonians are shared.
        copy = copies.get((gs_sym, act_sym, fs_sym), 0)
        copies[(gs_sym, act_sym, fs_sym)] = copy + 1
        triad = _assemble_one_triad(
            rac, cowan, ban,
            gs_sym, act_sym, fs_sym,
            c["operators"], c["hybr_channels"],
            c["xham"], c["xmix"],
            c["eg_offsets"], c["ef_offsets"],
            c["gs_dims"], c["fs_dims"],
            c["irrep_dim"],
            c["gs_cowan_sec"], c["fs_cowan_sec"],
            c["nconf"],
            device,
            memo=memo,
            copy=copy,
        )
        if triad is not None:
            results.append(triad)

    return BanResult(triads=results, ban=ban)


def _assembly_context(rac: RACFileFull, ban: BanData) -> dict:
    """Operator names, parameters, offsets, sections and per-config dimensions shared by all triads."""
    nconf = ban.nconf_gs
    xham = ban.xham[0].values if ban.xham else [1.0]
    xmix = ban.xmix[0].values if ban.xmix else []

    # Energy offsets
    eg_offsets = [ban.eg.get(i, 0.0) for i in range(1, nconf + 1)]
    ef_offsets = [ban.ef.get(i, 0.0) for i in range(1, nconf + 1)]

    # Determine operator and hybridization channel names from XHAM length
    n_ops = len(xham)
    if n_ops == 4:
        operators = OPERATOR_ORDER_D4H
        hybr_channels = HYBR_ORDER_D4H
    elif n_ops == 2:
        operators = OPERATOR_ORDER_OH
        hybr_channels = HYBR_ORDER_OH
    else:
        operators = OPERATOR_ORDER_D4H[:n_ops]
        hybr_channels = HYBR_ORDER_D4H[:len(xmix)]

    # Get irrep dimensions for the 1/sqrt(IDIM) factor
    irrep_dim = {}
    for ir in rac.irreps:
        irrep_dim[ir.name] = ir.dim

    # Get ground and excited irrep multiplicities (state counts per irrep per config)
    gs_irrep_info = {}  # irrep -> {'kind': 'GROUND', 'mult': N}
    fs_irrep_info = {}
    for ir in rac.irreps:
        if ir.kind == 'GROUND':
            gs_irrep_info[ir.name] = ir.multiplicity
        elif ir.kind == 'EXCITE':
            fs_irrep_info[ir.name] = ir.multiplicity

    # Determine COWAN sections:
    # For 2-config CT: sections [0,1] = dipole RME, [2] = ground mixing, [3] = excited mixing
    gs_cowan_sec = 2 if nconf >= 2 else 0
    fs_cowan_sec = 3 if nconf >= 2 else 0

    # Determine per-config dimensions from the TRANSI blocks in .rme_rac.
    # TRANSI blocks appear in order: conf 1 then conf 2. The bra dimension
    # gives the ground state size, the ket dimension gives the final state size.
    gs_dims, fs_dims = _get_config_dims_from_transi(rac, nconf)

    return dict(
        nconf=nconf, xham=xham, xmix=xmix, eg_offsets=eg_offsets, ef_offsets=ef_offsets,
        operators=operators, hybr_channels=hybr_channels, irrep_dim=irrep_dim,
        gs_dims=gs_dims, fs_dims=fs_dims, gs_cowan_sec=gs_cowan_sec, fs_cowan_sec=fs_cowan_sec,
    )


def assemble_ground_hamiltonians(cowan, rac: RACFileFull, ban: BanData, device=None) -> Dict[str, tuple]:
    """Ground-state Hamiltonian of every ground irrep, without the final states.

    Returns ``{gs_sym: (H_gs, conf_labels, conf_sizes)}``, each ``H_gs``
    exactly the matrix :func:`assemble_and_diagonalize_in_memory` diagonalises
    for that irrep (configuration blocks, energy offsets, hybridisation).
    Used for ground-state analysis (:mod:`multitorch.analysis.ground_state`),
    where assembling the much larger final-state blocks would be wasted work.
    """
    c = _assembly_context(rac, ban)
    out: Dict[str, tuple] = {}
    for gs_sym, _, _ in ban.triads:
        if gs_sym in out:
            continue
        built = _ground_hamiltonian(
            rac, cowan, gs_sym, c["operators"], c["hybr_channels"], c["xham"], c["xmix"],
            c["eg_offsets"], c["gs_dims"], c["irrep_dim"], c["gs_cowan_sec"], c["nconf"], device,
        )
        if built is not None:
            out[gs_sym] = built
    return out


def _get_config_dims_from_transi(
    rac: RACFileFull, nconf: int,
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """
    Determine per-config dimensions for ground and final state irreps
    from TRANSI blocks.

    TRANSI blocks appear in config order in the .rme_rac file. For a 2-config
    CT calculation, the first set of TRANSI blocks has conf 1 dimensions
    and the second set has conf 2 dimensions.

    Returns
    -------
    gs_dims : dict mapping gs_irrep → [dim_conf1, dim_conf2, ...]
    fs_dims : dict mapping fs_irrep → [dim_conf1, dim_conf2, ...]
    """
    # Collect TRANSI blocks grouped by (gs_sym, act_sym, fs_sym, geometry)
    transi_groups: Dict[Tuple[str, str, str, str], List[RACBlockFull]] = {}
    for b in rac.blocks:
        if b.kind == 'TRANSI' and b.add_entries:
            # Skip HYBR blocks (they're also labeled TRANSI)
            if 'HYBR' in b.geometry:
                continue
            key = (b.bra_sym, b.op_sym, b.ket_sym, b.geometry)
            if key not in transi_groups:
                transi_groups[key] = []
            transi_groups[key].append(b)

    gs_dims: Dict[str, List[int]] = {}
    fs_dims: Dict[str, List[int]] = {}

    for (gs_sym, act_sym, fs_sym, geom), blocks in transi_groups.items():
        # blocks are in config order: conf 1, conf 2, ... A triad with a
        # repeated (PRMULT) coupling lists every copy, grouped by configuration
        # (conf 1 copy 1, conf 1 copy 2, conf 2 copy 1, ...). Counting each copy
        # as a configuration duplicated the ligand-hole block without its energy
        # offset or hybridisation for the Γ8 irreps of every half-integer-J
        # fixture (Cr3+, Mn2+, Fe3+, Co2+): dark phantom states, a 3810- instead
        # of 1410-state Fe3+ final block (ttban prints 1410).
        if len(blocks) > nconf and len(blocks) % nconf == 0:
            blocks = blocks[::len(blocks) // nconf]
        if gs_sym not in gs_dims:
            gs_dims[gs_sym] = []
        if fs_sym not in fs_dims:
            fs_dims[fs_sym] = []

        for i, blk in enumerate(blocks):
            # Append dimension if we don't have enough configs yet
            if i >= len(gs_dims[gs_sym]):
                gs_dims[gs_sym].append(blk.n_bra)
            if i >= len(fs_dims[fs_sym]):
                fs_dims[fs_sym].append(blk.n_ket)

    return gs_dims, fs_dims


def _ground_hamiltonian(
    rac, cowan, gs_sym,
    operators, hybr_channels,
    xham, xmix,
    eg_offsets, gs_dims, irrep_dim,
    gs_cowan_sec, nconf, device,
):
    """Assemble one ground irrep's Hamiltonian: ``(H_gs, conf_labels, conf_sizes)`` or ``None``."""
    # Get dimensions per config for this irrep
    gs_conf_sizes = gs_dims.get(gs_sym, [])

    if not gs_conf_sizes:
        return None

    idim_gs = irrep_dim.get(gs_sym, 1)
    idim_scale_gs = 1.0 / math.sqrt(idim_gs)

    # ── Build ground state Hamiltonian ──
    n_gs = sum(gs_conf_sizes)
    H_gs = torch.zeros(n_gs, n_gs, dtype=DTYPE, device=device)

    sec = cowan[gs_cowan_sec] if gs_cowan_sec < len(cowan) else []

    # Diagonal blocks for each configuration
    offset = 0
    conf_labels_gs = torch.zeros(n_gs, dtype=torch.int64, device=device)
    for ic, d in enumerate(gs_conf_sizes):
        # In the .rme_rac file, the first config's operator blocks are labeled
        # GROUND and the second config's are labeled EXCITE (even though both
        # contribute to the ground-state manifold in a CT calculation).
        kind = 'GROUND' if ic == 0 else 'EXCITE'
        op_blocks = _find_operator_blocks(rac, kind, gs_sym, operators, d)

        H_block = torch.zeros(d, d, dtype=DTYPE, device=device)
        for blk, xv in zip(op_blocks, xham):
            if blk is not None and blk.add_entries and not _is_constant_zero(xv):
                H_block += assemble_matrix_from_adds(blk.add_entries, sec, d, d, scale=xv, device=device)

        # Symmetrize and apply IDIM scaling
        H_block = 0.5 * (H_block + H_block.T) * idim_scale_gs

        # Add energy offset (NOT scaled by IDIM)
        eg = eg_offsets[ic] if ic < len(eg_offsets) else 0.0
        H_block += eg * torch.eye(d, dtype=DTYPE, device=device)

        H_gs[offset:offset + d, offset:offset + d] = H_block
        conf_labels_gs[offset:offset + d] = ic + 1  # 1-based
        offset += d

    # Off-diagonal mixing blocks (if nconf >= 2)
    if nconf >= 2 and len(gs_conf_sizes) >= 2 and xmix:
        d1 = gs_conf_sizes[0]
        d2 = gs_conf_sizes[1]
        hybr_blocks = _find_hybr_blocks(rac, gs_sym, hybr_channels, d1, d2)

        V = torch.zeros(d1, d2, dtype=DTYPE, device=device)
        for blk, xv in zip(hybr_blocks, xmix):
            if blk is not None and blk.add_entries and not _is_constant_zero(xv):
                V += assemble_matrix_from_adds(blk.add_entries, sec, d1, d2, scale=xv, device=device)

        V *= idim_scale_gs
        H_gs[:d1, d1:d1 + d2] = V
        H_gs[d1:d1 + d2, :d1] = V.T

    return H_gs, conf_labels_gs, gs_conf_sizes



def _final_hamiltonian(
    rac, cowan, fs_sym,
    operators, hybr_channels,
    xham, xmix,
    ef_offsets, fs_conf_sizes, irrep_dim,
    fs_cowan_sec, nconf, device,
):
    """Assemble one final irrep's Hamiltonian: ``(H_fs, conf_labels)``."""
    idim_scale_fs = 1.0 / math.sqrt(irrep_dim.get(fs_sym, 1))
    n_fs = sum(fs_conf_sizes)
    H_fs = torch.zeros(n_fs, n_fs, dtype=DTYPE, device=device)

    sec_fs = cowan[fs_cowan_sec] if fs_cowan_sec < len(cowan) else []

    offset = 0
    conf_labels_fs = torch.zeros(n_fs, dtype=torch.int64, device=device)
    for ic, d in enumerate(fs_conf_sizes):
        kind = 'GROUND' if ic == 0 else 'EXCITE'
        op_blocks = _find_operator_blocks(rac, kind, fs_sym, operators, d)

        H_block = torch.zeros(d, d, dtype=DTYPE, device=device)
        for blk, xv in zip(op_blocks, xham):
            if blk is not None and blk.add_entries and not _is_constant_zero(xv):
                H_block += assemble_matrix_from_adds(blk.add_entries, sec_fs, d, d, scale=xv, device=device)

        H_block = 0.5 * (H_block + H_block.T) * idim_scale_fs
        ef = ef_offsets[ic] if ic < len(ef_offsets) else 0.0
        H_block += ef * torch.eye(d, dtype=DTYPE, device=device)

        H_fs[offset:offset + d, offset:offset + d] = H_block
        conf_labels_fs[offset:offset + d] = ic + 1
        offset += d

    # Excited state mixing
    if nconf >= 2 and len(fs_conf_sizes) >= 2 and xmix:
        d1 = fs_conf_sizes[0]
        d2 = fs_conf_sizes[1]
        # Excited state HYBR blocks have - parity
        hybr_blocks = _find_hybr_blocks(rac, fs_sym, hybr_channels, d1, d2)

        V_fs = torch.zeros(d1, d2, dtype=DTYPE, device=device)
        for blk, xv in zip(hybr_blocks, xmix):
            if blk is not None and blk.add_entries and not _is_constant_zero(xv):
                V_fs += assemble_matrix_from_adds(blk.add_entries, sec_fs, d1, d2, scale=xv, device=device)

        V_fs *= idim_scale_fs
        H_fs[:d1, d1:d1 + d2] = V_fs
        H_fs[d1:d1 + d2, :d1] = V_fs.T

    return H_fs, conf_labels_fs


def _assemble_one_triad(
    rac, cowan, ban,
    gs_sym, act_sym, fs_sym,
    operators, hybr_channels,
    xham, xmix,
    eg_offsets, ef_offsets,
    gs_dims, fs_dims,
    irrep_dim,
    gs_cowan_sec, fs_cowan_sec,
    nconf, device,
    memo: Optional[dict] = None,
    copy: int = 0,
) -> Optional[TriadResult]:
    """Assemble and diagonalize one symmetry triad.

    ``memo`` (one dict per assembly) caches each ground and final irrep's
    diagonalised Hamiltonian across the triads that share it.
    """
    memo = {} if memo is None else memo
    if ('gs', gs_sym) not in memo:
        built = _ground_hamiltonian(
            rac, cowan, gs_sym, operators, hybr_channels, xham, xmix,
            eg_offsets, gs_dims, irrep_dim, gs_cowan_sec, nconf, device,
        )
        memo[('gs', gs_sym)] = None if built is None else (built, safe_eigh(built[0]))
    if memo[('gs', gs_sym)] is None:
        return None
    (H_gs, conf_labels_gs, gs_conf_sizes), (Eg, Ug) = memo[('gs', gs_sym)]
    n_gs = H_gs.shape[0]
    fs_conf_sizes = fs_dims.get(fs_sym, [])

    # ── Final state Hamiltonian ──
    if not fs_conf_sizes:
        # No final state for this triad
        n_fs = 0
        Ef = torch.zeros(0, dtype=DTYPE, device=device)
        Uf = torch.zeros(0, 0, dtype=DTYPE, device=device)
        conf_labels_fs = torch.zeros(0, dtype=torch.int64, device=device)
        T_eig = torch.zeros(n_gs, 0, dtype=DTYPE, device=device)
    else:
        if ('fs', fs_sym) not in memo:
            H_fs, conf_labels_fs = _final_hamiltonian(
                rac, cowan, fs_sym, operators, hybr_channels, xham, xmix, ef_offsets,
                fs_conf_sizes, irrep_dim, fs_cowan_sec, nconf, device,
            )
            memo[('fs', fs_sym)] = (conf_labels_fs, safe_eigh(H_fs))
        conf_labels_fs, (Ef, Uf) = memo[('fs', fs_sym)]
        n_fs = sum(fs_conf_sizes)

        # ── Build transition matrix ──
        # T_raw[gs_state, fs_state] assembled from TRANSI blocks
        T_raw = torch.zeros(n_gs, n_fs, dtype=DTYPE, device=device)

        # For each (gs_conf, fs_conf) pair that has transitions
        gs_offset = 0
        for icg, dg in enumerate(gs_conf_sizes):
            fs_offset = 0
            for icf, df in enumerate(fs_conf_sizes):
                # Check if this pair has a transition (from TRAN in .ban)
                has_tran = (icg + 1, icf + 1) in ban.tran
                if has_tran:
                    # Use the appropriate COWAN section and geometry
                    # Transitions use sections 0 (conf1) and 1 (conf2)
                    tran_cowan_sec = icg  # section 0 for conf1, 1 for conf2
                    tran_sec = cowan[tran_cowan_sec] if tran_cowan_sec < len(cowan) else []

                    # Determine geometry from actor symmetry
                    if act_sym in ('1-',):
                        geom = 'PERP'
                    elif act_sym in ('^0-',):
                        geom = 'PARA'
                    else:
                        geom = 'PERP'

                    tran_block = _find_transi_block(
                        rac, gs_sym, act_sym, fs_sym, geom, dg, df, copy=copy,
                    )
                    if tran_block is not None:
                        T_sub = assemble_matrix_from_adds(
                            tran_block.add_entries, tran_sec, dg, df, scale=1.0, device=device,
                        )
                        T_raw[gs_offset:gs_offset + dg, fs_offset:fs_offset + df] = T_sub

                fs_offset += df
            gs_offset += dg

        # Transform to eigenbasis: T_eig = Ug^T @ T_raw @ Uf
        T_eig = Ug.T @ T_raw @ Uf

    return TriadResult(
        gs_sym=gs_sym,
        act_sym=act_sym,
        fs_sym=fs_sym,
        Eg=Eg, Ug=Ug,
        gs_conf_labels=conf_labels_gs,
        gs_conf_sizes=gs_conf_sizes,
        n_gs=n_gs,
        Ef=Ef, Uf=Uf,
        fs_conf_labels=conf_labels_fs,
        fs_conf_sizes=fs_conf_sizes,
        n_fs=n_fs,
        T=T_eig,
    )
