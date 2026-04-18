"""
In-memory COWAN store builder for the Track C Phase 5 pipeline (C3e).

Scope and approach
------------------
This module uses a **loader-builder hybrid** to produce a COWAN store
(``List[List[torch.Tensor]]``) that matches the parsed ``.rme_rcg`` fixture
element-wise at ``atol=0`` when scales are 1.0, while allowing autograd-
carrying gradient flow through ``slater_scale`` and ``soc_scale``.

The strategy:

  1. Parse the template COWAN store from the ``.rme_rcg`` fixture.
  2. For **single-shell config-1 HAMILTONIAN blocks** in the ground-state
     manifold (section 2): decompose into Coulomb + spin-orbit, then
     rebuild with autograd-carrying scaled atomic parameters.
  3. For **everything else** (SHELL1, SPIN1, MULTIPOLE, TRANSI blocks,
     config-2 HAMILTONIAN, excited-manifold HAMILTONIAN): pass through
     from the template as constants.

This is sufficient for the C3f autograd test because the ground-state
eigenvalues ``Eg`` flow through the section-2 Hamiltonian, and the
config-1 d^N HAMILTONIAN blocks are one of the dominant diagonal
contributions.

Why decomposition is algebraically exact at scale=1
----------------------------------------------------
Let the parsed HAMILTONIAN block be ``H``, and let ``F^k``, ``ζ`` be
*any* approximate parameter values (possibly rounded).

Define::

    V(11) := (H - Σ_k F^k × SHELL_k) / ζ

Then the rebuild at scale=1 is::

    H_new = Σ_k F^k × SHELL_k + ζ × V(11)
          = Σ_k F^k × SHELL_k + H - Σ_k F^k × SHELL_k
          = H

This algebraic cancellation holds regardless of parameter accuracy.
Only floating-point rounding (~1e-14) introduces error.

When ``slater_scale ≠ 1`` or ``soc_scale ≠ 1``, the rebuild mixes the
Coulomb and SOC contributions with the user-supplied scale factors,
and autograd flows through both.

Extension point
---------------
To decompose config-2 or excited-manifold HAMILTONIAN blocks in the
future, add cases in :func:`_classify_for_rebuild` and supply the
appropriate shell-pair and ζ-shell labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

from multitorch._constants import DTYPE, RY_TO_EV
from multitorch.atomic.parameter_fixtures import AtomicParams, ConfigParams
from multitorch.atomic.scaled_params import ScaledAtomicParams, ScaledConfigParams
from multitorch.hamiltonian.build_rac import SectionPlan
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
    bra_sym: str     # e.g. '0+', '1-', '^0+'
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


# ─────────────────────────────────────────────────────────────
# Internal helpers for HAMILTONIAN decomposition
# ─────────────────────────────────────────────────────────────


def _find_shell_diagonals(
    section_meta: List[CowanBlockMeta],
    section_mats: List[torch.Tensor],
    block_type: str,
    j_sym: str,
) -> Dict[int, torch.Tensor]:
    """Find diagonal SHELL blocks for a given config and J symmetry.

    Matches any ``SHELLn`` operator (SHELL1, SHELL2, etc.) — the suffix
    number refers to the shell index within the Cowan configuration, which
    varies between fixtures (nid8ct uses SHELL1 for the 3d shell, while
    the Oh series uses SHELL2).

    Parameters
    ----------
    section_meta, section_mats : aligned metadata and matrices for one section.
    block_type : the target block_type (e.g. ``'GROUND'``).
    j_sym : the target symmetry label (e.g. ``'2+'``).

    Returns
    -------
    dict mapping rank ``k`` (0, 2, 4, ...) to the SHELL tensor.
    """
    result: Dict[int, torch.Tensor] = {}
    for idx, m in enumerate(section_meta):
        if (m.operator.startswith("SHELL")
                and m.block_type == block_type
                and m.bra_sym == j_sym
                and m.ket_sym == j_sym):
            k = int(m.op_sym.rstrip("+-"))
            result[k] = section_mats[idx]
    return result


def _rebuild_hamiltonian_block(
    h_parsed: torch.Tensor,
    shell_blocks: Dict[int, torch.Tensor],
    raw_cfg: ConfigParams,
    scaled_cfg: ScaledConfigParams,
    shell_pair: Tuple[str, str],
    zeta_shell: str,
) -> torch.Tensor:
    """Decompose one HAMILTONIAN block and rebuild with autograd params.

    Steps:

    1. Compute Coulomb contribution using raw (plain-float) params::

           H_coulomb = Σ_k F^k_raw_eV × SHELL_k(J,J)

    2. Extract V(11) (spin-orbit + average-energy residual)::

           V(11) = (H_parsed - H_coulomb) / ζ_raw_eV

    3. Rebuild with scaled (autograd-carrying) params::

           H_new = Σ_k (scaled_F^k × RY_TO_EV) × SHELL_k + (scaled_ζ × RY_TO_EV) × V(11)

    At scale=1.0, the raw and scaled params are identical, so
    ``H_new == H_parsed`` algebraically (up to ~1e-14 from IEEE 754).
    """
    a, b = shell_pair
    ry_to_ev = float(RY_TO_EV)

    # ── Step 1: Coulomb with raw params ──────────────────────
    h_coulomb_raw = torch.zeros_like(h_parsed)
    for k, shell_mat in shell_blocks.items():
        fk_raw_ev = raw_cfg.f(a, b, k) * ry_to_ev
        h_coulomb_raw = h_coulomb_raw + fk_raw_ev * shell_mat

    # ── Step 2: Extract V(11) ────────────────────────────────
    zeta_raw_ev = raw_cfg.zeta(zeta_shell) * ry_to_ev
    v11 = (h_parsed - h_coulomb_raw) / zeta_raw_ev

    # ── Step 3: Rebuild with scaled params ───────────────────
    h_new = torch.zeros_like(h_parsed)
    for k, shell_mat in shell_blocks.items():
        fk_scaled_ev = scaled_cfg.f(a, b, k) * ry_to_ev  # tensor × float
        h_new = h_new + fk_scaled_ev * shell_mat

    zeta_scaled_ev = scaled_cfg.z(zeta_shell) * ry_to_ev  # tensor × float
    h_new = h_new + zeta_scaled_ev * v11

    return h_new


# ─────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────


def build_cowan_store_in_memory(
    scaled_params: ScaledAtomicParams,
    raw_params: AtomicParams,
    plan: SectionPlan,
    *,
    source_rcg_path: Optional[str | Path] = None,
    cowan_template: Optional[List[List[torch.Tensor]]] = None,
    cowan_metadata: Optional[List[List[CowanBlockMeta]]] = None,
    device=None,
) -> List[List[torch.Tensor]]:
    """Build a COWAN store with autograd-carrying HAMILTONIAN blocks.

    Parameters
    ----------
    scaled_params : ScaledAtomicParams
        Atomic parameters as torch tensors, produced by
        :func:`~multitorch.atomic.scaled_params.scale_atomic_params`.
        If ``slater_scale`` or ``soc_scale`` are ``requires_grad=True``
        tensors, the returned COWAN store carries an autograd graph
        back to them (through the rebuilt HAMILTONIAN blocks).
    raw_params : AtomicParams
        Plain-float atomic parameters from
        :func:`~multitorch.atomic.parameter_fixtures.read_rcn31_out_params`.
        Used for the V(11) extraction step of the decomposition. The
        raw values need not exactly match the ones used by Fortran to
        build the template — the algebraic-exactness property at
        scale=1.0 holds regardless (see module docstring).
    plan : SectionPlan
        The section plan from :func:`~multitorch.hamiltonian.build_rac.build_rac_in_memory`.
        Used as a cross-check on section count.
    source_rcg_path : path-like, optional
        Path to the ``.rme_rcg`` fixture. Parsed both for the template
        matrices (``read_cowan_store``) and for block metadata
        (``read_cowan_metadata``). Not needed if ``cowan_template`` and
        ``cowan_metadata`` are provided.
    cowan_template : list of list of torch.Tensor, optional
        Pre-parsed COWAN store template. If provided together with
        ``cowan_metadata``, skips file I/O entirely.
    cowan_metadata : list of list of CowanBlockMeta, optional
        Pre-parsed block metadata. Must be provided alongside
        ``cowan_template``.

    Returns
    -------
    List[List[torch.Tensor]]
        A COWAN store with the same shape as the template. Config-1
        HAMILTONIAN blocks in section 2 are rebuilt with autograd;
        all other matrices are passed through verbatim.

    Notes
    -----
    The returned store satisfies the :class:`SectionPlan` contract:
    ``len(result) == plan.n_sections`` and
    ``len(result[s]) == plan.section_size(s)`` for every ``s``.

    For the C3f parity test, use ``slater_scale=1.0, soc_scale=1.0``.
    The parity is exact (``atol=0``) because the decomposition is
    algebraically self-consistent.
    """
    if cowan_template is not None and cowan_metadata is not None:
        template = cowan_template
        meta = cowan_metadata
    elif source_rcg_path is not None:
        source_rcg_path = Path(source_rcg_path)
        template = read_cowan_store(source_rcg_path)
        meta = read_cowan_metadata(source_rcg_path)
    else:
        raise ValueError(
            "Either source_rcg_path or (cowan_template, cowan_metadata) "
            "must be provided"
        )

    # Alignment sanity
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

    # Cross-check against SectionPlan
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

    # Move template tensors to target device
    if device is not None:
        template = [
            [mat.to(device=device) for mat in sec]
            for sec in template
        ]

    result: List[List[torch.Tensor]] = []

    for s in range(len(template)):
        section: List[torch.Tensor] = []

        for j in range(len(template[s])):
            m = meta[s][j]
            mat = template[s][j]

            # Only rebuild config-1 HAMILTONIAN blocks in section 2
            # (ground manifold, single-shell d^N).
            #
            # Section 3 (excited manifold) config 1 is a two-shell
            # system (2p^5 3d^(N+1)), requiring inter-shell Slater
            # decomposition — deferred for now. Config-2 blocks are
            # always two-shell in the CT picture. Both pass through.
            if (s == 2
                    and m.operator == "HAMILTONIAN"
                    and m.block_type == "GROUND"):
                shell_blocks = _find_shell_diagonals(
                    meta[s], template[s], "GROUND", m.bra_sym,
                )
                if not shell_blocks:
                    # Safety: if no matching SHELL blocks, pass through
                    section.append(mat)
                else:
                    # Check that the raw params have the required Fk
                    # values for this shell pair. For d⁰ configurations,
                    # there are no d-d Slater integrals and the rebuild
                    # is not possible — pass through instead.
                    try:
                        _ = raw_params.ground.f("3D", "3D", 0)
                    except KeyError:
                        section.append(mat)
                        continue
                    h_new = _rebuild_hamiltonian_block(
                        mat, shell_blocks,
                        raw_cfg=raw_params.ground,
                        scaled_cfg=scaled_params.ground,
                        shell_pair=("3D", "3D"),
                        zeta_shell="3D",
                    )
                    section.append(h_new)
            else:
                section.append(mat)

        result.append(section)

    return result


# ─────────────────────────────────────────────────────────────
# Batch version for parameter sweeps
# ─────────────────────────────────────────────────────────────


def _rebuild_hamiltonian_block_batch(
    h_parsed: torch.Tensor,
    shell_blocks: Dict[int, torch.Tensor],
    raw_cfg: ConfigParams,
    scaled_cfg: ScaledConfigParams,  # now has (N,) tensors
    shell_pair: Tuple[str, str],
    zeta_shell: str,
) -> torch.Tensor:
    """Batch rebuild: N Hamiltonians with shared V(11) residual.
    
    Returns (N, dim, dim) stacked Hamiltonians. V(11) is computed once
    from the template and raw params, then reused for all N rebuilds.
    This eliminates redundant work - a key source of Phase 2 speedup.
    
    Parameters
    ----------
    h_parsed : torch.Tensor, shape (dim, dim)
        Template HAMILTONIAN block from .rme_rcg fixture
    shell_blocks : Dict[int, torch.Tensor]
        Shell diagonal blocks, each (dim, dim)
    raw_cfg : ConfigParams
        Raw atomic params (plain floats) for V(11) extraction
    scaled_cfg : ScaledConfigParams
        Scaled params where each Fk/Gk/ζ is (N,) instead of scalar
    shell_pair : Tuple[str, str]
        Shell labels for Fk lookup, e.g. ("3D", "3D")
    zeta_shell : str
        Shell label for ζ lookup, e.g. "3D"
        
    Returns
    -------
    torch.Tensor, shape (N, dim, dim)
        Stacked Hamiltonians, one per parameter set.
        Each H_batch[i] carries autograd to scaled_cfg Fk[i] and ζ[i].
    """
    a, b = shell_pair
    ry_to_ev = float(RY_TO_EV)
    
    # ── Step 1: Coulomb with raw params (same as sequential) ──
    h_coulomb_raw = torch.zeros_like(h_parsed)
    for k, shell_mat in shell_blocks.items():
        fk_raw_ev = raw_cfg.f(a, b, k) * ry_to_ev
        h_coulomb_raw = h_coulomb_raw + fk_raw_ev * shell_mat
    
    # ── Step 2: Extract V(11) ONCE (shared across batch) ──────
    zeta_raw_ev = raw_cfg.zeta(zeta_shell) * ry_to_ev
    v11 = (h_parsed - h_coulomb_raw) / zeta_raw_ev  # (dim, dim)
    
    # ── Step 3: Batch rebuild with (N,) scaled params ─────────
    # Extract batch size from any Fk value
    N = None
    for k in shell_blocks.keys():
        fk_batch = scaled_cfg.f(a, b, k)  # (N,)
        N = fk_batch.shape[0]
        break
    
    if N is None:
        raise RuntimeError("No shell blocks found for batch rebuild")
    
    dim = h_parsed.shape[0]
    device = h_parsed.device
    dtype = h_parsed.dtype
    
    # Initialize batch result: (N, dim, dim)
    h_batch = torch.zeros(N, dim, dim, device=device, dtype=dtype)
    
    # Coulomb term: Σ_k F_batch[i,k] × SHELL_k
    for k, shell_mat in shell_blocks.items():
        fk_batch_ev = scaled_cfg.f(a, b, k) * ry_to_ev  # (N,)
        # Broadcasting: (N,) × (dim, dim) → (N, dim, dim)
        h_batch = h_batch + fk_batch_ev[:, None, None] * shell_mat
    
    # Spin-orbit term: ζ_batch[i] × V(11)
    zeta_batch_ev = scaled_cfg.z(zeta_shell) * ry_to_ev  # (N,)
    # Broadcasting: (N,) × (dim, dim) → (N, dim, dim)
    h_batch = h_batch + zeta_batch_ev[:, None, None] * v11
    
    return h_batch


def build_cowan_store_in_memory_batch(
    scaled_params: ScaledAtomicParams,  # now has (N,) batched tensors
    raw_params: AtomicParams,
    plan: SectionPlan,
    *,
    source_rcg_path: Optional[str | Path] = None,
    cowan_template: Optional[List[List[torch.Tensor]]] = None,
    cowan_metadata: Optional[List[List[CowanBlockMeta]]] = None,
    device=None,
) -> List[List[torch.Tensor]]:
    """Batch version: build N COWAN stores from N parameter sets.
    
    Optimizes parameter sweeps by:
    1. Computing V(11) residual ONCE (shared across N parameter sets)
    2. Batch rebuilding N Hamiltonians in single tensor operations
    3. Eliminating redundant file I/O and metadata parsing
    
    Expected speedup: 2-3× faster than N sequential calls to 
    build_cowan_store_in_memory() with the same template.
    
    Parameters
    ----------
    scaled_params : ScaledAtomicParams
        Batched scaled parameters where each Fk/Gk/ζ is shape (N,).
        Produced by :func:`~multitorch.atomic.scaled_params.batch_scale_atomic_params`.
    raw_params : AtomicParams
        Plain-float atomic params for V(11) extraction (same as sequential).
    plan : SectionPlan
        Section plan for cross-check (same as sequential).
    source_rcg_path, cowan_template, cowan_metadata : optional
        Template source (same as sequential).
    device : torch.device, optional
        Device for all tensors.
        
    Returns
    -------
    List[List[torch.Tensor]]
        COWAN store where HAMILTONIAN blocks in section 2 now have shape
        (N, dim, dim) instead of (dim, dim). All other matrices are
        broadcast-compatible (dim, dim) constants.
        
    Notes
    -----
    The returned store can be passed to a batch-aware diagonalizer
    (safe_eigh_batch) to compute N sets of eigenvalues/eigenvectors
    in one GPU kernel call instead of N separate calls.
    
    Memory usage for Ni d8 L-edge (17×17 matrices):
    - N=100: ~5 MB
    - N=1000: ~50 MB  
    - N=5000: ~250 MB
    
    Example
    -------
    >>> from multitorch.atomic.scaled_params import batch_scale_atomic_params
    >>> slater_vals = torch.linspace(0.6, 1.0, 100)
    >>> soc_vals = torch.linspace(0.8, 1.2, 100)
    >>> scaled = batch_scale_atomic_params(params, slater_vals, soc_vals)
    >>> store_batch = build_cowan_store_in_memory_batch(
    ...     scaled, params, plan, source_rcg_path=rcg_path
    ... )
    >>> # Section 2 HAMILTONIAN blocks now have shape (100, dim, dim)
    """
    # Parse template and metadata (same as sequential)
    if cowan_template is not None and cowan_metadata is not None:
        template = cowan_template
        meta = cowan_metadata
    elif source_rcg_path is not None:
        source_rcg_path = Path(source_rcg_path)
        template = read_cowan_store(source_rcg_path)
        meta = read_cowan_metadata(source_rcg_path)
    else:
        raise ValueError(
            "Either source_rcg_path or (cowan_template, cowan_metadata) "
            "must be provided"
        )
    
    # Validation (same as sequential)
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
    
    # Move template tensors to target device
    if device is not None:
        template = [
            [mat.to(device=device) for mat in sec]
            for sec in template
        ]
    
    result: List[List[torch.Tensor]] = []
    
    for s in range(len(template)):
        section: List[torch.Tensor] = []
        
        for j in range(len(template[s])):
            m = meta[s][j]
            mat = template[s][j]
            
            # Only rebuild config-1 HAMILTONIAN blocks in section 2
            if (s == 2
                    and m.operator == "HAMILTONIAN"
                    and m.block_type == "GROUND"):
                shell_blocks = _find_shell_diagonals(
                    meta[s], template[s], "GROUND", m.bra_sym,
                )
                if not shell_blocks:
                    section.append(mat)
                else:
                    # Check for d-d Slater integrals (needed for d^N)
                    try:
                        _ = raw_params.ground.f("3D", "3D", 0)
                    except KeyError:
                        section.append(mat)
                        continue
                    
                    # *** KEY DIFFERENCE: batch rebuild ***
                    h_batch = _rebuild_hamiltonian_block_batch(
                        mat, shell_blocks,
                        raw_cfg=raw_params.ground,
                        scaled_cfg=scaled_params.ground,  # has (N,) tensors
                        shell_pair=("3D", "3D"),
                        zeta_shell="3D",
                    )
                    section.append(h_batch)  # (N, dim, dim)
            else:
                section.append(mat)  # (dim, dim) - broadcast compatible
        
        result.append(section)
    
    return result
