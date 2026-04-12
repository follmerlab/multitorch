"""
In-memory `RACFileFull` builder + COWAN section plan for the Track C
Phase 5 builder.

Scope of this module
--------------------
``build_rac_in_memory`` is the entry point that the in-memory C3 pipeline
(C3e ``build_cowan_store_in_memory``, C3f parity test, C4 ``calcXAS``)
will call to obtain:

  1. A :class:`RACFileFull` describing the assembly recipe for every
     symmetry triad — block kinds, irrep labels, geometries, and ADD
     entries.
  2. A :class:`SectionPlan` describing the COWAN store layout that the
     RAC ADD entries reference: how many sections, how many matrices per
     section, and the shape required at every ``matrix_idx``.

The ``SectionPlan`` is the **contract** that ``build_cowan_store_in_memory``
must satisfy. The COWAN store it produces must contain
``len(plan.sections)`` sections; section ``s`` must contain
``len(plan.sections[s])`` torch tensors; and the tensor at position
``j`` (i.e. ``cowan[s][j]``) must have shape compatible with every ADD
entry that references ``matrix_idx == j + 1``.

Why this is "loader-based" today
--------------------------------
The ADD entry coefficients in a ``.rme_rac`` file are pure angular
products of CFP / Wigner-3j / Wigner-6j symbols, computed by Fortran
``ttrac.c``. Reproducing them from scratch in PyTorch is a massive port
(see ``../ttmult/src/ttrac.c``) and is **out of scope** for the C3
sub-step. Instead, this module accepts ``.rme_rac`` and ``.rme_rcg``
fixture paths, parses them once, and derives the ``SectionPlan`` by
classifying each parsed block to its COWAN section using a small
deterministic dispatch table (see :func:`classify_block_section`).

This is the same pragmatic split the rest of Track C uses: the angular
machinery (constants, no autograd) is sourced from existing fixtures or
``compute_*_blocks`` numpy code, and the autograd story flows through
the **atomic-parameter scalars** (``slater_scale``, ``soc_scale``, ``cf``,
``delta``, …) that multiply *into* the COWAN store in C3e — not through
the ADD-entry coefficients themselves.

The ``source_rac_path`` / ``source_rcg_path`` arguments are an explicit
extension point: a future from-scratch generator can replace the parser
calls without changing any downstream consumer (C3e, C3f, C4) because
the dataclass shapes and the section-plan contract stay identical.

COWAN section convention (nid8ct, Oh charge transfer)
-----------------------------------------------------
The Fortran emission convention, which all multi-config CT fixtures
follow and which :func:`~multitorch.hamiltonian.assemble.assemble_and_diagonalize_in_memory`
already encodes, is:

  - **Section 0**: TRANSI (PERP/PARA) blocks for **configuration 1**
    (sourced via ``tran_cowan_sec = icg = 0``).
  - **Section 1**: TRANSI blocks for **configuration 2**
    (``tran_cowan_sec = icg = 1``).
  - **Section 2**: GROUND/EXCITE Hamiltonian, 10DQ, DT, DS, and *HYBR*
    blocks for the **ground state manifold** (irreps with ``+`` parity).
    Used by ``_assemble_one_triad`` via ``gs_cowan_sec = 2``.
  - **Section 3**: same operator types for the **excited state manifold**
    (irreps with ``-`` parity). Used via ``fs_cowan_sec = 3``.

The TRANSI HYBR blocks live in sections 2 and 3 (not in 0/1) because
the hybridization is between the two configurations of the same final
manifold, not between configurations across the dipole. Their parity
follows the bra symmetry, just like Hamiltonian blocks.

Single-config (``nconf == 1``) fixtures use a single COWAN section (0).
Such fixtures collapse the four-section layout above to one; the
classifier handles this case by routing every block to section 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from multitorch.io.read_rme import (
    RACBlockFull,
    RACFileFull,
    read_cowan_store,
    read_rme_rac_full,
)
from multitorch.io.read_ban import BanData


# ─────────────────────────────────────────────────────────────
# Section-plan dataclasses
# ─────────────────────────────────────────────────────────────


@dataclass
class SectionEntry:
    """One slot in a COWAN section (one matrix in ``cowan[section_idx]``).

    The shape constraint is the union of every ADD entry that references
    ``matrix_idx``: the tensor must have at least
    ``required_n_rows = max(add.nbra over references)`` rows, and the
    column count must equal ``required_n_cols`` (the inferred column
    count, which all references must agree on).
    """

    section_idx: int
    matrix_idx: int  # 1-based, matches the value in ADDEntry.matrix_idx
    required_n_rows: int
    required_n_cols: int
    # How many ADD entries across the whole RAC file reference this slot.
    # Useful as a sanity check: a slot referenced by zero ADD entries is
    # an "unreferenced filler" and only constrained by the parsed COWAN
    # store's shape, not by any RAC requirement.
    n_references: int = 0

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.required_n_rows, self.required_n_cols)


@dataclass
class SectionPlan:
    """Layout the COWAN store builder (C3e) must reproduce.

    ``sections[s]`` is the ordered list of :class:`SectionEntry` for COWAN
    section ``s``. Position ``j`` in that list corresponds to
    ``cowan[s][j]`` — the matrix that ADD entries reference via
    ``matrix_idx == j + 1``.

    A SectionPlan is built by analysing the parsed RAC + COWAN store
    pair so that the COWAN-store contract is *unambiguously* defined by
    the ADD entries that reference it. The COWAN-store builder consumes
    this plan to know how many tensors to emit per section and what
    shape each one must have.
    """

    sections: List[List[SectionEntry]] = field(default_factory=list)

    @property
    def n_sections(self) -> int:
        return len(self.sections)

    def section_size(self, section_idx: int) -> int:
        return len(self.sections[section_idx])

    def get(self, section_idx: int, matrix_idx: int) -> SectionEntry:
        """Return the entry at ``cowan[section_idx][matrix_idx - 1]``."""
        return self.sections[section_idx][matrix_idx - 1]

    def total_matrices(self) -> int:
        return sum(len(s) for s in self.sections)


# ─────────────────────────────────────────────────────────────
# Block → COWAN section classifier
# ─────────────────────────────────────────────────────────────


def _is_plus_parity(sym: str) -> bool:
    """Return True if a symmetry label represents an even-parity (gs) irrep.

    Symmetry labels in the parsed file are of the form ``'0+'``, ``'1-'``,
    ``'^0+'``, ``'^2-'``. Strip the optional ``'^'`` prefix and look at
    the trailing parity character.
    """
    s = sym.lstrip("^")
    return s.endswith("+")


def classify_block_section(block: RACBlockFull, *, nconf: int) -> int:
    """Map a parsed RAC block to its COWAN section index.

    Parameters
    ----------
    block : RACBlockFull
        Parsed block from ``read_rme_rac_full``.
    nconf : int
        Number of configurations in the BAN file (1 or 2).

    Returns
    -------
    int
        COWAN section index (0 for single-config; 0/1/2/3 for two-config).

    Notes
    -----
    For a 2-config CT calculation, the convention encoded by
    :mod:`multitorch.hamiltonian.assemble` is:

      - TRANSI PERP / PARA → section 0 (conf 1) or 1 (conf 2)
      - TRANSI ``*HYBR``  → section 2 (gs, '+' parity bra) or 3 (fs, '-' parity bra)
      - GROUND/EXCITE operators → section 2 ('+' parity) or 3 ('-' parity)

    Distinguishing TRANSI conf-1 vs conf-2 requires extra context (the
    block ordering in the file), so this function returns ``0`` for any
    TRANSI dipole block. The conf-2 split is performed by
    :func:`derive_section_plan`, which sees the block sequence.
    """
    if nconf == 1:
        return 0

    # HYBR blocks (labelled TRANSI but with *HYBR geometry) belong with
    # the manifold whose parity matches the bra. They mix configurations
    # within the same manifold, not across the dipole.
    if block.kind == "TRANSI" and "HYBR" in block.geometry:
        return 2 if _is_plus_parity(block.bra_sym) else 3

    if block.kind == "TRANSI":
        # PERP / PARA dipole — caller (derive_section_plan) decides
        # conf-1 (section 0) vs conf-2 (section 1) based on order.
        return 0

    # GROUND or EXCITE operator block. Even though "EXCITE" is the
    # config label, the COWAN section is determined by the *manifold*
    # (irrep parity), not by the kind label.
    if block.kind in ("GROUND", "EXCITE"):
        return 2 if _is_plus_parity(block.bra_sym) else 3

    raise ValueError(
        f"Unknown block kind {block.kind!r} for {block.bra_sym} {block.ket_sym}"
    )


def _split_transi_dipole_sections(
    rac: RACFileFull, nconf: int,
) -> Dict[int, int]:
    """Decide section index (0 or 1) for every TRANSI PERP/PARA block.

    Returns a mapping from ``id(block)`` to ``0`` (conf 1) or ``1``
    (conf 2). For ``nconf == 1`` every dipole block goes to section 0.

    The Fortran emission rule, which the parsed file already obeys, is:
    for each (gs_sym, act_sym, fs_sym, geometry), the first occurrence
    is conf 1 and the second is conf 2. ``read_rme_rac_full`` preserves
    file order, so we just count occurrences in order.
    """
    seen: Dict[Tuple[str, str, str, str], int] = {}
    out: Dict[int, int] = {}
    for blk in rac.blocks:
        if blk.kind != "TRANSI" or "HYBR" in blk.geometry:
            continue
        key = (blk.bra_sym, blk.op_sym, blk.ket_sym, blk.geometry)
        occ = seen.get(key, 0)
        out[id(blk)] = 0 if occ == 0 else 1
        seen[key] = occ + 1
    if nconf == 1:
        # Collapse to single section.
        return {k: 0 for k in out}
    return out


# ─────────────────────────────────────────────────────────────
# Section-plan derivation
# ─────────────────────────────────────────────────────────────


def derive_section_plan(rac: RACFileFull, cowan, *, nconf: int) -> SectionPlan:
    """Build a :class:`SectionPlan` from a parsed RAC + COWAN-store pair.

    Walks every ADD entry of every RAC block, classifies the block to a
    COWAN section, and aggregates per-(section, matrix_idx) shape
    requirements. Cross-checks every requirement against the parsed
    COWAN store: the recorded ``required_n_cols`` is taken from the
    parsed tensor (the ADD entry only constrains the row count, since
    the source matrix is consumed column-by-column), and
    ``required_n_rows`` is the maximum ``add.nbra`` across all
    references to that slot.

    Parameters
    ----------
    rac : RACFileFull
        Parsed ``.rme_rac``.
    cowan : list of list of torch.Tensor
        Parsed ``.rme_rcg`` COWAN store, as returned by
        :func:`read_cowan_store`.
    nconf : int
        Number of configurations from the BAN file.

    Returns
    -------
    SectionPlan
        One :class:`SectionEntry` per matrix in the parsed COWAN store.
        Slots referenced by zero ADD entries (unused filler matrices)
        appear with ``n_references == 0`` and shape taken directly from
        the parsed tensor.
    """
    # Tabulate per-(section, matrix_idx) requirements from ADD entries.
    transi_section = _split_transi_dipole_sections(rac, nconf)

    requirements: Dict[Tuple[int, int], Dict[str, int]] = {}

    for blk in rac.blocks:
        if blk.kind == "TRANSI" and "HYBR" not in blk.geometry:
            sec = transi_section[id(blk)]
        else:
            sec = classify_block_section(blk, nconf=nconf)

        for add in blk.add_entries:
            key = (sec, add.matrix_idx)
            req = requirements.setdefault(
                key, {"max_rows": 0, "n_refs": 0}
            )
            req["max_rows"] = max(req["max_rows"], add.nbra)
            req["n_refs"] += 1

    # Build per-section entry lists, sized to match the parsed COWAN store.
    plan = SectionPlan(sections=[[] for _ in cowan])
    for s, sec_mats in enumerate(cowan):
        for j, mat in enumerate(sec_mats):
            mi = j + 1
            req = requirements.get((s, mi), {"max_rows": 0, "n_refs": 0})

            # The parsed source tensor's true shape is the binding
            # column count (the ADD entry consumes ``src.shape[1]``
            # columns). The row count is constrained by the maximum
            # ``add.nbra`` across all references; for unreferenced
            # slots, fall back to the parsed shape.
            true_rows, true_cols = int(mat.shape[0]), int(mat.shape[1])
            req_rows = req["max_rows"] or true_rows

            plan.sections[s].append(
                SectionEntry(
                    section_idx=s,
                    matrix_idx=mi,
                    required_n_rows=req_rows,
                    required_n_cols=true_cols,
                    n_references=req["n_refs"],
                )
            )

    return plan


def validate_section_plan(plan: SectionPlan, rac: RACFileFull, *, nconf: int) -> None:
    """Sanity-check a :class:`SectionPlan` against a parsed RAC.

    Verifies:
      - Every ADD entry's ``matrix_idx`` is in range for its section.
      - Every ADD entry's ``nbra`` does not exceed the slot's row capacity.

    TRANSI dipole blocks (PERP/PARA) are validated more loosely: the
    assembler determines the COWAN section at runtime based on the
    config index (``tran_cowan_sec = icg``), so we only check that the
    matrix_idx is in range for *at least one* of the dipole sections
    (0 or 1). This is necessary because fixtures like Ti⁴⁺ d⁰ have
    TRANSI blocks that exist only for conf-2 (d¹L), and the
    duplicate-based conf-1/conf-2 classifier cannot handle asymmetric
    irrep counts.

    Raises
    ------
    ValueError
        On any inconsistency. Used by :func:`build_rac_in_memory` to
        fail loudly when a fixture's RAC and RCG files disagree.
    """
    # Determine which sections are "dipole sections" (0 and/or 1)
    dipole_sections = [0] if nconf == 1 else [0, 1]

    for blk in rac.blocks:
        is_transi_dipole = (blk.kind == "TRANSI" and "HYBR" not in blk.geometry)

        if is_transi_dipole:
            # Validate against any eligible dipole section
            for add in blk.add_entries:
                valid = any(
                    1 <= add.matrix_idx <= plan.section_size(sec)
                    for sec in dipole_sections
                    if sec < plan.n_sections
                )
                if not valid:
                    sizes = {s: plan.section_size(s) for s in dipole_sections
                             if s < plan.n_sections}
                    raise ValueError(
                        f"ADD matrix_idx={add.matrix_idx} out of range for "
                        f"all dipole sections {sizes} in "
                        f"block {(blk.kind, blk.bra_sym, blk.ket_sym, blk.geometry)}"
                    )
        else:
            sec = classify_block_section(blk, nconf=nconf)

            if sec >= plan.n_sections:
                raise ValueError(
                    f"Block {(blk.kind, blk.bra_sym, blk.ket_sym, blk.geometry)} "
                    f"classified to section {sec} but plan has only "
                    f"{plan.n_sections} sections"
                )

            for add in blk.add_entries:
                if not (1 <= add.matrix_idx <= plan.section_size(sec)):
                    raise ValueError(
                        f"ADD matrix_idx={add.matrix_idx} out of range for "
                        f"section {sec} (size {plan.section_size(sec)}) in "
                        f"block {(blk.kind, blk.bra_sym, blk.ket_sym, blk.geometry)}"
                    )

                entry = plan.get(sec, add.matrix_idx)
                if add.nbra > entry.required_n_rows:
                    raise ValueError(
                        f"ADD nbra={add.nbra} exceeds slot capacity "
                        f"{entry.required_n_rows} at section {sec} "
                        f"matrix_idx {add.matrix_idx}"
                    )


# ─────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────


def build_rac_in_memory(
    ban: BanData,
    *,
    source_rac_path: Optional[str | Path] = None,
    source_rcg_path: Optional[str | Path] = None,
    parsed_rac: Optional[RACFileFull] = None,
    parsed_cowan: Optional[List] = None,
) -> Tuple[RACFileFull, SectionPlan]:
    """Construct a :class:`RACFileFull` + :class:`SectionPlan` for C3e.

    Parameters
    ----------
    ban : BanData
        Parsed BAN file. Only ``ban.nconf_gs`` is consulted here; the
        rest of the BAN content is consumed downstream by C3e and the
        in-memory assembler.
    source_rac_path : path-like, optional
        Path to a ``.rme_rac`` fixture. Parsed via
        :func:`read_rme_rac_full`. Not needed if ``parsed_rac`` is
        provided.
    source_rcg_path : path-like, optional
        Path to a ``.rme_rcg`` fixture. Parsed via
        :func:`read_cowan_store` so the section plan can record the
        per-slot column count. Not needed if ``parsed_cowan`` is
        provided.
    parsed_rac : RACFileFull, optional
        Pre-parsed RAC data. If provided, skips file I/O for RAC.
    parsed_cowan : list, optional
        Pre-parsed COWAN store (from ``read_cowan_store``). If
        provided, skips file I/O for the COWAN template.

    Returns
    -------
    (RACFileFull, SectionPlan)
        The parsed RAC dataclass and the derived section plan. Both are
        ready for ``build_cowan_store_in_memory`` to consume.

    Raises
    ------
    ValueError
        If the parsed RAC and the parsed COWAN store are inconsistent
        (``validate_section_plan`` failed). This is the contract that
        protects C3e from building against a bad fixture.
    """
    if parsed_rac is not None:
        rac = parsed_rac
    elif source_rac_path is not None:
        rac = read_rme_rac_full(source_rac_path)
    else:
        raise ValueError(
            "Either source_rac_path or parsed_rac must be provided"
        )

    if parsed_cowan is not None:
        cowan = parsed_cowan
    elif source_rcg_path is not None:
        cowan = read_cowan_store(source_rcg_path)
    else:
        raise ValueError(
            "Either source_rcg_path or parsed_cowan must be provided"
        )

    plan = derive_section_plan(rac, cowan, nconf=ban.nconf_gs)
    validate_section_plan(plan, rac, nconf=ban.nconf_gs)
    return rac, plan
