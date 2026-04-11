"""
Paired absorption / emission .oba (.ban_out) reader for RIXS.

A RIXS calculation needs two ttban runs: one for absorption (ground →
intermediate) and one for emission (intermediate → final). The two output
files share their intermediate-state diagonalization, so for every
absorption triad ``(g_sym, op_sym, i_sym)`` there is at least one emission
triad ``(i_sym, op_sym', f_sym)`` whose own ground states are the same
intermediate eigenstates that the absorption triad lands in.

This module groups those triads, stacks the matrix elements into the
``(Eg, TA, Ei, TE, Ef)`` layout that
:func:`multitorch.spectrum.rixs.kramers_heisenberg` expects, and exposes the
result as :class:`RIXSStore`.

Mirrors the structure of ``pyctm.rixs.read_abs_ems`` and
``pyctm.rixs.prep_kh`` (`../pyctm/pyctm/rixs.py:42-234`) but produces
``torch.Tensor`` outputs in float64 from the start.

Unit conventions
----------------
``read_ban_output`` parses ``.ban_out`` files literally:

* ``Eg`` (the bra eigenvalues) come out in **Ry**.
* ``Ef`` (the ket eigenvalues) come out in **eV**.
* ``M`` has shape ``(n_bra, n_ket)`` and is pre-squared (intensity).

For RIXS we keep the absolute ground-state energies in Ry (matching what
``kramers_heisenberg`` expects for its ``Eg`` argument) and use the
absorption file's ket energies as the canonical intermediate-state grid in
eV. The emission file's bra energies are intentionally **not** used: they
parameterize the same intermediate states but in Ry, and trusting the
eigenvalue ordering from the file is more robust than re-aligning two unit
systems.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from multitorch._constants import DTYPE
from multitorch.io.read_oba import BanOutput, TriadData, read_ban_output


# Channel key: (ground_sym, abs_op, intermediate_sym, ems_op, final_sym).
# Two channels with the same key share their (Eg, TA, Ei, TE, Ef) tensors.
ChannelKey = Tuple[str, str, str, str, str]


@dataclass
class RIXSChannel:
    """One Kramers-Heisenberg channel ready to feed into ``kramers_heisenberg``.

    A channel is identified by its full symmetry path
    ``(g_sym, abs_op, i_sym, ems_op, f_sym)``. All ground states with the
    same channel key are stacked along axis 0 of ``Eg``/``TA``; all
    intermediate and final states are stacked along the remaining axes.
    """

    key: ChannelKey
    Eg: torch.Tensor   # (n_g,)        ground state energies, Ry
    TA: torch.Tensor   # (n_g, n_i)    pre-squared absorption matrix elements
    Ei: torch.Tensor   # (n_i,)        intermediate state energies, eV
    TE: torch.Tensor   # (n_i, n_f)    pre-squared emission matrix elements
    Ef: torch.Tensor   # (n_f,)        final state energies, eV

    @property
    def n_g(self) -> int:
        return int(self.Eg.shape[0])

    @property
    def n_i(self) -> int:
        return int(self.Ei.shape[0])

    @property
    def n_f(self) -> int:
        return int(self.Ef.shape[0])


@dataclass
class RIXSStore:
    """Top-level container for one paired absorption/emission RIXS dataset."""

    channels: List[RIXSChannel] = field(default_factory=list)
    abs_path: Path | None = None
    ems_path: Path | None = None

    @property
    def min_gs_ry(self) -> float:
        """Lowest ground-state energy across all channels (Ry).

        Useful as the ``min_gs`` argument to ``kramers_heisenberg`` for
        Boltzmann population referencing.
        """
        if not self.channels:
            return 0.0
        return float(min(float(ch.Eg.min()) for ch in self.channels))

    def channels_for_ground_sym(self, g_sym: str) -> List[RIXSChannel]:
        return [ch for ch in self.channels if ch.key[0] == g_sym]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _group_by_triad(
    bo: BanOutput,
) -> Dict[Tuple[str, str, str, str], List[TriadData]]:
    """Group ``TriadData`` entries by ``(g_sym, op_sym, f_sym, actor)``.

    A single ``.ban_out`` typically emits one ``TriadData`` per
    ``(triad, actor)`` pair, so the resulting lists usually have length 1
    on the absorption side. The grouping is here for safety when files
    contain multiple charge-transfer slices for the same triad.
    """
    out: Dict[Tuple[str, str, str, str], List[TriadData]] = {}
    for t in bo.triad_list:
        key = (t.ground_sym, t.op_sym, t.final_sym, t.actor)
        out.setdefault(key, []).append(t)
    return out


def _stack_triad_group(group: List[TriadData]) -> TriadData:
    """Concatenate same-triad ``TriadData`` entries along the bra axis.

    Each entry contributes its ground states (rows of ``M``); the ket
    energies must match across the group. Returns a new ``TriadData``
    whose ``Eg`` and ``M`` are the row-stacked union and whose ``Ef``
    is the (shared) ket grid.
    """
    if len(group) == 1:
        return group[0]

    Ef_ref = group[0].Ef
    for t in group[1:]:
        if t.Ef.shape != Ef_ref.shape or not torch.allclose(t.Ef, Ef_ref):
            raise ValueError(
                f"Cannot stack triad group {group[0].ground_sym}/"
                f"{group[0].op_sym}/{group[0].final_sym}: ket grids differ "
                f"between entries"
            )

    Eg = torch.cat([t.Eg for t in group], dim=0)
    M = torch.cat([t.M for t in group], dim=0)
    return TriadData(
        ground_sym=group[0].ground_sym,
        op_sym=group[0].op_sym,
        final_sym=group[0].final_sym,
        actor=group[0].actor,
        Eg=Eg,
        Ef=Ef_ref,
        M=M,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_abs_ems_pair(
    abs_path: str | Path,
    ems_path: str | Path,
) -> RIXSStore:
    """Read a paired absorption/emission ``.ban_out`` set into a ``RIXSStore``.

    Parameters
    ----------
    abs_path
        Path to the absorption ``.ban_out`` file (the one whose ``Ef``
        column gives intermediate-state energies in eV).
    ems_path
        Path to the emission ``.ban_out`` file (the one whose ``Eg``
        column gives the same intermediate states in Ry, written from a
        ttban run that uses the intermediate-state manifold as its ground
        manifold).

    Returns
    -------
    RIXSStore
        One ``RIXSChannel`` per (g_sym, abs_op, i_sym, ems_op, f_sym)
        path with all eigenvalues / matrix elements as float64 tensors.

    Notes
    -----
    Pairing rule: an absorption triad ``(g, opA, i)`` is matched with every
    emission triad ``(i, opE, f)`` whose ``ground_sym`` equals the
    absorption triad's ``final_sym``. Channels with mismatched
    intermediate-state counts (``abs.n_i != ems.n_g``) are truncated to
    the smaller of the two, mirroring ``pyctm.rixs.read_abs_ems``
    (`../pyctm/pyctm/rixs.py:80-86`).
    """
    abs_path = Path(abs_path)
    ems_path = Path(ems_path)

    abs_bo = read_ban_output(abs_path)
    ems_bo = read_ban_output(ems_path)

    abs_groups = _group_by_triad(abs_bo)
    ems_groups = _group_by_triad(ems_bo)

    # Index emission triads by their ground (intermediate) symmetry so we
    # can find all emission channels reached from a given absorption
    # intermediate.
    ems_by_ground_sym: Dict[str, List[Tuple[Tuple[str, str, str, str], TriadData]]] = {}
    for key, group in ems_groups.items():
        merged = _stack_triad_group(group)
        ems_by_ground_sym.setdefault(merged.ground_sym, []).append((key, merged))

    channels: List[RIXSChannel] = []

    for abs_key, abs_group in abs_groups.items():
        abs_t = _stack_triad_group(abs_group)
        i_sym = abs_t.final_sym

        if i_sym not in ems_by_ground_sym:
            continue

        for ems_key, ems_t in ems_by_ground_sym[i_sym]:
            n_i_abs = int(abs_t.M.shape[1])
            n_i_ems = int(ems_t.M.shape[0])
            n_i = min(n_i_abs, n_i_ems)

            if n_i == 0:
                continue

            # Truncate matching how pyctm does it: keep the lower indices
            # of both sides. The eigenvalue ordering is shared across the
            # two ttban runs; trimming a tail just discards the highest
            # intermediate states.
            Eg = abs_t.Eg                                  # (n_g,)  Ry
            TA = abs_t.M[:, :n_i].contiguous()             # (n_g, n_i)
            Ei = abs_t.Ef[:n_i].contiguous()               # (n_i,)  eV
            TE = ems_t.M[:n_i, :].contiguous()             # (n_i, n_f)
            Ef = ems_t.Ef                                  # (n_f,)  eV

            channels.append(
                RIXSChannel(
                    key=(
                        abs_t.ground_sym,
                        abs_t.op_sym,
                        i_sym,
                        ems_t.op_sym,
                        ems_t.final_sym,
                    ),
                    Eg=Eg.to(dtype=DTYPE),
                    TA=TA.to(dtype=DTYPE),
                    Ei=Ei.to(dtype=DTYPE),
                    TE=TE.to(dtype=DTYPE),
                    Ef=Ef.to(dtype=DTYPE),
                )
            )

    return RIXSStore(channels=channels, abs_path=abs_path, ems_path=ems_path)
