"""
Parser for ttban .oba and .ban_out output files.

These files contain:
  - For each symmetry triad (ground_sym, transition_sym, final_sym):
    - Ground state energies (Ry)
    - Final state energies (eV, converted to absolute using edge energy)
    - Transition matrix elements (intensities, pre-squared)

Ported from pyttmult/pyttmult/read_output.py with numpy replaced by
torch.Tensor and dataclasses for structured returns.

File format (BANAUG4/7 output):
  TRANSFORMED MATRIX FOR TRIAD ( SYM1  SYM2  SYM3 ) (xdim*ydim) DIM : ... ACTOR <type>
  BRA/KET :  E1  E2  E3  ...
  Eg:  M1  M2  M3  ...
  ...
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple
import torch

from multitorch._constants import DTYPE


@dataclass
class TriadData:
    """All matrices for one (ground_sym, op_sym, final_sym) triad."""
    ground_sym: str
    op_sym: str
    final_sym: str
    actor: str
    # Shape: (n_ground,) ground state energies in Ry
    Eg: torch.Tensor
    # Shape: (n_final,) final state energies in eV
    Ef: torch.Tensor
    # Shape: (n_ground, n_final) transition matrix elements (pre-squared intensities)
    M: torch.Tensor


@dataclass
class BanOutput:
    """
    Complete parsed output of a ttban .oba / .ban_out file.

    Triads are stored as a flat list because charge-transfer calculations
    produce multiple entries for the same (gs_sym, op_sym, fs_sym) triad
    (one per CT configuration), potentially with different n_final counts.
    Use get() for the first match or get_all() for all entries.
    """
    # Flat list of all parsed triads in file order
    triad_list: List[TriadData] = field(default_factory=list)

    def get(self, actor: str, sym: Tuple[str, str, str]) -> TriadData | None:
        """Return the first TriadData matching actor and sym."""
        for t in self.triad_list:
            if t.actor == actor and (t.ground_sym, t.op_sym, t.final_sym) == sym:
                return t
        return None

    def get_all(self, actor: str, sym: Tuple[str, str, str]) -> List[TriadData]:
        """Return all TriadData matching actor and sym (charge transfer: multiple)."""
        return [
            t for t in self.triad_list
            if t.actor == actor and (t.ground_sym, t.op_sym, t.final_sym) == sym
        ]

    @property
    def triads(self):
        """Dict view for backward compatibility: {actor: {sym: first_match}}."""
        result: Dict[str, Dict[Tuple[str, str, str], TriadData]] = {}
        for t in self.triad_list:
            sym = (t.ground_sym, t.op_sym, t.final_sym)
            if t.actor not in result:
                result[t.actor] = {}
            if sym not in result[t.actor]:
                result[t.actor][sym] = t
        return result

    def all_triads(self):
        for t in self.triad_list:
            yield t.actor, (t.ground_sym, t.op_sym, t.final_sym), t


def read_ban_output(path: str | Path) -> BanOutput:
    """
    Parse a .oba or .ban_out file into a BanOutput dataclass.

    Parameters
    ----------
    path : str or Path
        Path to the .oba or .ban_out file.

    Returns
    -------
    BanOutput
        Parsed triad data with torch.Tensor fields.
    """
    path = Path(path)
    result = BanOutput()

    with open(path, "r") as f:
        line = f.readline()
        while line:
            if "TRANSFORMED" in line:
                xdim, ydim, actor, sym_list = _parse_mat_header(line)
                # Determine file type by extension
                if str(path).endswith(".ora"):
                    bras, kets, mat = _read_ora_matrix(f, xdim, ydim)
                else:
                    bras, kets, mat = _read_oba_matrix(f, xdim, ydim)

                # Normalize sym tuple (drop trailing '0' if present for 4-element)
                if len(sym_list) == 4 and sym_list[-1] == "0":
                    sym = tuple(sym_list[:3])
                else:
                    sym = tuple(sym_list[:3])

                actor_stripped = actor.strip()
                triad = TriadData(
                    ground_sym=sym[0],
                    op_sym=sym[1],
                    final_sym=sym[2],
                    actor=actor_stripped,
                    Eg=torch.tensor(bras, dtype=DTYPE),
                    Ef=torch.tensor(kets, dtype=DTYPE),
                    M=torch.tensor(mat, dtype=DTYPE),
                )
                result.triad_list.append(triad)

            line = f.readline()

    return result


def _parse_mat_header(line: str):
    """
    Parse the TRANSFORMED MATRIX header line.

    Returns (xdim, ydim, actor, sym_list).
    """
    idx_star = line.index("*")
    idx_close = line.index(")")
    idx_close2 = line.index(")", idx_star)
    idx_open = line.index("(")
    idx_open2 = line.index("(", idx_close)
    idx_actor = line.index("ACTOR")

    sym_list = line[idx_open + 1 : idx_close].split()
    xdim = int(line[idx_open2 + 1 : idx_star])
    ydim = int(line[idx_star + 1 : idx_close2])
    actor = line[idx_actor + 5 :].lstrip().rstrip()
    return xdim, ydim, actor, sym_list


def _read_oba_matrix(f, xdim: int, ydim: int):
    """
    Read one oba matrix block.

    The .oba format stores one ground state per TRIAD block (xdim=1 always
    for oba files). Multiple BRA/KET rows are printed 7 columns at a time;
    the ground state energy is the same on every data row — we only record
    it once.

    Returns (bras, kets, mat) where:
      bras: list of length 1 (single ground state Ry)
      kets: list of length ydim (final state energies eV)
      mat:  nested list of shape (1, ydim)
    """
    import numpy as np
    kets = []
    bra_val = None
    mat_chunks = []
    while len(kets) < ydim:
        b, k, m = _read_oba_submatrix(f)
        kets.extend(k)
        if bra_val is None:
            bra_val = b[0]   # record ground state once (same each row)
        mat_chunks.append(m)
    full_mat = np.hstack(mat_chunks)   # shape (1, ydim)
    return (
        [float(bra_val)],
        [float(x) for x in kets],
        full_mat.tolist(),
    )


def _read_oba_submatrix(f):
    import numpy as np
    line = f.readline()
    while "BRA/KET" not in line:
        line = f.readline()
    kets = line.split(":")[-1].split()
    b_str, r_str = f.readline().split(":")
    bras = [b_str.strip()]
    row = np.fromstring(r_str, sep=" ")
    return bras, kets, np.array([row])


def _read_ora_matrix(f, xdim: int, ydim: int):
    import numpy as np
    kets = []
    bras = []
    mat = []
    while len(kets) < ydim:
        b, k, m = _read_ora_submatrix(f, xdim)
        kets.extend(k)
        bras.extend(b)
        mat.append(m)
    return (
        [float(x) for x in bras],
        [float(x) for x in kets],
        np.hstack(mat).tolist(),
    )


def _read_ora_submatrix(f, dim: int):
    import numpy as np
    line = f.readline()
    while "BRA/KET" not in line:
        line = f.readline()
    kets = line.split(":")[-1].split()
    f.readline()  # blank line after BRA/KET in ora files
    bras = []
    rows = []
    for _ in range(dim):
        b_str, r_str = f.readline().split(":")
        bras.append(b_str.strip())
        rows.append(np.fromstring(r_str, sep=" "))
    return bras, kets, np.array(rows)
