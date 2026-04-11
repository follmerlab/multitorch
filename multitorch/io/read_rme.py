"""
Parsers for ttrcg .rme_rcg (fort.14/.m14) and ttrac .rme_rac (fort.15/.m15) files.

.rme_rcg format (ttrcg output, high-symmetry RME):
  - Header: % lines, IRREP lines listing symmetries and dimensions
  - RME blocks:  RME <type> <sym1> <sym2> <sym3> <operator> <ndim1> <ndim2> <n_nz>
    followed by sparse matrix entries:
      row col  ,  col  val , col  val ... ;

.rme_rac format (ttrac output, lower-symmetry RME, REDUCEDMATRIX format):
  - IRREP lines
  - REDUCEDMATRIX TRANSI/HAMIL <sym1> <sym2> <sym3> <geometry> <n_bra> <n_ket>
    followed by:
      ADD  config_idx  bra_row  ket_col  mult1  mult2  value
    terminated by:
      END <geometry>
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch

from multitorch._constants import DTYPE


@dataclass
class IrrepInfo:
    """Irreducible representation metadata."""
    name: str         # e.g. '0+', '1-', '^0+'
    kind: str         # 'GROUND' or 'EXCITE'
    multiplicity: int
    dim: int          # dimension of the irrep (2J+1)


@dataclass
class RMEBlock:
    """One reduced matrix element block: operator between two irreps."""
    bra_sym: str      # e.g. '0+', '^0+'
    op_sym: str       # e.g. '1-'
    ket_sym: str      # e.g. '1-'
    operator: str     # e.g. 'MULTIPOLE', 'HAMILTONIAN'
    n_bra: int
    n_ket: int
    # Dense (n_bra, n_ket) matrix of RME values
    matrix: torch.Tensor


@dataclass
class RMEData:
    """All RME data for one configuration block in a .rme_rcg or .rme_rac file."""
    irreps: List[IrrepInfo] = field(default_factory=list)
    blocks: List[RMEBlock] = field(default_factory=list)
    config_label: Optional[str] = None

    def get_block(self, bra_sym: str, op_sym: str, ket_sym: str,
                  operator: str = "MULTIPOLE") -> Optional[RMEBlock]:
        for b in self.blocks:
            if (b.bra_sym == bra_sym and b.op_sym == op_sym
                    and b.ket_sym == ket_sym and b.operator == operator):
                return b
        return None


@dataclass
class RMEFile:
    """Complete parsed .rme_rcg file, which may contain multiple configurations."""
    configs: List[RMEData] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# .rme_rcg parser (ttrcg output)
# ─────────────────────────────────────────────────────────────

def read_rme_rcg(path: str | Path) -> RMEFile:
    """
    Parse a .rme_rcg file (fort.14 / .m14) into an RMEFile.

    Each configuration block starts with a `%` header line followed by
    `IRREP` lines, then `RME` blocks.  Multiple configurations are
    separated by a `FINISHED` line.
    """
    path = Path(path)
    result = RMEFile()

    with open(path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        # Skip until we hit an IRREP block or RME block
        line = lines[i]

        if line.startswith(" IRREP ") or line.startswith("IRREP "):
            # Start of a new configuration
            config = RMEData()
            # Read IRREP lines
            while i < len(lines) and ("IRREP" in lines[i]):
                config.irreps.append(_parse_irrep_line(lines[i]))
                i += 1
            result.configs.append(config)

        elif line.strip().startswith("RME "):
            # Parse the RME block
            block, i = _parse_rme_block(lines, i)
            if result.configs:
                result.configs[-1].blocks.append(block)
            else:
                # No IRREP header yet — create a default config
                result.configs.append(RMEData())
                result.configs[-1].blocks.append(block)

        else:
            i += 1

    return result


def _parse_irrep_line(line: str) -> IrrepInfo:
    """Parse:  IRREP GROUND   0+   MULT      2   DIM    1"""
    parts = line.split()
    # parts: ['IRREP', kind, sym, 'MULT', mult, 'DIM', dim]
    return IrrepInfo(
        name=parts[2],
        kind=parts[1],
        multiplicity=int(parts[4]),
        dim=int(parts[6]),
    )


def _parse_rme_block(lines: List[str], start: int):
    """
    Parse one RME block starting at lines[start].

    RME block header:
      RME <type>  <sym1>  <sym2>  <sym3>  <operator>  <n_bra>  <n_ket>  <n_nz>
    Followed by sparse entries (may span multiple lines per row):
      row  nnz  ,  col  val , col  val ... ;
    A row ends with ';'. If a row wraps to the next line, the continuation
    starts directly with col val pairs (no row/nnz header).
    Block ends at the first blank or RME/IRREP/FINISHED/%/END line.
    """
    import numpy as np

    line = lines[start]
    parts = line.split()
    # e.g.: ['RME', 'TRANSITION', '0+', '1-', '1-', 'MULTIPOLE', '2', '3', '3']
    block_type = parts[1]   # TRANSITION / GROUND / EXCITE / HAMILTONIAN / etc.
    bra_sym = parts[2]
    op_sym = parts[3]
    ket_sym = parts[4]
    operator = parts[5]
    n_bra = int(parts[6])
    n_ket = int(parts[7])

    matrix = np.zeros((n_bra, n_ket), dtype=np.float64)

    row = -1  # current row being parsed
    i = start + 1
    while i < len(lines):
        l = lines[i].strip()
        if not l or l.startswith("%") or l.startswith("RME") or \
                l.startswith("IRREP") or l.startswith("FINISHED"):
            break

        # Check if this line ends the current row
        has_semicolon = l.endswith(";")
        l = l.rstrip(";").strip()
        if not l:
            i += 1
            continue

        # Split on ','
        tokens = [t.strip() for t in l.split(",")]
        if not tokens[0]:
            i += 1
            continue

        # Determine if this is a new row header or a continuation.
        # New row: "row  nnz , col val , ..."  (first token has 2 integers)
        # Continuation: "col val , col val , ..."  (first token has int + float)
        first = tokens[0].split()
        is_new_row = False
        if len(first) >= 2:
            # Check if second element is an integer (nnz) vs a float (value)
            try:
                int(first[1])
                is_new_row = True
            except ValueError:
                is_new_row = False

        if is_new_row:
            row = int(first[0]) - 1  # 1-indexed → 0-indexed
            col_val_tokens = tokens[1:]
        else:
            # Continuation line: first token is a col-val pair
            col_val_tokens = tokens

        for token in col_val_tokens:
            t = token.split()
            if len(t) >= 2:
                col = int(t[0]) - 1
                val = float(t[1])
                if 0 <= row < n_bra and 0 <= col < n_ket:
                    matrix[row, col] = val
        i += 1

    block = RMEBlock(
        bra_sym=bra_sym,
        op_sym=op_sym,
        ket_sym=ket_sym,
        operator=operator,
        n_bra=n_bra,
        n_ket=n_ket,
        matrix=torch.tensor(matrix, dtype=DTYPE),
    )
    return block, i


# ─────────────────────────────────────────────────────────────
# .rme_rac parser (ttrac output, REDUCEDMATRIX format)
# ─────────────────────────────────────────────────────────────

@dataclass
class RACBlock:
    """One REDUCEDMATRIX block from the .rme_rac file."""
    kind: str          # 'TRANSI' or 'HAMIL'
    bra_sym: str
    op_sym: str
    ket_sym: str
    geometry: str      # 'PERP', 'PARA', etc.
    n_bra: int
    n_ket: int
    # Dense (n_bra, n_ket) matrix of RME values
    matrix: torch.Tensor


@dataclass
class RACFile:
    """Complete parsed .rme_rac file."""
    irreps: List[IrrepInfo] = field(default_factory=list)
    blocks: List[RACBlock] = field(default_factory=list)

    def get_block(self, bra_sym: str, op_sym: str, ket_sym: str,
                  kind: str = "TRANSI") -> Optional[RACBlock]:
        for b in self.blocks:
            if (b.bra_sym == bra_sym and b.op_sym == op_sym
                    and b.ket_sym == ket_sym and b.kind == kind):
                return b
        return None


def read_rme_rac(path: str | Path) -> RACFile:
    """
    Parse a .rme_rac file (fort.15 / .m15 / rme_out.dat) into an RACFile.

    This format is produced by ttrac when run in 'racer' mode (nid8ct-style).
    If the file is in 'butler' mode (.ora format), returns an empty RACFile.

    Format (racer mode):
      IRREP <kind>  <sym>  MULT <n>  DIM <d>
      REDUCEDMATRIX TRANSI/HAMIL <sym1> <sym2> <sym3> <geom> <n_bra> <n_ket>
        ADD  cfg  bra  ket  mult1  mult2  value
        ...
      END <geom>
    """
    import numpy as np

    path = Path(path)
    result = RACFile()
    current_block: Optional[dict] = None
    current_matrix: Optional[np.ndarray] = None

    # Quick check: if file doesn't contain REDUCEDMATRIX it's in .ora format
    with open(path, "r") as f:
        content_preview = f.read(4096)
    if "REDUCEDMATRIX" not in content_preview:
        # .ora (butler mode) format — not yet parsed, return empty
        return result

    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()

            if stripped.startswith("IRREP ") and "BASIS" not in stripped:
                try:
                    result.irreps.append(_parse_irrep_rac_line(line))
                except (ValueError, IndexError):
                    pass

            elif stripped.startswith("REDUCEDMATRIX "):
                parts = stripped.split()
                # ['REDUCEDMATRIX', kind, sym1, sym2, sym3, geom, n_bra, n_ket]
                kind = parts[1]
                bra_sym = parts[2]
                op_sym = parts[3]
                ket_sym = parts[4]
                geom = parts[5]
                n_bra = int(parts[6])
                n_ket = int(parts[7])
                current_block = dict(
                    kind=kind, bra_sym=bra_sym, op_sym=op_sym,
                    ket_sym=ket_sym, geometry=geom, n_bra=n_bra, n_ket=n_ket,
                )
                current_matrix = np.zeros((n_bra, n_ket), dtype=np.float64)

            elif stripped.startswith("ADD ") and current_matrix is not None:
                parts = stripped.split()
                # ADD  cfg  bra  ket  mult1  mult2  value
                bra = int(parts[2]) - 1  # 1-indexed
                ket = int(parts[3]) - 1
                val = float(parts[6])
                if 0 <= bra < current_block["n_bra"] and 0 <= ket < current_block["n_ket"]:
                    current_matrix[bra, ket] += val

            elif stripped.startswith("END ") and current_block is not None:
                result.blocks.append(RACBlock(
                    kind=current_block["kind"],
                    bra_sym=current_block["bra_sym"],
                    op_sym=current_block["op_sym"],
                    ket_sym=current_block["ket_sym"],
                    geometry=current_block["geometry"],
                    n_bra=current_block["n_bra"],
                    n_ket=current_block["n_ket"],
                    matrix=torch.tensor(current_matrix, dtype=DTYPE),
                ))
                current_block = None
                current_matrix = None

    return result


def _parse_irrep_rac_line(line: str) -> IrrepInfo:
    """Parse: IRREP GROUND   0+  MULT 9   DIM 1  (or with '^' prefix on sym)"""
    parts = line.split()
    return IrrepInfo(
        name=parts[2],
        kind=parts[1],
        multiplicity=int(parts[4]),
        dim=int(parts[6]),
    )


# ─────────────────────────────────────────────────────────────
# COWAN store: flat indexed list of all RME matrices from .rme_rcg
# ─────────────────────────────────────────────────────────────

import numpy as np


def read_cowan_store(path: str | Path) -> List[List[torch.Tensor]]:
    """
    Read all RME blocks from a .rme_rcg file, grouped by FINISHED-delimited sections.

    The Fortran COWAN subroutine reads .rme_rcg sections sequentially.
    Each section (between FINISHED markers) corresponds to one PAIRIN call.
    Matrix indices in .rme_rac ADD entries are 1-based within each section.

    Parameters
    ----------
    path : str or Path
        Path to .rme_rcg file.

    Returns
    -------
    sections : list of list of torch.Tensor
        sections[i] is a list of dense matrices (torch.Tensor) for COWAN section i.
        sections[i][j] is the (j+1)-th matrix (0-indexed; ADD entries use 1-indexed).
    """
    path = Path(path)
    with open(path) as f:
        lines = f.readlines()

    sections: List[List[torch.Tensor]] = []
    current_section: List[torch.Tensor] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == 'FINISHED':
            sections.append(current_section)
            current_section = []
            i += 1
            continue

        if line.startswith('RME ') or (len(line) > 4 and ' RME ' in line[:10]):
            # Parse this RME block
            block, i = _parse_rme_block(lines, i)
            current_section.append(block.matrix)
        else:
            i += 1

    # Append any trailing section
    if current_section:
        sections.append(current_section)

    return sections


# ─────────────────────────────────────────────────────────────
# Enhanced .rme_rac reader with subblock ADD semantics
# ─────────────────────────────────────────────────────────────

@dataclass
class ADDEntry:
    """One ADD instruction from a REDUCEDMATRIX block."""
    matrix_idx: int    # 1-based index into COWAN store section
    bra: int           # 1-based bra start position in the full matrix
    ket: int           # 1-based ket start position in the full matrix
    nbra: int          # number of bra rows in the subblock
    nket: int          # number of ket columns in the subblock
    coeff: float       # angular coefficient


@dataclass
class RACBlockFull:
    """One REDUCEDMATRIX block with raw ADD entries preserved."""
    kind: str          # 'TRANSI', 'GROUND', 'EXCITE'
    bra_sym: str
    op_sym: str
    ket_sym: str
    geometry: str      # operator name: 'PERP', 'HAMILTONIAN', '10DQ', 'DT', 'DS', etc.
    n_bra: int
    n_ket: int
    add_entries: List[ADDEntry]


@dataclass
class RACFileFull:
    """Complete parsed .rme_rac file with full ADD entry data."""
    irreps: List[IrrepInfo] = field(default_factory=list)
    blocks: List[RACBlockFull] = field(default_factory=list)

    def get_blocks(self, kind: str, bra_sym: str, op_sym: str, ket_sym: str,
                   geometry: Optional[str] = None) -> List[RACBlockFull]:
        """Get all matching blocks (there may be multiple for multi-config)."""
        result = []
        for b in self.blocks:
            if (b.kind == kind and b.bra_sym == bra_sym
                    and b.op_sym == op_sym and b.ket_sym == ket_sym):
                if geometry is None or b.geometry == geometry:
                    result.append(b)
        return result


def read_rme_rac_full(path: str | Path) -> RACFileFull:
    """
    Parse a .rme_rac file preserving full ADD entry semantics.

    Unlike read_rme_rac() which accumulates ADD values into a flat matrix,
    this function preserves the (matrix_idx, bra, ket, nbra, nket, coeff)
    structure needed for Hamiltonian assembly via COWAN store lookup.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    RACFileFull
    """
    path = Path(path)
    result = RACFileFull()

    with open(path) as f:
        content_preview = f.read(4096)
    if "REDUCEDMATRIX" not in content_preview:
        return result

    current_block: Optional[dict] = None
    current_adds: List[ADDEntry] = []

    with open(path) as f:
        for line in f:
            stripped = line.strip()

            if stripped.startswith("IRREP ") and "BASIS" not in stripped:
                try:
                    result.irreps.append(_parse_irrep_rac_line(stripped))
                except (ValueError, IndexError):
                    pass

            elif stripped.startswith("REDUCEDMATRIX "):
                parts = stripped.split()
                # Two formats exist:
                #   Old: REDUCEDMATRIX TRANSI  0+  1-  1-  PERP  9 15
                #   New: REDUCEDMATRIX TRANSI  0+  1-  1-  0  MULTIPOLE  4 7
                # Detect by trying to parse parts[6] as int
                try:
                    n_bra = int(parts[6])
                    n_ket = int(parts[7])
                    geometry = parts[5]
                except ValueError:
                    # New format: parts[5] is an int flag, parts[6] is geometry string
                    geometry = parts[6]
                    n_bra = int(parts[7])
                    n_ket = int(parts[8])
                current_block = dict(
                    kind=parts[1], bra_sym=parts[2], op_sym=parts[3],
                    ket_sym=parts[4], geometry=geometry,
                    n_bra=n_bra, n_ket=n_ket,
                )
                current_adds = []

            elif stripped.startswith("ADD ") and current_block is not None:
                parts = stripped.split()
                # ADD  matrix_idx  bra  ket  nbra  nket  coeff
                entry = ADDEntry(
                    matrix_idx=int(parts[1]),
                    bra=int(parts[2]),
                    ket=int(parts[3]),
                    nbra=int(parts[4]),
                    nket=int(parts[5]),
                    coeff=float(parts[6]),
                )
                current_adds.append(entry)

            elif stripped.startswith("END ") and current_block is not None:
                result.blocks.append(RACBlockFull(
                    kind=current_block["kind"],
                    bra_sym=current_block["bra_sym"],
                    op_sym=current_block["op_sym"],
                    ket_sym=current_block["ket_sym"],
                    geometry=current_block["geometry"],
                    n_bra=current_block["n_bra"],
                    n_ket=current_block["n_ket"],
                    add_entries=current_adds,
                ))
                current_block = None
                current_adds = []

    return result


def assemble_matrix_from_adds(
    add_entries: List[ADDEntry],
    cowan_section: List[torch.Tensor],
    n_bra: int,
    n_ket: int,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Assemble a dense matrix from ADD entries using COWAN store matrices.

    Each ADD entry says: take the subblock of cowan_section[matrix_idx-1],
    starting at row (bra-1) with nbra rows, and place it at column (ket-1)
    in the full matrix, multiplied by coeff × scale.

    The implementation is vectorized: each ADD entry contributes one
    sliced add (`mat[r0:r0+nr, c0:c0+nc] += factor * src[...]`) instead
    of an element-wise Python loop. This is essential for autograd —
    the previous implementation called `float(src[jr, jc])`, which
    severed the gradient graph from `src` to the assembled matrix.

    Parameters
    ----------
    add_entries : list of ADDEntry
    cowan_section : list of torch.Tensor (0-indexed; ADD uses 1-indexed)
    n_bra, n_ket : full matrix dimensions
    scale : additional scaling factor (e.g., XHAM value).
        Can be a Python float or a torch scalar tensor; if a tensor with
        `requires_grad=True`, gradient flows through it. Per-entry
        ``add.coeff`` is a Python float (parser output), so it does not
        carry gradient.

    Returns
    -------
    torch.Tensor  shape (n_bra, n_ket)
        If any source tensor in `cowan_section` (or `scale`) has
        `requires_grad=True`, the returned tensor participates in the
        autograd graph.
    """
    mat = torch.zeros(n_bra, n_ket, dtype=DTYPE)

    for add in add_entries:
        idx = add.matrix_idx - 1  # 1-based → 0-based
        if idx < 0 or idx >= len(cowan_section):
            continue
        src = cowan_section[idx]
        if src.ndim < 2 or src.shape[1] == 0:
            continue

        r0 = add.bra - 1
        c0 = add.ket - 1
        # The subblock advertised by the ADD entry is (add.nbra, src_cols);
        # the actual source matrix may have fewer rows than add.nbra, in
        # which case the loop in the legacy implementation would silently
        # truncate. Preserve that behavior.
        src_rows = min(add.nbra, src.shape[0])
        src_cols = src.shape[1]

        # Clip the source slice so that the destination slice
        # mat[r0+sr_lo:r0+sr_hi, c0+sc_lo:c0+sc_hi] stays in bounds.
        # This reproduces the per-element bounds check in the legacy
        # implementation, including the case where r0 or c0 is negative
        # (only ever happens with malformed ADD entries, but the legacy
        # code tolerated it).
        sr_lo = max(0, -r0)
        sr_hi = min(src_rows, n_bra - r0)
        sc_lo = max(0, -c0)
        sc_hi = min(src_cols, n_ket - c0)

        if sr_hi <= sr_lo or sc_hi <= sc_lo:
            continue

        factor = add.coeff * scale  # float or tensor; broadcasts cleanly
        mat[r0 + sr_lo:r0 + sr_hi, c0 + sc_lo:c0 + sc_hi] = (
            mat[r0 + sr_lo:r0 + sr_hi, c0 + sc_lo:c0 + sc_hi]
            + factor * src[sr_lo:sr_hi, sc_lo:sc_hi]
        )

    return mat
