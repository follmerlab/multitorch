"""
Coefficients of Fractional Parentage (CFP) for d^n shells.

Two implementations:
  1. Binary file parser: reads rcg_cfp72/73/74 Fortran unformatted binary files
     produced by ttrcg's gen_cfp calculation. Exact match to Cowan code.
  2. Hard-coded tables: d^n CFP values from Cowan (1981) / Sugano-Tanabe-Kamimura (1970)
     for the most common cases (only accessible after binary parsing is complete).

The rcg_cfp72 file format (Fortran sequential unformatted):
  For each (l, n) configuration block:
    Record A: char(LL) + int(NI) + float(FLL) + int(NOT) + int(NLT) + int(NORCD)
    Record B: [int(MULT)+char(LBCD)+char*2(ALF)+float(FL)+float(S)]×NLT + int×4 (NLP,NLGP,NOCFP,NOCFGP)
    If NOCFP > 0:
      Record C: float×lsparse  (sparse CFP matrix: [lsparse, rows, (ri, ncols, col, val×ncols)×rows])
    If NLGP > 0:
      Record D: [int+char+char*2+float+float]×NLGP  (parent state info)
    If NOCFGP > 0:
      Record E: float×lsparse  (sparse CFGP matrix)

Reference:
  Cowan (1981) The Theory of Atomic Structure and Spectra, App. 2.
  Sugano, Tanabe, Kamimura (1970) Multiplets of Transition-Metal Ions, App. A6.
"""
from __future__ import annotations
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from multitorch._constants import DTYPE


# ─────────────────────────────────────────────────────────────
# Fortran unformatted binary reader
# ─────────────────────────────────────────────────────────────

class _FortranReader:
    """Sequential reader for Fortran unformatted binary files (standard 4-byte markers)."""

    def __init__(self, path: str | Path):
        self._f = open(path, 'rb')

    def close(self):
        self._f.close()

    def read_record(self) -> bytes:
        """Read one Fortran record: [4-byte len][data][4-byte len]."""
        marker_bytes = self._f.read(4)
        if not marker_bytes:
            raise EOFError
        n = struct.unpack('<i', marker_bytes)[0]
        data = self._f.read(n)
        self._f.read(4)  # trailing marker (ignore)
        return data

    def read_record_as_floats(self) -> np.ndarray:
        data = self.read_record()
        return np.frombuffer(data, dtype='<f4')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _parse_sparse(raw: np.ndarray) -> np.ndarray:
    """
    Expand a sparse float32 array (from sparsex format) into a dense matrix.

    Format (all values stored as float32):
      raw[0]  = lsparse (total array length, as float → cast to int)
      raw[1]  = rows    (number of non-empty rows)
      Then for each row:
        raw[ix]   = ri     (1-indexed row number, as float)
        raw[ix+1] = ncols  (number of non-zero columns)
        For each nonzero:
          raw[ix+2j]   = col_j  (1-indexed column)
          raw[ix+2j+1] = value
    """
    rows = int(raw[1])
    if rows == 0:
        return np.zeros((0, 0), dtype=np.float32)

    ix = 2
    entries = []
    max_row = 0
    max_col = 0
    for _ in range(rows):
        ri = int(raw[ix])
        ncols = int(raw[ix + 1])
        ix += 2
        for _ in range(ncols):
            rj = int(raw[ix])
            val = float(raw[ix + 1])
            entries.append((ri, rj, val))
            max_row = max(max_row, ri)
            max_col = max(max_col, rj)
            ix += 2

    mat = np.zeros((max_row, max_col), dtype=np.float64)
    for ri, rj, val in entries:
        mat[ri - 1, rj - 1] = val
    return mat


@dataclass
class LSTermInfo:
    """Quantum numbers for one LS-coupled term of an l^n configuration."""
    index: int        # 0-based index within this block
    seniority: int    # seniority number (from record B integer field)
    L_label: str      # L label character ('S','P','D','F','G','H','I',...)
    alpha: str        # seniority designation (2-char ALF field)
    L: float          # orbital angular momentum quantum number
    S: float          # spin quantum number


class CFPBlock:
    """One (l, n) CFP block parsed from the binary file."""
    def __init__(self):
        self.ll: str = ''        # Shell label ('S', 'P', 'D', ...)
        self.ni: int = 0         # Number of electrons
        self.fll: float = 0.0   # l quantum number
        self.nlt: int = 0        # Number of LS terms for n-electron shell
        self.nlp: int = 0        # Number of terms for (n-1)-electron shell
        self.nocfp: int = 0
        self.nocfgp: int = 0
        # Sparse CFP matrix: shape (nlt, nlp) — coefficients <l^n αSL | l^(n-1) α'S'L'; l>
        self.cfp: Optional[np.ndarray] = None
        self.cfgp: Optional[np.ndarray] = None
        # Per-term quantum numbers
        self.terms: List[LSTermInfo] = []
        self.parent_terms: List[LSTermInfo] = []


def parse_cfp_file(path: str | Path) -> List[CFPBlock]:
    """
    Parse a rcg_cfp72/73/74 Fortran unformatted binary file into CFPBlock objects.

    File structure (CUVFD in ttrcg.f writes this):
      For each (l, n) configuration block:
        Record A (21 bytes): char(LL) + int(NI) + float(FLL) + int(NOT) + int(NLT) + int(NORCD)
        Record B (NLT×15 + 16 bytes): per-term data + [NLP, NLGP, NOCFP, NOCFGP]
        NORCD more records: CFP sparse matrix(ces) and parent term data interspersed

    NORCD counts the number of records that follow Record B. Sparse float records
    contain CFP/CFGP data; shorter records are parent term data.

    Parameters
    ----------
    path : str or Path
        Path to the rcg_cfp7x file.

    Returns
    -------
    List[CFPBlock] — one per (l, n) configuration block in the file.
    """
    blocks: List[CFPBlock] = []

    with _FortranReader(path) as fr:
        while True:
            try:
                # ─ Record A: shell header ─────────────────────────────────
                recA = fr.read_record()
                if len(recA) < 21:
                    break

                block = CFPBlock()
                # char(1) + int32 + float32 + int32 + int32 + int32 = 21 bytes
                block.ll = chr(recA[0])
                block.ni = struct.unpack_from('<i', recA, 1)[0]
                block.fll = struct.unpack_from('<f', recA, 5)[0]
                _not = struct.unpack_from('<i', recA, 9)[0]
                block.nlt = struct.unpack_from('<i', recA, 13)[0]
                norcd = struct.unpack_from('<i', recA, 17)[0]  # records after RecB

                if block.nlt <= 0:
                    # Skip NORCD records (which start with RecB) and continue
                    for _ in range(norcd):
                        fr.read_record()
                    continue

                # NORCD = total records for this block (RecB + additional).
                # Read RecB as first of NORCD records.
                all_recs = []
                for i in range(norcd):
                    all_recs.append(fr.read_record())

                # First record is RecB
                recB = all_recs[0]
                tail_off = block.nlt * 15

                # Extract per-term quantum numbers from RecB
                # Each term: int32(MULT/seniority) + char(LBCD) + char*2(ALF) + float(FL=L) + float(S)
                block.terms = []
                for t_idx in range(block.nlt):
                    off = t_idx * 15
                    if off + 15 <= len(recB):
                        sen = struct.unpack_from('<i', recB, off)[0]
                        lbcd = chr(recB[off + 4]) if recB[off + 4] > 0 else '?'
                        alf = recB[off + 5:off + 7].decode('ascii', errors='replace').strip()
                        fl = struct.unpack_from('<f', recB, off + 7)[0]
                        s_val = struct.unpack_from('<f', recB, off + 11)[0]
                        block.terms.append(LSTermInfo(
                            index=t_idx, seniority=sen,
                            L_label=lbcd, alpha=alf,
                            L=float(fl), S=float(s_val),
                        ))

                if tail_off + 16 <= len(recB):
                    block.nlp = struct.unpack_from('<i', recB, tail_off)[0]
                    _nlgp = struct.unpack_from('<i', recB, tail_off + 4)[0]
                    block.nocfp = struct.unpack_from('<i', recB, tail_off + 8)[0]
                    block.nocfgp = struct.unpack_from('<i', recB, tail_off + 12)[0]
                else:
                    _nlgp = 0
                    block.nocfp = 0
                    block.nocfgp = 0

                # Remaining records: find sparse CFP/CFGP matrices and parent term data
                # Sparse float records have length divisible by 4 and
                # their first float32 (as int) equals the record length in floats.
                cfp_found = False
                cfgp_found = False
                parent_found = False
                for rec in all_recs[1:]:
                    if len(rec) % 4 == 0 and len(rec) >= 24:
                        raw = np.frombuffer(rec, dtype='<f4')
                        lsparse_candidate = int(raw[0])
                        if lsparse_candidate == len(raw):
                            if not cfp_found and block.nocfp > 0:
                                block.cfp = _parse_sparse(raw)
                                cfp_found = True
                            elif not cfgp_found and block.nocfgp > 0:
                                block.cfgp = _parse_sparse(raw)
                                cfgp_found = True
                    # Parent term data record: same 15-byte-per-term format
                    elif not parent_found and _nlgp > 0 and len(rec) == _nlgp * 15:
                        block.parent_terms = []
                        for t_idx in range(_nlgp):
                            off = t_idx * 15
                            sen = struct.unpack_from('<i', rec, off)[0]
                            lbcd = chr(rec[off + 4]) if rec[off + 4] > 0 else '?'
                            alf = rec[off + 5:off + 7].decode('ascii', errors='replace').strip()
                            fl = struct.unpack_from('<f', rec, off + 7)[0]
                            s_val = struct.unpack_from('<f', rec, off + 11)[0]
                            block.parent_terms.append(LSTermInfo(
                                index=t_idx, seniority=sen,
                                L_label=lbcd, alpha=alf,
                                L=float(fl), S=float(s_val),
                            ))
                        parent_found = True

                blocks.append(block)

            except (EOFError, struct.error):
                break

    return blocks


# ─────────────────────────────────────────────────────────────
# Cached access to CFP tables
# ─────────────────────────────────────────────────────────────

_cfp_cache: Optional[List[CFPBlock]] = None
_cfp_lock = threading.Lock()


def load_cfp_tables(binpath: Optional[str] = None) -> List[CFPBlock]:
    """
    Load CFP tables from the rcg_cfp72 file.

    Searches for the file in:
      1. binpath (if provided)
      2. MULTITORCH_BINPATH environment variable
      3. Adjacent ttmult/bin directory relative to this file

    Returns cached results on repeated calls.  Thread-safe.
    """
    global _cfp_cache
    if _cfp_cache is not None:
        return _cfp_cache

    with _cfp_lock:
        # Double-check after acquiring lock
        if _cfp_cache is not None:
            return _cfp_cache

        import os
        search_paths = []
        if binpath:
            search_paths.append(Path(binpath) / 'rcg_cfp72')
        env_path = os.environ.get('MULTITORCH_BINPATH')
        if env_path:
            search_paths.append(Path(env_path) / 'rcg_cfp72')
        # Bundled package data: multitorch/data/cfp/
        pkg_data = Path(__file__).resolve().parent.parent / 'data' / 'cfp' / 'rcg_cfp72'
        search_paths.append(pkg_data)
        # Legacy: relative to this file: ../../../../ttmult/bin/
        here = Path(__file__).resolve()
        for _ in range(5):
            here = here.parent
            candidate = here / 'ttmult' / 'bin' / 'rcg_cfp72'
            if candidate.exists():
                search_paths.append(candidate)
                break

        for p in search_paths:
            if p.exists():
                _cfp_cache = parse_cfp_file(p)
                return _cfp_cache

    raise FileNotFoundError(
        "rcg_cfp72 not found. Install multitorch with package data, "
        "set binpath or MULTITORCH_BINPATH, or place rcg_cfp72 in multitorch/data/cfp/."
    )


def get_cfp_for_shell(l: int, n: int,
                      binpath: Optional[str] = None) -> Optional[torch.Tensor]:
    """
    Get the CFP matrix for l^n → l^(n-1) transitions.

    The shell label 'l' maps to: 0='S', 1='P', 2='D', 3='F'

    Parameters
    ----------
    l : int
        Orbital quantum number (2 for d-electrons).
    n : int
        Number of electrons.
    binpath : str or None
        Path to ttmult/bin/ directory.

    Returns
    -------
    torch.Tensor  shape (n_states_n, n_states_n_minus_1)  or None
    """
    shell_labels = {0: 'S', 1: 'P', 2: 'D', 3: 'F', 4: 'G'}
    label = shell_labels.get(l, '?')

    blocks = load_cfp_tables(binpath)
    for block in blocks:
        if block.ll.upper() == label and block.ni == n and block.cfp is not None:
            return torch.tensor(block.cfp, dtype=DTYPE)

    return None


def get_cfp_block(l: int, n: int,
                  binpath: Optional[str] = None) -> Optional[CFPBlock]:
    """
    Get the full CFP block (including term info) for l^n.

    Parameters
    ----------
    l : int
        Orbital quantum number (2 for d-electrons).
    n : int
        Number of electrons.
    binpath : str or None
        Path to ttmult/bin/ directory.

    Returns
    -------
    CFPBlock or None
    """
    shell_labels = {0: 'S', 1: 'P', 2: 'D', 3: 'F', 4: 'G'}
    label = shell_labels.get(l, '?')

    blocks = load_cfp_tables(binpath)
    for block in blocks:
        if block.ll.upper() == label and block.ni == n:
            return block

    return None


# ─────────────────────────────────────────────────────────────
# rcg_cfp73 parser: pre-tabulated U^(k) matrices
# ─────────────────────────────────────────────────────────────

@dataclass
class UkBlock:
    """Unit tensor U^(k) and V^(k) matrices for one (l, n) configuration.

    U^(k): orbital unit tensor (δ_SS' selection rule, connects same spin)
    V^(k): spin-orbit unit tensor (ΔS=0,±1, connects different spins)
    """
    ll: str        # Shell label
    ni: int        # Number of electrons
    fll: float     # l quantum number
    nlt: int       # Number of LS terms
    # uk[k] = np.ndarray of shape (nlt, nlt) for orbital tensor rank k
    uk: Dict[int, np.ndarray]
    # vk[k] = np.ndarray of shape (nlt, nlt) for spin-orbit tensor rank k
    vk: Dict[int, np.ndarray]


_uk_cache: Optional[List[UkBlock]] = None
_uk_lock = threading.Lock()


def parse_cfp73_file(path: str | Path) -> List[UkBlock]:
    """
    Parse rcg_cfp73 to extract pre-tabulated U^(k) matrices.

    The file has the same block-header structure as rcg_cfp72.
    After the header, records contain sparse U^(k) matrices for k=0,1,...,2l.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    List[UkBlock]
    """
    blocks: List[UkBlock] = []

    with _FortranReader(path) as fr:
        while True:
            try:
                recA = fr.read_record()
                if len(recA) < 21:
                    break

                ll = chr(recA[0])
                ni = struct.unpack_from('<i', recA, 1)[0]
                fll = struct.unpack_from('<f', recA, 5)[0]
                _not = struct.unpack_from('<i', recA, 9)[0]
                nlt = struct.unpack_from('<i', recA, 13)[0]
                norcd = struct.unpack_from('<i', recA, 17)[0]

                all_recs = []
                for i in range(norcd):
                    all_recs.append(fr.read_record())

                if nlt <= 0:
                    continue

                # rcg_cfp73 has NO RecB (unlike rcg_cfp72).
                # All NORCD records are sparse matrices:
                #   Records 0..(NOR-1): U^(k) matrices for k=0..NOR-1
                #   Records NOR..(2*NOR-1): V^(k) matrices for k=0..NOR-1
                # where NOR = NORCD/2 = 2l+1
                l_val = int(round(fll))
                nor = norcd // 2  # number of U matrices (= 2l+1)
                uk_dict: Dict[int, np.ndarray] = {}
                vk_dict: Dict[int, np.ndarray] = {}
                sparse_idx = 0
                for rec in all_recs:
                    if len(rec) % 4 == 0 and len(rec) >= 8:
                        raw = np.frombuffer(rec, dtype='<f4')
                        lsparse_candidate = int(raw[0])
                        if lsparse_candidate == len(raw) and lsparse_candidate > 1:
                            mat = _parse_sparse(raw)
                            if sparse_idx < nor:
                                uk_dict[sparse_idx] = mat
                            else:
                                vk_dict[sparse_idx - nor] = mat
                            sparse_idx += 1

                blocks.append(UkBlock(
                    ll=ll, ni=ni, fll=fll, nlt=nlt,
                    uk=uk_dict, vk=vk_dict,
                ))

            except (EOFError, struct.error):
                break

    return blocks


def load_uk_tables(binpath: Optional[str] = None) -> List[UkBlock]:
    """Load U^(k) tables from rcg_cfp73 file (cached).  Thread-safe."""
    global _uk_cache
    if _uk_cache is not None:
        return _uk_cache

    with _uk_lock:
        # Double-check after acquiring lock
        if _uk_cache is not None:
            return _uk_cache

        import os
        search_paths = []
        if binpath:
            search_paths.append(Path(binpath) / 'rcg_cfp73')
        env_path = os.environ.get('MULTITORCH_BINPATH')
        if env_path:
            search_paths.append(Path(env_path) / 'rcg_cfp73')
        # Bundled package data: multitorch/data/cfp/
        pkg_data = Path(__file__).resolve().parent.parent / 'data' / 'cfp' / 'rcg_cfp73'
        search_paths.append(pkg_data)
        # Legacy: relative to this file: ../../../../ttmult/bin/
        here = Path(__file__).resolve()
        for _ in range(5):
            here = here.parent
            candidate = here / 'ttmult' / 'bin' / 'rcg_cfp73'
            if candidate.exists():
                search_paths.append(candidate)
                break

        for p in search_paths:
            if p.exists():
                _uk_cache = parse_cfp73_file(p)
                return _uk_cache

    raise FileNotFoundError(
        "rcg_cfp73 not found. Install multitorch with package data, "
        "set binpath or MULTITORCH_BINPATH, or place rcg_cfp73 in multitorch/data/cfp/."
    )


def get_uk_for_shell(l: int, n: int,
                     binpath: Optional[str] = None) -> Optional[UkBlock]:
    """Get the U^(k) matrices for l^n configuration."""
    shell_labels = {0: 'S', 1: 'P', 2: 'D', 3: 'F', 4: 'G'}
    label = shell_labels.get(l, '?')

    blocks = load_uk_tables(binpath)
    for block in blocks:
        if block.ll.upper() == label and block.ni == n:
            return block

    return None
