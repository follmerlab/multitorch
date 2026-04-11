"""
Parser for .ban input files (ttban configuration).

The .ban file specifies how to combine multiple electronic configurations
into a charge-transfer multiplet calculation:
  NCONF:  number of configurations (ground, excited)
  N2/N3:  band discretization parameters
  DEF:    energy offsets (EG, EF per configuration)
  XMIX:   charge transfer mixing parameters
  XHAM:   Hamiltonian operator strengths
  TRAN:   which configurations are connected by transitions
  TRIADS: symmetry triads (ground, actor, final irreps)

Reference: pyctm/write_BAN.py
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import re


@dataclass
class XMIXEntry:
    """One XMIX block: mixing parameters for charge transfer."""
    values: List[float]           # V values per channel (b1, a1, b2, e for D4h)
    combos: List[Tuple[int, int, int]]  # (state_type, conf_from, conf_to) per combo


@dataclass
class XHAMEntry:
    """One XHAM block: Hamiltonian operator strengths."""
    values: List[float]           # [1.0, tendq_eff, dt, ds] for D4h
    combos: List[Tuple[int, int]]  # (state_type, conf) per combo


@dataclass
class BanData:
    """Parsed .ban file data."""
    erange: float = 0.3
    nconf_gs: int = 1             # Number of ground state configurations
    nconf_fs: int = 1             # Number of final state configurations
    n_band: dict = field(default_factory=dict)  # N2, N3 etc.
    eg: dict = field(default_factory=dict)  # {conf_idx: energy} (1-indexed)
    ef: dict = field(default_factory=dict)  # {conf_idx: energy} (1-indexed)
    xmix: List[XMIXEntry] = field(default_factory=list)
    xham: List[XHAMEntry] = field(default_factory=list)
    tran: List[Tuple[int, int]] = field(default_factory=list)  # (conf_gs, conf_fs) pairs
    triads: List[Tuple[str, str, str]] = field(default_factory=list)
    prmult: bool = False


def read_ban(path: str | Path) -> BanData:
    """
    Parse a .ban input file.

    Parameters
    ----------
    path : str or Path
        Path to the .ban file.

    Returns
    -------
    BanData
    """
    path = Path(path)
    data = BanData()

    with open(path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        tokens = line.split()
        if not tokens:
            i += 1
            continue

        keyword = tokens[0].upper()

        if keyword == 'PRMULT':
            data.prmult = True
            i += 1

        elif keyword == 'ERANGE':
            data.erange = float(tokens[1])
            i += 1

        elif keyword == 'NCONF':
            data.nconf_gs = int(tokens[1])
            data.nconf_fs = int(tokens[2])
            i += 1

        elif keyword.startswith('N') and keyword[1:].isdigit():
            conf_idx = int(keyword[1:])
            n_val = int(tokens[1])
            data.n_band[conf_idx] = n_val
            i += 1

        elif keyword == 'DEF':
            # DEF EG2 = 5.0 UNITY or DEF EF2 = 4.0 UNITY
            name = tokens[1].upper()
            val = float(tokens[3])
            if name.startswith('EG'):
                idx = int(name[2:])
                data.eg[idx] = val
            elif name.startswith('EF'):
                idx = int(name[2:])
                data.ef[idx] = val
            i += 1

        elif keyword == 'XMIX':
            n_vals = int(tokens[1])
            values = [float(t) for t in tokens[2:2 + n_vals]]
            i += 1
            # Next line: n_combos  (state conf_from conf_to) ...
            combo_line = lines[i].strip().split()
            n_combos = int(combo_line[0])
            combos = []
            idx = 1
            for _ in range(n_combos):
                st = int(combo_line[idx])
                cf = int(combo_line[idx + 1])
                ct = int(combo_line[idx + 2])
                combos.append((st, cf, ct))
                idx += 3
            data.xmix.append(XMIXEntry(values=values, combos=combos))
            i += 1

        elif keyword == 'XHAM':
            n_vals = int(tokens[1])
            values = [float(t) for t in tokens[2:2 + n_vals]]
            i += 1
            # Next line: n_combos  (state conf) ...
            combo_line = lines[i].strip().split()
            n_combos = int(combo_line[0])
            combos = []
            idx = 1
            for _ in range(n_combos):
                st = int(combo_line[idx])
                co = int(combo_line[idx + 1])
                combos.append((st, co))
                idx += 2
            data.xham.append(XHAMEntry(values=values, combos=combos))
            i += 1

        elif keyword == 'TRAN':
            n_pairs = int(tokens[1])
            pairs = []
            idx = 2
            for _ in range(n_pairs):
                cg = int(tokens[idx])
                cf = int(tokens[idx + 1])
                pairs.append((cg, cf))
                idx += 2
            data.tran = pairs
            i += 1

        elif keyword == 'TRIADS':
            i += 1
            while i < len(lines):
                tline = lines[i].strip()
                if not tline:
                    i += 1
                    break
                # Parse triad: 3 symmetry labels (may have ^ prefix)
                parts = tline.split()
                if len(parts) >= 3:
                    data.triads.append((parts[0], parts[1], parts[2]))
                else:
                    break
                i += 1

        else:
            i += 1

    return data
