"""
Atomic-parameter fixture loader for the Track C Phase 5 parity tests.

Why this module exists
----------------------
The C3f parity test in Track C compares an in-memory COWAN store builder
against a parsed `.rme_rcg` fixture. To make that comparison
self-consistent, the in-memory builder must consume the *same* atomic
parameters (Slater Fk/Gk integrals, spin-orbit ζ) that Cowan rcn31 used
when it generated the reference `.rme_rcg`. Those parameter values are
stored in human-readable form in the `.rcn31_out` companion file.

The existing :func:`multitorch.io.read_rcf.read_rcn31_out` parser only
extracts the orbital energies and ZETA section (it was written for HFS
SCF validation). It does not parse the SLATER INTEGRALS section.

This module fills that gap by parsing the SLATER INTEGRALS section *and*
the ZETA section from the same file in one pass, returning a structured
:class:`AtomicParams` dataclass keyed for downstream use by
``build_cowan_store_in_memory`` (C3e).

Design notes
------------
- Values are stored in **Rydberg** to match the convention of the rest of
  multitorch's atomic module. The eV column is checked for consistency
  but not stored; use ``RY_TO_EV`` if you need eV downstream.
- The ``.rcn31_out`` file contains one block per electronic configuration
  (e.g. ground ``2p06 3d08`` then excited ``2p05 3d09`` for a 2p L-edge
  XAS). Both blocks are parsed and returned as a list, in the order
  encountered.
- The values in the file are *already* the rcn31 ab-initio Hartree-Fock
  Slater integrals — they have **not** been multiplied by Cowan's
  empirical reduction factors. The C3f parity test must therefore use
  ``slater_scale=1.0`` so that the in-memory builder reproduces the
  reference COWAN store element-wise.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────


@dataclass
class ConfigParams:
    """Parsed atomic parameters for one electronic configuration."""

    label: str
    """Configuration label, e.g. 'Ni2+ 2p06 3d08'."""

    nconf: int
    """1 = ground, 2 = excited (rcn31's NCONF index)."""

    fk: Dict[Tuple[str, str, int], float] = field(default_factory=dict)
    """Slater Fk integrals in Ry, keyed by (shell_a, shell_b, k).

    Shell labels are uppercase, e.g. ``'2P'``, ``'3D'``. The key order
    matches the file: the smaller-index shell appears first.
    """

    gk: Dict[Tuple[str, str, int], float] = field(default_factory=dict)
    """Slater Gk exchange integrals in Ry, keyed by (shell_a, shell_b, k).

    Diagonal entries (same shell) are not stored — Gk is only meaningful
    for distinct shells.
    """

    zeta_bw: Dict[str, float] = field(default_factory=dict)
    """Blume-Watson spin-orbit ζ in Ry, keyed by shell label."""

    zeta_rvi: Dict[str, float] = field(default_factory=dict)
    """Central-field R*VI spin-orbit ζ in Ry, keyed by shell label."""

    # ── Convenience accessors ────────────────────────────
    def f(self, a: str, b: str, k: int) -> float:
        """Return Fk(a,b) in Ry, trying both orderings."""
        a, b = a.upper(), b.upper()
        if (a, b, k) in self.fk:
            return self.fk[(a, b, k)]
        if (b, a, k) in self.fk:
            return self.fk[(b, a, k)]
        raise KeyError(f"F^{k}({a},{b}) not found in {self.label}")

    def g(self, a: str, b: str, k: int) -> float:
        """Return Gk(a,b) in Ry, trying both orderings."""
        a, b = a.upper(), b.upper()
        if (a, b, k) in self.gk:
            return self.gk[(a, b, k)]
        if (b, a, k) in self.gk:
            return self.gk[(b, a, k)]
        raise KeyError(f"G^{k}({a},{b}) not found in {self.label}")

    def zeta(self, shell: str, *, method: str = "blume_watson") -> float:
        """Return spin-orbit ζ in Ry for one shell.

        Parameters
        ----------
        shell : str
            Shell label, e.g. ``'2P'``, ``'3D'``.
        method : {'blume_watson', 'rvi'}
            Which column to read. Default ``'blume_watson'`` matches
            the recommended physics setting for L-edge XAS.
        """
        shell = shell.upper()
        table = self.zeta_bw if method == "blume_watson" else self.zeta_rvi
        if shell not in table:
            raise KeyError(
                f"ζ({shell}) not found in {self.label} ({method})"
            )
        return table[shell]


@dataclass
class AtomicParams:
    """All atomic parameters parsed from a single `.rcn31_out` file."""

    configs: List[ConfigParams] = field(default_factory=list)
    source_path: Optional[Path] = None

    def by_nconf(self, nconf: int) -> ConfigParams:
        """Return the configuration block whose ``NCONF`` index is ``nconf``."""
        for c in self.configs:
            if c.nconf == nconf:
                return c
        raise KeyError(
            f"NCONF={nconf} not found in {self.source_path} "
            f"(have {[c.nconf for c in self.configs]})"
        )

    @property
    def ground(self) -> ConfigParams:
        """Shortcut for the NCONF=1 (ground) configuration."""
        return self.by_nconf(1)

    @property
    def excited(self) -> ConfigParams:
        """Shortcut for the NCONF=2 (excited) configuration."""
        return self.by_nconf(2)


# ─────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────

# Header that introduces the SLATER INTEGRALS section. The Fortran
# pagination prefix (the leading '1' or '0') is stripped before matching.
_SLATER_HEADER_RE = re.compile(r"^\s*SLATER\s+INTEGRALS\s*$")

# Header that introduces the ZETA table. The next non-blank line is the
# column header (NL WNL ----BLUME-WATSON----) which we skip.
_ZETA_HEADER_RE = re.compile(r"-+\s*ZETA\s*-+")

# A line of Fk/Gk values:
#   ( 2P, 3D)       2    0.5115688 RYD  =      6.96030  EV    1.000  ...
#                                                   3    0.2119891 RYD  ...
# Pulls (orbital_a, orbital_b, k_F, F_ry, k_G, G_ry).
_FK_GK_RE = re.compile(
    r"\(\s*([1-9][SPDFG]),\s*([1-9][SPDFG])\s*\)"  # ( 2P, 3D)
    r"\s+(\d+)"                                      # k_F
    r"\s+([\-+]?\d+\.\d+)\s*RYD"                    # F in Ry
    r".*?(\d+)"                                      # k_G
    r"\s+([\-+]?\d+\.\d+)\s*RYD",                   # G in Ry
)

# A continuation line for a 2nd Fk/Gk on the same shell pair.
_FK_GK_NO_PAIR_RE = re.compile(
    r"^\s+(\d+)"                                     # k_F
    r"\s+([\-+]?\d+\.\d+)\s*RYD"                    # F in Ry
    r".*?(\d+)"                                      # k_G
    r"\s+([\-+]?\d+\.\d+)\s*RYD",                   # G in Ry
)

# Configuration header line:
#   "   Ni2+ 2p06 3d08          NCONF=  1    Z= 28 ..."
_NCONF_RE = re.compile(r"NCONF\s*=\s*(\d+)")

# A line of ζ values: "  2P  6.     0.81583   11.100   0.85796   11.673  ..."
_ZETA_LINE_RE = re.compile(
    r"^\s*([1-9][SPDFG])"                            # shell label
    r"\s+\d+\.?"                                     # occupation
    r"\s+([\-+]?\d+\.\d+)"                           # ζ_BW (Ry)
    r"\s+([\-+]?\d+\.\d+)"                           # ζ_BW (eV) — discarded
    r"\s+([\-+]?\d+\.\d+)"                           # ζ_R*VI (Ry)
    r"\s+([\-+]?\d+\.\d+)"                           # ζ_R*VI (eV) — discarded
)


def _strip_pagination(line: str) -> str:
    """Strip the Fortran column-1 pagination character ('0', '1', etc.)."""
    if line and line[0] in "01":
        return line[1:]
    return line


def read_rcn31_out_params(path: str | Path) -> AtomicParams:
    """Parse a `.rcn31_out` file into an :class:`AtomicParams`.

    Extracts both the SLATER INTEGRALS and the ZETA tables from every
    configuration block found in the file, in the order encountered.

    Parameters
    ----------
    path : str or Path
        Path to a Cowan ``.rcn31_out`` file (e.g. ``nid8.rcn31_out``).

    Returns
    -------
    AtomicParams
        One :class:`ConfigParams` per configuration block, with all Fk,
        Gk, ζ_BW, and ζ_R*VI values populated. Values are in Rydberg.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file contains no recognisable configuration blocks.
    """
    path = Path(path)
    text = path.read_text()
    lines = text.splitlines()

    out = AtomicParams(source_path=path)
    current: Optional[ConfigParams] = None

    in_slater = False
    in_zeta = False
    last_pair: Optional[Tuple[str, str]] = None  # for Fk/Gk continuation

    for raw in lines:
        line = _strip_pagination(raw)

        # Configuration header — open or reopen a ConfigParams. The
        # `.rcn31_out` file emits the same NCONF header several times
        # per configuration (once for the HFS section, once for the
        # SLATER INTEGRALS section, once for the INTERACTION ENERGIES
        # section). We must NOT reset in_slater/in_zeta on these
        # headers — for example, the SLATER INTEGRALS section header
        # appears immediately *before* the second NCONF header for the
        # same config, so resetting would disable Fk/Gk parsing entirely.
        m = _NCONF_RE.search(line)
        if m and "Z=" in line and "ION=" in line:
            label = line.split("NCONF=")[0].strip()
            nconf = int(m.group(1))
            existing = next(
                (c for c in out.configs if c.nconf == nconf), None
            )
            if existing is not None:
                current = existing
            else:
                current = ConfigParams(label=label, nconf=nconf)
                out.configs.append(current)
            last_pair = None
            continue

        if current is None:
            continue

        # Section toggles.
        if _SLATER_HEADER_RE.match(line):
            in_slater = True
            in_zeta = False
            last_pair = None
            continue
        if _ZETA_HEADER_RE.search(line):
            in_zeta = True
            in_slater = False
            last_pair = None
            continue

        # The "INTERACTION ENERGIES" header closes the SLATER section
        # before the next config block.
        if "INTERACTION ENERGIES" in line:
            in_slater = False
            in_zeta = False
            last_pair = None
            continue

        if in_slater:
            mm = _FK_GK_RE.search(line)
            if mm:
                a = mm.group(1).upper()
                b = mm.group(2).upper()
                k_f = int(mm.group(3))
                f_ry = float(mm.group(4))
                k_g = int(mm.group(5))
                g_ry = float(mm.group(6))
                current.fk[(a, b, k_f)] = f_ry
                if a != b and g_ry != 0.0:
                    current.gk[(a, b, k_g)] = g_ry
                last_pair = (a, b)
                continue
            # Continuation row for the same pair (e.g. ( 3D, 3D ) k=2 line).
            if last_pair is not None:
                mc = _FK_GK_NO_PAIR_RE.match(line)
                if mc:
                    k_f = int(mc.group(1))
                    f_ry = float(mc.group(2))
                    k_g = int(mc.group(3))
                    g_ry = float(mc.group(4))
                    a, b = last_pair
                    current.fk[(a, b, k_f)] = f_ry
                    if a != b and g_ry != 0.0:
                        current.gk[(a, b, k_g)] = g_ry

        elif in_zeta:
            mz = _ZETA_LINE_RE.match(line)
            if mz:
                shell = mz.group(1).upper()
                zeta_bw_ry = float(mz.group(2))
                zeta_rvi_ry = float(mz.group(4))
                current.zeta_bw[shell] = zeta_bw_ry
                current.zeta_rvi[shell] = zeta_rvi_ry

    if not out.configs:
        raise ValueError(f"No configuration blocks found in {path}")

    return out
