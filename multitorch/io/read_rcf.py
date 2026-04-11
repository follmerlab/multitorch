"""
Parsers for rcn31 and rcn2 output files.

rcn31 output (nid8.rcn31_out):
  Contains self-consistent orbital energies (EE, in Ry), radial expectation
  values, spin-orbit coupling constants (ZETA, in Ry and eV), and relativistic
  corrections per orbital per configuration.

rcn2 output (nid8.rcn2_out):
  Contains Slater-Condon Fk and Gk integrals (in Ry and eV) plus the
  radial electric multipole integral R1 for dipole transitions.
  Also contains the RCG-ready input deck (the FACT= block).

These parsers extract the physically meaningful quantities as torch.Tensor
for unit test validation of the atomic module (Phase 4).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch

from multitorch._constants import DTYPE, RY_TO_EV


@dataclass
class OrbitalData:
    """Data for one orbital from rcn31 output."""
    nl: str           # e.g. '2P', '3D'
    wn: float         # occupancy
    ee_ry: float      # orbital energy in Ry
    zeta_ry: float    # spin-orbit coupling in Ry (Blume-Watson)
    zeta_ev: float    # spin-orbit coupling in eV


@dataclass
class HFSOutput:
    """
    Parsed output from one rcn31 HFS configuration block.

    Holds orbital energies and spin-orbit coupling constants for one
    electronic configuration (e.g. Ni2+ 2p06 3d08).
    """
    config_label: str                 # e.g. 'Ni2+ 2p06 3d08'
    orbitals: List[OrbitalData] = field(default_factory=list)
    mesh: int = 641

    def orbital(self, nl: str) -> Optional[OrbitalData]:
        """Return the OrbitalData for orbital nl (e.g. '2P', '3D')."""
        for o in self.orbitals:
            if o.nl.strip().lower() == nl.strip().lower():
                return o
        return None

    def zeta(self, nl: str) -> torch.Tensor:
        """Return spin-orbit coupling in eV for orbital nl."""
        o = self.orbital(nl)
        if o is None:
            raise KeyError(f"Orbital {nl} not found in {self.config_label}")
        return torch.tensor(o.zeta_ev, dtype=DTYPE)

    def energy(self, nl: str) -> torch.Tensor:
        """Return orbital energy in eV."""
        o = self.orbital(nl)
        if o is None:
            raise KeyError(f"Orbital {nl} not found in {self.config_label}")
        return torch.tensor(o.ee_ry, dtype=DTYPE) * RY_TO_EV


def read_rcn31_out(path: str | Path) -> List[HFSOutput]:
    """
    Parse a rcn31 output file (nid8.rcn31_out).

    Returns a list of HFSOutput — one per configuration block found.

    Each block contains the table:
      NL  WNL   EE   AZ   ...
      ...
      ZETA section:
      NL WNL  --BLUME-WATSON--  ...
    """
    path = Path(path)
    configs: List[HFSOutput] = []
    current: Optional[HFSOutput] = None
    in_zeta = False

    with open(path, "r") as f:
        for line in f:
            # Detect config header, e.g.:
            # "   Ni2+ 2p06 3d08          NCONF=  1    Z= 28 ..."
            if "NCONF=" in line and "Z=" in line:
                label = line.split("NCONF=")[0].strip()
                current = HFSOutput(config_label=label)
                configs.append(current)
                in_zeta = False
                continue

            if current is None:
                continue

            # Detect start of ZETA table
            if "----------ZETA----------" in line or "ZETA" in line.upper() and "----" in line:
                in_zeta = True
                continue

            if in_zeta:
                # Parse ZETA line:
                # "  2P  6.     0.81583      11.100   0.85796 ..."
                parts = line.split()
                if len(parts) >= 4 and _is_nl(parts[0]):
                    nl = parts[0]
                    try:
                        zeta_ry = float(parts[2])
                        zeta_ev = float(parts[3])
                    except (ValueError, IndexError):
                        continue
                    # Update or create orbital entry
                    orb = current.orbital(nl)
                    if orb is not None:
                        orb.zeta_ry = zeta_ry
                        orb.zeta_ev = zeta_ev
                    else:
                        current.orbitals.append(OrbitalData(
                            nl=nl, wn=0.0, ee_ry=0.0,
                            zeta_ry=zeta_ry, zeta_ev=zeta_ev,
                        ))
                # Stop ZETA section at blank or at non-orbital line
                if not parts:
                    in_zeta = False
            else:
                # Parse EE table:  "  2P  6.   -67.66589  ..."
                parts = line.split()
                if len(parts) >= 3 and _is_nl(parts[0]):
                    try:
                        wn = float(parts[1].rstrip("."))
                        ee = float(parts[2])
                    except ValueError:
                        continue
                    orb = current.orbital(parts[0])
                    if orb is not None:
                        orb.wn = wn
                        orb.ee_ry = ee
                    else:
                        current.orbitals.append(OrbitalData(
                            nl=parts[0], wn=wn, ee_ry=ee,
                            zeta_ry=0.0, zeta_ev=0.0,
                        ))

    return configs


def _is_nl(s: str) -> bool:
    """Check if a string looks like an nl orbital label, e.g. '1S', '2P', '3D'."""
    if len(s) < 2:
        return False
    return s[0].isdigit() and s[1].upper() in ("S", "P", "D", "F", "G")


# ─────────────────────────────────────────────────────────────
# rcn2 output parser — Slater integrals
# ─────────────────────────────────────────────────────────────

@dataclass
class SlaterParams:
    """
    Slater-Condon and SOC parameters for one configuration pair from rcn2.

    All energies in eV.
    """
    config_label: str
    # Fk(dd): Coulomb integrals, k=0,2,4  (eV)
    Fk_dd: torch.Tensor     # shape (3,)
    # Fk(pd): Coulomb integrals, k=0,2    (eV)
    Fk_pd: torch.Tensor     # shape (2,)
    # Gk(pd): Exchange integrals, k=1,3   (eV)
    Gk_pd: torch.Tensor     # shape (2,)
    # Spin-orbit coupling (eV)
    zeta_2p: float
    zeta_3d: float
    # Radial electric dipole integral (a.u.)
    R1: float
    # Reduction factors applied (FACT line)
    factors: List[float] = field(default_factory=list)


def read_rcn2_out(path: str | Path) -> List[SlaterParams]:
    """
    Parse a rcn2 output file (nid8.rcn2_out) into SlaterParams objects.

    Extracts one SlaterParams per configuration line.

    The FACT= block in rcn2 output looks like:
      0Ni2+ 2p06 3d08   0.00000  12.23388  7.59758  0.08260  0.00000  HR  FACT= 1.00 ...
    Where values are: [F0dd E_shift  F2dd  F4dd  F0pd  ...]
    See Thole/Cowan code documentation for column ordering.

    The radial dipole integral is on the line starting with the atomic number Z.
    """
    path = Path(path)
    results: List[SlaterParams] = []

    with open(path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        # Configuration data lines start with '0' (Fortran page break) followed by config label
        # e.g.: "0Ni2+ 2p06 3d08               0.00000    12.23388     7.59758 ..."
        stripped = line.lstrip("0").strip()
        if ("2p06" in line or "2p05" in line) and "FACT=" in line:
            parts = line.split("FACT=")
            # Extract numerical values before FACT=
            # Remove leading '0' and label
            label_end = _find_label_end(line)
            val_str = line[label_end:parts[0].rfind("HR")].strip()
            vals = val_str.split()
            try:
                floats = [float(v) for v in vals if _is_float(v)]
            except ValueError:
                continue

            # Get FACT multipliers
            fact_parts = parts[1].replace("1", "1").split()
            facts = [float(v) for v in fact_parts if _is_float(v)]

            # Build a SlaterParams — column ordering from rcn2:
            # [F0dd, F2dd, F4dd, F0pd or edge_energy, F2pd, G1pd, G3pd, R1...]
            # The exact columns depend on the configuration
            label = _extract_config_label(line)
            sp = _build_slater_params(label, floats, facts)
            if sp is not None:
                results.append(sp)

        # Second config (excited state) — e.g. "0Ni2+ 2p05 3d09  860.16587 ..."
        # Contains edge energy + excited state parameters

    return results


def _find_label_end(line: str) -> int:
    """Find where the config label ends and numerical values begin."""
    # After stripping leading '0', label is something like 'Ni2+ 2p06 3d08'
    # then whitespace then numbers
    i = 0
    while i < len(line) and not line[i].isdigit():
        i += 1
    # Skip past the leading '0' or whitespace
    # Find first digit that's part of numbers (not config label)
    # Config labels look like: Ni2+ 2p06 3d08
    # After label there are ~15+ spaces then values
    # Find column where we have 5+ consecutive spaces
    for j in range(i, len(line) - 5):
        if line[j:j+5] == "     ":
            return j + 5
    return i


def _extract_config_label(line: str) -> str:
    """Extract config label from a rcn2 line."""
    stripped = line.lstrip()
    if stripped.startswith("0"):
        stripped = stripped[1:].lstrip()
    # Label ends before the long whitespace
    parts = stripped.split()
    if len(parts) >= 3:
        return " ".join(parts[:3])
    return stripped[:30].strip()


def _build_slater_params(label: str, vals: list, facts: list) -> Optional[SlaterParams]:
    """
    Build SlaterParams from raw float list extracted from rcn2 output.

    The rcn2 output line for a 2p→3d L-edge has the format:
      E0  V1  F2(dd)  F4(dd)  V4  [edge_energy  V6  F2(pd)  G1(pd)  G3(pd)  ...]

    For the GROUND STATE (2p06 3d08), 5 values after label:
      vals[0] = E0 = 0.0 (energy offset)
      vals[1] = F0(pd) or another Coulomb parameter (~12 eV)
      vals[2] = F2(dd) ≈ 7.60 eV   ← USE THIS
      vals[3] = F4(dd) ≈ 0.083 eV
      vals[4] = 0.0 (ζ or placeholder)

    For the EXCITED STATE (2p05 3d09), more values including edge energy.

    NOTE: Full decoding of the rcf digit-suffix encoding is implemented
    in Phase 4 (atomic parameters). This simplified parser extracts the
    key F2(dd), F4(dd) values by position.
    """
    if len(vals) < 3:
        return None
    try:
        return SlaterParams(
            config_label=label,
            Fk_dd=torch.tensor([0.0,
                                 vals[2] if len(vals) > 2 else 0.0,  # F2(dd)
                                 vals[3] if len(vals) > 3 else 0.0], # F4(dd)
                                dtype=DTYPE),
            Fk_pd=torch.tensor([0.0, vals[7] if len(vals) > 7 else 0.0], dtype=DTYPE),
            Gk_pd=torch.tensor([vals[8] if len(vals) > 8 else 0.0,
                                 vals[9] if len(vals) > 9 else 0.0], dtype=DTYPE),
            zeta_2p=0.0,
            zeta_3d=0.0,
            R1=0.0,
            factors=facts,
        )
    except (IndexError, ValueError):
        return None


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False
