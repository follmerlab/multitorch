"""
Atomic data tables for 3d transition metals.

Contains:
  - Default electronic configurations for ground and L-edge excited states
  - Atomic numbers
  - Orbital quantum numbers

Supports: Ti(22), V(23), Cr(24), Mn(25), Fe(26), Co(27), Ni(28), Cu(29), Zn(30)
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple


# Atomic numbers for supported elements
ATOMIC_NUMBERS = {
    'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25,
    'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
}

# Ground state d-electron counts for M^n+ ions
# Format: (element, valence_string) → n_d_electrons
# Valence strings: 'i' = +1, 'ii' = +2, 'iii' = +3, etc.
D_ELECTRON_COUNT = {
    ('Ti', 'ii'): 2,  ('Ti', 'iii'): 1,  ('Ti', 'iv'): 0,
    ('V',  'ii'): 3,  ('V',  'iii'): 2,  ('V',  'iv'): 1,
    ('Cr', 'ii'): 4,  ('Cr', 'iii'): 3,
    ('Mn', 'ii'): 5,  ('Mn', 'iii'): 4,
    ('Fe', 'ii'): 6,  ('Fe', 'iii'): 5,
    ('Co', 'ii'): 7,  ('Co', 'iii'): 6,
    ('Ni', 'ii'): 8,  ('Ni', 'iii'): 7,
    ('Cu', 'i'):  10, ('Cu', 'ii'): 9,
    ('Zn', 'ii'): 10,
}


def get_d_electrons(element: str, valence: str) -> int:
    """
    Return the number of d-electrons for an element/valence combination.

    Parameters
    ----------
    element : str
        Element symbol (e.g. 'Ni', 'Fe').
    valence : str
        Roman numeral valence string ('i', 'ii', 'iii', 'iv').

    Returns
    -------
    int : number of d-electrons (0-10).
    """
    key = (element.capitalize(), valence.lower())
    if key not in D_ELECTRON_COUNT:
        raise ValueError(f"Unknown element/valence combination: {element} {valence}")
    return D_ELECTRON_COUNT[key]


def get_l_edge_configs(element: str, valence: str) -> Tuple[str, str]:
    """
    Return standard L-edge electronic configurations for ground and excited states.

    Parameters
    ----------
    element : str
        Element symbol.
    valence : str
        Oxidation state (Roman numeral).

    Returns
    -------
    gs_config : str
        Ground state configuration string (e.g. '2P06 3D08').
    fs_config : str
        Final (core-hole) configuration string (e.g. '2P05 3D09').
    """
    n_d = get_d_electrons(element, valence)
    # Core: 1s2 2s2 2p6 3s2 3p6 (all filled below d shell)
    gs = f'2P06 3D{n_d:02d}'
    # L-edge: remove one 2p electron, add one 3d electron
    fs = f'2P05 3D{min(n_d + 1, 10):02d}'
    return gs, fs


def get_atomic_number(element: str) -> int:
    """Return atomic number for an element symbol."""
    elem = element.capitalize()
    if elem not in ATOMIC_NUMBERS:
        raise ValueError(f"Element '{element}' not in supported range (Ti-Zn).")
    return ATOMIC_NUMBERS[elem]


# Orbital quantum numbers
ORBITAL_QN = {
    's': 0, 'p': 1, 'd': 2, 'f': 3,
    'S': 0, 'P': 1, 'D': 2, 'F': 3,
}


def parse_config_string(config: str) -> Dict[str, float]:
    """
    Parse a configuration string like '2P06 3D08' into a dict.

    Returns {'2p': 6.0, '3d': 8.0} etc.
    """
    result = {}
    for tok in config.split():
        if len(tok) >= 3:
            n = int(tok[0])
            l = tok[1].lower()
            occ = float(tok[2:])
            result[f'{n}{l}'] = occ
    return result
