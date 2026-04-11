"""
High-level calculation API — the primary user interface for multitorch.

These functions wire together:
  atomic  → angular  → hamiltonian  → spectrum

Each function has two modes of operation:
  1. Full PyTorch pipeline (Phase 5 complete): all physics computed natively
  2. Fortran bootstrap mode: reads pre-computed reference data from .ban_out
     files (requires the ttmult Fortran binaries and pyttmult installed)

The API matches pyctm's function signatures for drop-in compatibility.

Usage:
    from multitorch import calcXAS
    x, y = calcXAS(element='Ni', valence='ii', sym='d4h', edge='l',
                   cf={'tendq': 1.0, 'ds': 0.0, 'dt': 0.01},
                   slater=0.8, soc=1.0, T=80)
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Union
import torch
import numpy as np

from multitorch._constants import DTYPE
from multitorch.spectrum.sticks import get_sticks
from multitorch.spectrum.broaden import pseudo_voigt


def calcXAS(
    element: str,
    valence: str,
    sym: str,
    edge: str,
    cf: dict,
    slater: float = 0.8,
    soc: float = 1.0,
    delta: Optional[dict] = None,
    u: Optional[list] = None,
    lmct: Optional[dict] = None,
    mlct: Optional[dict] = None,
    T: float = 80.0,
    beam_fwhm: float = 0.2,
    gamma1: float = 0.2,
    gamma2: float = 0.4,
    med_energy: float = 25.0,
    max_gs: int = 1,
    broaden_mode: str = "legacy",
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    nbins: int = 2000,
    return_sticks: bool = False,
    device: str = "cpu",
    ban_output_path: Optional[str] = None,
    **kwargs,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Calculate an X-ray absorption spectrum.

    Parameters
    ----------
    element : str
        Element symbol (e.g. 'Ni', 'Fe', 'Ti').
    valence : str
        Oxidation state ('i', 'ii', 'iii', 'iv').
    sym : str
        Crystal symmetry ('oh', 'd4h', 'c4h').
    edge : str
        X-ray edge ('l' for L-edge 2p→3d, 'k' for K-edge 1s→3p).
    cf : dict
        Crystal field parameters: {'tendq': float, 'ds': float, 'dt': float}.
    slater : float
        Slater integral reduction factor (0-1, default 0.8).
    soc : float
        Spin-orbit coupling reduction factor (0-1, default 1.0).
    delta : dict or None
        Charge transfer energies: {'lmct': float, 'mlct': float}.
    u : list or None
        Coulomb repulsion parameters.
    lmct : dict or None
        LMCT configuration mixing parameters.
    mlct : dict or None
        MLCT configuration mixing parameters.
    T : float
        Temperature in Kelvin (default 80 K).
    beam_fwhm : float
        Gaussian beam FWHM (eV).
    gamma1 : float
        L3 lifetime FWHM (eV).
    gamma2 : float
        L2 lifetime FWHM (eV).
    med_energy : float
        L3/L2 crossover energy (eV relative to sticks).
    max_gs : int
        Number of ground states to include.
    broaden_mode : str
        'legacy' or 'correct' pseudo-Voigt mode.
    xmin, xmax : float or None
        Energy range for output spectrum. Auto-detected if None.
    nbins : int
        Number of energy bins in output spectrum.
    return_sticks : bool
        If True, also return the stick spectrum (E, I).
    device : str
        PyTorch device ('cpu', 'cuda:0', etc.)
    ban_output_path : str or None
        Path to a pre-computed .ban_out file. If provided, skips the full
        PyTorch physics pipeline and reads the Fortran output directly.
        This is the bootstrap mode for early validation.

    Returns
    -------
    x : torch.Tensor  shape (nbins,)
        Energy axis (eV).
    y : torch.Tensor  shape (nbins,)
        Absorption intensity.
    sticks : torch.Tensor  shape (N, 2)  (only if return_sticks=True)
        Columns: [energy (eV), intensity].
    """
    if ban_output_path is not None:
        # Bootstrap mode: read pre-computed Fortran output
        return _calcXAS_from_ban(
            ban_output_path, T=T, beam_fwhm=beam_fwhm,
            gamma1=gamma1, gamma2=gamma2, med_energy=med_energy,
            max_gs=max_gs, broaden_mode=broaden_mode,
            xmin=xmin, xmax=xmax, nbins=nbins,
            return_sticks=return_sticks, device=device,
        )

    # Full PyTorch pipeline (Phase 5)
    # Step 1: Compute atomic parameters
    from multitorch.atomic.tables import get_atomic_number, get_l_edge_configs
    from multitorch.atomic.hfs import hfs_scf

    Z = get_atomic_number(element)
    gs_config, fs_config = get_l_edge_configs(element, valence)

    # NOTE: Full HFS SCF not yet implemented (Phase 4 complete infrastructure,
    # SCF loop pending). For now raise informative error.
    raise NotImplementedError(
        "Full PyTorch physics pipeline (HFS SCF → RME → Hamiltonian) not yet "
        "complete for Phase 5. Use ban_output_path to specify a pre-computed "
        ".ban_out file from the Fortran ttmult codes, or run pyttmult directly.\n\n"
        "Example:\n"
        "  x, y = calcXAS(..., ban_output_path='mycomplex.ban_out')"
    )


def _calcXAS_from_ban(
    ban_path: str,
    T: float = 80.0,
    beam_fwhm: float = 0.2,
    gamma1: float = 0.2,
    gamma2: float = 0.4,
    med_energy: float = 25.0,
    max_gs: int = 1,
    broaden_mode: str = "legacy",
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    nbins: int = 2000,
    return_sticks: bool = False,
    device: str = "cpu",
):
    """
    Compute XAS spectrum from a pre-computed .ban_out file.

    This is the primary path for Phase 5 validation — reads Fortran output
    and applies the PyTorch spectrum layer (Phase 1).
    """
    from multitorch.io.read_oba import read_ban_output

    ban = read_ban_output(ban_path)

    # Get Boltzmann-weighted sticks
    E_sticks, M_sticks, _ = get_sticks(ban, T=T, max_gs=max_gs, device=device)

    if E_sticks.numel() == 0:
        raise ValueError(f"No transitions found in {ban_path}")

    # Set energy range
    E_min = float(E_sticks.min())
    E_max = float(E_sticks.max())
    if xmin is None:
        xmin = E_min - 5.0
    if xmax is None:
        xmax = E_max + 5.0

    x = torch.linspace(xmin, xmax, nbins, dtype=DTYPE, device=device)

    # Apply pseudo-Voigt broadening
    med = 0.5 * (E_min + E_max)
    y = pseudo_voigt(
        x, E_sticks, M_sticks,
        fwhm_g=beam_fwhm, fwhm_l=gamma1, fwhm_l2=gamma2,
        med_energy=med, mode=broaden_mode,
    )

    if return_sticks:
        sticks = torch.stack([E_sticks, M_sticks], dim=1)
        return x, y, sticks
    return x, y


def calcXES(
    element: str,
    valence: str,
    sym: str,
    edge: str,
    cf: dict,
    slater: float = 0.8,
    soc: float = 1.0,
    ban_output_path: Optional[str] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate an X-ray emission spectrum.

    See calcXAS for parameter documentation. For XES, the ground/final
    state assignments are swapped (emission from a core-hole state).
    """
    if ban_output_path is not None:
        return _calcXAS_from_ban(ban_output_path, **kwargs)
    raise NotImplementedError("Full XES pipeline pending. Use ban_output_path.")


def calcRIXS(
    element: str,
    valence: str,
    sym: str,
    edge: str,
    cf: dict,
    Einc: Optional[torch.Tensor] = None,
    Efin: Optional[torch.Tensor] = None,
    ban_abs_path: Optional[str] = None,
    ban_ems_path: Optional[str] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate a RIXS (resonant inelastic X-ray scattering) plane.

    Returns (Einc_grid, Efin_grid, intensity_2D) arrays.
    Full Kramers-Heisenberg calculation available in multitorch.spectrum.rixs.
    """
    raise NotImplementedError("RIXS pipeline not yet complete. See multitorch.spectrum.rixs.")


def calcDOC(
    element: str,
    valence: str,
    sym: str,
    edge: str,
    cf: dict,
    **kwargs,
) -> dict:
    """
    Calculate the degree of covalency (DOC).

    Returns a dict with metal orbital character fractions.
    """
    raise NotImplementedError("DOC calculation not yet implemented.")
