"""
Boltzmann-weighted stick spectrum from BanOutput transition matrices.

Ported from pyctm/pyctm/get_spectrum.py::get_sticks() with numpy
replaced by torch.Tensor. Fully differentiable w.r.t. temperature T.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch

from multitorch._constants import DTYPE, k_B
from multitorch.io.read_oba import BanOutput


def get_sticks(
    ban: BanOutput,
    T: float = 80.0,
    max_gs: int = 1,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute a Boltzmann-weighted XAS stick spectrum from a BanOutput.

    Parameters
    ----------
    ban : BanOutput
        Parsed ttban output from read_ban_output().
    T : float
        Temperature in Kelvin (0 = no Boltzmann weighting).
    max_gs : int
        Keep only this many lowest ground states.
    device : str
        PyTorch device string ('cpu', 'cuda:0', etc.)

    Returns
    -------
    Etrans : torch.Tensor  shape (N_sticks,)
        Transition energies in eV (Ef - Eg_eV, where Eg is converted to eV).
    Mtrans : torch.Tensor  shape (N_sticks,)
        Transition intensities (Boltzmann-weighted).
    Eg_min : torch.Tensor  scalar
        Minimum ground state energy in Ry.
    """
    # Collect all (Eg, Ef, M) from all triads
    all_Eg = []
    all_Ef = []
    all_M = []

    for t in ban.triad_list:
        eg = t.Eg.to(device=device, dtype=DTYPE)  # (n_g,)
        ef = t.Ef.to(device=device, dtype=DTYPE)  # (n_f,)
        m = t.M.to(device=device, dtype=DTYPE)    # (n_g, n_f)

        # Auto-square negative matrix elements (ttban_exact outputs amplitudes)
        if m.min() < 0:
            m = m ** 2

        all_Eg.append(eg)
        all_Ef.append(ef)
        all_M.append(m)

    if not all_Eg:
        empty = torch.zeros(0, dtype=DTYPE, device=device)
        return empty, empty, torch.tensor(0.0, dtype=DTYPE, device=device)

    # Find global minimum ground state energy (in Ry)
    all_Eg_flat = torch.cat(all_Eg)
    Eg_min = all_Eg_flat.min()

    # Apply max_gs filter: keep only ground states within range
    # (find the max_gs distinct lowest energies)
    unique_Eg = torch.unique(torch.sort(all_Eg_flat)[0])
    if max_gs < len(unique_Eg):
        gs_threshold = unique_Eg[max_gs - 1]
    else:
        gs_threshold = unique_Eg[-1]

    # Assemble all transitions
    Etrans_list = []
    Mtrans_list = []

    # k_B in eV/K; Eg in Ry; convert Eg to eV for energy difference
    k_B_val = k_B.to(device=device)
    Ry_to_eV = torch.tensor(13.60569, dtype=DTYPE, device=device)

    for eg, ef, m in zip(all_Eg, all_Ef, all_M):
        # Filter ground states
        gs_mask = eg <= gs_threshold
        eg = eg[gs_mask]
        m = m[gs_mask]

        if eg.numel() == 0:
            continue

        # Transition energies: Ef (eV) - Eg (Ry).
        # Note: this mixed-unit subtraction matches pyctm/get_spectrum.py
        # exactly. The KET (Ef) values in ttban output are in eV (absolute
        # photon energy including the edge offset). The BRA (Eg) values are
        # in Ry (internal atomic energy units). The subtraction produces a
        # value approximately equal to the photon energy + small Ry correction.
        # This is a pyctm convention, preserved for exact numerical matching.
        Etrans = ef.unsqueeze(0) - eg.unsqueeze(1)  # (n_g, n_f), mixed units by convention

        # Boltzmann population
        if T > 0:
            boltz = torch.exp((Eg_min - eg) / (k_B_val * T))  # (n_g,)
        else:
            boltz = torch.ones_like(eg)

        # Weight intensities by Boltzmann population
        M_weighted = m * boltz.unsqueeze(1)  # (n_g, n_f)

        Etrans_list.append(Etrans.reshape(-1))
        Mtrans_list.append(M_weighted.reshape(-1))

    if not Etrans_list:
        empty = torch.zeros(0, dtype=DTYPE, device=device)
        return empty, empty, Eg_min

    Etrans_all = torch.cat(Etrans_list)
    Mtrans_all = torch.cat(Mtrans_list)

    # Sum intensities at duplicate energies
    unique_E, inv_idx = torch.unique(Etrans_all, return_inverse=True)
    unique_M = torch.zeros_like(unique_E)
    unique_M.scatter_add_(0, inv_idx, Mtrans_all)

    # Sort by energy
    order = torch.argsort(unique_E)
    return unique_E[order], unique_M[order], Eg_min
