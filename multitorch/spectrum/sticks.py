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

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from multitorch.hamiltonian.assemble import BanResult


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
        Keep only this many lowest *distinct* ground state energies in the
        Boltzmann population pool. Higher values include more thermally
        accessible states; the rest are dropped before weighting.

        **Important — the temperature trap.** With ``max_gs=1`` (the
        default, matching pyctm for byte-exact reproducibility) only one
        energy is in the pool, so the Boltzmann weight is trivially 1.0
        and the spectrum is **T-independent regardless of the value of T**.
        To see any thermal effect you must pass ``max_gs >= 2``, and the
        next-up state must actually be within a few kT of the lowest
        (kT ≈ 0.026 eV at 300 K, ≈ 0.43 eV at 5000 K). For typical 3d
        L-edge fixtures with d-d splittings of ~1 eV the effect is
        invisible until ~5000 K with ``max_gs=1``; passing
        ``max_gs=10, T=300`` is the recommended way to enable physical
        thermal redistribution at room temperature for systems with a
        low-lying multiplet ladder.
    device : str
        PyTorch device string ('cpu', 'cuda:0', etc.)

    Returns
    -------
    Etrans : torch.Tensor  shape (N_sticks,)
        Transition energies in eV (Ef - Eg).
    Mtrans : torch.Tensor  shape (N_sticks,)
        Transition intensities (Boltzmann-weighted).
    Eg_min : torch.Tensor  scalar
        Minimum ground state energy in eV (despite the historical "Ry"
        labels in some upstream parser docstrings, ttban .ban_out / .oba
        ground state energies are in eV — d-d crystal-field splittings
        are O(1 eV), not O(1 Ry).)
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

    # Find global minimum ground state energy (in eV — see docstring note)
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

    # k_B is in eV/K; ttban Eg/Ef are both in eV (despite some upstream
    # docstrings labeling Eg as "Ry"). The subtraction below and the
    # Boltzmann factor are both unit-consistent in eV.
    k_B_val = k_B.to(device=device)

    for eg, ef, m in zip(all_Eg, all_Ef, all_M):
        # Filter ground states
        gs_mask = eg <= gs_threshold
        eg = eg[gs_mask]
        m = m[gs_mask]

        if eg.numel() == 0:
            continue

        # Transition energies: Ef (eV, absolute incl. edge offset) - Eg (eV).
        # Matches pyctm/get_spectrum.py:77 exactly.
        Etrans = ef.unsqueeze(0) - eg.unsqueeze(1)  # (n_g, n_f), eV

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


def get_sticks_from_banresult(
    result: 'BanResult',
    T: float = 80.0,
    max_gs: int = 1,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute a Boltzmann-weighted XAS stick spectrum from a BanResult.

    This is the Phase 5 analog of :func:`get_sticks`.  The input is a
    :class:`~multitorch.hamiltonian.assemble.BanResult` (from the in-memory
    assembler) rather than a :class:`~multitorch.io.read_oba.BanOutput`
    (from a ``.ban_out`` file).

    The key difference: ``TriadResult.T`` contains transition *amplitudes*
    in the eigenbasis (real-valued, possibly negative).  Intensities are
    ``T ** 2`` (the absolute square).

    Parameters
    ----------
    result : BanResult
        Output of ``assemble_and_diagonalize_in_memory``.
    T : float
        Temperature in Kelvin (0 = no Boltzmann weighting).
    max_gs : int
        Number of lowest distinct ground-state energies to keep.
    device : str
        PyTorch device.

    Returns
    -------
    Etrans, Mtrans, Eg_min : same as :func:`get_sticks`.
    """
    all_Eg = []
    all_Ef = []
    all_M = []

    for t in result.triads:
        eg = t.Eg.to(device=device, dtype=DTYPE)
        ef = t.Ef.to(device=device, dtype=DTYPE)
        m = t.T.to(device=device, dtype=DTYPE) ** 2  # amplitude → intensity

        all_Eg.append(eg)
        all_Ef.append(ef)
        all_M.append(m)

    if not all_Eg:
        empty = torch.zeros(0, dtype=DTYPE, device=device)
        return empty, empty, torch.tensor(0.0, dtype=DTYPE, device=device)

    all_Eg_flat = torch.cat(all_Eg)
    Eg_min = all_Eg_flat.min()

    unique_Eg = torch.unique(torch.sort(all_Eg_flat)[0])
    if max_gs < len(unique_Eg):
        gs_threshold = unique_Eg[max_gs - 1]
    else:
        gs_threshold = unique_Eg[-1]

    Etrans_list = []
    Mtrans_list = []
    k_B_val = k_B.to(device=device)

    for eg, ef, m in zip(all_Eg, all_Ef, all_M):
        gs_mask = eg <= gs_threshold
        eg = eg[gs_mask]
        m = m[gs_mask]

        if eg.numel() == 0:
            continue

        Etrans = ef.unsqueeze(0) - eg.unsqueeze(1)

        if T > 0:
            boltz = torch.exp((Eg_min - eg) / (k_B_val * T))
        else:
            boltz = torch.ones_like(eg)

        M_weighted = m * boltz.unsqueeze(1)

        Etrans_list.append(Etrans.reshape(-1))
        Mtrans_list.append(M_weighted.reshape(-1))

    if not Etrans_list:
        empty = torch.zeros(0, dtype=DTYPE, device=device)
        return empty, empty, Eg_min

    Etrans_all = torch.cat(Etrans_list)
    Mtrans_all = torch.cat(Mtrans_list)

    # Skip deduplication (unlike get_sticks) to preserve autograd flow.
    # Broadening is linear: summing duplicate sticks before broadening
    # gives the same result as broadening individually and summing.
    order = torch.argsort(Etrans_all)
    return Etrans_all[order], Mtrans_all[order], Eg_min
