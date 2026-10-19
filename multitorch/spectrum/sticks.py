"""
Boltzmann-weighted stick spectrum from BanOutput transition matrices.

Ported from pyctm/pyctm/get_spectrum.py::get_sticks() with numpy
replaced by torch.Tensor. Fully differentiable w.r.t. temperature T.
"""
from __future__ import annotations

import math
import numbers
from typing import Optional, Tuple
import torch

from multitorch._constants import DTYPE, K_B_FLOAT
from multitorch.io.read_oba import BanOutput

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from multitorch.hamiltonian.assemble import BanResult


def _check_pool(T, max_gs) -> None:
    """Validate the Boltzmann pool: T >= 0; max_gs a positive integer; T = 0 only with max_gs = 1.

    ``T = 0`` applies no Boltzmann weighting, so with ``max_gs > 1`` every pooled
    level would count fully: not the T → 0 limit (nid8ct max_gs=10: 6.6x the
    intensity of T = 1e-3 K).
    """
    if isinstance(max_gs, bool) or not isinstance(max_gs, numbers.Integral) or max_gs < 1:
        raise ValueError(f"max_gs must be a positive integer, got {max_gs!r}")
    T = float(T)
    if not math.isfinite(T) or T < 0:
        raise ValueError(f"T must be a finite temperature >= 0 K, got {T!r}")
    if T == 0 and max_gs > 1:
        raise ValueError("T = 0 applies no Boltzmann weighting, so max_gs > 1 would weight every pooled level "
                         "equally; use max_gs=1 or a small T > 0")


# ─────────────────────────────────────────────────────────────
# Ground-level resolution for the Boltzmann pool (residual 15)
# ─────────────────────────────────────────────────────────────

#: Two ground energies belong to the same *level* when they lie within this
#: many eV of the level's lowest member. Same value, same anchored rule and
#: same meaning as ``degeneracy_tol`` in
#: :func:`multitorch.analysis.ground_state.analyze_ground_state`, so that
#: "the lowest ``max_gs`` levels" here and "the lowest ``n_levels`` levels"
#: there describe the same partition of the manifold.
#:
#: Why a resolution and not a noise test: a symmetry degeneracy comes out of
#: ``eigh`` scattered, and the scatter is not separable from physics by any
#: fixed number — measured on 2026-10-18 over all eight bundled fixtures in
#: both regimes and the from-scratch Oh/D4h paths with and without charge
#: transfer, the ground-manifold gaps populate the whole band from 1.7e-8 eV
#: up. What *is* clean is the gap around this value: the components of a
#: degenerate ground manifold scatter by at most 8.0e-7 eV (Fe³⁺ d5 from
#: scratch; the Fortran stores' 6-decimal RMEs give 1.7e-8 to 2.4e-8, and the
#: real Mn²⁺ ⁶A₁g cubic zero-field splitting is 2.8e-7), while the smallest
#: gap that separates two genuinely different ground levels anywhere in that
#: survey is 7.0e-5 eV (the Fortran ``nid8ct`` axial splitting, 0.56 cm⁻¹).
#: 1e-5 eV sits an order of magnitude clear of both ends, and every one of the
#: 24 surveyed cases gives the same level 0 at 1e-5 as at 1e-4.
#:
#: It is deliberately *not* a fraction of kT, although the plan suggested one:
#: a window that moves with temperature makes the pool *count* jump wherever
#: ``window`` crosses a real gap (1 % of kT reaches the ``nid8ct`` 7.0e-5 eV
#: gap at 81 K), which would put a discontinuity in T into every spectrum. The
#: thermal argument still bounds the cost: members of one level are weighted by
#: the level's energy, so a merge misweights them by at most ``window``/kT —
#: 0.8 % at 15 K for the window, and 0.06 % for the spreads actually measured.
LEVEL_WINDOW_EV = 1e-5


def _ground_level_pool(
    Eg_flat: torch.Tensor, max_gs: int, window: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a ground manifold into levels and keep the lowest ``max_gs`` of them.

    ``Eg_flat`` is every ground energy of every triad, in one 1-D tensor.
    Returns ``(keep, level_energy)``, both aligned with ``Eg_flat``: a boolean
    mask of the states in the pool, and the energy of each state's level (its
    lowest member), which is what the Boltzmann factor uses. Weighting by the
    level rather than by the state is what makes the pool exact — the dipole
    sum rule over a degenerate ground level is then an exact integer multiple
    of the hole count, where per-state weights would scatter it by ~1e-4.

    Grouping is anchored, not chained: a state joins the open level while it is
    within ``window`` of that level's *first* member, so a long ladder of
    closely spaced states cannot drift one level arbitrarily wide.
    ``window = 0`` recovers exact-float level counting (what this did until
    2026-10-18, residual 15) bit for bit.

    Differentiability: the grouping itself is a comparison and carries no
    gradient, but ``level_energy`` is gathered *from* ``Eg_flat``, so the
    gradient of a level's weight flows to the state that anchors it. For a
    true degeneracy every member moves together, so this agrees with a finite
    difference.
    """
    order = torch.argsort(Eg_flat.detach())
    vals = Eg_flat.detach()[order]

    # level id of each sorted position, and the sorted position anchoring it
    level_of_sorted = torch.zeros(len(vals), dtype=torch.long, device=Eg_flat.device)
    anchor_positions = [0]
    for i in range(1, len(vals)):
        if float(vals[i] - vals[anchor_positions[-1]]) > window:
            anchor_positions.append(i)
        level_of_sorted[i] = len(anchor_positions) - 1

    anchors = torch.as_tensor(anchor_positions, dtype=torch.long, device=Eg_flat.device)

    # scatter back to the original order
    level_of = torch.empty_like(level_of_sorted)
    level_of[order] = level_of_sorted

    keep = level_of < max_gs
    level_energy = Eg_flat[order[anchors[level_of]]]
    return keep, level_energy


def get_sticks(
    ban: BanOutput,
    T: float = 80.0,
    max_gs: int = 1,
    device: str = "cpu",
    level_window: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute a Boltzmann-weighted XAS stick spectrum from a BanOutput.

    Parameters
    ----------
    ban : BanOutput
        Parsed ttban output from read_ban_output().
    T : float
        Temperature in Kelvin (0 = no Boltzmann weighting; only with ``max_gs=1``).
    max_gs : int
        Keep only this many lowest ground state *levels* in the Boltzmann
        population pool. Higher values include more thermally accessible
        states; the rest are dropped before weighting. A level is a group of
        energies within ``level_window`` of its lowest member, so a
        symmetry-degenerate manifold counts once however its components come
        out of ``eigh`` — before 2026-10-18 this counted exactly equal floats
        and so kept 2 of the 6 components of an Oh d5 ground sextet (residual
        15).

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

        **Weight convention (decision D-1, docs/DEVELOPMENT_PLAN_2026-09.md).**
        Weights are ``exp((E_min − E_g)/kT)``, *not* divided by the partition
        function, as in pyctm. Total intensity therefore varies with the
        pool, and the spectrum has a kink (one-sided derivatives) wherever the
        lowest ground level changes branch, e.g. D4h at dt = ds = 0 or ζ = 0.
        Gradients are reliable only away from such exact crossings. Kept for
        pyctm/Fortran parity; to be revisited.
    device : str
        PyTorch device string ('cpu', 'cuda:0', etc.)
    level_window : float, optional
        Level resolution in eV; ``None`` uses :data:`LEVEL_WINDOW_EV` (1e-5),
        which is :func:`~multitorch.analysis.ground_state.analyze_ground_state`'s
        ``degeneracy_tol``. ``0.0`` restores exact-float level counting.

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

        # ttban_exact may output amplitudes (with sign, needs squaring) or
        # pre-squared intensities (all non-negative). Detect from data:
        # if any element is negative, it must be amplitudes.
        if (m < 0).any():
            m = m ** 2

        all_Eg.append(eg)
        all_Ef.append(ef)
        all_M.append(m)

    if not all_Eg:
        empty = torch.zeros(0, dtype=DTYPE, device=device)
        return empty, empty, torch.tensor(0.0, dtype=DTYPE, device=device)

    # Find global minimum ground state energy (in eV — see docstring note)
    _check_pool(T, max_gs)
    all_Eg_flat = torch.cat(all_Eg)
    Eg_min = all_Eg_flat.min()

    # Apply max_gs filter: keep only ground states within range
    # (find the max_gs distinct lowest energies)
    window = LEVEL_WINDOW_EV if level_window is None else float(level_window)
    if not (window >= 0.0):
        raise ValueError(f"level_window must be a non-negative number of eV, got {level_window!r}")
    keep_flat, level_E_flat = _ground_level_pool(all_Eg_flat, max_gs, window)
    sizes = [len(e) for e in all_Eg]
    keep_per = list(torch.split(keep_flat, sizes))
    level_E_per = list(torch.split(level_E_flat, sizes))

    # Assemble all transitions
    Etrans_list = []
    Mtrans_list = []

    # K_B_FLOAT is in eV/K; ttban Eg/Ef are both in eV (despite some upstream
    # docstrings labeling Eg as "Ry"). The subtraction below and the
    # Boltzmann factor are both unit-consistent in eV.
    for eg, ef, m, gs_mask, level_E in zip(all_Eg, all_Ef, all_M, keep_per, level_E_per):
        # Filter ground states: keep the states of the lowest max_gs levels
        eg = eg[gs_mask]
        m = m[gs_mask]
        level_E = level_E[gs_mask]

        if eg.numel() == 0:
            continue

        # Transition energies: Ef (eV, absolute incl. edge offset) - Eg (eV).
        # Matches pyctm/get_spectrum.py:77 exactly.
        Etrans = ef.unsqueeze(0) - eg.unsqueeze(1)  # (n_g, n_f), eV

        # Boltzmann population
        # weighted by the *level's* energy, not the state's (see
        # :func:`_ground_level_pool`)
        if T > 0:
            boltz = torch.exp((Eg_min - level_E) / (K_B_FLOAT * T))  # (n_g,)
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
    level_window: Optional[float] = None,
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
        Temperature in Kelvin (0 = no Boltzmann weighting; only with ``max_gs=1``).
    max_gs : int
        Number of lowest ground-state *levels* to keep (see :func:`get_sticks`).
        Boltzmann weights are unnormalised (pyctm convention, decision D-1):
        see :func:`get_sticks` for the consequences for gradients.
    device : str
        PyTorch device.
    level_window : float, optional
        Level resolution in eV; see :func:`get_sticks`.

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

    _check_pool(T, max_gs)
    all_Eg_flat = torch.cat(all_Eg)
    Eg_min = all_Eg_flat.min()

    window = LEVEL_WINDOW_EV if level_window is None else float(level_window)
    if not (window >= 0.0):
        raise ValueError(f"level_window must be a non-negative number of eV, got {level_window!r}")
    keep_flat, level_E_flat = _ground_level_pool(all_Eg_flat, max_gs, window)
    sizes = [len(e) for e in all_Eg]
    keep_per = list(torch.split(keep_flat, sizes))
    level_E_per = list(torch.split(level_E_flat, sizes))

    Etrans_list = []
    Mtrans_list = []
    for eg, ef, m, gs_mask, level_E in zip(all_Eg, all_Ef, all_M, keep_per, level_E_per):
        eg = eg[gs_mask]
        m = m[gs_mask]
        level_E = level_E[gs_mask]

        if eg.numel() == 0:
            continue

        Etrans = ef.unsqueeze(0) - eg.unsqueeze(1)

        if T > 0:
            boltz = torch.exp((Eg_min - level_E) / (K_B_FLOAT * T))
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
