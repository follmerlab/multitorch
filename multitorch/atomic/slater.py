"""
Slater-Condon parameter calculation from radial wavefunctions.

Given a set of radial wavefunctions P_nl(r) on the logarithmic mesh,
computes:
  Fk(nl, n'l') = ∫∫ r_<^k/r_>^(k+1) × P_nl²(r1) × P_n'l'²(r2) dr1 dr2
  Gk(nl, n'l') = ∫∫ r_<^k/r_>^(k+1) × P_nl(r1) × P_n'l'(r2) × P_nl(r2) × P_n'l'(r1) dr1 dr2

These are computed efficiently via the Y^k potential function:
  Y^k_b(r) = r^k × ∫_0^r P_b²(r')/r'^(k+1) dr'
            + r^(-(k+1)) × ∫_r^∞ P_b²(r') × r'^k dr'

which is the solution to the Poisson equation:
  d²Y^k/dr² - k(k+1)/r² × Y^k = -P_b²/r

Then:
  Fk(a,b) = ∫ P_a²(r) × Y^k_b(r) / r dr

All integrals are evaluated using torch.trapezoid on the logarithmic mesh
(with dr = r*h).

This module is differentiable w.r.t. the wavefunction P — enabling
gradient-based optimization of the Slater parameters.

Reference:
  Cowan (1981) The Theory of Atomic Structure and Spectra, Chap. 6.
"""
from __future__ import annotations
from typing import Tuple
import torch

from multitorch._constants import DTYPE


def _cum_trap(f: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """
    Cumulative trapezoidal integral ∫_{r[0]}^{r[i]} f(r') dr' on an arbitrary
    (possibly non-uniform) mesh. Output shape matches f; output[0] = 0.
    """
    dr = r[1:] - r[:-1]
    # Trapezoidal slab values: 0.5*(f[i]+f[i-1]) * (r[i] - r[i-1])
    slabs = 0.5 * (f[1:] + f[:-1]) * dr
    cum = torch.cat([torch.zeros(1, dtype=f.dtype, device=f.device),
                     torch.cumsum(slabs, dim=0)])
    return cum


def _trap(f: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Scalar trapezoidal integral on a non-uniform mesh."""
    return torch.trapezoid(f, r)


def compute_yk(
    Pb: torch.Tensor,
    r: torch.Tensor,
    k: int,
    h: float,                    # retained for API compatibility; not used
) -> torch.Tensor:
    """
    Compute the Y^k_b(r) potential function via two-pass cumulative integration.

    Y^k_b(r) = r^k × I_forward(r) + r^{-(k+1)} × I_backward(r)

    where:
      I_forward(r) = ∫_0^r P_b²(r')/r'^(k+1) dr'
      I_backward(r) = ∫_r^∞ P_b²(r') × r'^k dr'

    On the log mesh: dr = r × h, so:
      I_forward[i] = h × Σ_{j≤i} P_b[j]² / r[j]^(k+1) × r[j]  = h × Σ_{j≤i} P_b[j]² / r[j]^k
      I_backward[i] = h × Σ_{j≥i} P_b[j]² × r[j]^k × r[j]   = h × Σ_{j≥i} P_b[j]² × r[j]^(k+1)

    Parameters
    ----------
    Pb : torch.Tensor  shape (N,)
        Radial wavefunction P_b(r) (NOT P(r)/r, but the actual P function).
    r : torch.Tensor  shape (N,)
        Logarithmic radial mesh.
    k : int
        Order of the Slater integral (0, 2, 4 for d-d; 1, 3 for p-d).
    h : float
        Mesh step size.

    Returns
    -------
    Yk : torch.Tensor  shape (N,)
        Y^k potential in Rydbergs.
    """
    Pb2 = Pb ** 2   # shape (N,)

    # Y^k_b(r) = r^{-(k+1)} × ∫_0^r r'^k Pb²(r') dr'  +  r^k × ∫_r^∞ r'^{-(k+1)} Pb²(r') dr'
    # Mesh-adaptive trapezoidal quadrature on arbitrary (non-uniform) mesh r.
    f_fwd = Pb2 * (r ** k)              # integrand of ∫_0^r r'^k Pb² dr'
    I_fwd = _cum_trap(f_fwd, r)         # shape (N,)

    f_bwd = Pb2 / (r ** (k + 1))        # integrand of ∫_0^r r'^{-(k+1)} Pb² dr'
    I_bwd_cum = _cum_trap(f_bwd, r)
    I_bwd = I_bwd_cum[-1] - I_bwd_cum   # tail: ∫_r^∞

    Yk = r ** (-(k + 1)) * I_fwd + r ** k * I_bwd
    return Yk


def compute_fk(
    Pa: torch.Tensor,
    Pb: torch.Tensor,
    r: torch.Tensor,
    k: int,
    h: float,
) -> torch.Tensor:
    """
    Compute the Slater F^k(a,b) Coulomb integral.

    F^k(a,b) = ∫ P_a²(r) × Y^k_b(r) / r dr
             = h × Σ_i P_a²[i] × Y^k_b[i]  (on log mesh, dr/r = h)

    Returns value in Rydbergs.
    """
    Yk = compute_yk(Pb, r, k, h)
    Pa2 = Pa ** 2
    # F^k = ∫ Pa²(r) × Y^k(r) / r dr  (mesh-adaptive)
    return _trap(Pa2 * Yk / r, r)


def compute_gk(
    Pa: torch.Tensor,
    Pb: torch.Tensor,
    r: torch.Tensor,
    k: int,
    h: float,
) -> torch.Tensor:
    """
    Compute the Slater G^k(a,b) exchange integral.

    G^k(a,b) = ∫∫ r_<^k/r_>^(k+1) × P_a(r1) × P_b(r1) × P_a(r2) × P_b(r2) dr1 dr2
             = ∫ P_a(r) × P_b(r) × Y^k_{ab}(r) / r dr

    where Y^k_{ab} uses P_a × P_b instead of P_b².

    Returns value in Rydbergs.
    """
    # Y^k using the mixed density P_a × P_b
    Pab = Pa * Pb
    Yk = compute_yk_cross(Pab, r, k, h)
    # G^k = ∫ Pab(r) × Y^k_{ab}(r) / r dr  (mesh-adaptive)
    return _trap(Pab * Yk / r, r)


def compute_yk_cross(
    Pab: torch.Tensor,
    r: torch.Tensor,
    k: int,
    h: float,
) -> torch.Tensor:
    """
    Compute Y^k for the cross-density P_a × P_b (used for G^k).
    Identical to compute_yk but uses Pab instead of Pb².
    """
    # Same structure as compute_yk but with Pab instead of Pb²
    f_fwd = Pab * (r ** k)
    I_fwd = _cum_trap(f_fwd, r)

    f_bwd = Pab / (r ** (k + 1))
    I_bwd_cum = _cum_trap(f_bwd, r)
    I_bwd = I_bwd_cum[-1] - I_bwd_cum

    return r ** (-(k + 1)) * I_fwd + r ** k * I_bwd


def compute_slater_from_wavefunctions(
    pnl: dict,
    r: torch.Tensor,
    h: float,
    config: str = "2p6_3d8",
) -> dict:
    """
    Compute all Slater integrals relevant for an L-edge X-ray calculation.

    For Ni2+ (2p6 3d8 → 2p5 3d9):
      Ground state parameters: F2(dd), F4(dd), ζ(3d)
      Excited state parameters: F2(dd), F4(dd), F2(pd), G1(pd), G3(pd), ζ(2p), ζ(3d)

    Parameters
    ----------
    pnl : dict
        Wavefunction dictionary: {'2p': P_2p tensor, '3d': P_3d tensor, ...}
        Each P_nl has shape (N,) on the mesh r.
    r : torch.Tensor  shape (N,)
        Radial mesh.
    h : float
        Mesh step size.
    config : str
        Electronic configuration string (determines which integrals to compute).

    Returns
    -------
    params : dict
        Slater integrals in Rydbergs:
          {'F0dd': ..., 'F2dd': ..., 'F4dd': ...,
           'F0pd': ..., 'F2pd': ..., 'G1pd': ..., 'G3pd': ...,
           'R1': ...}
    """
    params = {}

    if '3d' in pnl:
        P3d = pnl['3d']
        # d-d Slater integrals (F^0 always = 0 by definition in Cowan)
        params['F0dd'] = torch.tensor(0.0, dtype=DTYPE)
        params['F2dd'] = compute_fk(P3d, P3d, r, 2, h)
        params['F4dd'] = compute_fk(P3d, P3d, r, 4, h)

    if '2p' in pnl and '3d' in pnl:
        P2p = pnl['2p']
        P3d = pnl['3d']
        # pd Slater integrals
        params['F0pd'] = torch.tensor(0.0, dtype=DTYPE)  # defined to be 0
        params['F2pd'] = compute_fk(P2p, P3d, r, 2, h)
        params['G1pd'] = compute_gk(P2p, P3d, r, 1, h)
        params['G3pd'] = compute_gk(P2p, P3d, r, 3, h)

        # Radial electric dipole integral R1 = ∫ P_2p(r) × P_3d(r) dr
        # (the dipole operator r in ⟨nl|r|n'l'⟩ cancels with P=rR convention)
        params['R1'] = _trap(P2p * P3d, r)

    return params
