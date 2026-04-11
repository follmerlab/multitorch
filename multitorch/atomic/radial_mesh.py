"""
Logarithmic radial mesh for atomic orbital calculations.

Cowan's rcn31.f uses a logarithmic mesh:
  r[i] = exp(-8.0 + (i-1) * h),   i = 1, ..., MESH
  h ≈ 0.0125   (default step size)
  MESH = 641   (default for L-edge calculations)

This gives a mesh spanning from r_min = exp(-8.0) ≈ 0.000335 a.u. to
r_max = exp(-8.0 + 640 * 0.0125) = exp(0.0) = 1.0 a.u. for MESH=641.

For heavier atoms or more precision, MESH can be up to 1801 with IDB=641
marking the boundary for extended mesh calculations.

The Numerov integration for the Schrödinger equation uses the substitution:
  P(r) = u(r) where the radial wavefunction is R(r) = P(r)/r
  The equation becomes: d²P/dr² = [V(r) - E] P(r)
  On the log mesh with t = ln(r): d²P/dt² - dP/dt = [V(r) - E] r² P

References:
  Cowan (1981) The Theory of Atomic Structure and Spectra, Chap. 6.
"""
from __future__ import annotations
import math
from typing import Optional, Tuple
import torch

from multitorch._constants import DTYPE


def make_log_mesh(
    mesh: int = 641,
    h: Optional[float] = None,
    r_start: float = -8.0,
    device: str = "cpu",
) -> Tuple[torch.Tensor, float]:
    """
    Create Cowan's logarithmic radial mesh.

    Parameters
    ----------
    mesh : int
        Number of mesh points (641 for standard L-edge, up to 1801).
    h : float or None
        Step size in log space. If None, computed from mesh and r_start to
        end at r=1910.724 a.u. (Cowan's default RDB value).
    r_start : float
        Starting value in log space: r[0] = exp(r_start). Default -8.0.
    device : str

    Returns
    -------
    r : torch.Tensor  shape (mesh,)
        Radial coordinate in atomic units (Bohr).
    h : float
        Mesh step size used.
    """
    if h is None:
        # Cowan default: RDB = 1910.724 a.u. at mesh point IDB=641
        # r[mesh-1] = exp(r_start + (mesh-1)*h)
        # → h = (ln(RDB) - r_start) / (IDB - 1) ≈ 0.012500
        h = (math.log(1910.724) - r_start) / (641 - 1)

    # r[i] = exp(r_start + i * h) for i = 0, ..., mesh-1
    indices = torch.arange(mesh, dtype=DTYPE, device=device)
    r = torch.exp(torch.tensor(r_start, dtype=DTYPE, device=device) + indices * h)
    return r, h


def make_integration_weights(r: torch.Tensor, h: float) -> torch.Tensor:
    """
    Compute integration weights for the logarithmic mesh.

    On the log mesh with r[i] = exp(r_start + i*h):
      dr = r * h  (so the weight for trapezoidal integration is r * h * Δi)

    For the radial integral ∫f(r)dr ≈ h × Σ_i r[i] × f[r[i]]

    Parameters
    ----------
    r : torch.Tensor  shape (N,)
        Radial mesh points.
    h : float
        Mesh step size.

    Returns
    -------
    w : torch.Tensor  shape (N,)
        Integration weights (r * h for each mesh point).
    """
    return r * h


def numerov_step(
    y_prev: torch.Tensor,
    y_curr: torch.Tensor,
    f_prev: torch.Tensor,
    f_curr: torch.Tensor,
    f_next: torch.Tensor,
    h: float,
) -> torch.Tensor:
    """
    One step of the Numerov integration algorithm.

    Solves y'' = f(x) × y via:
      y_next = (2*y_curr*(1 - 5/12*h²*f_curr) - y_prev*(1 + 1/12*h²*f_prev))
               / (1 + 1/12*h²*f_next)

    This is accurate to O(h^6) and is the method used in rcn31.f for
    integrating the radial Schrödinger equation.

    Parameters
    ----------
    y_prev, y_curr : torch.Tensor
        Wavefunction at previous and current mesh points.
    f_prev, f_curr, f_next : torch.Tensor
        Coefficient function at previous, current, and next mesh points.
    h : float
        Mesh step size.

    Returns
    -------
    y_next : torch.Tensor
        Wavefunction at next mesh point.
    """
    h2 = h * h
    # Correct Numerov: y_{n+1}(1 - h²f_{n+1}/12) = 2y_n(1 + 5h²f_n/12) - y_{n-1}(1 - h²f_{n-1}/12)
    return (
        2.0 * y_curr * (1.0 + 5.0 / 12.0 * h2 * f_curr)
        - y_prev * (1.0 - 1.0 / 12.0 * h2 * f_prev)
    ) / (1.0 - 1.0 / 12.0 * h2 * f_next)
