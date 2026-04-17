"""
Spectral broadening functions.

Vectorized pseudo-Voigt broadening (no Python for loops over sticks).
Two modes:
  'legacy'  — matches pyctm/get_spectrum.py exactly, including the known bug
              where the second term of the eta mixing parameter is missing
              its exponent: `- 0.47719*(FWHML/f)` instead of `- 0.47719*(FWHML/f)**2`
  'correct' — uses the Thompson et al. (1987) formula correctly

Reference:
  Thompson P., Cox D.E., Hastings J.B. (1987) J. Appl. Cryst. 20, 79.
  Ida T., Ando M., Toraya H. (2000) J. Appl. Cryst. 33, 1311.

Broadening parameters (fwhm_g, fwhm_l) can be either plain floats or
scalar ``torch.Tensor`` values.  When tensors are passed, all arithmetic
stays on the autograd tape so gradients w.r.t. beam width or lifetime
broadening are available.
"""
from __future__ import annotations
from typing import Optional, Union
import torch
import math

from multitorch._constants import DTYPE

# Type alias for parameters that can be float or scalar tensor.
_Scalar = Union[float, torch.Tensor]

# Precomputed constants
_SQRT_2_LN2 = math.sqrt(2 * math.log(2))   # ≈ 1.17741
_SQRT_2_PI = math.sqrt(2 * math.pi)         # ≈ 2.50663


def _to_float(v: _Scalar) -> float:
    """Extract a plain float (for comparisons/guards only)."""
    return float(v) if isinstance(v, torch.Tensor) else v


def pseudo_voigt(
    x: torch.Tensor,
    x0: torch.Tensor,
    amp: torch.Tensor,
    fwhm_g: _Scalar,
    fwhm_l: _Scalar,
    fwhm_l2: Optional[_Scalar] = None,
    med_energy: Optional[float] = None,
    mode: str = "legacy",
) -> torch.Tensor:
    """
    Vectorized pseudo-Voigt broadening of a stick spectrum.

    Computes a pseudo-Voigt profile for each stick and sums over all sticks.
    No Python for loops — all sticks processed simultaneously via broadcasting.

    Parameters
    ----------
    x : torch.Tensor  shape (N_grid,)
        Energy grid to evaluate spectrum on (eV).
    x0 : torch.Tensor  shape (N_sticks,)
        Stick center positions (eV).
    amp : torch.Tensor  shape (N_sticks,)
        Stick amplitudes (intensities).
    fwhm_g : float or torch.Tensor (scalar)
        Gaussian FWHM (beam width, eV).
    fwhm_l : float or torch.Tensor (scalar)
        Lorentzian FWHM for low-energy region (L3 lifetime, eV).
    fwhm_l2 : float, torch.Tensor, or None
        Lorentzian FWHM for high-energy region (L2 lifetime, eV).
        If None, uses fwhm_l for all sticks.
    med_energy : float or None
        Crossover energy between L3 and L2 broadening.
        If None, uses median of x.
    mode : str
        'legacy' (matches pyctm bug) or 'correct' (Thompson 1987).

    Returns
    -------
    y : torch.Tensor  shape (N_grid,)
        Broadened spectrum.
    """
    if fwhm_l2 is None:
        fwhm_l2 = fwhm_l
    if med_energy is None:
        med_energy = float(x.median())

    # Compute pseudo-Voigt parameters for each broadening region
    f, s, n = _pv_params(fwhm_g, fwhm_l, mode)
    f2, s2, n2 = _pv_params(fwhm_g, fwhm_l2, mode)

    # Split sticks into L3 and L2 regions
    low_mask = (x0 - med_energy) < 0

    y = torch.zeros_like(x, dtype=DTYPE)

    # Low-energy sticks (L3 broadening)
    if low_mask.any():
        y = y + _convolve_sticks(x, x0[low_mask], amp[low_mask], f, s, n)

    # High-energy sticks (L2 broadening)
    high_mask = ~low_mask
    if high_mask.any():
        y = y + _convolve_sticks(x, x0[high_mask], amp[high_mask], f2, s2, n2)

    return y


def _pv_params(fwhm_g: _Scalar, fwhm_l: _Scalar, mode: str):
    """
    Compute pseudo-Voigt shape parameters f (total FWHM), sigma, eta.

    Supports both plain float and scalar tensor inputs.  When tensors are
    used, the result stays on the autograd tape.

    Thompson et al. (1987) combined FWHM:
      f^5 = fG^5 + 2.69269*fG^4*fL + 2.42843*fG^3*fL^2
            + 4.47163*fG^2*fL^3 + 0.07842*fG*fL^4 + fL^5

    Mixing parameter eta (fraction Lorentzian):
      eta = 1.36603*(fL/f) - 0.47719*(fL/f)^2 + 0.11116*(fL/f)^3  [CORRECT]

    Legacy bug in pyctm/get_spectrum.py line 500:
      n = 1.36603*(fL/f) - 0.47719*(fL/f) + 0.11116*(fL/f)**3
      (missing **2 on second term → linear, not quadratic)
    """
    f = (fwhm_g**5
         + 2.69269 * fwhm_g**4 * fwhm_l
         + 2.42843 * fwhm_g**3 * fwhm_l**2
         + 4.47163 * fwhm_g**2 * fwhm_l**3
         + 0.07842 * fwhm_g * fwhm_l**4
         + fwhm_l**5) ** 0.2

    # FWHM → sigma conversion: σ = FWHM / (2√(2 ln 2))
    # Use the precomputed constant — works for both float and tensor f.
    s = f / (2 * _SQRT_2_LN2)

    ratio = fwhm_l / f
    if mode == "legacy":
        # Bug: second term is linear, not quadratic
        n = 1.36603 * ratio - 0.47719 * ratio + 0.11116 * ratio**3
    else:
        # Correct Thompson 1987 formula
        n = 1.36603 * ratio - 0.47719 * ratio**2 + 0.11116 * ratio**3

    return f, s, n


def _convolve_sticks(
    x: torch.Tensor,        # (N_grid,)
    x0: torch.Tensor,       # (N_sticks,)
    amp: torch.Tensor,      # (N_sticks,)
    f: _Scalar,
    s: _Scalar,
    n: _Scalar,
) -> torch.Tensor:
    """
    Sum pseudo-Voigt profiles for all sticks via broadcasting.

    No Python loops. Uses outer (x - x0) difference matrix.
    Shape:  (N_grid, 1) - (1, N_sticks) → (N_grid, N_sticks) → sum over sticks.
    """
    dx = x.unsqueeze(1) - x0.unsqueeze(0)   # (N_grid, N_sticks)

    # Lorentzian: 1/π * (HWHM / (dx² + HWHM²))
    hwhm = 0.5 * f
    if _to_float(hwhm) > 0:
        lor = (1.0 / math.pi) * hwhm / (dx**2 + hwhm**2)
    else:
        lor = torch.zeros_like(dx)

    # Gaussian: 1/(σ√2π) * exp(-dx²/(2σ²))
    if _to_float(s) > 0:
        gau = (1.0 / (s * _SQRT_2_PI)) * torch.exp(-0.5 * (dx / s)**2)
    else:
        # Delta function limit
        gau = torch.zeros_like(lor)

    # Pseudo-Voigt: η * Lorentzian + (1-η) * Gaussian
    profile = n * lor + (1.0 - n) * gau   # (N_grid, N_sticks)

    # Weight by stick amplitudes and sum over sticks
    return (profile * amp.unsqueeze(0)).sum(dim=1)   # (N_grid,)


def broaden_gaussian(E: torch.Tensor, D: torch.Tensor, fwhm: _Scalar) -> torch.Tensor:
    """
    Convolve a 2D spectrum matrix D with a Gaussian of given FWHM.

    Used in RIXS calculation to broaden along incident energy axis.
    Matches pyctm/get_spectrum.py::broaden() exactly.

    Parameters
    ----------
    E : torch.Tensor  shape (N,)
        Energy axis.
    D : torch.Tensor  shape (N, M)
        Spectrum to broaden along axis 0.
    fwhm : float or torch.Tensor
        Gaussian FWHM in eV. When a tensor with requires_grad=True,
        gradients flow through the broadening width.

    Returns
    -------
    Y : torch.Tensor  shape (N, M)
    """
    n, m = D.shape
    sigma = fwhm / (2 * _SQRT_2_LN2)
    n2 = n // 2
    E0 = E[n2]  # keep on tape when E requires grad
    G = _gauss_torch(E, sigma, E0)   # (N,)

    # Batch all columns in a single conv1d call (GPU-friendly, no Python loop)
    # D: (N, M) → (1, M, N) for grouped conv1d
    kernel = G.flip(0).unsqueeze(0).unsqueeze(0).expand(m, 1, -1)  # (M, 1, N)
    D_batch = D.T.unsqueeze(0)  # (1, M, N)
    Y_batch = torch.nn.functional.conv1d(
        D_batch, kernel, padding=n - 1, groups=m,
    )  # (1, M, 2N-1)
    Y = Y_batch.squeeze(0)[:, n2 : n2 + n].T  # (N, M)
    return Y


def _gauss_torch(x: torch.Tensor, sigma: _Scalar, x0: _Scalar) -> torch.Tensor:
    if not isinstance(sigma, torch.Tensor):
        sigma = torch.tensor(sigma, dtype=x.dtype, device=x.device)
    if not isinstance(x0, torch.Tensor):
        x0 = torch.tensor(x0, dtype=x.dtype, device=x.device)
    return (1.0 / (sigma * _SQRT_2_PI)) * torch.exp(
        -0.5 * ((x - x0) / sigma) ** 2
    )
