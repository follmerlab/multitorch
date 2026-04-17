"""
Background subtraction and saturation correction for XAS spectra.

Ported from pyctm/pyctm/background.py with numpy replaced by torch.Tensor.
"""
from __future__ import annotations
from typing import Optional
import math
import torch

_SQRT_2_LN2 = math.sqrt(2 * math.log(2))

from multitorch._constants import DTYPE


def add_background(
    E: torch.Tensor,
    I: torch.Tensor,
    edge_energy: float,
    step_height: float = 1.0,
    width: float = 0.5,
    linear_slope: float = 0.0,
    gaussian_amp: float = 0.0,
    gaussian_center: float = 0.0,
    gaussian_fwhm: float = 1.0,
) -> torch.Tensor:
    """
    Add a background (edge step + optional Gaussian + linear baseline) to a spectrum.

    Parameters
    ----------
    E : torch.Tensor  shape (N,)
        Energy axis (eV).
    I : torch.Tensor  shape (N,)
        Spectrum to modify.
    edge_energy : float
        Center of the arctangent edge step (eV).
    step_height : float
        Height of the edge step.
    width : float
        Width (arctangent scale) of the edge step (eV).
    linear_slope : float
        Linear baseline slope.
    gaussian_amp, gaussian_center, gaussian_fwhm : float
        Optional Gaussian pre-edge feature.

    Returns
    -------
    I_with_bg : torch.Tensor  shape (N,)
    """
    bg = _arctangent_step(E, edge_energy, step_height, width)
    if linear_slope != 0.0:
        bg = bg + linear_slope * (E - edge_energy)
    if gaussian_amp != 0.0:
        sigma = gaussian_fwhm / (2 * _SQRT_2_LN2)
        bg = bg + gaussian_amp * torch.exp(
            -0.5 * ((E - gaussian_center) / sigma) ** 2
        )
    return I + bg


def _arctangent_step(E: torch.Tensor, E0: float, height: float, width: float) -> torch.Tensor:
    """Double arctangent step function for XAS edge."""
    # Matches pyctm/background.py::bkg()
    pi = math.pi
    return height * (
        0.5 + (1.0 / pi) * torch.atan((E - E0) / width)
    )


def saturate(
    E: torch.Tensor,
    I: torch.Tensor,
    thickness: float,
    angle: float = 45.0,
    independent: bool = False,
) -> torch.Tensor:
    """
    Apply X-ray self-absorption saturation correction to a fluorescence-mode spectrum.

    Saturation reduces peaks: I_sat = (1 - exp(-μ * t)) / (μ * t) × I_unsaturated

    Parameters
    ----------
    E : torch.Tensor  shape (N,)
        Energy axis (eV).
    I : torch.Tensor  shape (N,)
        Spectrum to saturate.
    thickness : float
        Effective sample thickness (absorption lengths).
    angle : float
        Incidence angle (degrees). Only used if not independent.
    independent : bool
        If True, use angle-independent saturation.

    Returns
    -------
    I_sat : torch.Tensor  shape (N,)
    """
    if independent:
        mu_t = thickness * I
    else:
        theta = angle * math.pi / 180.0
        mu_t = thickness * I / math.sin(theta)

    # Avoid division by zero
    safe_mu_t = mu_t.clamp(min=1e-10)
    return (1.0 - torch.exp(-safe_mu_t)) / safe_mu_t * I
