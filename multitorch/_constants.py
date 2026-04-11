"""
Physical constants used throughout multitorch.

All values in SI-derived units unless noted. All tensors default to float64
to match Fortran real*8 precision.
"""
import torch

DTYPE = torch.float64

# Boltzmann constant in eV/K
k_B = torch.tensor(8.6173303e-05, dtype=DTYPE)

# Rydberg to eV conversion
RY_TO_EV = torch.tensor(13.60569, dtype=DTYPE)

# eV to Rydberg
EV_TO_RY = 1.0 / RY_TO_EV

# Electron mass * c^2 in eV (for relativistic corrections)
M_E_EV = torch.tensor(510998.950, dtype=DTYPE)

# Speed of light in atomic units
C_AU = torch.tensor(137.035999084, dtype=DTYPE)
