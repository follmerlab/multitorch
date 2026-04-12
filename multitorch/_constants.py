"""
Physical constants used throughout multitorch.

All values in SI-derived units unless noted. All tensors default to float64
to match Fortran real*8 precision.

Constants are stored as plain Python floats for device-agnostic use.
Torch tensor versions are available via the ``_t`` suffixed names for
backward compatibility (always on CPU). When running on GPU, prefer the
plain float versions — PyTorch auto-promotes floats in tensor operations.
"""
import torch

DTYPE = torch.float64

# ── Plain float constants (device-agnostic, preferred) ──

# Boltzmann constant in eV/K
K_B_FLOAT = 8.6173303e-05

# Rydberg to eV conversion
RY_TO_EV_FLOAT = 13.60569

# eV to Rydberg
EV_TO_RY_FLOAT = 1.0 / RY_TO_EV_FLOAT

# Electron mass * c^2 in eV (for relativistic corrections)
M_E_EV_FLOAT = 510998.950

# Speed of light in atomic units
C_AU_FLOAT = 137.035999084


# ── Torch tensor constants (CPU, backward compatibility) ──

k_B = torch.tensor(K_B_FLOAT, dtype=DTYPE)
RY_TO_EV = torch.tensor(RY_TO_EV_FLOAT, dtype=DTYPE)
EV_TO_RY = 1.0 / RY_TO_EV
M_E_EV = torch.tensor(M_E_EV_FLOAT, dtype=DTYPE)
C_AU = torch.tensor(C_AU_FLOAT, dtype=DTYPE)
