"""
multitorch.spectrum — Spectral broadening and analysis (Phase 1).

Replaces pyctm/get_spectrum.py, rixs.py, background.py with
fully vectorized PyTorch implementations.

Phase 1 modules:
  sticks      — Boltzmann-weighted stick spectrum from transition matrices
  broaden     — Pseudo-Voigt broadening (legacy + correct modes)
  rixs        — Kramers-Heisenberg RIXS (fully broadcast, no Python loops)
  background  — Saturation correction, edge step, arctangent baseline
"""
