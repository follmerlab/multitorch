"""
multitorch.angular — Angular momentum algebra (Phase 3).

Implements ttrcg.f (reduced matrix elements) and ttrac.c (symmetry
reduction) in PyTorch using Racah-Wigner algebra.

Phase 3 modules:
  wigner    — Wigner 3j, 6j, 9j symbols (Racah-Schwinger recurrence)
  cfp       — Coefficients of fractional parentage for d^n
  rme       — Reduced matrix element builder (Wigner-Eckart + CFP)
  symmetry  — O3 → Oh → D4h → C4h subduction coefficients
"""
