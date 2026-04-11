"""
multitorch.hamiltonian — Hamiltonian assembly and diagonalization (Phase 2).

Implements ttban_exact.f / ttban.f in PyTorch.

Phase 2 modules:
  assemble        — Full Hamiltonian assembly from .rme_rcg/.rme_rac/.ban
  crystal_field   — H_CF from Butler-Wybourne branch coefficients
  charge_transfer — Block-structured H with LMCT/MLCT off-diagonals
  diagonalize     — torch.linalg.eigh + optional Lanczos
  transitions     — Transition matrix T[g,f] from eigenvectors + RME
"""
