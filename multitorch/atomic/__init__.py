"""
multitorch.atomic — HFS SCF atomic orbital solver (Phase 4).

Implements rcn31.f (Hartree-Fock-Slater self-consistent field) and
rcn2.f (Slater-Condon parameter calculation) in PyTorch.

Phase 4 modules:
  radial_mesh  — logarithmic radial mesh (Cowan's r[i] = exp(-8 + (i-1)*h))
  hfs          — HFS SCF loop, Numerov integrator
  slater       — Fk/Gk via Yk potential + torch.trapezoid
  tables       — atomic data (Z, default electronic configurations)
"""
