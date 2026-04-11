"""
multitorch.io — parsers for ttmult/ttrcg/ttban file formats.

Used for:
  1. Loading pre-computed Fortran reference outputs into torch.Tensor
     for unit test validation.
  2. Bootstrapping the hamiltonian layer before the full angular momentum
     algebra (Phase 3) and atomic parameters (Phase 4) are implemented.
"""
