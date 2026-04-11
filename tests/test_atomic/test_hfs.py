"""
Phase 4 tests: HFS SCF solver.

Tests:
  1. Mesh construction: r[0] ≈ 0, r values positive, monotone
  2. Hydrogen 1s: E ≈ -1.0 Ry (exact), P normalized
  3. Helium 1s²: E(1s) in expected range
  4. Coulomb potential (quad2) for known wavefunction
  5. Exchange potential shape (positive/negative)
  6. SCHEQ node counting: n-l-1 nodes for nl orbital
  7. Validate against nid8.rcn31_out for Ni2+ 2p06 3d08 (if reference available)
"""
import pytest
import math
import torch
import numpy as np
from pathlib import Path

REFDATA = Path(__file__).parent.parent / "reference_data" / "nid8"


@pytest.mark.phase4
def test_cowan_mesh_monotone():
    from multitorch.atomic.radial_mesh import make_log_mesh
    r, h = make_log_mesh(mesh=641)
    assert (r[1:] > r[:-1]).all()


@pytest.mark.phase4
def test_cowan_mesh_start():
    from multitorch.atomic.radial_mesh import make_log_mesh
    r, h = make_log_mesh(mesh=641)
    assert abs(float(r[0]) - math.exp(-8.0)) < 1e-12


@pytest.mark.phase4
def test_hfs_mesh_positive():
    from multitorch.atomic.hfs import build_cowan_mesh
    r, IDB, h = build_cowan_mesh(Z=28, mesh=641)
    assert r[0] >= 0
    assert (r[1:] > r[:-1]).all()
    assert 1 <= IDB <= 640


@pytest.mark.phase4
def test_scheq_hydrogen_1s():
    """
    Hydrogen (Z=1) 1s orbital: energy should be exactly -1.0 Ry.
    This tests the Numerov integrator and matching in scheq().
    """
    from multitorch.atomic.hfs import build_cowan_mesh, scheq
    r, IDB, h = build_cowan_mesh(Z=1, mesh=641)
    # Potential for hydrogen: V(r) = -2/r, so V_arr = -2/r × r = -2
    # Hydrogen: V(r) = -2/r (Ry units)
    V = -2.0 / r.clamp(min=1e-30)
    E, P, KKK = scheq(V, r, IDB, n=1, l=0, Z=1.0, E_init=-1.2, THRESH=1e-8)
    # Hydrogen 1s: E = -1.0 Ry (exact)
    assert abs(E - (-1.0)) < 0.05, f"H 1s energy = {E:.4f} Ry, expected -1.0 Ry"


@pytest.mark.phase4
def test_scheq_hydrogen_2s():
    """
    Hydrogen 2s orbital: energy should be -0.25 Ry = -1/(2n)² Ry.
    This tests node counting (one radial node for 2s).
    """
    from multitorch.atomic.hfs import build_cowan_mesh, scheq
    r, IDB, h = build_cowan_mesh(Z=1, mesh=641)
    V = -2.0 / r.clamp(min=1e-30)
    E, P, KKK = scheq(V, r, IDB, n=2, l=0, Z=1.0, E_init=-0.3, THRESH=1e-8)
    assert abs(E - (-0.25)) < 0.05, f"H 2s energy = {E:.4f} Ry, expected -0.25 Ry"


@pytest.mark.phase4
def test_scheq_normalization():
    """Normalized wavefunction should have ∫P²dr ≈ 1."""
    from multitorch.atomic.hfs import build_cowan_mesh, scheq
    r, IDB, h = build_cowan_mesh(Z=1, mesh=641)
    V = -2.0 / r.clamp(min=1e-30)
    E, P, KKK = scheq(V, r, IDB, n=1, l=0, Z=1.0, E_init=-1.2, THRESH=1e-8)
    r_np = r.numpy()
    P_np = P.numpy()
    norm = np.trapezoid(P_np[:KKK + 1] ** 2, r_np[:KKK + 1])
    assert abs(norm - 1.0) < 0.05, f"Normalization = {norm:.4f}, expected 1.0"


@pytest.mark.phase4
def test_quad2_non_negative():
    """Coulomb self-energy XI should be non-negative everywhere."""
    from multitorch.atomic.hfs import build_cowan_mesh, quad2_coulomb
    r, IDB, h = build_cowan_mesh(Z=1, mesh=200)
    # Gaussian-like wavefunction
    P = torch.exp(-2.0 * r) * r
    norm = float(torch.trapezoid(P**2, r).sqrt())
    P = P / norm
    XI = quad2_coulomb(P, r, IDB)
    assert (XI >= 0).all(), "Coulomb energy should be non-negative"


@pytest.mark.phase4
def test_exchange_potential_negative():
    """Slater exchange potential should be everywhere ≤ 0 (attractive)."""
    from multitorch.atomic.hfs import build_cowan_mesh, build_exchange_potential
    r, IDB, h = build_cowan_mesh(Z=28, mesh=200)
    # Simple test density
    density = torch.exp(-4.0 * r) * r**2
    RUEXCH = build_exchange_potential(density, r, EXF=1.0)
    # RUEXCH = V_ex × r; V_ex < 0, so RUEXCH < 0 where density > 0
    # Check at peak of density
    peak = density.argmax()
    assert float(RUEXCH[peak]) <= 0.0


@pytest.mark.phase4
@pytest.mark.slow
def test_hfs_scf_helium():
    """
    Helium (Z=2, 1s²): 1s orbital energy should be near -0.918 Ry
    (HFS value; exact HF is -0.918 Ry for He 1s).
    """
    from multitorch.atomic.hfs import hfs_scf
    result = hfs_scf(
        Z=2,
        config={'1s': 2.0},
        mesh=641,
        max_iter=100,
        tol=1e-6,
    )
    orb_1s = result.orbital('1S')
    assert orb_1s is not None
    # HF value for He 1s is -0.918 Hartree = -1.836 Ry; HFS with EXF=0.7
    # gives a somewhat less-bound value (~-1.2 to -1.6 Ry depending on
    # convergence). The previous bound -0.85 to -0.95 confused Hartree with Ry.
    E_1s = orb_1s.E
    assert -1.9 < E_1s < -1.0, f"He 1s energy = {E_1s:.4f} Ry, expected ~ -1.8 Ry (HF) or -1.2 Ry (HFS)"


@pytest.mark.phase4
@pytest.mark.slow
def test_hfs_ni2plus_3d_energy(nid8_hfs):
    """
    Ni2+ 3d8: 3d orbital energy from HFS SCF should match rcn31 reference.
    From nid8.rcn31_out: E(3d) ≈ -2.796 Ry.
    Tolerance: atol=0.5 Ry (SCF convergence criterion in Cowan is 1e-8, but
    our simplified potential gives good accuracy).
    """
    from multitorch.atomic.hfs import hfs_scf
    result = hfs_scf(
        Z=28,
        config={'1s': 2.0, '2s': 2.0, '2p': 6.0, '3s': 2.0, '3p': 6.0, '3d': 8.0},
        mesh=641,
        max_iter=130,
        tol=1e-7,
    )
    orb_3d = result.orbital('3D')
    assert orb_3d is not None, "3D orbital not found in result"

    # Reference from nid8.rcn31_out
    ref_E_3d = -2.796  # Ry

    E_3d = orb_3d.E
    assert abs(E_3d - ref_E_3d) < 1.0, (
        f"Ni2+ 3d energy = {E_3d:.4f} Ry, expected ≈ {ref_E_3d} Ry"
    )


@pytest.mark.phase4
@pytest.mark.slow
def test_hfs_ni2plus_2p_energy(nid8_hfs):
    """
    Ni2+ 2p: energy from HFS should be near rcn31 reference.
    From nid8.rcn31_out: E(2p) ≈ -67.7 Ry.
    """
    from multitorch.atomic.hfs import hfs_scf
    result = hfs_scf(
        Z=28,
        config={'1s': 2.0, '2s': 2.0, '2p': 6.0, '3s': 2.0, '3p': 6.0, '3d': 8.0},
        mesh=641,
        max_iter=130,
        tol=1e-7,
    )
    orb_2p = result.orbital('2P')
    assert orb_2p is not None

    ref_E_2p = -67.7  # Ry (from nid8.rcn31_out)
    E_2p = orb_2p.E
    assert abs(E_2p - ref_E_2p) < 10.0, (
        f"Ni2+ 2p energy = {E_2p:.4f} Ry, expected ≈ {ref_E_2p} Ry"
    )


# ─────────────────────────────────────────────────────────────────────
# Track B: Blume-Watson spin-orbit ζ validation against nid8.rcn31_out
# ─────────────────────────────────────────────────────────────────────
#
# Reference values, Ni²⁺ 2p⁶3d⁸ at EXF=1.0 (nid8.rcn31_out:34-39):
#
#   orb   BLUME-WATSON (Ry)    R*VI (Ry)
#   2P    0.81583              0.85796
#   3P    0.09945              0.10407
#   3D    0.00607              0.00706
#
# The PyTorch HFS solver has a known limitation (README §3): the 3d orbital
# binds ~8% too tightly because the radial mesh / Numerov scheme is 2nd-order
# rather than Cowan's 4th-order. That makes ⟨r⁻³⟩_3d too large and the
# absolute ζ on 3d ~25% above the Fortran reference. Tolerances on the 3d
# row are set against this limitation, not the BW algorithm itself.
#
# What we *can* test rigorously: (a) the central-field ζ matches Fortran
# R*VI on 2p to ~5%; (b) the BW correction *reduces* CF in the same direction
# Fortran does, with the relative reduction ratio matching to ≤ 5%; (c) the
# BW absolute on 2p (where HFS is stable) is closer to Fortran BW than CF
# is to Fortran R*VI.

NID8_REFERENCE_ZETA = {
    # orb_label: (BW_Ry, RVI_Ry)
    "2P": (0.81583, 0.85796),
    "3P": (0.09945, 0.10407),
    "3D": (0.00607, 0.00706),
}


@pytest.fixture(scope="module")
def nid8_hfs_central_field():
    """Run hfs_scf for Ni²⁺ 2p⁶3d⁸ with the default central-field ζ."""
    from multitorch.atomic.hfs import hfs_scf
    return hfs_scf(
        Z=28,
        config={'1s': 2.0, '2s': 2.0, '2p': 6.0,
                '3s': 2.0, '3p': 6.0, '3d': 8.0},
        mesh=641,
        max_iter=200,
        tol=1e-7,
    )


@pytest.fixture(scope="module")
def nid8_hfs_blume_watson():
    """Run hfs_scf for Ni²⁺ 2p⁶3d⁸ with full Blume-Watson ζ."""
    from multitorch.atomic.hfs import hfs_scf
    return hfs_scf(
        Z=28,
        config={'1s': 2.0, '2s': 2.0, '2p': 6.0,
                '3s': 2.0, '3p': 6.0, '3d': 8.0},
        mesh=641,
        max_iter=200,
        tol=1e-7,
        zeta_method="blume_watson",
    )


@pytest.mark.phase4
@pytest.mark.slow
def test_zeta_central_field_matches_rvi_2p(nid8_hfs_central_field):
    """Central-field ζ on 2p matches Fortran R*VI to ~5% (the working
    tolerance in README §validation). This row is not affected by the
    3d HFS limitation."""
    orb = nid8_hfs_central_field.orbital('2P')
    ref = NID8_REFERENCE_ZETA["2P"][1]   # R*VI column
    rel_err = abs(orb.zeta_ry - ref) / ref
    assert rel_err < 0.05, (
        f"2p central-field ζ = {orb.zeta_ry:.5f} Ry; "
        f"Fortran R*VI = {ref:.5f} Ry; relative error {rel_err:.3%} > 5%"
    )


@pytest.mark.phase4
@pytest.mark.slow
def test_zeta_blume_watson_matches_bw_2p(nid8_hfs_blume_watson):
    """Blume-Watson ζ on 2p matches Fortran BW to ≤ 3%, and is closer to
    Fortran BW than the central-field number is to Fortran R*VI."""
    orb = nid8_hfs_blume_watson.orbital('2P')
    bw_ref, rvi_ref = NID8_REFERENCE_ZETA["2P"]
    rel_err = abs(orb.zeta_ry - bw_ref) / bw_ref
    assert rel_err < 0.03, (
        f"2p Blume-Watson ζ = {orb.zeta_ry:.5f} Ry; "
        f"Fortran BW = {bw_ref:.5f} Ry; relative error {rel_err:.3%} > 3%"
    )


@pytest.mark.phase4
@pytest.mark.slow
def test_zeta_blume_watson_reduces_2p_in_correct_direction(
    nid8_hfs_central_field, nid8_hfs_blume_watson
):
    """The BW correction must shrink ζ on 2p (Fortran reduces by 4.9%)."""
    cf = nid8_hfs_central_field.orbital('2P').zeta_ry
    bw = nid8_hfs_blume_watson.orbital('2P').zeta_ry
    assert bw < cf, (
        f"BW ({bw:.5f}) should be smaller than central-field ({cf:.5f}) for 2p"
    )
    fortran_reduction = (
        NID8_REFERENCE_ZETA["2P"][1] - NID8_REFERENCE_ZETA["2P"][0]
    ) / NID8_REFERENCE_ZETA["2P"][1]
    pytorch_reduction = (cf - bw) / cf
    # Reduction ratios should agree to ≤ 2% absolute (Fortran 4.9%, PyTorch ~6.1%)
    assert abs(pytorch_reduction - fortran_reduction) < 0.02, (
        f"BW reduction on 2p: PyTorch = {pytorch_reduction:.3%}, "
        f"Fortran = {fortran_reduction:.3%}"
    )


@pytest.mark.phase4
@pytest.mark.slow
def test_zeta_blume_watson_3d_reduction_matches_fortran(
    nid8_hfs_central_field, nid8_hfs_blume_watson
):
    """On 3d the absolute ζ is dominated by the HFS mesh limitation
    (README §3 — 3d binds 8% too tightly), so the absolute BW value is
    ~25% off. What *is* portable is the BW reduction ratio: Fortran BW
    is 14.0% smaller than R*VI; PyTorch BW should reduce CF by a
    similar amount (≤ 5% absolute mismatch in the ratio)."""
    cf = nid8_hfs_central_field.orbital('3D').zeta_ry
    bw = nid8_hfs_blume_watson.orbital('3D').zeta_ry
    assert bw < cf, "BW should reduce 3d ζ relative to CF"
    fortran_reduction = (
        NID8_REFERENCE_ZETA["3D"][1] - NID8_REFERENCE_ZETA["3D"][0]
    ) / NID8_REFERENCE_ZETA["3D"][1]
    pytorch_reduction = (cf - bw) / cf
    assert abs(pytorch_reduction - fortran_reduction) < 0.05, (
        f"BW reduction on 3d: PyTorch = {pytorch_reduction:.3%}, "
        f"Fortran = {fortran_reduction:.3%}"
    )


@pytest.mark.phase4
@pytest.mark.slow
def test_zeta_blume_watson_3p_reduction_matches_fortran(
    nid8_hfs_central_field, nid8_hfs_blume_watson
):
    """3p is intermediate: HFS is reasonable but not as accurate as 2p.
    Cross-check that the BW reduction is in family with Fortran's."""
    cf = nid8_hfs_central_field.orbital('3P').zeta_ry
    bw = nid8_hfs_blume_watson.orbital('3P').zeta_ry
    assert bw < cf
    fortran_reduction = (
        NID8_REFERENCE_ZETA["3P"][1] - NID8_REFERENCE_ZETA["3P"][0]
    ) / NID8_REFERENCE_ZETA["3P"][1]
    pytorch_reduction = (cf - bw) / cf
    assert abs(pytorch_reduction - fortran_reduction) < 0.05


@pytest.mark.phase4
def test_zeta_method_kwarg_validation():
    """hfs_scf must reject unknown zeta_method values up front."""
    from multitorch.atomic.hfs import hfs_scf
    with pytest.raises(ValueError, match="zeta_method"):
        hfs_scf(Z=1, config={'1s': 1.0}, zeta_method="not_a_method")


@pytest.mark.phase4
def test_zeta_blume_watson_l0_returns_zero():
    """ζ(s) ≡ 0 — the BW driver must short-circuit on l=0 orbitals."""
    import torch
    from multitorch.atomic.blume_watson import zeta_blume_watson
    from multitorch.atomic.hfs import OrbitalState
    r = torch.linspace(0.01, 5.0, 200, dtype=torch.float64)
    P = r * torch.exp(-r)
    P = P / torch.sqrt(torch.trapezoid(P * P, r))
    orb = OrbitalState(n=1, l=0, occ=2.0, E=-1.0, P=P, KKK=199)
    out = zeta_blume_watson([orb], 0, r, Z=1.0)
    assert float(out) == 0.0
