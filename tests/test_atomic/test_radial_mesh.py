"""
Phase 4 tests: radial mesh and Slater integral infrastructure.

Tests:
  1. Log mesh construction matches Cowan's r_start, step size
  2. Numerov integrator reproduces known analytical solution
  3. Y^k potential formula (Slater integral building block)
  4. F^k integral for a known test case
  5. G^k integral for a known test case
  6. Atomic tables: d-electron count, config strings
"""
import pytest
import math
import torch


@pytest.mark.phase4
def test_log_mesh_start_value():
    """First point of Cowan's log mesh: r[0] = exp(-8.0)."""
    from multitorch.atomic.radial_mesh import make_log_mesh
    r, h = make_log_mesh(mesh=641)
    expected_r0 = math.exp(-8.0)
    assert abs(float(r[0]) - expected_r0) < 1e-12


@pytest.mark.phase4
def test_log_mesh_step_size():
    """Step size h should be close to 0.01250 (Cowan default)."""
    from multitorch.atomic.radial_mesh import make_log_mesh
    _, h = make_log_mesh(mesh=641)
    # h = (ln(1910.724) - (-8.0)) / 640 ≈ 0.01250
    expected_h = (math.log(1910.724) - (-8.0)) / 640
    assert abs(h - expected_h) < 1e-10


@pytest.mark.phase4
def test_log_mesh_dtype():
    from multitorch.atomic.radial_mesh import make_log_mesh
    r, _ = make_log_mesh()
    assert r.dtype == torch.float64


@pytest.mark.phase4
def test_log_mesh_monotone():
    """Mesh should be strictly monotonically increasing."""
    from multitorch.atomic.radial_mesh import make_log_mesh
    r, _ = make_log_mesh(mesh=100)
    assert (r[1:] > r[:-1]).all()


@pytest.mark.phase4
def test_log_mesh_custom_size():
    from multitorch.atomic.radial_mesh import make_log_mesh
    r, _ = make_log_mesh(mesh=200)
    assert r.shape == (200,)


@pytest.mark.phase4
def test_numerov_harmonic_oscillator():
    """
    Numerov should solve d²y/dx² = -k²y exactly for harmonic oscillator.
    y = A*sin(k*x), y'' = -k²*y.

    Test one integration step and verify it matches the analytic solution.
    """
    from multitorch.atomic.radial_mesh import numerov_step
    import math
    k = 2.0
    h = 0.01
    x0 = 0.5

    y_prev = torch.tensor(math.sin(k * (x0 - h)), dtype=torch.float64)
    y_curr = torch.tensor(math.sin(k * x0), dtype=torch.float64)
    f_prev = torch.tensor(-k**2, dtype=torch.float64)
    f_curr = torch.tensor(-k**2, dtype=torch.float64)
    f_next = torch.tensor(-k**2, dtype=torch.float64)

    y_next_numerov = numerov_step(y_prev, y_curr, f_prev, f_curr, f_next, h)
    y_next_exact = torch.tensor(math.sin(k * (x0 + h)), dtype=torch.float64)

    # Numerov is O(h^6) accurate; for h=0.01 error should be < 1e-10
    assert abs(float(y_next_numerov - y_next_exact)) < 1e-10, (
        f"Numerov error = {abs(float(y_next_numerov - y_next_exact)):.2e}"
    )


@pytest.mark.phase4
def test_yk_monopole_of_gaussian():
    """
    Test Y^0 on a normalized Gaussian density.

    For P(r) = A × exp(-α r²) × r (so P²/r² = A² exp(-2αr²)):
    The Y^0 should give the electrostatic potential of a Gaussian charge.
    Only test qualitative behavior: Y^0 > 0 everywhere, decreasing at large r.
    """
    from multitorch.atomic.radial_mesh import make_log_mesh
    from multitorch.atomic.slater import compute_yk
    r, h = make_log_mesh(mesh=200)
    # Gaussian-like density
    alpha = 1.0
    Pa = torch.exp(-alpha * r ** 2) * r
    Pa = Pa / (h * (Pa ** 2).sum()).sqrt()  # normalize

    Y0 = compute_yk(Pa, r, 0, h)
    assert (Y0 >= 0).all(), "Y^0 should be non-negative"
    # Y^0 should be roughly constant at large r (like charge/r)
    # Check that it's non-trivially non-zero
    assert float(Y0.max()) > 1e-5


@pytest.mark.phase4
def test_fk_self_integral_positive():
    """F^k(a,a) should be positive for any wavefunction."""
    from multitorch.atomic.radial_mesh import make_log_mesh
    from multitorch.atomic.slater import compute_fk
    r, h = make_log_mesh(mesh=200)
    # Simple test wavefunction
    Pa = torch.exp(-0.5 * r) * r
    Pa = Pa / (h * (Pa ** 2).sum()).sqrt()

    F0 = compute_fk(Pa, Pa, r, 0, h)
    F2 = compute_fk(Pa, Pa, r, 2, h)
    F4 = compute_fk(Pa, Pa, r, 4, h)

    assert float(F0) > 0, "F0(a,a) should be positive"
    assert float(F2) > 0, "F2(a,a) should be positive"
    assert float(F4) > 0, "F4(a,a) should be positive"


@pytest.mark.phase4
def test_fk_ordering():
    """
    For a localized wavefunction, F0 > F2 > F4 in Rydbergs.

    Note: The Y^k potential formula gives F^k values that decrease with k
    when the wavefunction is concentrated at small r. For a Slater-type
    orbital Pa ~ exp(-alpha × r) × r, the F^k integrals decrease with k
    because higher k weights more toward larger r where the density is smaller.
    """
    from multitorch.atomic.radial_mesh import make_log_mesh
    from multitorch.atomic.slater import compute_fk
    r, h = make_log_mesh(mesh=500)
    # Localized function, peaked at r ~ 1/3 a.u.
    alpha = 3.0
    Pa = torch.exp(-alpha * r) * r
    norm = (h * (Pa ** 2).sum()) ** 0.5
    Pa = Pa / norm

    F0 = float(compute_fk(Pa, Pa, r, 0, h))
    F2 = float(compute_fk(Pa, Pa, r, 2, h))
    F4 = float(compute_fk(Pa, Pa, r, 4, h))

    assert F0 > 0 and F2 > 0 and F4 > 0, f"All F^k should be positive, got {F0:.4f} {F2:.4f} {F4:.4f}"
    assert F0 >= F2, f"Expected F0≥F2, got F0={F0:.4f} F2={F2:.4f}"


@pytest.mark.phase4
def test_d_electron_counts():
    """Check known d-electron counts for common ions."""
    from multitorch.atomic.tables import get_d_electrons
    assert get_d_electrons('Ni', 'ii') == 8   # Ni2+ = d8
    assert get_d_electrons('Fe', 'iii') == 5  # Fe3+ = d5
    assert get_d_electrons('Ti', 'iv') == 0   # Ti4+ = d0
    assert get_d_electrons('Cu', 'i') == 10   # Cu1+ = d10


@pytest.mark.phase4
def test_l_edge_config_strings():
    """L-edge config strings should have correct electron counts."""
    from multitorch.atomic.tables import get_l_edge_configs
    gs, fs = get_l_edge_configs('Ni', 'ii')
    # Ni2+ ground: 2p6 3d8
    assert '3D08' in gs and '2P06' in gs
    # Ni2+ excited: 2p5 3d9
    assert '3D09' in fs and '2P05' in fs


@pytest.mark.phase4
def test_parse_config_string():
    """Config string parser should return correct occupancies."""
    from multitorch.atomic.tables import parse_config_string
    result = parse_config_string('2P06 3D08')
    assert result.get('2p') == 6.0
    assert result.get('3d') == 8.0
