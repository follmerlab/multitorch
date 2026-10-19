"""
Phase 1 tests: verify the Boltzmann-weighted stick spectrum generation.

Given nid8ct.ban_out reference data, verify that get_sticks() produces
energies and intensities consistent with the Fortran output.

Key reference values from nid8ct.ban_out (triad 0+, 1-, 1-):
  Ground state: Eg = -2.19045 eV (relative to config average)
  First final state: Ef = 852.30042 eV (absolute, edge-shifted)
  First intensity: M[0,0] = 0.043971
  Total row intensity: 0.27102
"""
import pytest
import torch
from pathlib import Path

REFDATA = Path(__file__).parent.parent / "reference_data"


@pytest.mark.phase1
def test_get_sticks_runs_without_error(nid8ct_ban):
    from multitorch.spectrum.sticks import get_sticks
    Etrans, Mtrans, Eg_min = get_sticks(nid8ct_ban, T=0)
    assert Etrans.numel() > 0
    assert Mtrans.numel() > 0


@pytest.mark.phase1
def test_get_sticks_dtype(nid8ct_ban):
    from multitorch.spectrum.sticks import get_sticks
    Etrans, Mtrans, _ = get_sticks(nid8ct_ban, T=0)
    assert Etrans.dtype == torch.float64
    assert Mtrans.dtype == torch.float64


@pytest.mark.phase1
def test_get_sticks_no_negative_intensities_at_T0(nid8ct_ban):
    """At T=0 with pre-squared intensities, all M should be >= 0."""
    from multitorch.spectrum.sticks import get_sticks
    _, Mtrans, _ = get_sticks(nid8ct_ban, T=0)
    assert (Mtrans >= -1e-10).all(), "Negative intensities found"


@pytest.mark.phase1
def test_get_sticks_energies_in_xas_range(nid8ct_ban):
    """
    Transition energies should be in the Ni L-edge range.

    Etrans = Ef - Eg (both eV). With Ef ≈ 852 eV (absolute) and
    Eg ≈ -2.19 eV (relative offset), Etrans ≈ 854 eV. Range extended to
    account for all charge-transfer configurations (CT states extend
    above 876 eV).
    """
    from multitorch.spectrum.sticks import get_sticks
    Etrans, _, _ = get_sticks(nid8ct_ban, T=0)
    assert float(Etrans.min()) > 845, "Minimum energy too low for Ni L-edge"
    assert float(Etrans.max()) < 930, "Maximum energy unreasonably high"


@pytest.mark.phase1
def test_get_sticks_first_transition_energy(nid8ct_ban):
    """
    Etrans = Ef - Eg, both in eV. For nid8ct CT calculation, max_gs=1
    selects the globally lowest ground state (in the 1+ symmetry sector at
    ~-3.45 eV). First transition: 852.18597 - (-3.45060) ≈ 855.64 eV.
    """
    from multitorch.spectrum.sticks import get_sticks
    Etrans, _, _ = get_sticks(nid8ct_ban, T=0, max_gs=1)
    min_E = float(Etrans.min())
    # Should be near 855-856 eV region (from the J=1+ ground state sector)
    assert abs(min_E - 855.5) < 1.5, f"First stick at {min_E:.2f} eV, expected ~855.5 eV"


@pytest.mark.phase1
def test_get_sticks_boltzmann_effect(nid8ct_ban):
    """Boltzmann weighting at T=300K should not change total intensity dramatically."""
    from multitorch.spectrum.sticks import get_sticks
    _, M0, _ = get_sticks(nid8ct_ban, T=0)
    _, M300, _ = get_sticks(nid8ct_ban, T=300)
    # Total intensity ratio should be in [0.5, 2.0]
    ratio = M300.sum() / (M0.sum() + 1e-10)
    assert 0.5 < float(ratio) < 2.0, f"T=300K / T=0 ratio = {ratio:.3f}, unexpected"


@pytest.mark.phase1
def test_get_sticks_max_gs_one_is_temperature_independent(nid8ct_ban):
    """
    Regression: with max_gs=1 (the default, matching pyctm), the spectrum
    must be bit-identical across temperatures because only one energy is in
    the population pool — the Boltzmann weight is trivially 1.0.

    This is a known design trap of the pyctm convention. If a user wants
    visible thermal redistribution they MUST pass max_gs >= 2. Documented
    in get_sticks() docstring.
    """
    from multitorch.spectrum.sticks import get_sticks
    _, M_low, _ = get_sticks(nid8ct_ban, T=10, max_gs=1)
    _, M_hi, _ = get_sticks(nid8ct_ban, T=5000, max_gs=1)
    assert torch.allclose(M_low, M_hi, atol=1e-12), (
        "max_gs=1 spectrum should be T-independent (only one state in pool); "
        "if this fails, check the Boltzmann normalization."
    )


@pytest.mark.phase1
def test_get_sticks_thermal_redistribution_with_max_gs_pool(nid8ct_ban):
    """
    With max_gs >= 2 the integral must actually move with temperature,
    proving the Boltzmann weighting is wired up correctly. We use max_gs=10
    so the entire low-lying multiplet ladder is in the pool, and probe at
    T=10K vs T=5000K — for nid8ct the next-up multiplet is 1.27 eV above
    the ground state, so a 5000K (kT≈0.43 eV) probe is required to see a
    significant shift.
    """
    from multitorch.spectrum.sticks import get_sticks
    _, M10, _ = get_sticks(nid8ct_ban, T=10, max_gs=10)
    _, M5000, _ = get_sticks(nid8ct_ban, T=5000, max_gs=10)
    int_low = float(M10.sum())
    int_high = float(M5000.sum())
    assert abs(int_high - int_low) / int_low > 0.05, (
        f"Thermal redistribution invisible: I(10K)={int_low:.4f}, "
        f"I(5000K)={int_high:.4f} — Boltzmann weighting may be broken."
    )


def test_boltzmann_weights_are_unnormalised_pyctm_convention():
    """Decision D-1 (docs/DEVELOPMENT_PLAN_2026-09.md): weight = exp((E_min − E_g)/kT), no 1/Z.

    Analytic oracle on a two-level ground manifold. This test is meant to fail
    if the convention is changed; update D-1 and the parity tests together.
    """
    import math
    from types import SimpleNamespace
    from multitorch._constants import DTYPE, K_B_FLOAT
    from multitorch.spectrum.sticks import get_sticks_from_banresult

    gap, T = 0.01, 300.0
    triad = SimpleNamespace(
        Eg=torch.tensor([0.0, gap], dtype=DTYPE),
        Ef=torch.tensor([10.0], dtype=DTYPE),
        T=torch.tensor([[1.0], [1.0]], dtype=DTYPE),
    )
    E, M, _ = get_sticks_from_banresult(SimpleNamespace(triads=[triad]), T=T, max_gs=2)
    lower, upper = M[(E - 10.0).abs().argmin()], M[(E - (10.0 - gap)).abs().argmin()]
    assert lower.item() == 1.0                     # lowest level: weight exactly 1
    assert upper.item() == pytest.approx(math.exp(-gap / (K_B_FLOAT * T)), rel=1e-12)
    assert M.sum().item() > 1.0                    # not normalised by Z


# ─────────────────────────────────────────────────────────────
# Ground-level resolution for the Boltzmann pool (residual 15)
# ─────────────────────────────────────────────────────────────

def _one_triad(energies, n_final=1):
    """A BanResult-shaped stand-in with unit amplitude from every ground state."""
    from types import SimpleNamespace
    from multitorch._constants import DTYPE
    eg = torch.tensor(energies, dtype=DTYPE)
    return SimpleNamespace(triads=[SimpleNamespace(
        Eg=eg,
        Ef=torch.full((n_final,), 10.0, dtype=DTYPE),
        T=torch.ones(len(eg), n_final, dtype=DTYPE),
    )])


def test_level_pool_merges_a_scattered_degeneracy():
    """Six components of one level, scattered the way eigh scatters them.

    Residual 15: exact-float counting made these six states six levels, so
    ``max_gs=1`` kept one and the dipole sum rule came out at 1/6 of the
    ground degeneracy. The measured scatter of a real degenerate manifold is
    1.7e-8 eV (Fortran stores) to 8.0e-7 eV (Fe3+ d5 from scratch); 3e-7 here.
    """
    from multitorch.spectrum.sticks import get_sticks_from_banresult

    scattered = [0.0, 1e-7, 1.5e-7, 2e-7, 2.5e-7, 3e-7]
    _, M, _ = get_sticks_from_banresult(_one_triad(scattered), T=80.0, max_gs=1)
    assert M.numel() == 6
    # every member weighted by the level's energy, so each weight is exactly 1
    assert float(M.sum()) == pytest.approx(6.0, abs=1e-12)

    _, M0, _ = get_sticks_from_banresult(_one_triad(scattered), T=80.0, max_gs=1, level_window=0.0)
    assert M0.numel() == 1, "level_window=0 must restore exact-float counting"


def test_level_pool_resolves_a_real_splitting():
    """A gap wider than the window stays two levels.

    7.0e-5 eV is the axial splitting of the Fortran ``nid8ct`` ground manifold
    (0.56 cm-1), the narrowest genuinely distinct pair of ground levels in the
    2026-10-18 survey, and it must survive the default 1e-5 eV window.
    """
    from multitorch.spectrum.sticks import get_sticks_from_banresult

    pair = [0.0, 7.0e-5]
    _, M1, _ = get_sticks_from_banresult(_one_triad(pair), T=80.0, max_gs=1)
    assert M1.numel() == 1
    _, M2, _ = get_sticks_from_banresult(_one_triad(pair), T=80.0, max_gs=2)
    assert M2.numel() == 2


def test_level_pool_anchors_rather_than_chains():
    """A ladder of closely spaced states cannot drift one level arbitrarily wide.

    Each step is 0.6 of the window, so chained grouping would swallow all six
    into one level spanning 3 windows. Anchoring on the level's first member
    closes it after the second state, as ``analyze_ground_state`` does.
    """
    from multitorch.spectrum.sticks import get_sticks_from_banresult

    w = 1e-5
    ladder = [0.6 * w * i for i in range(6)]
    _, M, _ = get_sticks_from_banresult(_one_triad(ladder), T=80.0, max_gs=1)
    assert M.numel() == 2


def test_level_pool_weights_by_the_level_not_the_state():
    """Within a level every member carries the level's Boltzmann factor.

    This is what keeps the dipole sum rule an exact integer: per-state weights
    over a manifold scattered by 8e-7 eV scatter the sum by ~1e-4 at 80 K,
    against the sum rule's 1e-6 tolerance. The level above still gets its own
    weight from its own energy, so D-1's convention is untouched.
    """
    import math
    from multitorch._constants import K_B_FLOAT
    from multitorch.spectrum.sticks import get_sticks_from_banresult

    T, gap = 80.0, 0.01
    _, M, _ = get_sticks_from_banresult(
        _one_triad([0.0, 8e-7, gap, gap + 8e-7]), T=T, max_gs=2)
    weights = sorted(float(x) for x in M)
    expected = math.exp(-gap / (K_B_FLOAT * T))
    assert weights[2:] == [pytest.approx(1.0, abs=1e-12)] * 2
    assert weights[:2] == [pytest.approx(expected, rel=1e-12)] * 2


def test_level_pool_rejects_a_negative_window():
    from multitorch.spectrum.sticks import get_sticks_from_banresult
    with pytest.raises(ValueError, match="non-negative"):
        get_sticks_from_banresult(_one_triad([0.0]), T=80.0, max_gs=1, level_window=-1e-5)


def test_level_pool_is_differentiable_through_the_level_energy():
    """The pool is a selection, but the weights must still carry a gradient."""
    from multitorch._constants import DTYPE, K_B_FLOAT
    from types import SimpleNamespace
    from multitorch.spectrum.sticks import get_sticks_from_banresult

    T, gap = 80.0, 0.01
    eg = torch.tensor([0.0, gap], dtype=DTYPE, requires_grad=True)
    result = SimpleNamespace(triads=[SimpleNamespace(
        Eg=eg, Ef=torch.full((1,), 10.0, dtype=DTYPE), T=torch.ones(2, 1, dtype=DTYPE))])
    _, M, _ = get_sticks_from_banresult(result, T=T, max_gs=2)
    (g,) = torch.autograd.grad(M.sum(), eg)
    # d/dE_1 exp((E_0 - E_1)/kT) = -exp(...)/kT, and the lowest level's weight is
    # pinned at 1 by E_min moving with it
    import math
    assert float(g[1]) == pytest.approx(-math.exp(-gap / (K_B_FLOAT * T)) / (K_B_FLOAT * T), rel=1e-10)
