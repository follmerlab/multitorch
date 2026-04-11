"""
Phase 3 tests: Wigner 3j and 6j symbols.

All tests validated against scipy.special.wigner_3j and
sympy.physics.quantum.cg.Wigner3j for exact reference values.

Key identities tested:
  1. Selection rules (return 0 for forbidden transitions)
  2. Known exact values for small j
  3. Symmetry relations (column permutations)
  4. Orthogonality sum rules
  5. Comparison with scipy.special.wigner_3j
"""
import pytest
import math

try:
    from sympy.physics.wigner import wigner_3j as sympy_w3j, wigner_6j as sympy_w6j
    from sympy import N as sympy_N
    HAS_SCIPY = True  # Using sympy as the reference
    def scipy_w3j(j1, j2, j3, m1, m2, m3):
        return float(sympy_N(sympy_w3j(j1, j2, j3, m1, m2, m3)))
    def scipy_w6j(j1, j2, j3, j4, j5, j6):
        return float(sympy_N(sympy_w6j(j1, j2, j3, j4, j5, j6)))
except ImportError:
    HAS_SCIPY = False


@pytest.mark.phase3
def test_wigner3j_selection_rule_m_sum():
    """3j should be zero if m1+m2+m3 != 0."""
    from multitorch.angular.wigner import wigner3j
    assert wigner3j(1, 1, 1, 1, 0, 0) == 0.0  # 1+0+0 != 0


@pytest.mark.phase3
def test_wigner3j_selection_rule_triangle():
    """3j should be zero if triangle inequality is violated."""
    from multitorch.angular.wigner import wigner3j
    assert wigner3j(3, 1, 1, 0, 0, 0) == 0.0  # |3-1|=2 > 1


@pytest.mark.phase3
def test_wigner3j_known_value_111():
    """
    W3j(1,1,1; -1,0,1) = 1/sqrt(6) = 0.408248...
    From Edmonds Table 2.
    """
    from multitorch.angular.wigner import wigner3j
    val = wigner3j(1, 1, 1, -1, 0, 1)
    expected = 1.0 / math.sqrt(6.0)  # 0.408248
    assert abs(val - expected) < 1e-10, f"W3j = {val}, expected {expected}"


@pytest.mark.phase3
def test_wigner3j_known_value_111_000():
    """
    W3j(1,1,1; 0,0,0) = 0 (since j1+j2+j3=3 is odd and all m=0 → parity rule).
    Actually: W3j(j1,j2,j3; 0,0,0) = 0 if j1+j2+j3 is odd.
    """
    from multitorch.angular.wigner import wigner3j
    val = wigner3j(1, 1, 1, 0, 0, 0)
    assert abs(val) < 1e-12, f"W3j(1,1,1;0,0,0) = {val}, expected 0"


@pytest.mark.phase3
def test_wigner3j_known_value_222():
    """
    W3j(2,2,2; 1,-2,1) = -sqrt(3/35) ≈ -0.29277.
    Exact: (-1)^(2-2-1) × algebraic Racah sum.
    Validated against sympy.physics.wigner.wigner_3j.
    """
    from multitorch.angular.wigner import wigner3j
    val = wigner3j(2, 2, 2, 1, -2, 1)
    # Exact value: -sqrt(3/35)
    expected = -math.sqrt(3.0 / 35.0)  # ≈ -0.29277
    assert abs(val - expected) < 1e-10, f"W3j = {val}, expected {expected}"


@pytest.mark.phase3
def test_wigner3j_symmetry_column_perm():
    """
    Even permutation of columns: W3j(j1,j2,j3; m1,m2,m3) = W3j(j2,j3,j1; m2,m3,m1)
    """
    from multitorch.angular.wigner import wigner3j
    v1 = wigner3j(2, 1, 3, 1, -1, 0)
    v2 = wigner3j(1, 3, 2, -1, 0, 1)
    assert abs(v1 - v2) < 1e-12, f"Column perm symmetry failed: {v1} != {v2}"


@pytest.mark.phase3
def test_wigner3j_sign_flip():
    """
    W3j(j1,j2,j3; -m1,-m2,-m3) = (-1)^(j1+j2+j3) * W3j(j1,j2,j3; m1,m2,m3)
    """
    from multitorch.angular.wigner import wigner3j
    j1, j2, j3 = 2.0, 2.0, 2.0
    m1, m2, m3 = 1.0, -1.0, 0.0
    v_pos = wigner3j(j1, j2, j3, m1, m2, m3)
    v_neg = wigner3j(j1, j2, j3, -m1, -m2, -m3)
    phase = 1.0 if int(j1 + j2 + j3) % 2 == 0 else -1.0
    assert abs(v_neg - phase * v_pos) < 1e-12


@pytest.mark.phase3
def test_wigner3j_orthogonality():
    """
    Orthogonality sum rule:
      Σ_{m1,m2: m1+m2=−m3} W3j(j1,j2,j3;m1,m2,m3)² = 1/(2j3+1)

    Test: j1=j2=1, j3=1, m3=0 → sum over m1,m2 with m1+m2=0.
    Expected: 1/(2×1+1) = 1/3.
    """
    from multitorch.angular.wigner import wigner3j
    j1, j2, j3 = 1.0, 1.0, 1.0
    m3 = 0.0

    total = sum(
        wigner3j(j1, j2, j3, float(m1), float(-m1 - m3), m3) ** 2
        for m1 in [-1, 0, 1]
        if abs(-m1 - m3) <= j2
    )
    expected = 1.0 / (2 * j3 + 1)
    assert abs(total - expected) < 1e-10, f"Sum = {total}, expected {expected}"


@pytest.mark.phase3
def test_wigner6j_known_value():
    """
    {1 1 2; 2 2 2} ≈ 0.15275.
    Exact algebraic value validated against sympy.physics.wigner.wigner_6j.
    (Not 1/(3*sqrt(5)) ≈ 0.14907; that is a different symbol.)
    """
    from multitorch.angular.wigner import wigner6j
    val = wigner6j(1, 1, 2, 2, 2, 2)
    # Validated against sympy: 0.15275252316519466
    expected = 0.15275252316519466
    assert abs(val - expected) < 1e-10, f"W6j = {val}, expected {expected}"


@pytest.mark.phase3
def test_wigner6j_selection_rule():
    """6j should be zero if triangle inequalities are violated."""
    from multitorch.angular.wigner import wigner6j
    assert wigner6j(3, 1, 1, 1, 1, 1) == 0.0  # Triangle(3,1,1) fails


@pytest.mark.phase3
@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
def test_wigner3j_scipy_agreement_batch():
    """Compare multitorch wigner3j against scipy for a batch of values."""
    from multitorch.angular.wigner import wigner3j

    test_cases = [
        (1, 1, 2, 1, -1, 0),
        (2, 2, 2, 2, -2, 0),
        (1.5, 0.5, 2, 0.5, 0.5, -1),
        (2, 1, 1, 1, 0, -1),
        (3, 2, 1, -1, 1, 0),
    ]
    for args in test_cases:
        j1, j2, j3, m1, m2, m3 = args
        our_val = wigner3j(j1, j2, j3, m1, m2, m3)
        scipy_val = float(scipy_w3j(j1, j2, j3, m1, m2, m3))
        assert abs(our_val - scipy_val) < 1e-10, (
            f"W3j{args}: ours={our_val}, scipy={scipy_val}"
        )


@pytest.mark.phase3
@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
def test_wigner6j_scipy_agreement_batch():
    """Compare multitorch wigner6j against scipy for a batch of values."""
    from multitorch.angular.wigner import wigner6j

    test_cases = [
        (1, 1, 2, 2, 2, 1),
        (2, 2, 2, 2, 2, 2),
        (1, 2, 1, 2, 1, 2),
        (0.5, 0.5, 1, 0.5, 0.5, 0),
    ]
    for args in test_cases:
        j1, j2, j3, j4, j5, j6 = args
        our_val = wigner6j(j1, j2, j3, j4, j5, j6)
        scipy_val = float(scipy_w6j(j1, j2, j3, j4, j5, j6))
        assert abs(our_val - scipy_val) < 1e-10, (
            f"W6j{args}: ours={our_val}, scipy={scipy_val}"
        )


@pytest.mark.phase3
def test_clebsch_gordan_known_value():
    """
    <1,1; 1,-1 | 0,0> = 1/sqrt(3) = 0.57735...
    """
    from multitorch.angular.wigner import clebsch_gordan
    val = clebsch_gordan(1, 1, 1, -1, 0, 0)
    expected = 1.0 / math.sqrt(3.0)
    assert abs(val - expected) < 1e-10, f"CG = {val}, expected {expected}"
