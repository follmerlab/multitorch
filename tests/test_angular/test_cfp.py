"""
Tests for CFP binary parser and sum rules.

Validates:
  1. Parser reads all d^n blocks (n=0..10)
  2. d^2 CFP: all parents are 1.0 (unique parent state)
  3. d^n norm rule: Σ_parent CFP[term, parent]^2 = 1 for each term
  4. get_cfp_for_shell returns correct shape
"""
import pytest
import math
import numpy as np
from pathlib import Path

BINPATH = Path(__file__).parent.parent.parent.parent / 'ttmult' / 'bin'
CFP72 = BINPATH / 'rcg_cfp72'


def _has_cfp_files():
    return CFP72.exists()


@pytest.mark.phase3
@pytest.mark.skipif(not _has_cfp_files(), reason="rcg_cfp72 not found")
def test_cfp_parser_loads_all_d_blocks():
    """All d^n blocks (n=0..10) should be parsed."""
    from multitorch.angular.cfp import parse_cfp_file
    blocks = parse_cfp_file(CFP72)
    d_blocks = [b for b in blocks if b.ll.upper() == 'D']
    n_values = sorted([b.ni for b in d_blocks])
    assert n_values == list(range(11)), f"Expected n=0..10, got {n_values}"


@pytest.mark.phase3
@pytest.mark.skipif(not _has_cfp_files(), reason="rcg_cfp72 not found")
def test_cfp_d2_all_ones():
    """
    d^2 has 5 LS terms (1S, 3P, 1D, 3F, 1G), all with unique parent d^1(2D).
    CFP values should all be 1.0.
    """
    from multitorch.angular.cfp import parse_cfp_file
    blocks = parse_cfp_file(CFP72)
    d2 = [b for b in blocks if b.ll.upper() == 'D' and b.ni == 2][0]
    assert d2.cfp is not None
    assert d2.cfp.shape == (5, 1)
    np.testing.assert_allclose(d2.cfp, np.ones((5, 1)), atol=1e-5)


@pytest.mark.phase3
@pytest.mark.skipif(not _has_cfp_files(), reason="rcg_cfp72 not found")
def test_cfp_norm_rule_d3():
    """
    CFP sum rule (normalization):
      Σ_{parent} |<d^n αSL | d^(n-1) α'S'L'; d>|^2 = 1
    for each d^n LS term α.
    """
    from multitorch.angular.cfp import parse_cfp_file
    blocks = parse_cfp_file(CFP72)
    d3 = [b for b in blocks if b.ll.upper() == 'D' and b.ni == 3][0]
    assert d3.cfp is not None
    # Each row norm should be 1.0
    norms = np.sum(d3.cfp ** 2, axis=1)
    np.testing.assert_allclose(norms, np.ones(d3.nlt), atol=1e-5,
                               err_msg="CFP norm rule failed for d^3")


@pytest.mark.phase3
@pytest.mark.skipif(not _has_cfp_files(), reason="rcg_cfp72 not found")
def test_cfp_norm_rule_all_d():
    """CFP norm rule should hold for all d^n (n=1..10)."""
    from multitorch.angular.cfp import parse_cfp_file
    blocks = parse_cfp_file(CFP72)
    d_blocks = [b for b in blocks if b.ll.upper() == 'D' and b.ni >= 1]
    for b in d_blocks:
        if b.cfp is not None and b.cfp.shape[0] > 0:
            norms = np.sum(b.cfp ** 2, axis=1)
            np.testing.assert_allclose(
                norms, np.ones(b.nlt), atol=1e-4,
                err_msg=f"CFP norm rule failed for d^{b.ni}"
            )


@pytest.mark.phase3
@pytest.mark.skipif(not _has_cfp_files(), reason="rcg_cfp72 not found")
def test_get_cfp_for_shell_d8():
    """get_cfp_for_shell(l=2, n=8) should return a (5, 8) tensor."""
    from multitorch.angular.cfp import get_cfp_for_shell
    import torch
    cfp = get_cfp_for_shell(l=2, n=8,
                             binpath=str(BINPATH))
    assert cfp is not None, "CFP for d^8 not found"
    assert cfp.dtype == torch.float64
    assert cfp.shape == (5, 8), f"Expected (5,8), got {cfp.shape}"


@pytest.mark.phase3
@pytest.mark.skipif(not _has_cfp_files(), reason="rcg_cfp72 not found")
def test_cfp_d1_trivial():
    """d^1 CFP: single state, single parent (vacuum), value = 1.0."""
    from multitorch.angular.cfp import parse_cfp_file
    blocks = parse_cfp_file(CFP72)
    d1 = [b for b in blocks if b.ll.upper() == 'D' and b.ni == 1][0]
    assert d1.cfp is not None
    assert d1.cfp.shape == (1, 1)
    assert abs(float(d1.cfp[0, 0]) - 1.0) < 1e-5
