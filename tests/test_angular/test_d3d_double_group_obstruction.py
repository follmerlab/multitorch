"""Why D3d's half-integer path is refused: there is no real-coefficient form.

This file measures the obstruction rather than asserting the refusal, because
the refusal is only honest if the obstruction is real. Three statements, each
independent of our implementation:

1. **Frobenius-Schur indicators.** Every D4h double irrep is *pseudo-real*
   (-1), which is the premise ``double_group``'s module docstring states and
   its gauge argument rests on. D3d's E1/2 is pseudo-real too. Its two
   **one-dimensional** spinor irreps are **0** -- a genuine complex-conjugate
   pair, which Butler writes ``S1 * -S1`` and which D4h simply does not have.
   The computation is self-validating: it reproduces -1 on the three irreps the
   package already documents as pseudo-real.

2. **The loop-product criterion.** A complex Hermitian matrix is reducible to
   real symmetric by a diagonal phase transformation iff every loop product
   ``H_ij H_jk H_ki`` is real -- and a loop product is *invariant* under
   exactly that phase freedom, which is the whole freedom gauge-fixing has (one
   phase per route, one per copy). D3d's E1/2 block passes; its ``1E3/2`` block
   fails at ~1e+00. So this is not a bad spanning tree or a missing seed
   operator, and no amount of route-phase pinning reaches it.

3. **The pair is reducible together, not separately.** ``H(2E3/2) =
   conj(H(1E3/2))`` with equal eigenvalues (Kramers), and the direct sum has an
   exact real form of twice the dimension while neither member alone does. That
   is the shape a fix would take -- the Kramers pair carried as one
   two-dimensional corepresentation -- and it is a data-model change, not a
   gauge choice.

Residual 68. Two things *not* claimed: that the partner vectors of the pair are
conjugates of one another (they are not -- measured exactly 0 overlap, since
Gamma8's carrier spaces are not related by plain conjugation in this basis),
and that the spectrum is wrong today (it is refused, not computed).
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from multitorch.angular import symmetry
from multitorch.angular.double_group import (
    gauged_basis, oh_to_target_double_subduction, target_double_irrep_matrices,
)
from multitorch.angular.finite_group import point_group
from multitorch.angular.symmetry import OH_ROTATION_SUBGROUP

#: J values spanning enough copies for the obstruction to appear; the witness
#: triangle is (3/2, 5/2, 9/2), so 9/2 has to be present.
JS = [0.5, 1.5, 2.5, 3.5, 4.5]


def _frobenius_schur(target_group):
    """``{irrep: indicator}`` over the double group.

    ``(1/|G|) Sum_g chi(g^2)``. A barred element's square equals its unbarred
    partner's, so the sum over all ``2n`` elements is twice the sum over the
    ``n`` proper rotations and the ``2n`` normalisation cancels the doubling.
    """
    group = point_group(OH_ROTATION_SUBGROUP[target_group])
    pairs = symmetry._target_rotations_in_oh(group)
    out = {}
    for name, gamma in target_double_irrep_matrices(target_group).items():
        out[name] = sum(np.trace(g @ g).real for g in gamma) / len(pairs)
    return out


def test_the_trigonal_one_dimensional_pair_is_complex_not_pseudo_real():
    """The premise of the whole refusal, checked against what the package documents."""
    d4h = _frobenius_schur("D4h")
    assert sorted(d4h) == ["E1/2", "E3/2"]
    for name, fs in d4h.items():
        assert fs == pytest.approx(-1.0, abs=1e-9), (name, fs)

    d3d = _frobenius_schur("D3d")
    assert d3d["E1/2"] == pytest.approx(-1.0, abs=1e-9), d3d
    for name in ("1E3/2", "2E3/2"):
        assert d3d[name] == pytest.approx(0.0, abs=1e-9), (name, d3d[name])


def _cf_block(basis, irrep, coefficients):
    """The D3d crystal-field matrix on one target irrep, over its route-nodes."""
    from multitorch.angular.crystal_field import cf_operator_vector
    from multitorch.angular.symmetry_plan import CONVENTIONS

    ops = {name: cf_operator_vector("D3d", name)
           for name in ("TENDQ", "DSIGMA", "DTAU")}
    nodes = [n for n in basis.nodes if n[0] == "gs"
             and irrep in oh_to_target_double_subduction(n[2], "D3d")]
    H = np.zeros((len(nodes),) * 2, dtype=np.complex128)
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            total = 0j
            for name, (k, vec) in ops.items():
                if abs(a[1] - b[1]) > k or a[1] + b[1] < k:
                    continue
                sa = oh_to_target_double_subduction(a[2], "D3d")[irrep]
                sb = oh_to_target_double_subduction(b[2], "D3d")[irrep]
                M = basis.operator_tensor(a, b, k, [vec])[:, 0, :]
                total += coefficients[name] * (sa[:, 0].conj() @ M @ sb[:, 0])
            H[i, j] = total
    return nodes, 0.5 * (H + H.conj().T)


def _worst_loop(H):
    """max |Im(loop)| / |loop| over every triangle; invariant under diag phases."""
    worst = 0.0
    for i, j, k in itertools.combinations(range(H.shape[0]), 3):
        loop = H[i, j] * H[j, k] * H[k, i]
        if abs(loop) < 1e-10:
            continue
        worst = max(worst, abs(loop.imag) / abs(loop))
    return worst


@pytest.fixture(scope="module")
def basis():
    return gauged_basis([("gs", JS), ("ex", JS)], d4h=False)


COEFFS = {"TENDQ": 1.0, "DSIGMA": 0.37, "DTAU": -0.21}


def test_the_one_dimensional_block_admits_no_real_form(basis):
    """The loop-product obstruction, which no diagonal phase can remove."""
    _nodes, H_e = _cf_block(basis, "E1/2", COEFFS)
    assert _worst_loop(H_e) < 1e-9, "E1/2 should be fixable; it is pseudo-real"

    nodes, H_1 = _cf_block(basis, "1E3/2", COEFFS)
    assert len(nodes) >= 3, nodes
    worst = _worst_loop(H_1)
    assert worst > 0.1, (
        f"the 1E3/2 obstruction measured {worst:.3e}; if this has become small the "
        f"refusal in _validate_from_scratch may no longer be justified")
    # and the block is not trivially complex -- it carries real weight
    assert float(np.abs(H_1.imag).max()) > 1e-3


def test_the_kramers_pair_is_reducible_together_but_not_separately(basis):
    """What a fix would look like: one 2-D corepresentation, not two 1-D irreps."""
    _n1, H1 = _cf_block(basis, "1E3/2", COEFFS)
    _n2, H2 = _cf_block(basis, "2E3/2", COEFFS)
    assert np.abs(H2 - np.conj(H1)).max() < 1e-12, "the pair should be conjugate"
    assert np.abs(np.linalg.eigvalsh(H1) - np.linalg.eigvalsh(H2)).max() < 1e-12, \
        "Kramers degeneracy"

    # neither member alone is real...
    assert float(np.abs(H1.imag).max()) > 1e-3
    # ...and the standard real form of the pair is exactly symmetric, with the
    # spectrum doubled.
    M = np.block([[H1.real, -H1.imag], [H1.imag, H1.real]])
    assert np.abs(M - M.T).max() == 0.0
    doubled = np.repeat(np.sort(np.linalg.eigvalsh(H1)), 2)
    assert np.abs(np.sort(np.linalg.eigvalsh(M)) - doubled).max() < 1e-10


def test_the_partner_vectors_of_the_pair_are_not_conjugates(basis):
    """Recorded because it is the plausible-looking claim that is false.

    The two irreps are conjugate *as representations*; their carrier spaces
    inside Gamma8 are not related by plain conjugation in this basis, and the
    overlap is exactly zero. Only the representation statement supports the
    labelling being a convention.
    """
    sub = oh_to_target_double_subduction("G3/2", "D3d")
    v1, v2 = sub["1E3/2"][:, 0], sub["2E3/2"][:, 0]
    assert abs(v1.conj() @ np.conj(v2)) < 1e-12
    gam = target_double_irrep_matrices("D3d")
    assert max(abs(a[0, 0] - np.conj(b[0, 0]))
               for a, b in zip(gam["1E3/2"], gam["2E3/2"])) < 1e-12
