"""WP-B5-9: Td end-to-end — the third group reachable from ``calcXAS``.

**Td is Oh relabelled, for this observable.** Residual 48 measured that before
any Td code was written, and it is what makes this increment cheap and its
oracle exact. Td is a subgroup of Oh with no inversion centre, so parity is
absorbed into the irrep label: gerade ``Xg -> X`` while the ungerade irreps
pick up the ``A1 <-> A2``, ``T1 <-> T2`` swap (and ``E1/2 <-> E5/2``,
``G3/2`` fixed, in the double group). The branching multiplicities are
identical at every J, Td admits Oh's own cubic invariant to 5.0e-16, the dipole
has one component in both, and the triad invariant counts agree in all 25
dipole and 5 scalar triads. So the subduction basis, the ADD coefficients and
the block dimensions are **Oh's**, and only the labels differ.

That gives two oracles and neither is code-vs-code in the bad sense:

1. **Exact equality with Oh** at the same parameters — ``torch.equal``, not a
   tolerance. This is a statement the group theory *forces*, and it is the
   strongest available: Td has no Fortran reference and will not until WP-B6.
2. **The dipole sum rule** (``tests/test_spectrum/test_dipole_sum_rule.py``),
   which is independent of Oh entirely and holds for either sign of ``tendq``.

What would be missed if only (1) were asserted: that Td is *distinguishable*
from Oh at all. So the labels are checked too — the RAC must carry Td's irreps,
not Oh's, or ``sym='td'`` would be an alias rather than a group.

**The sign convention, which has no oracle** (user decision, 2026-10-20):
``tendq`` is the coefficient of the rank-4 cubic invariant, identically to Oh.
A physical tetrahedron therefore needs a **negative** ``tendq``, around -4/9 of
the octahedral value for the same ligands. Flipping it internally so that
positive meant "physically tetrahedral" would have been a per-group convention
asserted without a reference — residual 39's trap — so the parameter means one
thing in every group and the sign is the user's to choose.
"""
from __future__ import annotations

import pytest
import torch

from multitorch.api.calc import calcXAS_from_scratch

GRID = dict(xmin=-12.0, xmax=28.0, nbins=800)
POOL = dict(slater=0.8, soc=1.0, T=80.0, max_gs=1)

IONS = [("Ti", "iii", 1), ("V", "iii", 2), ("Cr", "iii", 3), ("Mn", "iii", 4),
        ("Mn", "ii", 5), ("Fe", "ii", 6), ("Co", "ii", 7), ("Ni", "ii", 8),
        ("Cu", "ii", 9)]
ION_IDS = [f"d{n}-{el}{va}" for el, va, n in IONS]


@pytest.mark.parametrize("element,valence,n_d", IONS, ids=ION_IDS)
def test_td_reproduces_oh_exactly_at_the_same_cubic_field(element, valence, n_d):
    """The headline oracle, over every electron count the generator accepts.

    Bit-identical, not merely close. The cubic invariant is the *same operator*
    in the two groups and the branching multiplicities agree at every J, so
    equality is forced; anything else means the relabelling has leaked into the
    numbers. Half-integer J (d1, d3, d5, d7, d9) goes through the double-group
    emitters, where the swap is E1/2 <-> E5/2 — so this covers both paths.
    """
    _, y_oh = calcXAS_from_scratch(element, valence, cf={"tendq": 1.0}, sym="oh", **POOL, **GRID)
    _, y_td = calcXAS_from_scratch(element, valence, cf={"tendq": 1.0}, sym="td", **POOL, **GRID)
    assert torch.equal(y_oh, y_td), (element, valence, (y_oh - y_td).abs().max().item())


def test_the_equality_is_not_an_artefact_of_a_trivial_spectrum():
    """Guard the oracle above: two zero spectra would also be `torch.equal`."""
    _, y = calcXAS_from_scratch("Ni", "ii", cf={"tendq": 1.0}, sym="td", **POOL, **GRID)
    assert torch.isfinite(y).all()
    assert float(y.sum()) > 0.0
    assert int((y > 0.01 * float(y.max())).sum()) > 10, "a spectrum with structure, not a spike"


def test_td_and_oh_carry_different_irrep_labels():
    """Td must be a group, not an alias for Oh.

    The numbers are Oh's by construction, so the only thing distinguishing the
    two in the RAC is the labelling — and if that were Oh's too, ``sym='td'``
    would be a synonym. The ground manifold's labels agree (``Xg -> X`` is the
    identity on Butler tags) and the excited manifold's are swapped, which is
    exactly the Oh -> Td correspondence.
    """
    from multitorch.angular.rac_generator import generate_ledge_template

    oh, _s1, _d1 = generate_ledge_template(2, 8, sym="oh")
    td, _s2, _d2 = generate_ledge_template(2, 8, sym="td")

    def labels(rac, kind):
        return [ir.name for ir in rac.irreps if ir.kind == kind]

    assert labels(oh, "GROUND") == labels(td, "GROUND")
    assert labels(oh, "EXCITE") != labels(td, "EXCITE")
    # the swap, named: 0- <-> ^0- and 1- <-> ^1-, 2- fixed
    swap = {"0-": "^0-", "^0-": "0-", "1-": "^1-", "^1-": "1-", "2-": "2-"}
    assert [swap[name] for name in labels(oh, "EXCITE")] == labels(td, "EXCITE")

    # and the dipole actor: Oh writes T1u ('1-'), Td writes T2 ('^1-')
    assert {b.op_sym for b in oh.blocks if b.kind == "TRANSI"} == {"1-"}
    assert {b.op_sym for b in td.blocks if b.kind == "TRANSI"} == {"^1-"}


def test_the_add_coefficients_are_oh_s_block_for_block():
    """What residual 48 licenses, asserted rather than assumed.

    Td needs no new angular machinery: the ADD coefficients and the COWAN store
    indices they point at are Oh's. If this ever stops holding, the Oh-equality
    oracle above stops being forced by the group theory and becomes a
    coincidence worth distrusting.
    """
    from multitorch.angular.rac_generator import generate_ledge_template

    for n in (2, 5, 8):
        oh, store_oh, _d1 = generate_ledge_template(2, n, sym="oh")
        td, store_td, _d2 = generate_ledge_template(2, n, sym="td")
        assert len(oh.blocks) == len(td.blocks), n
        for a, b in zip(oh.blocks, td.blocks):
            assert (a.n_bra, a.n_ket) == (b.n_bra, b.n_ket)
            assert a.geometry == b.geometry
            assert len(a.add_entries) == len(b.add_entries)
            for x, y in zip(a.add_entries, b.add_entries):
                assert x.matrix_idx == y.matrix_idx
                assert x.coeff == y.coeff          # exactly, not approximately
        for sec_oh, sec_td in zip(store_oh, store_td):
            assert len(sec_oh) == len(sec_td)
            for m_oh, m_td in zip(sec_oh, sec_td):
                assert torch.equal(m_oh, m_td)


def test_a_negative_tendq_is_the_physical_tetrahedral_field():
    """The sign convention, with the consequence a user would actually meet.

    ``tendq`` is the cubic invariant's coefficient in every group, so a
    tetrahedron needs a negative value and the two signs must give *different*
    spectra — if they did not, the parameter would not be doing anything.
    """
    _, y_pos = calcXAS_from_scratch("Ni", "ii", cf={"tendq": 1.0}, sym="td", **POOL, **GRID)
    _, y_neg = calcXAS_from_scratch("Ni", "ii", cf={"tendq": -1.0}, sym="td", **POOL, **GRID)
    assert not torch.allclose(y_pos, y_neg, atol=1e-8)

    # and the same sign convention in Oh, which is the point of having one
    _, y_oh_neg = calcXAS_from_scratch("Ni", "ii", cf={"tendq": -1.0}, sym="oh", **POOL, **GRID)
    assert torch.equal(y_neg, y_oh_neg)


def test_td_refuses_the_crystal_field_keys_it_has_no_operator_for():
    """One cubic parameter, so `dt`/`ds` are errors — as in Oh (residual 43)."""
    from multitorch.api.calc import calcXAS_cached, preload_from_scratch

    cache = preload_from_scratch("Ni", "ii", "td")
    for bad in ({"tendq": 1.0, "dt": 0.1}, {"ds": 0.1}):
        with pytest.raises(ValueError, match="crystal-field keys"):
            calcXAS_cached(cache, cf=bad, nbins=200)


def test_td_charge_transfer_is_refused_rather_than_wrong():
    """CT beyond Oh/D4h is carried-forward work, so it must raise."""
    from multitorch.api.calc import preload_from_scratch

    with pytest.raises(ValueError, match="unsupported symmetry"):
        preload_from_scratch("Ni", "ii", "td", charge_transfer=True)
