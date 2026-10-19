"""Transition-operator components adapted to a point group (WP-B4).

Oracles:

1. **Fortran, through the live path.** The derived dipole weights must equal the
   constants the generators had hard-coded — ``rac_generator``'s
   ``PERP_FACTOR = sqrt(2/3)`` and ``PARA_FACTOR = sqrt(1/3)``, and the same
   pair repeated in ``ct_generator`` — together with their irreps and ttrac's
   geometry strings. Those numbers are what make the D4h RAC match ``nid8ct``
   and the Oh RACs match their fixtures.

2. **Analytic.** The partners of a group's components must number exactly
   ``2k+1``, and the weights must square to 1: splitting an operator by
   symmetry neither creates nor destroys isotropic intensity.

3. **Physical, and derived rather than assumed.** In a uniaxial group the
   one-dimensional dipole component lies purely on ``q = 0`` and the
   two-dimensional one purely on ``q = +-1``. That is what makes "PARA"
   z-polarised and "PERP" in-plane, and it is forced by the symmetry, not
   chosen. Oh is the exception, correctly: no direction is special there.

4. **Consistency with WP-B3.** The totally symmetric components at ``k = 2, 4``
   must be exactly the crystal-field invariants, since the crystal field is the
   totally symmetric special case of this module.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from multitorch.angular.crystal_field import cf_invariant_vectors, totally_symmetric_irrep
from multitorch.angular.finite_group import point_group, wigner_from_quaternion
from multitorch.angular.operator_subduction import (
    DIPOLE, GEOMETRY_NAMES, QUADRUPOLE, dipole_geometries, isotropic_weight,
    operator_components, polarization_axis)

GROUPS = ["Oh", "O", "Td", "T", "D4h", "D4", "C4v", "D2d",
          "D3d", "D3", "C3v", "D2h", "D2", "C2v"]

#: Groups with a unique principal axis, where the dipole must separate into a
#: z-polarised channel and in-plane channels. Cubic groups are excluded because
#: no axis is distinguished in them.
UNIAXIAL = ["D4h", "D4", "C4v", "D2d", "D3d", "D3", "C3v", "D2h", "D2", "C2v"]


# ─────────────────────────────────────────────────────────────
# Oracle 1: the hand-coded generator constants
# ─────────────────────────────────────────────────────────────

def test_dipole_geometries_reproduce_the_hand_coded_generator_constants():
    """The two numbers ``generate_ledge_template`` had written out by hand."""
    perp_factor = math.sqrt(2.0 / 3.0)      # rac_generator PERP_FACTOR
    para_factor = math.sqrt(1.0 / 3.0)      # rac_generator PARA_FACTOR

    assert dipole_geometries("D4h") == (("Eu", "PERP", perp_factor),
                                        ("A2u", "PARA", para_factor))
    assert dipole_geometries("Oh") == (("T1u", "MULTIPOLE", 1.0),)


def test_the_generator_takes_its_dipole_blocks_from_this_module():
    """The derivation is load-bearing, not shelf-ware.

    ``generate_ledge_template`` used to carry ``PERP_FACTOR``/``PARA_FACTOR``
    and a two-row tuple of ``(irrep, butler, geometry, factor)``. It now loops
    over :func:`dipole_geometries`, so the whole Fortran-parity suite stands
    behind these numbers — if the derivation drifts, the fixtures fail, not
    just this file. The literals above are kept here as the independent record
    of what the Fortran-validated generator held.
    """
    import inspect

    from multitorch.angular import rac_generator

    src = inspect.getsource(rac_generator.generate_ledge_template)
    assert "for op_d4h_target, geometry, factor in dipole_geometries(" in src
    # the literals are gone as *code*; the words may still appear in comments
    assert "PERP_FACTOR = math.sqrt" not in src
    assert "PARA_FACTOR = math.sqrt" not in src
    assert "'PERP', PERP_FACTOR)" not in src
    assert "'PARA', PARA_FACTOR)" not in src


def test_the_charge_transfer_generator_uses_the_same_two_factors():
    """``ct_generator`` repeats the pair; it must stay the same pair."""
    import inspect

    from multitorch.angular import ct_generator

    src = inspect.getsource(ct_generator)
    assert "('Eu', '1-', 'PERP', math.sqrt(2.0 / 3.0))" in src
    assert "('A2u', '^0-', 'PARA', math.sqrt(1.0 / 3.0))" in src


# ─────────────────────────────────────────────────────────────
# Oracle 2: the analytic sum rules
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("k,parity", [DIPOLE, QUADRUPOLE, (4, +1)],
                         ids=["dipole", "quadrupole", "rank4"])
def test_components_account_for_every_operator_component(group, k, parity):
    components = operator_components(group, k, parity)
    assert sum(c.dim for c in components) == 2 * k + 1


@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("k,parity", [DIPOLE, QUADRUPOLE, (4, +1)],
                         ids=["dipole", "quadrupole", "rank4"])
def test_the_weights_conserve_isotropic_intensity(group, k, parity):
    """``sum w^2 = 1``: the split neither creates nor loses intensity."""
    components = operator_components(group, k, parity)
    assert sum(c.weight ** 2 for c in components) == pytest.approx(1.0, abs=1e-12)


def test_isotropic_weight_is_the_dimension_share():
    assert isotropic_weight(3, 1) == pytest.approx(1.0)
    assert isotropic_weight(2, 1) == pytest.approx(math.sqrt(2.0 / 3.0))
    assert isotropic_weight(1, 1) == pytest.approx(math.sqrt(1.0 / 3.0))


@pytest.mark.parametrize("group", GROUPS)
def test_component_vectors_are_orthonormal_and_span_the_operator(group):
    components = operator_components(group, *DIPOLE)
    S = np.column_stack([c.vectors for c in components])
    assert S.shape == (3, 3)
    np.testing.assert_allclose(S.conj().T @ S, np.eye(3), atol=1e-10)


@pytest.mark.parametrize("group", GROUPS)
def test_each_component_transforms_within_itself(group):
    """``D^(1)(g)`` maps a component's partner space onto itself, for every g.

    The defining property of a component: it is a subrepresentation. Checked as
    a projector commuting with the representation, so it is gauge-free.
    """
    g = point_group(group)
    for component in operator_components(group, *DIPOLE):
        V = np.asarray(component.vectors)
        P = V @ V.conj().T
        for e in g.elements:
            D = wigner_from_quaternion(1.0, e.quat) * (e.parity if e.parity < 0 else 1)
            assert np.allclose(D @ P, P @ D, atol=1e-10), (group, component.irrep, e)


# ─────────────────────────────────────────────────────────────
# Oracle 3: polarisation, read off rather than assumed
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("group", UNIAXIAL)
def test_a_uniaxial_group_separates_z_from_in_plane_polarisation(group):
    """Exactly one z channel, and every other channel purely in-plane.

    This is the physics behind the PARA/PERP names: the one-dimensional
    component of a uniaxial group comes out on ``q = 0`` alone, forced by the
    symmetry. Measured ``|v|`` on ``q = 0`` is exactly ``[0, 1, 0]`` for D4h.
    """
    axes = [polarization_axis(c) for c in operator_components(group, *DIPOLE)]
    assert axes.count("z") == 1, (group, axes)
    assert all(a == "xy" for a in axes if a != "z"), (group, axes)
    assert None not in axes, (group, axes)


@pytest.mark.parametrize("group", ["Oh", "O", "Td", "T"])
def test_a_cubic_group_has_no_distinguished_polarisation_axis(group):
    """One three-dimensional component, mixing every ``q`` — the honest answer."""
    components = operator_components(group, *DIPOLE)
    assert len(components) == 1 and components[0].dim == 3
    assert polarization_axis(components[0]) is None


def test_the_para_channel_of_d4h_is_exactly_the_q_zero_component():
    """Sharper than the axis label: the vector itself is ``|1, 0>`` up to phase."""
    para = [c for c in operator_components("D4h", *DIPOLE) if c.irrep == "A2u"]
    assert len(para) == 1
    v = np.asarray(para[0].vectors)[:, 0]
    assert abs(abs(v[1]) - 1.0) < 1e-12
    assert abs(v[0]) < 1e-12 and abs(v[2]) < 1e-12


def test_biaxial_groups_split_x_from_y():
    """D2h and C2v give three one-dimensional channels, not two.

    x, y and z are mutually inequivalent in a biaxial group, so the dipole
    needs three geometries. This is the structural statement that a
    PERP/PARA-shaped convention cannot express, and the reason B5 must take the
    geometry list from the group rather than from a two-entry tuple.
    """
    for group in ("D2h", "C2v"):
        components = operator_components(group, *DIPOLE)
        assert [c.dim for c in components] == [1, 1, 1], group
        assert len({c.irrep for c in components}) == 3, group


# ─────────────────────────────────────────────────────────────
# Oracle 4: the crystal field is the totally symmetric special case
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("k", [2, 4])
def test_the_totally_symmetric_components_are_the_crystal_field_invariants(group, k):
    """Ties WP-B3 and WP-B4 together: one subduction, two consumers."""
    totally_symmetric = totally_symmetric_irrep(group)
    components = [c for c in operator_components(group, k, +1)
                  if c.irrep == totally_symmetric]
    V_cf = np.asarray(cf_invariant_vectors(group, k))
    assert sum(c.dim for c in components) == V_cf.shape[1], (group, k)
    if V_cf.shape[1] == 0:
        return
    mine = np.column_stack([c.vectors for c in components])
    P_mine = mine @ mine.conj().T
    P_cf = V_cf @ V_cf.conj().T
    np.testing.assert_allclose(P_mine, P_cf, atol=1e-12)


# ─────────────────────────────────────────────────────────────
# Refusals and injected bugs
# ─────────────────────────────────────────────────────────────

def test_a_group_with_no_ttrac_convention_is_refused_not_guessed():
    """The structure is available; the *name* is not invented."""
    with pytest.raises(ValueError, match="no ttrac geometry convention"):
        dipole_geometries("C4v")
    # but the structure is there
    assert [c.irrep for c in operator_components("C4v", *DIPOLE)] == ["A1", "E"]


def test_a_geometry_table_that_drifts_from_the_group_is_refused(monkeypatch):
    """Inject a convention missing a block and watch the cross-check fail.

    Without it, a dropped entry would silently emit one RAC block instead of
    two and lose a polarisation channel — a wrong spectrum, not a crash.
    """
    import multitorch.angular.operator_subduction as osub

    monkeypatch.setattr(osub, "GEOMETRY_NAMES",
                        {**GEOMETRY_NAMES, "D4h": (("Eu", "PERP"),)})
    with pytest.raises(RuntimeError, match="the convention and the group theory disagree"):
        osub.dipole_geometries("D4h")


def test_a_geometry_table_naming_an_absent_irrep_is_refused(monkeypatch):
    import multitorch.angular.operator_subduction as osub

    monkeypatch.setattr(osub, "GEOMETRY_NAMES",
                        {**GEOMETRY_NAMES,
                         "D4h": (("Eu", "PERP"), ("A2u", "PARA"), ("B1u", "NOPE"))})
    with pytest.raises(RuntimeError, match="the convention and the group theory disagree"):
        osub.dipole_geometries("D4h")


def test_polarization_axis_refuses_a_non_dipole_component():
    quadrupole = operator_components("D4h", *QUADRUPOLE)[0]
    with pytest.raises(ValueError, match="rank-1"):
        polarization_axis(quadrupole)


def test_the_cached_component_vectors_cannot_be_mutated():
    component = operator_components("D4h", *DIPOLE)[0]
    with pytest.raises(ValueError, match="read-only"):
        component.vectors[0, 0] = 99.0
