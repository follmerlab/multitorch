"""The symmetry plan, and why the length-based group guesses have to go (WP-B5).

Oracles:

1. **The constants the production code holds today.** The plan's
   ``operator_order`` and ``hybr_channels`` must equal
   ``assemble.OPERATOR_ORDER_OH/_D4H`` and ``assemble.HYBR_ORDER_OH/_D4H``, its
   ``hybr_parameter_names`` the ``lmct`` keys ``ct_generator`` uses, and its
   ``dipole_blocks`` the generator's dipole rows. Those four lists are what
   make the Oh and D4h RAC output match Fortran.

2. **Derived structure against named convention.** The number and ranks of the
   crystal-field operators come from WP-B3 and the hybridisation channels from
   ``branching(2, +1)``; the names and orders are looked up. The two sides are
   cross-checked, and a drifted convention is refused.

3. **The ambiguity itself.** ``n_ops`` and the channel count are *not* group
   identifiers, and the test says so with numbers rather than by assertion.
"""
from __future__ import annotations

import pytest

from multitorch.angular.crystal_field import cf_parameter_count
from multitorch.angular.finite_group import point_group
from multitorch.angular.operator_subduction import dipole_geometries
from multitorch.angular.symmetry_plan import (
    CONVENTIONS, D_SHELL, HAMILTONIAN, SYM_TO_GROUP, GroupConventions,
    plan_for_sym, symmetry_plan)
from multitorch.hamiltonian.assemble import (HYBR_ORDER_D4H, HYBR_ORDER_OH,
                                             OPERATOR_ORDER_D4H, OPERATOR_ORDER_OH)


# ─────────────────────────────────────────────────────────────
# Oracle 1: today's production constants
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("group,operators,channels", [
    ("Oh", OPERATOR_ORDER_OH, HYBR_ORDER_OH),
    ("D4h", OPERATOR_ORDER_D4H, HYBR_ORDER_D4H),
])
def test_the_plan_reproduces_the_assembler_constants(group, operators, channels):
    """Including the order, which is the order of blocks in the RAC file."""
    plan = symmetry_plan(group)
    assert list(plan.operator_order) == operators
    assert list(plan.hybr_channels) == channels


def test_the_plan_reproduces_the_public_lmct_channel_keys():
    """``ct_generator`` holds ``{'oh': ('eg','t2g'), 'd4h': ('b1','a1','b2','e')}``."""
    import inspect

    from multitorch.angular import ct_generator

    src = inspect.getsource(ct_generator)
    assert "names = {'oh': ('eg', 't2g'), 'd4h': ('b1', 'a1', 'b2', 'e')}[sym]" in src
    assert symmetry_plan("Oh").hybr_parameter_names == ("eg", "t2g")
    assert symmetry_plan("D4h").hybr_parameter_names == ("b1", "a1", "b2", "e")


@pytest.mark.parametrize("group", ["Oh", "D4h"])
def test_the_plan_reproduces_the_dipole_blocks(group):
    assert symmetry_plan(group).dipole_blocks == dipole_geometries(group)


def test_the_hamiltonian_leads_the_operator_order():
    for group in CONVENTIONS:
        assert symmetry_plan(group).operator_order[0] == HAMILTONIAN


# ─────────────────────────────────────────────────────────────
# Oracle 2: derived structure vs named convention
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("group", ["Oh", "D4h"])
def test_the_crystal_field_operator_count_is_the_derived_one(group):
    """One named operator per invariant WP-B3 finds, and the ranks agree."""
    plan = symmetry_plan(group)
    derived = cf_parameter_count(group)
    assert plan.n_cf_parameters == sum(derived.values())
    by_rank = {}
    for _name, rank in plan.cf_operators:
        by_rank[rank] = by_rank.get(rank, 0) + 1
    assert by_rank == {k: v for k, v in derived.items() if v}


@pytest.mark.parametrize("group", ["Oh", "D4h"])
def test_the_hybridisation_channels_are_the_d_orbital_irreps(group):
    """The channels are what the d orbitals span: ``branching(2, +1)``.

    Oh gives Eg + T2g, which is ``eg``/``t2g``; D4h gives A1g + B1g + B2g + Eg,
    which is ``b1``/``a1``/``b2``/``e``. Derived, not listed.
    """
    plan = symmetry_plan(group)
    branching = point_group(group).branching(*D_SHELL)
    expected = sum(branching.values())
    assert len(plan.hybr_irreps) == expected
    assert len(plan.hybr_channels) == expected
    assert sorted(i for i, _c in plan.hybr_irreps) == sorted(
        i for i, c in branching.items() for _ in range(c))


def test_a_convention_naming_the_wrong_ranks_is_refused(monkeypatch):
    """Inject a convention that claims a rank-2 operator for Oh."""
    import multitorch.angular.symmetry_plan as sp

    bad = dict(CONVENTIONS)
    bad["Oh"] = GroupConventions(cf_operators=(("DS", 2),),
                                 hybr=CONVENTIONS["Oh"].hybr)
    monkeypatch.setattr(sp, "CONVENTIONS", bad)
    sp.symmetry_plan.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="crystal-field parameters of ranks"):
            sp.symmetry_plan("Oh")
    finally:
        sp.symmetry_plan.cache_clear()


def test_a_convention_naming_the_wrong_channels_is_refused(monkeypatch):
    """Inject a channel list for irreps the d orbitals do not span in Oh."""
    import multitorch.angular.symmetry_plan as sp

    bad = dict(CONVENTIONS)
    bad["Oh"] = GroupConventions(
        cf_operators=CONVENTIONS["Oh"].cf_operators,
        hybr=(("Eg", "EGHYBR", "eg"), ("A1g", "A1HYBR", "a1")))
    monkeypatch.setattr(sp, "CONVENTIONS", bad)
    sp.symmetry_plan.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="the d orbitals span"):
            sp.symmetry_plan("Oh")
    finally:
        sp.symmetry_plan.cache_clear()


# ─────────────────────────────────────────────────────────────
# Oracle 3: the ambiguity the plan exists to remove
# ─────────────────────────────────────────────────────────────

#: (group, n_ops the length-sniff would see, number of hybridisation channels).
#: Counts are derived here, not copied, so this table cannot go stale.
AMBIGUITY = ["Oh", "Td", "D4h", "C4v", "D3d", "D2h", "C2v"]


def _n_ops(group: str) -> int:
    return 1 + sum(cf_parameter_count(group).values())


def _n_channels(group: str) -> int:
    return sum(point_group(group).branching(*D_SHELL).values())


def test_the_operator_count_does_not_identify_the_group():
    """``assemble``'s ``n_ops = len(xham)`` collides across WP-B's targets.

    D4h, C4v and D3d all give 4; Oh and Td both give 2; D2h and C2v both give
    6 — while their operator names, channel names and channel order differ. A
    length-based guess would hand D4h's DT/DS and B1/A1/B2/E to any of them.
    """
    by_count = {}
    for group in AMBIGUITY:
        by_count.setdefault(_n_ops(group), []).append(group)
    collisions = {n: g for n, g in by_count.items() if len(g) > 1}
    assert collisions, "if this ever becomes empty, the sniffing is safe and B5 can relax"
    assert set(collisions[4]) == {"D4h", "C4v", "D3d"}
    assert set(collisions[2]) == {"Oh", "Td"}
    assert set(collisions[6]) == {"D2h", "C2v"}


def test_the_channel_count_does_not_identify_the_group_either():
    """The same holds for ``api.calc``'s ``len(ban.xmix[0].values)`` guess."""
    by_count = {}
    for group in AMBIGUITY:
        by_count.setdefault(_n_channels(group), []).append(group)
    assert set(by_count[4]) == {"D4h", "C4v"}
    assert set(by_count[2]) == {"Oh", "Td"}
    assert set(by_count[5]) == {"D2h", "C2v"}
    # D3d is the one the two counts disagree about: 4 operators but 3 channels
    assert _n_ops("D3d") == 4 and _n_channels("D3d") == 3


def test_the_plan_distinguishes_groups_the_counts_cannot():
    """Oh and D4h differ in name, not merely in length — that is the point."""
    oh, d4h = symmetry_plan("Oh"), symmetry_plan("D4h")
    assert oh.group != d4h.group
    assert oh.operator_order != d4h.operator_order
    assert oh.hybr_channels != d4h.hybr_channels
    assert oh.hybr_parameter_names != d4h.hybr_parameter_names
    assert oh.dipole_blocks != d4h.dipole_blocks


# ─────────────────────────────────────────────────────────────
# Surface
# ─────────────────────────────────────────────────────────────

def test_plan_for_sym_maps_the_public_api_strings():
    assert plan_for_sym("oh").group == "Oh"
    assert plan_for_sym("d4h").group == "D4h"
    assert set(SYM_TO_GROUP) == {"oh", "d4h"}


def test_an_unconventional_group_is_refused_with_a_route_to_the_derived_parts():
    with pytest.raises(ValueError, match="no store convention recorded"):
        symmetry_plan("C4v")
    with pytest.raises(ValueError, match="unknown symmetry"):
        plan_for_sym("c4v")
    # the derived structure needs no convention
    assert sum(cf_parameter_count("C4v").values()) == 3
    assert [c.irrep for c in __import__(
        "multitorch.angular.operator_subduction", fromlist=["x"]
    ).operator_components("C4v", 1, -1)] == ["A1", "E"]


def test_the_plan_is_immutable_and_cached():
    plan = symmetry_plan("D4h")
    assert symmetry_plan("D4h") is plan
    with pytest.raises(Exception):
        plan.group = "Oh"          # frozen dataclass


# ─────────────────────────────────────────────────────────────
# Behaviour preservation: the plan path equals the inference path
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("sym,n", [("oh", 2), ("oh", 8), ("d4h", 2), ("d4h", 8)])
def test_declaring_the_symmetry_changes_nothing_for_oh_and_d4h(sym, n):
    """The whole point of B5-2: a declared group must reproduce the inference.

    ``assemble`` used to infer the operator and channel names from
    ``len(xham)``. It now prefers ``ban.symmetry`` when the .ban declares one,
    and the from-scratch builders declare it. For Oh and D4h the two must agree
    *exactly* — the inference is correct there, and the plan reproduces its
    lists order and all — so this asserts bit-identical eigenvalues and
    transition matrices rather than a tolerance.

    Beyond Oh and D4h they cannot agree, because the inference cannot tell
    those groups apart; that is what the ambiguity tests above measure.
    """
    import dataclasses

    import torch

    from multitorch.angular.rac_generator import generate_ledge_rac
    from multitorch.api.calc import _build_ban_from_rac
    from multitorch.hamiltonian.assemble import assemble_and_diagonalize_in_memory

    rac, cowan = generate_ledge_rac(2, n, sym=sym)
    declared = _build_ban_from_rac(rac, tendq=1.0, dt=0.05, ds=0.1, sym=sym)
    assert declared.symmetry == sym, "the builder must declare the group"
    inferred = dataclasses.replace(declared, symmetry=None)

    a = assemble_and_diagonalize_in_memory(cowan, rac, declared)
    b = assemble_and_diagonalize_in_memory(cowan, rac, inferred)

    assert len(a.triads) == len(b.triads) > 0
    for ta, tb in zip(a.triads, b.triads):
        where = (ta.gs_sym, ta.act_sym, ta.fs_sym)
        assert where == (tb.gs_sym, tb.act_sym, tb.fs_sym)
        assert torch.equal(ta.Eg, tb.Eg), ("Eg", where)
        assert torch.equal(ta.Ef, tb.Ef), ("Ef", where)
        assert torch.equal(ta.T, tb.T), ("T", where)


def test_a_declared_symmetry_inconsistent_with_xham_is_refused():
    """Inject the mismatch the declaration exists to make impossible.

    A .ban declaring 'oh' but carrying D4h's four XHAM operators would, before
    the check, assemble DT and DS as though they were Oh operators. Silent.
    """
    import dataclasses

    from multitorch.angular.rac_generator import generate_ledge_rac
    from multitorch.api.calc import _build_ban_from_rac
    from multitorch.hamiltonian.assemble import assemble_and_diagonalize_in_memory

    rac, cowan = generate_ledge_rac(2, 8, sym="d4h")
    ban = _build_ban_from_rac(rac, tendq=1.0, dt=0.05, ds=0.1, sym="d4h")
    lying = dataclasses.replace(ban, symmetry="oh")
    with pytest.raises(ValueError, match="declares symmetry 'oh'"):
        assemble_and_diagonalize_in_memory(cowan, rac, lying)
