"""One object answering "what does this group need?" (WP-B5).

Today two places work out the group from the *length* of a list:

* ``hamiltonian.assemble`` (``n_ops = len(xham)``; 4 means D4h, 2 means Oh),
* ``api.calc`` (``_HYBR_CHANNELS.get(len(ban.xmix[0].values))``).

Both are ambiguous for the groups WP-B targets, and demonstrably so rather than
as a matter of taste. Counting ``HAMILTONIAN`` plus the crystal-field
parameters derived in :mod:`multitorch.angular.crystal_field`:

=====================  =======  ==================================
group                  n_ops    hybridisation channels
=====================  =======  ==================================
Oh                     2        2  (Eg, T2g)
Td                     2        2  (E, T2)
D4h                    4        4  (A1g, B1g, B2g, Eg)
C4v                    4        4  (A1, B1, B2, E)
D3d                    4        3  (A1g, Eg x2)
D2h                    6        5  (Ag x2, B1g, B2g, B3g)
C2v                    6        5  (A1 x2, A2, B1, B2)
=====================  =======  ==================================

so ``n_ops`` cannot tell D4h from C4v or D3d, nor Oh from Td, nor D2h from
C2v — while their operator names, channel names and channel *order* all differ.
A length-based guess would hand D4h's ``DT``/``DS`` and its
``B1/A1/B2/E`` channels to any three-parameter group. :class:`SymmetryPlan`
names them instead.

**What is derived and what is a convention**, the same split WP-B3 and WP-B4
settled on:

* *derived* — how many crystal-field parameters and of which rank
  (:func:`crystal_field.cf_parameter_count`); which irreps the d orbitals span,
  which is the hybridisation channel list (``branching(2, +1)``); which irreps
  the dipole spans and their isotropic weights
  (:func:`operator_subduction.dipole_geometries`).
* *convention* — the store's operator strings (``10DQ``, ``DT``, ``DS``), the
  ``*HYBR`` channel strings, the public ``lmct`` keys, and the **order** of
  each list, because that order is the order of blocks in the RAC file.

:data:`CONVENTIONS` carries the second kind for Oh and D4h, the two groups with
a Fortran oracle, and :func:`symmetry_plan` refuses any other group rather than
inventing names. The derived and named sides are cross-checked on construction,
so a convention that has drifted from the group theory is an error and not a
silently dropped block.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Tuple

from multitorch.angular.crystal_field import CF_RANKS, cf_invariant_vectors
from multitorch.angular.finite_group import point_group
from multitorch.angular.operator_subduction import dipole_geometries

#: Rank and parity of the valence shell whose irreps are the hybridisation
#: channels: the d orbitals, so rank 2 and gerade.
D_SHELL = (2, +1)


@dataclass(frozen=True)
class GroupConventions:
    """The store's names and orders for one group — looked up, not derived.

    ``cf_operators`` is ``(store name, rank)`` in store order; ``hybr`` is
    ``(irrep, HYBR string, public lmct key)`` in store order.
    """

    cf_operators: Tuple[Tuple[str, int], ...]
    hybr: Tuple[Tuple[str, str, str], ...]


#: ttrac's names and orders. Oh carries a single rank-4 crystal-field operator
#: and two channels; D4h three operators and four channels, in the orders the
#: existing ``OPERATOR_ORDER_D4H`` and ``HYBR_ORDER_D4H`` use.
CONVENTIONS: Dict[str, GroupConventions] = {
    "Oh": GroupConventions(
        cf_operators=(("10DQ", 4),),
        hybr=(("Eg", "EGHYBR", "eg"), ("T2g", "T2GHYBR", "t2g")),
    ),
    "D4h": GroupConventions(
        cf_operators=(("10DQ", 4), ("DT", 4), ("DS", 2)),
        hybr=(("B1g", "B1HYBR", "b1"), ("A1g", "A1HYBR", "a1"),
              ("B2g", "B2HYBR", "b2"), ("Eg", "EHYBR", "e")),
    ),
}

#: The operator that is always first in the store, before any crystal field.
HAMILTONIAN = "HAMILTONIAN"


@dataclass(frozen=True)
class SymmetryPlan:
    """What one point group needs, in one place.

    ``operator_order`` and ``hybr_channels`` are exactly what
    ``hamiltonian.assemble`` currently guesses from list lengths, and
    ``hybr_parameter_names`` what ``api.calc`` guesses; ``dipole_blocks`` is
    what the generator had as a two-row literal.
    """

    group: str
    cf_operators: Tuple[Tuple[str, int], ...]
    operator_order: Tuple[str, ...]
    hybr_irreps: Tuple[Tuple[str, int], ...]
    hybr_channels: Tuple[str, ...]
    hybr_parameter_names: Tuple[str, ...]
    dipole_blocks: Tuple[Tuple[str, str, float], ...]

    @property
    def n_cf_parameters(self) -> int:
        return len(self.cf_operators)

    @property
    def n_operators(self) -> int:
        """What ``assemble``'s ``n_ops = len(xham)`` sees. Not a group identifier."""
        return len(self.operator_order)


def _derived_cf_ranks(group: str) -> Tuple[int, ...]:
    """Ranks of the group's crystal-field parameters, with multiplicity."""
    out = []
    for k in CF_RANKS:
        out.extend([k] * cf_invariant_vectors(group, k).shape[1])
    return tuple(out)


def _derived_hybr_irreps(group: str) -> Tuple[Tuple[str, int], ...]:
    """``(irrep, copy)`` the d orbitals span — the hybridisation channels."""
    g = point_group(group)
    out = []
    for irrep, copies in g.branching(*D_SHELL).items():
        out.extend((irrep, copy) for copy in range(copies))
    return tuple(out)


@lru_cache(maxsize=None)
def symmetry_plan(group: str) -> SymmetryPlan:
    """The plan for *group*, derived where possible and named where necessary.

    Raises for a group with no recorded store convention. The derived
    structure is still reachable without one, from
    :func:`crystal_field.cf_parameter_count`,
    :func:`operator_subduction.operator_components` and
    ``point_group(group).branching(2, +1)``.
    """
    try:
        conventions = CONVENTIONS[group]
    except KeyError:
        raise ValueError(
            f"no store convention recorded for point group {group!r}; available: "
            f"{sorted(CONVENTIONS)}. The derived structure does not need one — see "
            f"crystal_field.cf_parameter_count and operator_subduction") from None

    # --- crystal field: the count and ranks are derived, the names are not ---
    derived_ranks = sorted(_derived_cf_ranks(group))
    named_ranks = sorted(rank for _name, rank in conventions.cf_operators)
    if derived_ranks != named_ranks:
        raise RuntimeError(
            f"{group}: the group has crystal-field parameters of ranks {derived_ranks} but "
            f"the convention names ranks {named_ranks}; they disagree")

    # --- hybridisation channels: which irreps is derived, the names are not ---
    derived_hybr = _derived_hybr_irreps(group)
    named_hybr_irreps = tuple(irrep for irrep, _channel, _key in conventions.hybr)
    if sorted(irrep for irrep, _copy in derived_hybr) != sorted(named_hybr_irreps):
        raise RuntimeError(
            f"{group}: the d orbitals span {sorted(i for i, _c in derived_hybr)} but the "
            f"convention names channels for {sorted(named_hybr_irreps)}; they disagree")

    return SymmetryPlan(
        group=group,
        cf_operators=conventions.cf_operators,
        operator_order=(HAMILTONIAN,) + tuple(name for name, _rank in conventions.cf_operators),
        hybr_irreps=derived_hybr,
        hybr_channels=tuple(channel for _irrep, channel, _key in conventions.hybr),
        hybr_parameter_names=tuple(key for _irrep, _channel, key in conventions.hybr),
        dipole_blocks=dipole_geometries(group),
    )


#: Mapping from the lowercase ``sym`` strings the public API uses to the
#: Schoenflies names :mod:`finite_group` registers. ``calcXAS(sym=...)`` takes
#: the former; everything derived here is keyed by the latter.
SYM_TO_GROUP: Dict[str, str] = {"oh": "Oh", "d4h": "D4h"}


def plan_for_sym(sym: str) -> SymmetryPlan:
    """The plan for one of the public API's ``sym`` strings (``'oh'``, ``'d4h'``)."""
    try:
        group = SYM_TO_GROUP[sym]
    except KeyError:
        raise ValueError(
            f"unknown symmetry {sym!r}; available: {sorted(SYM_TO_GROUP)}") from None
    return symmetry_plan(group)
