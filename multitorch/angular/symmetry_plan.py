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
from typing import Dict, List, Mapping, Optional, Tuple

from multitorch.angular.crystal_field import CF_RANKS, cf_invariant_vectors
from multitorch.angular.finite_group import point_group
from multitorch.angular.operator_subduction import dipole_geometries

#: Rank and parity of the valence shell whose irreps are the hybridisation
#: channels: the d orbitals, so rank 2 and gerade.
D_SHELL = (2, +1)


@dataclass(frozen=True)
class GroupConventions:
    """The store's names and orders for one group — looked up, not derived.

    ``cf_operators`` is ``(store name, rank, public cf key, default)`` in store
    order; ``hybr`` is ``(irrep, HYBR string, public lmct key)`` in store order.

    The public key and the default are here because they were previously
    restated at four sites, which is how the Oh path came to accept ``dt`` and
    ``ds`` and silently drop them (residual 43). The default is the value the
    XHAM carries when the caller's ``cf`` dict omits that operator: 1.0 for the
    cubic field, so an unspecified 10Dq is 1 eV rather than no crystal field at
    all, and 0.0 for every distortion, so an unspecified distortion is absent.
    """

    cf_operators: Tuple[Tuple[str, int, str, float], ...]
    hybr: Tuple[Tuple[str, str, str], ...]
    #: Butler label of each dipole component, in the order
    #: :func:`~multitorch.angular.operator_subduction.dipole_geometries`
    #: returns them. A store string like the rest of this class, and the
    #: thing the assembler used to recover by testing ``act_sym == '^0-'``
    #: (WP-B5-7).
    dipole_butler: Tuple[str, ...]


#: ttrac's names and orders. Oh carries a single rank-4 crystal-field operator
#: and two channels; D4h three operators and four channels, in the orders the
#: existing ``OPERATOR_ORDER_D4H`` and ``HYBR_ORDER_D4H`` use.
CONVENTIONS: Dict[str, GroupConventions] = {
    "Oh": GroupConventions(
        cf_operators=(("10DQ", 4, "tendq", 1.0),),
        hybr=(("Eg", "EGHYBR", "eg"), ("T2g", "T2GHYBR", "t2g")),
        dipole_butler=("1-",),                      # T1u
    ),
    # Td. No Fortran reference, and no invented convention either: one cubic
    # crystal-field parameter and one dipole component, exactly as Oh (residual
    # 48 measures the correspondence). **`tendq` is the coefficient of the
    # rank-4 cubic invariant, identically to Oh** (user decision, 2026-10-20) --
    # so `sym='td', tendq=+1.0` reproduces Oh's level ordering, and a physical
    # tetrahedron needs a *negative* value, around -4/9 of the octahedral one
    # for the same ligands. The alternative, flipping the sign internally so
    # positive means "physically tetrahedral", would have been a per-group
    # convention asserted without an oracle, which is residual 39's trap.
    # The channel names follow the irreps the d orbitals span here, E + T2.
    "Td": GroupConventions(
        cf_operators=(("10DQ", 4, "tendq", 1.0),),
        hybr=(("E", "EHYBR", "e"), ("T2", "T2HYBR", "t2")),
        dipole_butler=("^1-",),                     # T2, Butler '^1'
    ),
    # D3d. Three crystal-field parameters of ranks 4, 4, 2 -- the same counts
    # as D4h -- and the `DSIGMA`/`DTAU` convention is **not** invented here:
    # residual 53 pinned it against the published trigonal one-electron
    # energies with zero free parameters, and `ONE_ELECTRON_PATTERNS['D3d']`
    # already carries it. The store order is rank 4 before rank 2, as D4h's
    # is, so `10DQ, DTAU, DSIGMA`.
    #
    # **The three hybridisation channel names are placeholders and say so.**
    # The d orbitals span A1g + Eg + Eg here, a *repeated* irrep that neither
    # Oh nor D4h has, so there is no existing name to carry over and no
    # Fortran deck to read one from. They are unreachable rather than
    # guessed-at: charge transfer beyond Oh/D4h is refused
    # (`generate_ct_ledge_template`), which is the only consumer of these
    # strings, and a test pins that refusal. Whoever implements trigonal CT
    # has to pin them against a reference then -- that is residual 49's
    # neutral-name route, used for exactly the piece that cannot be derived.
    "D3d": GroupConventions(
        cf_operators=(("10DQ", 4, "tendq", 1.0),
                      ("DTAU", 4, "dtau", 0.0),
                      ("DSIGMA", 2, "dsigma", 0.0)),
        hybr=(("A1g", "A1HYBR", "a1"),
              ("Eg", "E1HYBR", "e1"),
              ("Eg", "E2HYBR", "e2")),
        dipole_butler=("1-", "^0-"),            # Eu, A2u -- as D4h, via T1u -> A2 + E
    ),
    "D4h": GroupConventions(
        cf_operators=(("10DQ", 4, "tendq", 1.0), ("DT", 4, "dt", 0.0),
                      ("DS", 2, "ds", 0.0)),
        hybr=(("B1g", "B1HYBR", "b1"), ("A1g", "A1HYBR", "a1"),
              ("B2g", "B2HYBR", "b2"), ("Eg", "EHYBR", "e")),
        dipole_butler=("1-", "^0-"),                # Eu, A2u
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
    cf_operators: Tuple[Tuple[str, int, str, float], ...]
    operator_order: Tuple[str, ...]
    hybr_irreps: Tuple[Tuple[str, int], ...]
    hybr_channels: Tuple[str, ...]
    hybr_parameter_names: Tuple[str, ...]
    dipole_blocks: Tuple[Tuple[str, str, float], ...]
    dipole_butler: Tuple[str, ...]

    @property
    def n_cf_parameters(self) -> int:
        return len(self.cf_operators)

    @property
    def n_operators(self) -> int:
        """What ``assemble``'s ``n_ops = len(xham)`` sees. Not a group identifier."""
        return len(self.operator_order)

    @property
    def cf_parameter_names(self) -> Tuple[str, ...]:
        """The public ``cf`` keys this group accepts, in store order."""
        return tuple(key for _name, _rank, key, _default in self.cf_operators)

    def xham_values(self, cf: Optional[Mapping[str, object]] = None) -> List[object]:
        """The XHAM values for one ``cf`` dict: ``[1.0]`` then one per operator.

        The leading 1.0 is HAMILTONIAN, which is never scaled. Each operator
        takes its value from *cf* under this group's public key, or its default
        when absent. **A key this group has no operator for is an error, not a
        silently dropped parameter** -- which is what three of the four sites
        this replaces used to do in Oh (residual 43). Values pass through
        untouched, so a tensor keeps its gradient.
        """
        cf = dict(cf or {})
        unknown = sorted(set(cf) - set(self.cf_parameter_names))
        if unknown:
            raise ValueError(
                f"unknown crystal-field keys {unknown} for {self.group}; "
                f"allowed {list(self.cf_parameter_names)}")
        return [1.0] + [cf.get(key, default)
                        for _name, _rank, key, default in self.cf_operators]

    @property
    def actor_geometries(self) -> Dict[str, Tuple[str, ...]]:
        """``{Butler actor label: the geometries ttrac writes for it}``.

        What the assembler needs to find a triad's TRANSI block, and the one
        thing it previously recovered by **testing the Butler string itself**
        — ``('PARA', 'MULTIPOLE') if act_sym == '^0-' else ('PERP',
        'MULTIPOLE')`` (WP-B5-7). That rule is correct for Oh and D4h and has
        nowhere to go beyond them: D2h and C2v split the dipole three ways, so
        a third actor would silently take the ``'PERP'`` branch.

        One actor can carry more than one geometry in principle, so the value
        is a tuple in store order, and :func:`_find_transi_block` tries them
        in that order.
        """
        out: Dict[str, Tuple[str, ...]] = {}
        for butler, (_irrep, geometry, _factor) in zip(self.dipole_butler, self.dipole_blocks):
            out[butler] = out.get(butler, ()) + (geometry,)
        return out

    @property
    def dipole_emission(self) -> Tuple[Tuple[str, str, str, float], ...]:
        """``(Butler label, irrep, geometry, isotropic factor)`` per component."""
        return tuple(
            (butler, irrep, geometry, factor)
            for butler, (irrep, geometry, factor) in zip(self.dipole_butler, self.dipole_blocks)
        )

    def hybr_values(self, hybridisation=None) -> List[object]:
        """The XMIX values: one per hybridisation channel, in store order.

        Accepts a dict keyed by this group's public channel names, a single
        value for every channel, a sequence in store order, or ``None`` for no
        hybridisation. An unknown channel name is an error.
        """
        names = self.hybr_parameter_names
        if hybridisation is None:
            return [0.0] * len(names)
        if isinstance(hybridisation, Mapping):
            unknown = sorted(set(hybridisation) - set(names))
            if unknown:
                raise ValueError(
                    f"unknown hybridisation channels {unknown}; "
                    f"{self.group} has {list(names)}")
            return [hybridisation.get(k, 0.0) for k in names]
        if isinstance(hybridisation, (list, tuple)):
            values = list(hybridisation)
            if len(values) != len(names):
                raise ValueError(
                    f"{self.group} needs {len(names)} hybridisation values {list(names)}, "
                    f"got {len(values)}")
            return values
        return [hybridisation] * len(names)      # one value for every channel


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
    named_ranks = sorted(rank for _name, rank, _key, _default in conventions.cf_operators)
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

    # --- dipole: which components is derived, their Butler labels are not ---
    derived_dipole = dipole_geometries(group)
    if len(conventions.dipole_butler) != len(derived_dipole):
        raise RuntimeError(
            f"{group}: the dipole spans {len(derived_dipole)} component(s) "
            f"{[i for i, _g, _f in derived_dipole]} but the convention names "
            f"{len(conventions.dipole_butler)} Butler label(s) "
            f"{list(conventions.dipole_butler)}; they disagree")

    return SymmetryPlan(
        group=group,
        cf_operators=conventions.cf_operators,
        operator_order=(HAMILTONIAN,) + tuple(
            name for name, _rank, _key, _default in conventions.cf_operators),
        hybr_irreps=derived_hybr,
        hybr_channels=tuple(channel for _irrep, channel, _key in conventions.hybr),
        hybr_parameter_names=tuple(key for _irrep, _channel, key in conventions.hybr),
        dipole_blocks=derived_dipole,
        dipole_butler=conventions.dipole_butler,
    )


#: Mapping from the lowercase ``sym`` strings the public API uses to the
#: Schoenflies names :mod:`finite_group` registers. ``calcXAS(sym=...)`` takes
#: the former; everything derived here is keyed by the latter.
SYM_TO_GROUP: Dict[str, str] = {"oh": "Oh", "d4h": "D4h", "td": "Td",
                               "d3d": "D3d"}


def plan_for_sym(sym: str) -> SymmetryPlan:
    """The plan for one of the public API's ``sym`` strings.

    The keys are :data:`SYM_TO_GROUP`'s -- ``'oh'``, ``'d4h'``, ``'td'`` and
    ``'d3d'`` -- and that mapping is the single source of truth the public
    entry points are checked against
    (``test_supported_symmetries_are_documented_everywhere``). A plan says
    nothing about which *electron counts* a group supports: ``'d3d'`` is
    even-only, which `_validate_from_scratch` enforces.
    """
    try:
        group = SYM_TO_GROUP[sym]
    except KeyError:
        raise ValueError(
            f"unknown symmetry {sym!r}; available: {sorted(SYM_TO_GROUP)}") from None
    return symmetry_plan(group)


def plan_for_rac(rac, sym: Optional[str] = None) -> SymmetryPlan:
    """The plan for a generated RAC: from the RAC's own group, else from *sym*.

    The from-scratch generators stamp ``RACFileFull.symmetry``, so a consumer
    holding a generated RAC need not be told the group again — which is what
    WP-B5-4 is for. *sym* remains accepted, because a ``.rme_rac`` parsed from
    a file carries no such field, and a *sym* that **contradicts** the RAC is
    refused rather than silently preferred either way: the two disagreeing is a
    caller bug, and picking a winner would hide it.
    """
    declared = getattr(rac, "symmetry", None)
    if declared is not None and sym is not None and declared != sym:
        raise ValueError(
            f"this RAC was generated for symmetry {declared!r} but {sym!r} was requested; "
            f"they must agree")
    group = declared if declared is not None else sym
    if group is None:
        raise ValueError(
            "this RAC does not record which point group it describes (a parsed .rme_rac "
            "has no such field), so pass sym='oh' or sym='d4h'")
    return plan_for_sym(group)
