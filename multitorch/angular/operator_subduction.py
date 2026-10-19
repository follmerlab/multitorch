"""Transition-operator components adapted to a point group (WP-B4).

Where :mod:`multitorch.angular.crystal_field` needs only the *totally
symmetric* part of a rank-k tensor, a transition operator needs all of it: the
dipole spans a whole irrep of G (or several), and each of those pieces is a
separate RAC block with its own geometry label and its own weight. This module
is the general version, and the crystal field is its totally symmetric special
case — :func:`operator_components` with ``k = 2, parity = +1`` returns the CF
invariants among its components, which a test asserts.

**What is derived.** The partner-adapted components of the rank-k, parity-p
operator are the columns of :meth:`PointGroup.subduction(k, p)`, grouped by
irrep and copy. Each component's weight in an isotropic (powder) spectrum is
``sqrt(d / (2k+1))``: the ``2k+1`` operator components divide among the irreps
in proportion to their dimensions, so the weights satisfy ``sum w^2 = 1``
exactly, which is the statement that no intensity is created or lost by
splitting the operator up.

For the dipole (``k = 1``, ``parity = -1``) that reproduces the two numbers the
Oh and D4h generators had hard-coded, and the comments that went with them:

====  ===========================  ==========================================
Oh    ``T1u``, d = 3               ``w = sqrt(3/3) = 1`` — one MULTIPOLE block
D4h   ``A2u`` d = 1, ``Eu`` d = 2  ``w = sqrt(1/3)`` (PARA), ``sqrt(2/3)`` (PERP)
====  ===========================  ==========================================

and it keeps going on its own: D2h and C2v split the dipole into **three**
one-dimensional components, because x, y and z are mutually inequivalent there,
so a biaxial group needs three geometries rather than two.

**What the group also fixes, and is worth stating because it is the physics.**
In every uniaxial group here (D4h, C4v, D3d, D2h, C2v) the one-dimensional
dipole component comes out lying *purely* on ``q = 0`` and the two-dimensional
one purely on ``q = +-1``. That is what makes "PARA" mean z-polarised and
"PERP" mean in-plane: it is not a labelling convention, it is forced by the
symmetry, and :func:`polarization_axis` reads it off rather than assuming it.
Oh is the exception and correctly so — nothing distinguishes a direction there,
so its three T1u partners each mix all three ``q``.

**What is *not* derived: the geometry names.** ``'MULTIPOLE'``, ``'PERP'`` and
``'PARA'`` are ttrac's strings, and which irrep carries which is a convention
of the Fortran store rather than a fact about the group. :data:`GEOMETRY_NAMES`
records them for Oh and D4h, the two groups with a Fortran oracle, and
:func:`dipole_geometries` refuses a group that has none instead of inventing
one. Same split as WP-B3: derive the structure, look up only the names.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

from multitorch.angular.finite_group import point_group

#: Rank and parity of the operators this module is used for. The dipole is a
#: polar vector, so odd parity; the quadrupole (K pre-edge) is even.
DIPOLE = (1, -1)
QUADRUPOLE = (2, +1)

#: Tolerance for reading a component's polarisation off its operator vectors.
POLARIZATION_TOL = 1e-9

#: ttrac's geometry strings by group, **in the order the generator emits the
#: blocks**. A *convention of the Fortran store* on both counts — the string and
#: the order — so both are looked up rather than derived, and only for the two
#: groups whose RAC output is validated against Fortran. Oh writes one
#: ``MULTIPOLE`` block (the dipole is a single irrep there); D4h splits it, and
#: plan residual 35 records that ttrac labels the Oh blocks ``MULTIPOLE``
#: rather than ``PERP``/``PARA``.
#:
#: The order matters and is not cosmetic: it is the order of the TRANSI blocks
#: in the RAC file, which WP-B5's "byte-equal RAC output" oracle compares. The
#: subduction lists D4h's components as ``A2u`` then ``Eu``; the generator emits
#: ``Eu`` (PERP) first, so that is the order recorded here.
GEOMETRY_NAMES: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "Oh": (("T1u", "MULTIPOLE"),),
    "D4h": (("Eu", "PERP"), ("A2u", "PARA")),
}


@dataclass(frozen=True)
class OperatorComponent:
    """One irrep's worth of a rank-k operator, with its partner basis.

    ``vectors`` is ``(2k+1, dim)``: column ``mu`` is partner ``mu``'s operator
    vector in the ``|k, q>`` basis, q = -k..k, ready to hand to
    ``rac_generator._build_coupling_operator``.
    """

    irrep: str
    copy: int
    dim: int
    weight: float
    vectors: np.ndarray

    def __post_init__(self) -> None:
        if self.vectors.shape[1] != self.dim:
            raise ValueError(
                f"{self.irrep} copy {self.copy}: {self.vectors.shape[1]} partner vectors "
                f"for a {self.dim}-dimensional irrep")


def isotropic_weight(dim: int, k: int) -> float:
    """``sqrt(dim / (2k+1))`` — a component's share of the isotropic spectrum.

    The ``2k+1`` components of the operator divide among the irreps in
    proportion to their dimensions, so these weights square to 1 over a
    group's components. For the dipole in D4h that is exactly the generators'
    ``PERP_FACTOR = sqrt(2/3)`` and ``PARA_FACTOR = sqrt(1/3)``.
    """
    return math.sqrt(dim / (2.0 * k + 1.0))


@lru_cache(maxsize=None)
def operator_components(group: str, k: int, parity: int) -> Tuple[OperatorComponent, ...]:
    """Partner-adapted components of the rank-k, parity-p operator under *group*.

    Returns one :class:`OperatorComponent` per (irrep, copy), in the order
    :meth:`PointGroup.subduction` lists them.
    """
    g = point_group(group)
    S, labels = g.subduction(float(k), int(parity))
    dims = {label: int(g.irrep_dims[i]) for i, label in enumerate(g.irrep_labels)}

    grouped: Dict[Tuple[str, int], List[int]] = {}
    for column, (irrep, copy, _partner) in enumerate(labels):
        grouped.setdefault((irrep, copy), []).append(column)

    out: List[OperatorComponent] = []
    for (irrep, copy), columns in grouped.items():
        vectors = np.ascontiguousarray(S[:, columns])
        vectors.flags.writeable = False          # cached: do not let callers edit
        out.append(OperatorComponent(irrep=irrep, copy=copy, dim=dims[irrep],
                                     weight=isotropic_weight(dims[irrep], k),
                                     vectors=vectors))

    total = sum(c.dim for c in out)
    if total != 2 * k + 1:
        raise RuntimeError(
            f"{group}: rank-{k} parity {parity:+d} components have {total} partners in all, "
            f"not the operator's {2 * k + 1}")
    sum_w2 = sum(c.weight ** 2 for c in out)
    if abs(sum_w2 - 1.0) > 1e-12:
        raise RuntimeError(
            f"{group}: rank-{k} component weights square to {sum_w2!r}, not 1; the operator's "
            f"intensity is not conserved by the split")
    return tuple(out)


def polarization_axis(component: OperatorComponent) -> Optional[str]:
    """``'z'``, ``'xy'`` or ``None`` for a dipole component, read off its vectors.

    ``'z'`` when the component lives entirely on ``q = 0`` and ``'xy'`` when it
    lives entirely on ``q = +-1``; ``None`` when it mixes them, which is the
    honest answer for a group in which no axis is special — Oh's single ``T1u``
    is exactly that case. This is measured, not assumed: it is what justifies
    calling the one-dimensional component of a uniaxial group the
    z-polarised (PARA) channel.
    """
    if component.vectors.shape[0] != 3:
        raise ValueError("polarization_axis is defined for the rank-1 (dipole) operator")
    q0 = float(np.max(np.abs(component.vectors[1, :])))
    qpm = float(np.max(np.abs(component.vectors[[0, 2], :])))
    if qpm <= POLARIZATION_TOL:
        return "z"
    if q0 <= POLARIZATION_TOL:
        return "xy"
    return None


def dipole_geometries(group: str) -> Tuple[Tuple[str, str, float], ...]:
    """``(irrep, geometry, weight)`` per dipole block, in ttrac's emission order.

    The weights and the component structure are derived; the geometry string
    and the block order come from :data:`GEOMETRY_NAMES`. Raises for a group
    with no recorded ttrac convention rather than inventing a name — the
    structure is still available from :func:`operator_components`.

    The two sides are cross-checked: the named irreps and the group's actual
    dipole components must be the same set, so a convention that has drifted
    from the group theory is refused rather than silently dropping a block.
    """
    try:
        names = GEOMETRY_NAMES[group]
    except KeyError:
        raise ValueError(
            f"no ttrac geometry convention recorded for {group!r}; available: "
            f"{sorted(GEOMETRY_NAMES)}. The component structure is still available from "
            f"operator_components({group!r}, 1, -1)") from None

    by_irrep = {c.irrep: c for c in operator_components(group, *DIPOLE)}
    named = [irrep for irrep, _geometry in names]
    if set(named) != set(by_irrep) or len(named) != len(by_irrep):
        raise RuntimeError(
            f"{group}: GEOMETRY_NAMES names {named} but the dipole subduces to "
            f"{sorted(by_irrep)}; the convention and the group theory disagree")
    return tuple((irrep, geometry, by_irrep[irrep].weight) for irrep, geometry in names)
