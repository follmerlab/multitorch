"""Numerical finite point groups: elements, classes, character table (WP-B1).

One construction for any finite subgroup G of O(3), replacing the hand-written
Oh tables of :mod:`~multitorch.angular.point_group` and the Oh→D4h second step
of :mod:`~multitorch.angular.symmetry`. Nothing here is tabulated: a group is
given by generators, and its elements, conjugacy classes, irreducible
characters and Mulliken labels are all computed.

**Elements are quaternions, so the double cover comes for free.** A proper
rotation is a unit quaternion ``q``, and ``q`` and ``-q`` are the same rotation
of R³ but different elements of SU(2) — which is exactly the distinction a
half-integer-J manifold sees. ``PointGroup`` therefore carries one element per
``±q`` pair and ``DoubleGroup`` carries both, and the single group's character
table is the double group's restricted to the integer-J irreps. Improper
operations are a rotation times inversion, kept as an explicit ``parity`` flag
rather than the ``'g'``/``'u'`` suffix the Oh code bolts on afterwards.

**The character table comes from the class-sum (Dixon) method**, which needs no
irrep matrices: the class multiplication matrices M_i commute, their
simultaneous eigenvalues are the central characters λ_i^μ = |C_i| χ^μ(g_i)/d_μ,
and the dimensions follow from orthogonality as
d_μ = √(|G| / Σ_i |λ_i^μ|²/|C_i|). Each M_i is (number of classes)² — a 10×10
matrix for Oh — so this is cheap and exact to ~1e-12.

Oracles (``tests/test_angular/test_finite_group.py``): group orders and class
counts; Σ d² = |G| and both character orthogonality relations; the character
table of O against the 24 matrices of
:func:`~multitorch.angular.point_group.octahedral_rotations`, built
independently and validated against Fortran; the isomorphisms O ≅ Td, D4 ≅ C4v
and D3 ≅ C3v, whose character tables must agree up to a relabelling of classes;
and the SO(3) branching Σ_μ mult·d_μ = 2J+1 for every J, with G = Oh
reproducing :func:`~multitorch.angular.point_group.oh_branching` exactly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Elements closer than this in quaternion/parity space are the same element.
#: Generators are exact products of π/2 and 2π/3 rotations, so the closure
#: accumulates only rounding error; the observed spread over Oh's 96 double-group
#: elements is ~1e-15.
ELEMENT_TOL = 1e-9

#: Character and class-sum arithmetic tolerance. The class multiplication
#: matrices are integer-valued, so their eigenvalues are algebraic numbers
#: computed to ~1e-13 here; characters round to that.
CHARACTER_TOL = 1e-7


# ─────────────────────────────────────────────────────────────
# Quaternions
# ─────────────────────────────────────────────────────────────

def quaternion(axis: Sequence[float], angle: float) -> np.ndarray:
    """Unit quaternion (w, x, y, z) of a rotation by ``angle`` about ``axis``."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    return np.concatenate([[math.cos(angle / 2.0)], math.sin(angle / 2.0) * a])


def quaternion_product(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions."""
    pw, pv = p[0], p[1:]
    qw, qv = q[0], q[1:]
    return np.concatenate([[pw * qw - pv @ qv], pw * qv + qw * pv + np.cross(pv, qv)])


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """SO(3) matrix of a unit quaternion. ``q`` and ``-q`` give the same matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """A unit quaternion of a proper rotation matrix (sign convention arbitrary).

    Picks the largest component first so the division never amplifies noise.
    """
    t = np.trace(R)
    if t > -0.5:
        w = 0.5 * math.sqrt(max(1.0 + t, 0.0))
        s = 0.25 / max(w, 1e-30)
        q = np.array([w, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s,
                      (R[1, 0] - R[0, 1]) * s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        u = 0.5 * math.sqrt(max(1.0 + R[i, i] - R[j, j] - R[k, k], 0.0))
        s = 0.25 / max(u, 1e-30)
        q = np.empty(4)
        q[0] = (R[k, j] - R[j, k]) * s
        q[1 + i] = u
        q[1 + j] = (R[j, i] + R[i, j]) * s
        q[1 + k] = (R[k, i] + R[i, k]) * s
    return q / np.linalg.norm(q)


@dataclass(frozen=True)
class Element:
    """One element of a subgroup of O(3): a quaternion and a parity.

    ``parity`` is ``+1`` for a proper rotation and ``-1`` for a rotation
    composed with inversion, so the O(3) matrix is ``parity * R(q)``. Inversion
    is central and squares to the identity, which is why it factors out as a
    sign and why the double cover only has to be tracked in ``q``.
    """
    q: Tuple[float, float, float, float]
    parity: int

    @property
    def quat(self) -> np.ndarray:
        return np.asarray(self.q, dtype=np.float64)

    @property
    def matrix(self) -> np.ndarray:
        """The 3x3 O(3) matrix."""
        return self.parity * quaternion_to_matrix(self.quat)

    @property
    def rotation(self) -> np.ndarray:
        """The proper-rotation part, with inversion divided out."""
        return quaternion_to_matrix(self.quat)

    @property
    def angle(self) -> float:
        """Rotation angle in [0, 2pi] of the quaternion (not of ±q)."""
        return 2.0 * math.acos(min(1.0, max(-1.0, float(self.q[0]))))

    @property
    def axis(self) -> np.ndarray:
        v = self.quat[1:]
        n = np.linalg.norm(v)
        return v / n if n > ELEMENT_TOL else np.array([0.0, 0.0, 1.0])

    def __mul__(self, other: "Element") -> "Element":
        q = quaternion_product(self.quat, other.quat)
        return Element(tuple(q), self.parity * other.parity)

    def inverse(self) -> "Element":
        w, v = self.q[0], self.quat[1:]
        return Element((w, -v[0], -v[1], -v[2]), self.parity)


def _key(e: Element, *, spinor: bool) -> Tuple:
    """Hashable identity of an element, to ``ELEMENT_TOL``.

    With ``spinor=False`` the quaternion is folded so that ``q`` and ``-q``
    collapse: that is the single group, where a 2pi rotation is the identity.
    With ``spinor=True`` they stay apart, which is the double group.
    """
    q = np.asarray(e.q, dtype=np.float64)
    if not spinor:
        # canonical sign: first component beyond tolerance decides
        for c in q:
            if abs(c) > ELEMENT_TOL:
                if c < 0:
                    q = -q
                break
    r = np.round(q / ELEMENT_TOL).astype(np.int64)
    # -0 and 0 must hash alike
    r = r + 0
    return (int(e.parity),) + tuple(int(x) for x in r)


def _close(generators: Sequence[Element], *, spinor: bool,
           max_order: int = 400) -> List[Element]:
    """Elements of the group generated by ``generators``, identity first."""
    identity = Element((1.0, 0.0, 0.0, 0.0), 1)
    found = {_key(identity, spinor=spinor): identity}
    order = [identity]
    frontier = [identity]
    gens = list(generators)
    if spinor:
        # a double group must contain the 2pi rotation, which is -1 in SU(2);
        # it is generated automatically whenever a generator has even order, but
        # adding it explicitly makes the closure independent of that accident
        gens = gens + [Element((-1.0, 0.0, 0.0, 0.0), 1)]
    while frontier:
        nxt = []
        for a in frontier:
            for g in gens:
                c = a * g
                k = _key(c, spinor=spinor)
                if k not in found:
                    found[k] = c
                    order.append(c)
                    nxt.append(c)
                    if len(order) > max_order:
                        raise ValueError(
                            f"group closure exceeded {max_order} elements; "
                            f"the generators probably do not generate a finite group")
        frontier = nxt
    return order


# ─────────────────────────────────────────────────────────────
# The group object
# ─────────────────────────────────────────────────────────────

class PointGroup:
    """A finite subgroup of O(3), built from generators.

    Parameters
    ----------
    name : str
        Schoenflies name, used only for reporting and for the Mulliken
        labelling conventions that depend on the principal axis.
    generators : sequence of Element
        Any generating set; the closure is taken.
    spinor : bool
        ``False`` for the single group (``q`` and ``-q`` identified, order
        |G|), ``True`` for the double group (order 2|G|), whose extra
        irreducible characters are the half-integer-J ones.
    principal_axis : array-like, optional
        The axis Mulliken A/B and the subscripts are referred to. Defaults to
        z, which is the standard setting for every group this module builds.
    """

    def __init__(self, name: str, generators: Sequence[Element], *,
                 spinor: bool = False, principal_axis=(0.0, 0.0, 1.0)):
        self.name = name
        self.spinor = spinor
        self.principal_axis = np.asarray(principal_axis, dtype=np.float64)
        self.principal_axis = self.principal_axis / np.linalg.norm(self.principal_axis)
        self.elements: List[Element] = _close(generators, spinor=spinor)
        self._index = {_key(e, spinor=spinor): i for i, e in enumerate(self.elements)}
        self._verify_group()

    # ---- structure ----

    @property
    def order(self) -> int:
        return len(self.elements)

    def index(self, e: Element) -> int:
        k = _key(e, spinor=self.spinor)
        if k not in self._index:
            raise KeyError(f"{self.name}: element is not in the group")
        return self._index[k]

    def _verify_group(self) -> None:
        """Closure, identity and inverses, on the hot path rather than assumed.

        WP-B2a's lesson, applied one level up: fix every free choice and then
        check it. A generator set that silently closes to the wrong group would
        otherwise surface as a wrong spectrum many layers away.
        """
        n = self.order
        if n == 0:
            raise ValueError(f"{self.name}: empty group")
        for i, a in enumerate(self.elements):
            if _key(a.inverse(), spinor=self.spinor) not in self._index:
                raise ValueError(f"{self.name}: element {i} has no inverse in the group")
        # closure, by sampling the full multiplication table
        for a in self.elements:
            for b in self.elements:
                if _key(a * b, spinor=self.spinor) not in self._index:
                    raise ValueError(f"{self.name}: not closed under multiplication")

    @cached_property
    def multiplication_table(self) -> np.ndarray:
        """``t[i, j]`` is the index of ``elements[i] * elements[j]``."""
        n = self.order
        t = np.empty((n, n), dtype=np.int64)
        for i, a in enumerate(self.elements):
            for j, b in enumerate(self.elements):
                t[i, j] = self.index(a * b)
        return t

    @cached_property
    def classes(self) -> List[List[int]]:
        """Conjugacy classes as lists of element indices, identity class first."""
        n = self.order
        seen = [False] * n
        out: List[List[int]] = []
        for i in range(n):
            if seen[i]:
                continue
            orbit = set()
            for g in self.elements:
                c = g * self.elements[i] * g.inverse()
                orbit.add(self.index(c))
            for j in orbit:
                seen[j] = True
            out.append(sorted(orbit))
        out.sort(key=lambda c: (0 if 0 in c else 1, len(c), c[0]))
        return out

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @cached_property
    def class_of(self) -> np.ndarray:
        """``class_of[i]`` is the class index of element ``i``."""
        out = np.empty(self.order, dtype=np.int64)
        for c, members in enumerate(self.classes):
            for i in members:
                out[i] = c
        return out

    # ---- characters ----

    @cached_property
    def class_multiplication_matrices(self) -> np.ndarray:
        """``M[i][j, k]``: the number of ``(x, y)`` in ``C_i x C_j`` with ``xy = g_k``.

        ``g_k`` is any fixed representative of class ``k``; the count does not
        depend on which, which is what makes these matrices well defined and
        what the Dixon method rests on.
        """
        r = self.n_classes
        table = self.multiplication_table
        reps = [c[0] for c in self.classes]
        M = np.zeros((r, r, r), dtype=np.int64)
        for i in range(r):
            for j in range(r):
                # a_{ijk}: how many (x, y) in C_i x C_j multiply to the fixed
                # representative g_k. Counting products once and bucketing by
                # "is this g_k" is O(|C_i||C_j|) rather than r times that.
                where = {reps[k]: k for k in range(r)}
                for x in self.classes[i]:
                    row = table[x]
                    for y in self.classes[j]:
                        k = where.get(int(row[y]))
                        if k is not None:
                            M[i, j, k] += 1
        return M

    @cached_property
    def _raw_character_table(self) -> np.ndarray:
        """Irreducible characters, shape ``(n_irreps, n_classes)``, complex.

        Rows are ordered by dimension and then by the labelling convention of
        :attr:`irrep_labels`; the first row is always the totally symmetric
        irrep. ``n_irreps == n_classes`` by construction.
        """
        r = self.n_classes
        M = self.class_multiplication_matrices
        sizes = np.array([len(c) for c in self.classes], dtype=np.float64)

        # The M_i commute, so one generic real combination separates every
        # simultaneous eigenspace; its eigenvectors are the central characters'
        # common eigenvectors. A fixed seed keeps this reproducible, and the
        # result is checked by orthogonality below, so a degenerate draw cannot
        # pass silently.
        rng = np.random.default_rng(20261018)
        for _ in range(16):
            weights = rng.normal(size=r)
            A = np.tensordot(weights, M.astype(np.float64), axes=(0, 0))  # (r, r)
            # omega_i^mu v_j = sum_k a_{ijk} v_k, i.e. M_i v = omega_i v, so the
            # central-character vectors are RIGHT eigenvectors of A
            evals, evecs = np.linalg.eig(A)
            # the whole complex eigenvalue, not just its real part: the abelian
            # groups (C3, C4, T) have conjugate pairs that share a real part
            keys = {(int(round(z.real / 1e-6)), int(round(z.imag / 1e-6))) for z in evals}
            if len(keys) == r:
                break
        else:
            raise RuntimeError(f"{self.name}: could not separate the class-sum eigenspaces")

        rows = []
        for t in range(r):
            v = evecs[:, t]
            # central characters: lambda_i = |C_i| chi(g_i) / d, normalised so
            # that the identity class (always index 0, size 1) gives lambda_0 = 1
            lam = v / v[0]
            d = math.sqrt(self.order / float(np.sum(np.abs(lam) ** 2 / sizes).real))
            rows.append(d * lam / sizes)
        chi = np.array(rows, dtype=np.complex128)

        # clean up and verify both orthogonality relations
        chi[np.abs(chi) < CHARACTER_TOL] = 0.0
        chi = np.where(np.abs(chi.imag) < CHARACTER_TOL, chi.real + 0j, chi)
        gram = (chi * sizes) @ chi.conj().T / self.order
        if not np.allclose(gram, np.eye(r), atol=1e-6):
            raise RuntimeError(f"{self.name}: irreducible characters are not orthonormal")
        col = chi.conj().T @ chi / self.order
        if not np.allclose(col * sizes[:, None], np.eye(r), atol=1e-6):
            raise RuntimeError(f"{self.name}: character columns are not orthogonal")

        dims = np.round(chi[:, 0].real).astype(np.int64)
        if int(np.sum(dims ** 2)) != self.order:
            raise RuntimeError(
                f"{self.name}: dimensions {sorted(dims)} sum of squares "
                f"{int(np.sum(dims ** 2))} != order {self.order}")

        # Orthonormality and Burnside are not enough. Measured on 2026-10-18:
        # corrupting a_{1,1,1} or a_{4,4,4} of D4h's class algebra gives a table
        # that is wrong in every row and still passes both of them. The check
        # that closes it asks whether these are characters *of this group*:
        # every tensor product must decompose into non-negative integers,
        # m[mu,nu,la] = (1/|G|) sum_i |C_i| chi_mu chi_nu conj(chi_la). It uses
        # only the class sizes, which come from the multiplication table rather
        # than from the class-sum coefficients, so it is independent of them.
        m = np.einsum("i,ui,vi,wi->uvw", sizes, chi, chi, chi.conj()) / self.order
        if np.max(np.abs(m.imag)) > 1e-6:
            raise RuntimeError(f"{self.name}: tensor-product multiplicities are not real")
        m = m.real
        if np.max(np.abs(m - np.round(m))) > 1e-6 or np.min(m) < -1e-6:
            worst = np.unravel_index(int(np.argmax(np.abs(m - np.round(m)))), m.shape)
            raise RuntimeError(
                f"{self.name}: these are not characters of the group -- the tensor "
                f"product of irreps {worst[0]} and {worst[1]} contains irrep "
                f"{worst[2]} {m[worst]:.6f} times, which is not a non-negative integer")
        order_idx = np.lexsort((-chi[:, 0].real, dims))
        return chi[order_idx]

    @property
    def character_table(self) -> np.ndarray:
        """Irreducible characters in :attr:`irrep_labels` order, ``(n_irreps, n_classes)``."""
        return self._raw_character_table[self.row_order]

    @property
    def irrep_dims(self) -> np.ndarray:
        """Irrep dimensions, aligned with :attr:`irrep_labels`."""
        return np.round(self.character_table[:, 0].real).astype(np.int64)

    @cached_property
    def _two_pi_class(self) -> Optional[int]:
        """Class of the 2pi rotation, the element that separates spinor irreps.

        It exists only in a double group, where chi(2pi) = -d for a
        half-integer-J (spinor) irrep and +d for an integer-J one. That is the
        sharp, computable version of "which irreps are the extra ones".
        """
        if not self.spinor:
            return None
        target = Element((-1.0, 0.0, 0.0, 0.0), 1)
        k = _key(target, spinor=True)
        if k not in self._index:
            return None
        return int(self.class_of[self._index[k]])

    @cached_property
    def spinor_irreps(self) -> List[bool]:
        """Per irrep (in label order): is this a half-integer-J representation?"""
        c = self._two_pi_class
        if c is None:
            return [False] * len(self.irrep_dims)
        chi = self.character_table
        return [bool(chi[t, c].real < 0) for t in range(chi.shape[0])]

    # ---- SO(3) branching ----

    def so3_character(self, J, parity: int = +1) -> np.ndarray:
        """Character of the (2J+1)-dimensional O(3) rep on each class.

        ``parity`` is the rep's own parity: ``+1`` for ``g`` (even l, or a
        pseudo-tensor), ``-1`` for ``u``. An improper group element contributes
        ``parity`` times the rotation character, which is the whole of the
        ``'g'``/``'u'`` bookkeeping the Oh code carries as a label suffix.

        For a half-integer J this needs the *spinor* group: chi depends on the
        quaternion, not just the rotation, and ``q``/``-q`` differ in sign.
        """
        two_j = int(round(2 * J))
        if two_j % 2 and not self.spinor:
            raise ValueError(
                f"{self.name}: J = {J} is half-integer and needs the double group "
                f"(PointGroup(..., spinor=True))")
        out = np.empty(self.n_classes, dtype=np.float64)
        for c, members in enumerate(self.classes):
            e = self.elements[members[0]]
            theta = e.angle
            # chi^J(theta) = sum_{m=-J}^{J} exp(i m theta), real
            ms = np.arange(-two_j / 2.0, two_j / 2.0 + 0.5, 1.0)
            out[c] = float(np.sum(np.cos(ms * theta)))
            if e.parity < 0:
                out[c] *= parity
        return out

    def branching(self, J, parity: int = +1) -> Dict[str, int]:
        """Multiplicity of each irrep in the O(3) rep of angular momentum ``J``.

        ``n_mu = (1/|G|) sum_g chi^J(g) conj(chi^mu(g))``, evaluated by class.
        """
        sizes = np.array([len(c) for c in self.classes], dtype=np.float64)
        chi_j = self.so3_character(J, parity)
        chi = self.character_table
        mult = (chi.conj() * sizes * chi_j).sum(axis=1).real / self.order
        rounded = np.round(mult).astype(np.int64)
        if np.max(np.abs(mult - rounded)) > 1e-6:
            raise RuntimeError(
                f"{self.name}: non-integer branching multiplicities for J = {J}: {mult}")
        total = int(np.sum(rounded * self.irrep_dims))
        if total != int(round(2 * J)) + 1:
            raise RuntimeError(
                f"{self.name}: branching of J = {J} spans {total} states, not {int(round(2*J)) + 1}")
        return {label: int(n) for label, n in zip(self.irrep_labels, rounded) if n}

    # ---- labels ----

    # ---- labels ----

    @cached_property
    def _probes(self) -> Dict[str, object]:
        """Classes the Mulliken rules refer to, and which family's rules apply.

        Found by geometry, not tabulated: the family follows from how many
        distinct C3 axes and proper C2 axes the group has, and each family names
        the classes whose characters carry its subscripts.
        """
        cls = self.classes
        reps = [self.elements[c[0]] for c in cls]
        z = self.principal_axis

        def first(pred):
            for c, e in enumerate(reps):
                if pred(e):
                    return c
            return None

        def rot_angle(e) -> float:
            """Rotation angle in [0, pi] of the O(3) operation, ignoring the spinor sign.

            Mulliken's rules are statements about geometry, so they must not
            see the difference between q and -q: a C4 stored as -q has
            ``e.angle`` 3pi/2, and inversion carries the 2pi quaternion.
            """
            return min(e.angle, 2 * math.pi - e.angle)

        def proper(e):
            return e.parity > 0 and rot_angle(e) > ELEMENT_TOL

        def turn(e):
            """Rotation order n of the O(3) operation, or None if not an exact 2pi/n."""
            a = rot_angle(e)
            if a < ELEMENT_TOL:
                return None
            n = 2 * math.pi / a
            return int(round(n)) if abs(n - round(n)) < 1e-6 else None

        # distinct proper C3 axes, as unsigned directions
        c3_axes, c2_axes = [], []
        for e in self.elements:
            if not proper(e):
                continue
            n = turn(e)
            if n == 3:
                bucket = c3_axes
            elif n == 2:
                bucket = c2_axes
            else:
                continue
            if not any(abs(abs(a @ e.axis) - 1.0) < 1e-6 for a in bucket):
                bucket.append(e.axis)

        inversion = first(lambda e: e.parity < 0 and rot_angle(e) < ELEMENT_TOL)
        on_axis = lambda e: abs(abs(e.axis @ z) - 1.0) < 1e-6
        perp = lambda e: abs(e.axis @ z) < 1e-6

        # The highest-order axis decides the family. An improper S_n with
        # n >= 3 counts (D2d and the S4 family have no proper axis above C2 but
        # are still axial); S2 is a mirror, not an axis.
        axis_order = 1
        for e in self.elements:
            n = turn(e)
            if n is None:
                continue
            if e.parity > 0 or n >= 3:
                axis_order = max(axis_order, n)
        if len(c3_axes) > 1:
            family = "cubic"
        elif axis_order >= 3:
            family = "axial"
        elif len(c2_axes) >= 2:
            family = "biaxial"
        else:
            family = "axial"

        out: Dict[str, object] = {"family": family, "inversion": inversion}

        if family == "cubic":
            # the order-4 class: proper C4 in O/Oh, improper S4 in Td
            out["quad"] = (first(lambda e: proper(e) and turn(e) == 4)
                           or first(lambda e: e.parity < 0 and turn(e) == 4))
        elif family == "biaxial":
            # D2/D2h: B1, B2, B3 are keyed to C2(z), C2(y), C2(x) in that order
            axes = [z, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])]
            if abs(abs(z @ axes[1]) - 1.0) < 1e-6:          # non-standard setting
                axes = [z] + [a for a in axes[1:] if abs(abs(z @ a) - 1.0) > 1e-6]
            out["c2_chain"] = [first(lambda e, a=a: proper(e) and turn(e) == 2
                                     and abs(abs(e.axis @ a) - 1.0) < 1e-6) for a in axes]
        else:
            # highest-order proper rotation about the principal axis
            best, principal = 0, None
            for c, e in enumerate(reps):
                if proper(e) and on_axis(e) and (turn(e) or 0) > best:
                    best, principal = turn(e), c
            out["principal"] = principal
            # An S_2n axis outranks it for A/B only in a group with no
            # inversion -- D2d and the S4 family, where A/B is an S4 label.
            # With inversion present (D3d's S6) the improper part is already
            # carried by g/u and using it here would mislabel A1u as B.
            if inversion is None:
                alt, alt_order = None, best
                for c, e in enumerate(reps):
                    if e.parity < 0 and on_axis(e) and (turn(e) or 0) > alt_order:
                        alt_order, alt = turn(e), c
                if alt is not None:
                    out["principal"] = alt
            # 1/2 subscript: a C2 perpendicular to the axis, else a vertical mirror
            out["secondary"] = (first(lambda e: proper(e) and turn(e) == 2 and perp(e))
                                or first(lambda e: e.parity < 0 and turn(e) == 2 and perp(e)))
        return out

    @cached_property
    def _labelling(self) -> Tuple[List[str], bool, List[int]]:
        """Mulliken labels, derived from dimension and character signature.

        Letter from the dimension (1 -> A or B, 2 -> E, 3 -> T, 4 -> G), the
        1/2 subscript from the class its family keys it to (C4 or S4 for a
        cubic group, a perpendicular C2 or vertical mirror for an axial one),
        B1/B2/B3 from which C2 is symmetric in D2/D2h, and a g/u suffix from
        inversion. Rows come out in the conventional order: totally symmetric
        first, then g before u, then by dimension, letter and subscript.

        Where a family's rules cannot separate two irreps -- the
        complex-conjugate pairs of C3, C4, C4h, T and Th, which Mulliken
        distinguishes with primes this does not attempt -- the labels get an
        ``a``/``b`` suffix in character-table order and
        :attr:`labels_are_conventional` is ``False``. The character table is
        the authority either way; the label is a convenience.
        """
        chi = self._raw_character_table
        dims = np.round(chi[:, 0].real).astype(np.int64)
        p = self._probes
        family, inversion = p["family"], p["inversion"]

        def ch(c: Optional[int], t: int) -> float:
            return 0.0 if c is None else float(chi[t, c].real)

        stems: List[Tuple[str, str]] = []            # (letter, subscript)
        for t in range(len(dims)):
            d = int(dims[t])
            letter = {1: "A", 2: "E", 3: "T", 4: "G"}[d] if d in (1, 2, 3, 4) else f"D{d}"
            sub = ""
            if family == "cubic":
                if d in (1, 3) and p.get("quad") is not None:
                    sub = "1" if ch(p["quad"], t) > CHARACTER_TOL else "2"
            elif family == "biaxial":
                chain = p["c2_chain"]
                if d == 1:
                    signs = [ch(c, t) > CHARACTER_TOL for c in chain]
                    if all(signs):
                        letter = "A"
                    else:
                        letter = "B"
                        sub = str(1 + signs.index(True)) if any(signs) else ""
            else:
                if d == 1:
                    letter = "A" if ch(p.get("principal"), t) > -CHARACTER_TOL else "B"
                    if p.get("secondary") is not None:
                        sub = "1" if ch(p["secondary"], t) > CHARACTER_TOL else "2"
            stems.append((letter, sub))

        parity = ["" if inversion is None else ("g" if ch(inversion, t) > 0 else "u")
                  for t in range(len(dims))]

        # A double group's extra irreps are named for the lowest half-integer J
        # that contains them -- E1/2, E5/2, G3/2 in Oh, which is exactly the
        # naming point_group.OH_DOUBLE_IRREP_DIM already uses. A spinor irrep is
        # one the 2pi rotation represents as -1, so the test is sharp rather
        # than a guess about which 2-dimensional irrep is which.
        if self.spinor and self._two_pi_class is not None:
            sizes = np.array([len(c) for c in self.classes], dtype=np.float64)
            two_pi = self._two_pi_class
            for t in range(len(dims)):
                if chi[t, two_pi].real >= 0:
                    continue                       # an integer-J irrep
                p_rep = -1 if parity[t] == "u" else +1
                for two_j in range(1, 32, 2):
                    chi_j = self.so3_character(two_j / 2.0, p_rep)
                    mult = float((chi[t].conj() * sizes * chi_j).sum().real) / self.order
                    if abs(mult) > 1e-6:
                        # 1- and 2-dimensional spinor irreps are both "E" in the
                        # double-group convention (a 1-dimensional one is one of
                        # a complex-conjugate pair, written 1E and 2E; the
                        # prefix is attached below), 4-dimensional ones are G.
                        letter = {1: "E", 2: "E", 4: "G"}.get(
                            int(dims[t]), f"D{int(dims[t])}")
                        stems[t] = (letter, f"{two_j}/2")
                        break
                else:
                    raise RuntimeError(
                        f"{self.name}: spinor irrep {t} appears in no J <= 15/2")

        labels = [a + b + c for (a, b), c in zip(stems, parity)]

        # A complex-conjugate pair of one-dimensional spinor irreps shares a
        # stem; the double-group convention distinguishes them as 1E and 2E.
        if self.spinor:
            groups: Dict[str, List[int]] = {}
            for t, lab in enumerate(labels):
                if int(dims[t]) == 1 and stems[t][1].endswith("/2"):
                    groups.setdefault(lab, []).append(t)
            for lab, members in groups.items():
                if len(members) == 2:
                    for k, t in enumerate(members):
                        labels[t] = f"{k + 1}{lab}"

        conventional = len(set(labels)) == len(labels)
        if not conventional:
            seen: Dict[str, int] = {}
            out = []
            for lab in labels:
                if labels.count(lab) > 1:
                    seen[lab] = seen.get(lab, 0) + 1
                    stem, suffix = (lab[:-1], lab[-1]) if lab[-1:] in ("g", "u") else (lab, "")
                    lab = f"{stem}{chr(ord('a') + seen[lab] - 1)}{suffix}"
                out.append(lab)
            labels = out

        # conventional row order
        letter_rank = {"A": 0, "B": 1, "E": 2, "T": 3, "G": 4}

        def sub_key(t) -> float:
            sub = stems[t][1]
            if not sub:
                return 0.0
            return float(sub.split("/")[0]) / 2.0 if "/" in sub else float(sub)

        order = sorted(range(len(labels)), key=lambda t: (
            0 if parity[t] != "u" else 1,
            1 if (self.spinor and chi[t, self._two_pi_class].real < 0
                  if self._two_pi_class is not None else False) else 0,
            int(dims[t]),
            letter_rank.get(stems[t][0][0], 9),
            sub_key(t),
            labels[t],
        ))
        return [labels[t] for t in order], conventional, order

    @property
    def irrep_labels(self) -> List[str]:
        """Mulliken labels in conventional row order (see :attr:`_labelling`)."""
        return self._labelling[0]

    @property
    def labels_are_conventional(self) -> bool:
        """False when a family's rules could not separate two irreps.

        True for every group WP-B targets. False for the complex-conjugate
        pairs of C3, C4, C4h, T and Th, where Mulliken uses primes this does
        not attempt; those labels carry an ``a``/``b`` suffix instead.
        """
        return self._labelling[1]

    @property
    def row_order(self) -> List[int]:
        """Permutation taking :attr:`character_table` rows to label order."""
        return self._labelling[2]

    def __repr__(self) -> str:
        kind = "double" if self.spinor else "single"
        return (f"PointGroup({self.name!r}, {kind}, order={self.order}, "
                f"classes={self.n_classes}, irreps={list(self.irrep_labels)})")


# ─────────────────────────────────────────────────────────────
# Generators of the groups WP-B needs
# ─────────────────────────────────────────────────────────────

_X, _Y, _Z = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
_DIAG, _XY = (1.0, 1.0, 1.0), (1.0, 1.0, 0.0)


def _rot(axis, n: int, k: int = 1) -> Element:
    """Proper rotation by ``k`` turns of ``2pi/n`` about ``axis``."""
    return Element(tuple(quaternion(axis, 2 * math.pi * k / n)), +1)


def _improper(axis, n: int, k: int = 1) -> Element:
    """``inversion x rotation``: a mirror for n=2 (normal ``axis``), S_n generally."""
    return Element(tuple(quaternion(axis, 2 * math.pi * k / n)), -1)


#: Inversion, written out rather than as ``_improper(z, 1)``: the latter builds
#: the 2pi quaternion (w = -1), which is the same O(3) operation but reports a
#: rotation angle of 2pi, and anything reading the quaternion angle would miss it.
INVERSION = Element((1.0, 0.0, 0.0, 0.0), -1)


#: Generators by Schoenflies name, in the standard setting (principal axis z).
#: A mirror plane with normal ``a`` is inversion times C2(a), so ``sigma_h`` is
#: ``_improper(z, 2)`` and a vertical ``sigma_v`` with normal x is
#: ``_improper(x, 2)``.
GENERATORS: Dict[str, Tuple[Element, ...]] = {
    # cubic
    "O":   (_rot(_Z, 4), _rot(_DIAG, 3)),
    "Oh":  (_rot(_Z, 4), _rot(_DIAG, 3), INVERSION),      # + inversion
    "Td":  (_rot(_Z, 2), _rot(_DIAG, 3), _improper(_Z, 4)),      # S4 about z
    "T":   (_rot(_Z, 2), _rot(_DIAG, 3)),
    "Th":  (_rot(_Z, 2), _rot(_DIAG, 3), INVERSION),
    # tetragonal
    "D4":  (_rot(_Z, 4), _rot(_X, 2)),
    "D4h": (_rot(_Z, 4), _rot(_X, 2), INVERSION),
    "C4v": (_rot(_Z, 4), _improper(_X, 2)),                      # sigma_v normal x
    "C4h": (_rot(_Z, 4), INVERSION),
    "C4":  (_rot(_Z, 4),),
    "D2d": (_rot(_Z, 2), _rot(_X, 2), _improper(_Z, 4)),
    # trigonal
    "D3":  (_rot(_Z, 3), _rot(_X, 2)),
    "D3d": (_rot(_Z, 3), _rot(_X, 2), INVERSION),
    "C3v": (_rot(_Z, 3), _improper(_X, 2)),
    "C3":  (_rot(_Z, 3),),
    # orthorhombic and lower
    "D2":  (_rot(_Z, 2), _rot(_X, 2)),
    "D2h": (_rot(_Z, 2), _rot(_X, 2), INVERSION),
    "C2v": (_rot(_Z, 2), _improper(_X, 2)),
    "C2h": (_rot(_Z, 2), INVERSION),
    "C2":  (_rot(_Z, 2),),
    "Ci":  (INVERSION,),
    "C1":  (),
}

#: Groups whose principal axis is not z (none at present, kept explicit so a
#: future non-standard setting cannot be introduced silently).
PRINCIPAL_AXIS: Dict[str, Tuple[float, float, float]] = {}


@lru_cache(maxsize=None)
def point_group(name: str, *, spinor: bool = False) -> PointGroup:
    """The named point group, cached. ``spinor=True`` gives the double group."""
    if name not in GENERATORS:
        raise KeyError(f"unknown point group {name!r}; known: {sorted(GENERATORS)}")
    return PointGroup(name, GENERATORS[name], spinor=spinor,
                      principal_axis=PRINCIPAL_AXIS.get(name, (0.0, 0.0, 1.0)))
