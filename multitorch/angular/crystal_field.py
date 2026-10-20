"""Symmetry-adapted crystal field by derivation (WP-B3).

The crystal-field Hamiltonian is a Wybourne expansion over spherical tensor
operators,

    H_CF = sum_k sum_q  B_kq T^(k)_q        (k = 2, 4 for d electrons)

and ``rac_generator._build_coupling_operator(J_b, J_k, k, a)`` is exactly that
sum with ``a[q] = B_kq``, so an "operator vector" in this package *is* a row of
Wybourne coefficients in the ``|k, q>`` basis, q = -k..k.

Two things then follow, and together they are the whole module.

**Which coefficients a group allows.** A tensor operator transforms as
``U(g) T^(k)_q U(g)^dag = sum_q' D^(k)_{q'q}(g) T^(k)_{q'}``, so
``H_CF`` is invariant under G exactly when ``D^(k)(g) B_k = B_k`` for every
``g``. That is the statement that ``B_k`` lies in the **totally symmetric
subspace of D^(k) restricted to G** — which :meth:`PointGroup.subduction`
already returns, as the columns labelled with the totally symmetric irrep. No
Butler table, no branching rules written out by hand, nothing tabulated. The
dimension of that subspace is the number of free crystal-field parameters, and
it comes out at the conventional count for every group: 1 for Oh/O/Td (10Dq),
3 for D4h/D4/C4v/D2d and D3d/D3/C3v, 5 for D2h/D2/C2v.

**Which element of that subspace a named parameter is.** The projection fixes
no particular basis inside a multi-dimensional invariant space, so a *name*
like 10Dq or Dt needs one more input, and that input is a convention rather
than group theory: the one-electron d-orbital energies the name is defined by.
Ballhausen's are in :data:`ONE_ELECTRON_PATTERNS`. Requiring the operator's
one-electron matrix to equal ``sqrt(2l+1)`` times that pattern determines the
operator vector uniquely — direction, scale *and* sign at once — as the
solution of a small linear system, which is what :func:`cf_operator_vector`
solves and then verifies.

That last point is the reason this module exists rather than a table. The
hand-coded predecessor, ``rac_generator._d4h_operator_vector_complex``, summed
Butler-weighted Oh-route contributions from the four tabulated coefficients of
``symmetry.D4H_BRANCHES_BY_OPERATOR`` and then needed **two hand-placed gauge
pins** to recover what the Butler weights could not fix: a per-route sign fix
requiring each route's m = 0 component to be positive (plan residual 21 —
without it a flipped LAPACK eigenvector sign mixed the cubic field into Dt by
up to 1.6 eV), and an overall sign pinned by the z^2 matrix element. Solving
against the one-electron pattern leaves no gauge for LAPACK to set, so both
pins disappear rather than being ported.

The ``sqrt(2l+1)`` is the one normalisation constant in the module. It is
measured, not assumed: all three of the hand-coded D4h operators carry exactly
that factor relative to their Ballhausen patterns (residuals <= 5e-15, and no
off-diagonal part at all), which is the unit-tensor normalisation implied by
``_build_coupling_operator``'s Clebsch-Gordan convention.

**A pattern is a matrix, and a matrix needs a frame.** The cubic eg/t2g
pattern is written in the registered axes, where the cubic field is diagonal.
Ballhausen's *axial* parameters -- D4h's Ds and Dt, and the trigonal Dsigma and
Dtau -- are written in a frame whose z is the group's own **principal axis**.
For D4h the two frames coincide and nothing had to be said; for the trigonal
groups, registered with their C3 along [111] since WP-B5-8, they do not, so
each entry in :data:`ONE_ELECTRON_PATTERNS` carries its frame and
:func:`cf_operator_from_pattern` takes a ``frame`` argument. The rotation is
the package's single ``quaternion_carrying_to_z`` convention at rank *l*, not a
transform written here -- residual 47 is what writing a second one costs.

**The trigonal convention, and why it is a derivation with an external pin
rather than a guess (residual 53).** Measured from the Fortran-validated D4h
operators: ``Ds`` is the pure axial rank-2 operator (``B_2,0`` alone,
coefficient ``sqrt(70)``) and ``Dt`` the pure axial rank-4 one (``B_4,0``
alone, ``-sqrt(630)``), while 10Dq is the cubic mix with
``B_4,+-4 / B_4,0 = sqrt(5/14)``. **Dsigma and Dtau are those same two
operators, rotated onto the trigonal axis** -- the same pattern tuples
:data:`AXIAL_RANK_2` and :data:`AXIAL_RANK_4` serve both groups, which is the
whole content of the claim. It reproduces the published trigonal one-electron
energies exactly, in every coefficient:

===============  =============================================
level            E / (published; our construction, identical)
===============  =============================================
``e(e_g)``       ``6Dq + (7/3) Dtau``
``e(t_2g)``      ``-4Dq + Dsigma + (2/3) Dtau``
``a_1(t_2g)``    ``-4Dq - 2Dsigma - 6Dtau``
===============  =============================================

Source: S. Kaizaki, *Ligand Field Theory*, chapter 1 of *Instrumental Analysis
of Coordination Compounds*, vol. 1, eds. H. Oshio and G. N. Newton, Royal
Society of Chemistry, 2024, section 1.2.5, doi 10.1039/9781837674978-00001.
Provenance, tier and caveats are in ``NUMBER_AUDIT.md``.

The table fixes **both** operators uniquely and so fixes the off-diagonal term
it does not itself list: the rank-2 invariant space is one-dimensional, so its
three diagonal entries pin scale and sign with nothing left over, and the map
from the two-dimensional rank-4 space to its three diagonal entries is
injective (Hermiticity cuts an ``n``-complex-dimensional invariant space to
``n`` real dimensions, so ``n`` named operators span it).
``tests/test_hamiltonian/test_cf_one_electron.py`` asserts that injectivity
rather than arguing it, and the ``e(t2g)`` / ``e(e_g)`` mixing -- which the
published table omits -- comes out proportional to ``Dsigma - (5/3) Dtau``.

What is *not* true, and was asserted here before being measured: the three
diagonal energies **together** do not invert to the parameter triple. Every
crystal field is traceless, so they carry only two independent numbers, and the
kernel is ``(10Dq, Dsigma, Dtau) ∝ (-35/9, -20/9, 1)`` -- a field with three
vanishing diagonal energies and an e-mixing of norm 70/9. The convention is
well defined per rank; a *fit* to level energies alone is not determined.

**Scope.** Oh, D4h and D3d carry named conventions here. Oh and D4h have a
Fortran-validated oracle and textbook Ballhausen formulae; D3d has the
published table above and the Oh descent limit
(``tests/test_angular/test_descent_to_oh.py``). D3/C3v share D3d's invariant
spaces exactly (the family argument, residual 49) and so would share these
patterns, but they are not listed until something needs them. Any other group
is usable through :func:`cf_invariant_vectors` and
:func:`cf_operator_from_pattern`, which take a pattern or return the raw
invariant basis; what they deliberately do *not* do is invent coefficients for
the five-parameter D2h/C2v sets. Those need a literature reference, and
guessing numbers that merely look plausible is the trap recorded as plan
residual 39.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

import numpy as np

from multitorch.angular.finite_group import point_group

#: Ranks of the crystal field for d electrons. k is even (the field is gerade)
#: and bounded by 2l, and k = 0 is the configuration-average constant, which is
#: carried by the HAMILTONIAN blocks rather than here.
CF_RANKS: Tuple[int, ...] = (2, 4)

#: Orbital angular momentum of the shell the named conventions are defined on.
D_SHELL_L = 2

#: Real-spherical-harmonic order of ``rac_generator._c2r_unitary(2)``, which is
#: m = -2..2. Read off the eg/t2g pattern of the cubic operator rather than
#: assumed: ``test_crystal_field.py`` pins it.
D_ORBITAL_ORDER: Tuple[str, ...] = ("xy", "yz", "z2", "xz", "x2-y2")

#: One-electron d-orbital energies per unit of each named parameter, in the
#: order of :data:`D_ORBITAL_ORDER`. These are conventions, not group theory:
#: they are what fix a *name* to one element of the invariant subspace.
#:
#: Ballhausen's D4h definitions, with Dq = 10Dq/10:
#:
#:   E(xy)      = -4Dq + 2Ds -  Dt        E(z2)      =  6Dq - 2Ds - 6Dt
#:   E(xz)      = -4Dq -  Ds + 4Dt        E(x2-y2)   =  6Dq + 2Ds -  Dt
#:   E(yz)      = -4Dq -  Ds + 4Dt
#:
#: Oh is the Dt = Ds = 0 restriction, i.e. E(eg) = 6Dq, E(t2g) = -4Dq. Every
#: pattern is traceless, as it must be: the trace is the k = 0 term, which is
#: not part of the crystal field.
#: Frames a pattern can be written in. ``"cubic"`` is the group's registered
#: axes; ``"axial"`` is a frame whose z is the group's principal axis. They
#: coincide for every group whose principal axis is already z, which is every
#: group with a ttrac convention except the trigonal ones.
FRAMES: Tuple[str, ...] = ("cubic", "axial")

#: The cubic eg/t2g pattern per unit 10Dq, in the registered axes.
CUBIC_RANK_4: Tuple[float, ...] = (-0.4, -0.4, 0.6, -0.4, 0.6)

#: Ballhausen's axial rank-2 pattern, in the axial frame: ``Ds`` in D4h and
#: ``Dsigma`` in D3d are **the same operator** in their own frames, which is why
#: one tuple serves both. ``sqrt(70) B_2,0`` and nothing else.
AXIAL_RANK_2: Tuple[float, ...] = (2.0, -1.0, -2.0, -1.0, 2.0)

#: Ballhausen's axial rank-4 pattern: ``Dt`` in D4h, ``Dtau`` in D3d.
#: ``-sqrt(630) B_4,0`` and nothing else.
AXIAL_RANK_4: Tuple[float, ...] = (-1.0, 4.0, -6.0, 4.0, -1.0)

ONE_ELECTRON_PATTERNS: Dict[str, Dict[str, Tuple[int, Tuple[float, ...], str]]] = {
    "Oh": {
        # (rank, pattern, frame)
        "10DQ": (4, CUBIC_RANK_4, "cubic"),
    },
    "D4h": {
        "TENDQ": (4, CUBIC_RANK_4, "cubic"),
        "DS": (2, AXIAL_RANK_2, "axial"),
        "DT": (4, AXIAL_RANK_4, "axial"),
    },
    # D3d. The axial patterns are D4h's, in the trigonal frame; the cubic one
    # is Oh's, in the registered (octahedral) axes. See the module docstring
    # for the published table this reproduces and ``NUMBER_AUDIT.md`` for its
    # provenance. Store strings and public keys are *not* set here -- that is
    # ``symmetry_plan.CONVENTIONS``, and D3d has no entry there yet.
    "D3d": {
        "TENDQ": (4, CUBIC_RANK_4, "cubic"),
        "DSIGMA": (2, AXIAL_RANK_2, "axial"),
        "DTAU": (4, AXIAL_RANK_4, "axial"),
    },
}

#: Tolerance on the invariance and the pattern solve. The subduction itself is
#: verified at ~1e-15, and the measured pattern residuals are <= 5e-15.
CF_TOL = 1e-11


@lru_cache(maxsize=None)
def totally_symmetric_irrep(group: str) -> str:
    """Label of the trivial irrep, found by its defining property.

    The character that is 1 on every class, rather than ``irrep_labels[0]``.
    The two agree for every group here — the row order guarantees it — but the
    property is what actually defines "totally symmetric", and relying on a
    position means a labelling change silently selects the wrong subspace.
    That is not hypothetical: T used to come out ``['Aa', 'Ab', 'Ac', 'T']``
    with the all-ones row *last*, because the complex-conjugate-pair families
    get their a/b/c suffixes in character-table order.
    """
    g = point_group(group)
    chi = g.character_table
    hits = [i for i in range(chi.shape[0])
            if np.allclose(chi[i], np.ones(g.n_classes), atol=1e-9)]
    if len(hits) != 1:
        raise RuntimeError(
            f"{group}: found {len(hits)} totally symmetric irreps, expected exactly one")
    return g.irrep_labels[hits[0]]


def _one_electron_matrix(k: int, op_vec: np.ndarray, l: int = D_SHELL_L) -> np.ndarray:
    """One-electron matrix of a rank-k operator vector, in the real-SH basis.

    Imported lazily: ``rac_generator`` imports a great deal of the angular
    layer, and this module is also used from tests that want nothing else.
    """
    from multitorch.angular.rac_generator import _operator_real_matrix

    return _operator_real_matrix(float(l), float(l), k, op_vec)


def _one_electron_matrix_complex(k: int, op_vec: np.ndarray, l: int = D_SHELL_L) -> np.ndarray:
    """One-electron matrix in the complex ``|l, m>`` basis — **no real extraction**.

    This, and not :func:`_one_electron_matrix`, is what the pattern solve must
    be built from. ``_operator_real_matrix`` keeps only ``Re(U O U^dag)`` (or
    ``-Im``, by the parity rule), which throws away exactly the information
    that makes the solve independent of the gauge the subduction happened to
    pick: multiply an invariant basis vector by a complex phase and the real
    part no longer determines it.

    That is not hypothetical. The invariant basis comes out of ``eigh``, which
    returns *real* vectors here when left alone (measured max imaginary part
    4.4e-16), so a real-extracted solve looks correct — and then the LAPACK
    scramble harness, which applies complex phases, broke four integration
    tests. Worse, with a phase of ``exp(0.7i)`` the answer was wrong by 21 in a
    vector of norm 25 while the solve's own residual stayed at 5.3e-15: a
    silently wrong result, not a caught one.

    Solved in the complex basis instead, a phase on a basis vector is absorbed
    by its coefficient — ``_build_coupling_operator`` is linear in the operator
    vector, so ``v -> v * phi`` gives ``c -> c / phi`` and ``c * v`` is
    unchanged — and so is any unitary mixing inside the invariant subspace.
    """
    from multitorch.angular.point_group import _build_coupling_operator

    return _build_coupling_operator(float(l), float(l), k, op_vec)


@lru_cache(maxsize=None)
def cf_invariant_vectors(group: str, k: int) -> np.ndarray:
    """Totally symmetric subspace of ``D^(k)`` restricted to *group*.

    Returns a ``(2k+1, n)`` array whose orthonormal columns are the allowed
    Wybourne coefficient vectors ``B_k`` in the ``|k, q>`` basis, q = -k..k.
    ``n`` is the number of free crystal-field parameters of that rank, so for a
    d shell ``n(k=2) + n(k=4)`` is the group's total.

    The basis *inside* that subspace is arbitrary when ``n > 1`` — the
    projection has no reason to prefer one rotation of it. Use
    :func:`cf_operator_vector` when a specific named parameter is wanted.
    """
    if k % 2:
        raise ValueError(f"the crystal field is gerade, so k must be even, got {k}")
    g = point_group(group)
    S, labels = g.subduction(k, +1)
    cols = [i for i, (irrep, _copy, _partner) in enumerate(labels)
            if irrep == totally_symmetric_irrep(group)]
    V = np.ascontiguousarray(S[:, cols]) if cols else np.zeros((2 * k + 1, 0), dtype=complex)

    # The defining property, checked rather than trusted: D^(k)(g) B = B for
    # every element. This is what makes these the *allowed* coefficients.
    from multitorch.angular.finite_group import wigner_from_quaternion

    for e in g.elements:
        D = wigner_from_quaternion(float(k), e.quat)
        if e.parity < 0:
            D = D * (+1)          # even k: the field is gerade either way
        if V.shape[1] and not np.allclose(D @ V, V, atol=CF_TOL):
            raise RuntimeError(
                f"{group}: a rank-{k} vector the subduction labelled totally symmetric "
                f"is not invariant under one of the group elements")

    # The result is cached, so hand back something a caller cannot mutate: an
    # in-place edit would otherwise poison every later call for this group.
    V.flags.writeable = False
    return V


def cf_parameter_count(group: str, ranks: Tuple[int, ...] = CF_RANKS) -> Dict[int, int]:
    """Number of free crystal-field parameters of *group*, by rank."""
    return {k: cf_invariant_vectors(group, k).shape[1] for k in ranks}


@lru_cache(maxsize=None)
def axial_frame_rotation(group: str, l: int = D_SHELL_L) -> np.ndarray:
    """Real-spherical-harmonic rotation ``D`` for shell *l* onto the axial frame.

    ``M_axial = D @ M_registered @ D.conj().T``, so a pattern given in the
    axial frame is brought back to the registered axes by ``D.conj().T @ M @ D``.

    Built from :func:`~multitorch.angular.finite_group.quaternion_carrying_to_z`
    on the group's own ``principal_axis``, which is the package's one such
    convention (residual 47). For a group whose principal axis is already z
    this is the identity, which is why D4h's ``Ds``/``Dt`` entries can be
    labelled ``"axial"`` without changing a single number.
    """
    from multitorch.angular.finite_group import (point_group as _pg,
                                                 quaternion_carrying_to_z,
                                                 wigner_from_quaternion)
    from multitorch.angular.point_group import _c2r_unitary

    axis = np.asarray(_pg(group).principal_axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    if abs(axis[2] - 1.0) < 1e-12:
        # Already z. Returned as the *exact* identity rather than the matrix the
        # rotation machinery produces, which differs from it by 2.2e-16 through
        # the complex-to-real round trip and would shift D4h's Fortran-validated
        # operator vectors by ~3e-14 for no reason. So labelling D4h's Ds and Dt
        # "axial" costs exactly nothing, and the rotating path is exercised only
        # where the axis really is off z.
        D = np.eye(2 * l + 1, dtype=complex)
    else:
        quat = quaternion_carrying_to_z(axis)
        U = _c2r_unitary(l)
        D = U @ wigner_from_quaternion(float(l), quat) @ U.conj().T
    D.flags.writeable = False
    return D


def cf_operator_from_pattern(group: str, k: int, pattern, l: int = D_SHELL_L,
                             *, frame: str = "cubic") -> np.ndarray:
    """The rank-k invariant operator whose one-electron matrix is *pattern*.

    *pattern* is the one-electron energy pattern per unit of the named
    parameter: either a length-(2l+1) diagonal in :data:`D_ORBITAL_ORDER`, or a
    full ``(2l+1, 2l+1)`` matrix. It is matched against ``sqrt(2l+1)`` times
    itself, which is the unit-tensor normalisation this package's operator
    vectors carry (see the module docstring).

    *frame* says which axes *pattern* is written in -- see :data:`FRAMES`. An
    ``"axial"`` pattern is rotated back to the registered axes by
    :func:`axial_frame_rotation` before the solve, so the returned operator
    vector is always in the registered basis, like every other vector here.

    Raises if the pattern is not realisable in this group at this rank: that is
    a statement about the convention, not a numerical tolerance, and it is the
    guard that would catch a named parameter being given to a group whose
    symmetry forbids it -- **including a pattern handed over in the wrong
    frame**, which for a trigonal group is not a near miss but a different
    operator.
    """
    if frame not in FRAMES:
        raise ValueError(f"unknown frame {frame!r}; available: {list(FRAMES)}")
    V = cf_invariant_vectors(group, k)
    if V.shape[1] == 0:
        raise ValueError(
            f"{group}: rank {k} carries no totally symmetric invariant, so it has no "
            f"crystal-field parameter of that rank")

    target = np.asarray(pattern, dtype=np.float64)
    dim = 2 * l + 1
    if target.shape == (dim,):
        target = np.diag(target)
    if target.shape != (dim, dim):
        raise ValueError(f"pattern must be ({dim},) or ({dim}, {dim}), got {target.shape}")
    if frame == "axial":
        D = axial_frame_rotation(group, l)
        target = np.asarray(D.conj().T @ target.astype(np.complex128) @ D)
        if np.max(np.abs(target.imag)) > CF_TOL:
            raise RuntimeError(
                f"{group}: rotating a real axial pattern out of the axial frame left an "
                f"imaginary part of {np.max(np.abs(target.imag)):.3e}")
        target = target.real
    target = target * np.sqrt(2.0 * l + 1.0)

    # Rotate the target into the complex |l, m> basis and solve there, so that
    # the answer does not depend on the gauge the subduction picked. See
    # :func:`_one_electron_matrix_complex` for why the real-SH form will not do.
    from multitorch.angular.point_group import _c2r_unitary

    U = _c2r_unitary(l)
    target_complex = U.conj().T @ target.astype(np.complex128) @ U

    # Each invariant's one-electron matrix, flattened into the columns of a
    # least-squares problem. The map is injective on the invariant subspace, so
    # the solve is exact whenever the convention is realisable.
    basis = np.column_stack([_one_electron_matrix_complex(k, V[:, i], l).ravel()
                             for i in range(V.shape[1])])
    coeffs, *_ = np.linalg.lstsq(basis, target_complex.ravel(), rcond=None)
    residual = float(np.max(np.abs(basis @ coeffs - target_complex.ravel())))
    scale = max(1.0, float(np.max(np.abs(target))))
    if residual > CF_TOL * scale:
        raise ValueError(
            f"{group}: the requested one-electron pattern is not a rank-{k} invariant of "
            f"this group (residual {residual:.3e}); the convention and the symmetry disagree")

    op_vec = V @ coeffs

    # The solve is only as good as its own claim: rebuild the pattern.
    check = _one_electron_matrix(k, op_vec, l)
    if not np.allclose(check, target, atol=CF_TOL * scale):
        raise RuntimeError(
            f"{group}: rank-{k} operator vector does not reproduce its own target pattern")
    return op_vec


def cf_operator_vector(group: str, operator: str, l: int = D_SHELL_L) -> Tuple[int, np.ndarray]:
    """``(rank, operator vector)`` for a named crystal-field parameter.

    Replaces ``rac_generator._d4h_operator_vector_complex`` and the Butler
    table behind it: the vector is solved from the group's invariant subspace
    and the convention in :data:`ONE_ELECTRON_PATTERNS`, with no tabulated
    branching coefficient and no gauge pin.
    """
    try:
        patterns = ONE_ELECTRON_PATTERNS[group]
    except KeyError:
        raise ValueError(
            f"no named crystal-field convention for {group!r}; available: "
            f"{sorted(ONE_ELECTRON_PATTERNS)}. Use cf_invariant_vectors or "
            f"cf_operator_from_pattern for a group with no named set here") from None
    try:
        k, pattern, frame = patterns[operator]
    except KeyError:
        raise ValueError(
            f"{group}: unknown crystal-field operator {operator!r}; "
            f"available: {sorted(patterns)}") from None
    return k, cf_operator_from_pattern(group, k, pattern, l, frame=frame)
