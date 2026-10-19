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

**Scope.** Oh and D4h carry named conventions here because they are the two
groups with a Fortran-validated oracle and textbook Ballhausen formulae. Any
other group is usable immediately through :func:`cf_invariant_vectors` and
:func:`cf_operator_from_pattern`, which take a pattern or return the raw
invariant basis; what they deliberately do *not* do is invent coefficients for
the conventional ``(10Dq, Dsigma, Dtau)`` of D3d/C3v or the five-parameter
D2h/C2v sets. Those need a literature reference, and guessing numbers that
merely look plausible is the trap recorded as plan residual 39.
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
ONE_ELECTRON_PATTERNS: Dict[str, Dict[str, Tuple[int, Tuple[float, ...]]]] = {
    "Oh": {
        # (rank, pattern per unit 10Dq)
        "10DQ": (4, (-0.4, -0.4, 0.6, -0.4, 0.6)),
    },
    "D4h": {
        "TENDQ": (4, (-0.4, -0.4, 0.6, -0.4, 0.6)),   # per unit 10Dq
        "DS": (2, (2.0, -1.0, -2.0, -1.0, 2.0)),      # per unit Ds
        "DT": (4, (-1.0, 4.0, -6.0, 4.0, -1.0)),      # per unit Dt
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


def cf_operator_from_pattern(group: str, k: int, pattern, l: int = D_SHELL_L) -> np.ndarray:
    """The rank-k invariant operator whose one-electron matrix is *pattern*.

    *pattern* is the one-electron energy pattern per unit of the named
    parameter: either a length-(2l+1) diagonal in :data:`D_ORBITAL_ORDER`, or a
    full ``(2l+1, 2l+1)`` matrix. It is matched against ``sqrt(2l+1)`` times
    itself, which is the unit-tensor normalisation this package's operator
    vectors carry (see the module docstring).

    Raises if the pattern is not realisable in this group at this rank: that is
    a statement about the convention, not a numerical tolerance, and it is the
    guard that would catch a named parameter being given to a group whose
    symmetry forbids it.
    """
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
        k, pattern = patterns[operator]
    except KeyError:
        raise ValueError(
            f"{group}: unknown crystal-field operator {operator!r}; "
            f"available: {sorted(patterns)}") from None
    return k, cf_operator_from_pattern(group, k, pattern, l)
