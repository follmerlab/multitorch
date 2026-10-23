"""
Gauge-fixed double-group partner bases for half-integer J (WP-B2a).

The projected emitters of :mod:`~multitorch.angular.rac_generator` build each
RAC coefficient as ⟨v_b|O|v_k⟩ between partner-0 basis vectors of an Oh irrep.
For integer J those vectors are real (:func:`~multitorch.angular.point_group._real_subduction_matrix`).
The double-group irreps E1/2, E5/2 and G3/2 are pseudo-real: they have no real
form, the projection returns complex vectors, and a RAC coefficient must
nevertheless be a real number.

It can be, and this module constructs the basis that makes it so:

1. **Partner-0 vector per copy** from the complex projection
   (:func:`~multitorch.angular.point_group._complex_subduction_matrix_half_int`).
2. **Canonical copies.** Where an irrep occurs more than once in D^J the
   projection returns an arbitrary basis of the multiplicity space (whatever
   the host's eigensolver produced). The copies are re-expressed in the
   eigenbasis of the rank-4 cubic operator inside that space and ordered by
   its eigenvalue, which is a property of the physics rather than of LAPACK.
3. **One phase per copy**, fixed by walking a spanning tree of the non-zero
   *scalar* couplings (ranks 0, 2, 4 — the Hamiltonian and crystal field) from
   a single seed, within each manifold.

Step 3 is what makes those coefficients real, and it is not obvious that it
can: the phases must come out consistent around every closed loop. They do —
every cycle phase is ±1 — so after the walk every scalar coupling is real,
verified to ~1e-13 for the ions of the bundled fixtures. :func:`gauged_basis`
re-checks it for each system it builds and raises if a residue survives, so a
system where the argument fails cannot pass silently.

**Scope.** Steps 1–3 cover the operators that are scalars in the point group:
the configuration Hamiltonian and the crystal field, whose coefficient is a
single number per (bra copy, ket copy). The dipole needs one step more. For
half-integer J the triad (Γ_bra, T1u, Γ_ket) can carry two independent
invariants — Γ8 ⊗ T1u ⊗ Γ8 does — so a transition needs one reduced element
per invariant rather than one number, which is why ttrac lists such a triad
twice in the .ban (the repeated coupling the assembler learned to read in the
residual-14 fix). :func:`real_invariant_coefficients` splits a triad's partner
matrix elements over those channels (group Wigner–Eckart), and
:meth:`GaugedBasis.operator_tensor` supplies them.

The overall sign of each vector is still free (a spanning tree fixes phases up
to the seed's own phase and to ±1 per vector); that freedom cancels in the
spectrum, as the seed's does.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from multitorch.angular.point_group import (
    OH_DOUBLE_IRREP_DIM,
    _build_coupling_operator,
    _complex_subduction_matrix_half_int,
    oh_branching,
)

DOUBLE_IRREPS = ("E1/2", "E5/2", "G3/2")
# D4h double group: Γ6 ↓ E1/2, Γ7 ↓ E3/2, Γ8 ↓ E1/2 ⊕ E3/2
D4H_DOUBLE_IRREPS = ("E1/2", "E3/2")
D4H_DOUBLE_IRREP_DIM = {"E1/2": 2, "E3/2": 2}


def target_double_irrep_dim(irrep_stem: str, target_group: str) -> int:
    """Dimension of a double-group irrep, by the table *target_group* uses.

    Oh's ``G3/2`` is four-dimensional and has no D4h counterpart under that
    name, so the two tables are genuinely different and the emitter cannot
    share one (WP-B5-5b step 3).

    The two tabulated groups keep their tables -- they are what the Fortran
    parity runs through -- and any other group is asked of ``point_group``,
    whose spinor irreps carry the parity suffix this function's callers have
    already stripped. **This is the only place the half-integer path needed
    widening for a target with a one-dimensional double irrep** (residual 62):
    everywhere a double-irrep dimension is used downstream it is used *as a
    variable* -- ``sqrt(dim / (2J + 1))`` in the operator coefficient,
    ``sub[:, copy * dim:(copy + 1) * dim]`` in the partner slice,
    ``range(dim)`` in the layout -- so nothing there assumes 2.
    """
    from multitorch.angular.finite_group import point_group
    from multitorch.angular.point_group import OH_DOUBLE_IRREP_DIM

    if target_group == "D4h":
        return D4H_DOUBLE_IRREP_DIM[irrep_stem]
    if target_group == "Oh":
        return OH_DOUBLE_IRREP_DIM[irrep_stem]

    group = point_group(target_group, spinor=True)
    dims = {label[:-1] if label[-1] in "gu" else label: int(group.irrep_dims[i])
            for i, is_spinor in enumerate(group.spinor_irreps) if is_spinor
            for label in (group.irrep_labels[i],)}
    if irrep_stem not in dims:
        raise ValueError(
            f"{target_group} has no double-group irrep {irrep_stem!r}; "
            f"known: {sorted(dims)}")
    return dims[irrep_stem]
REALITY_TOL = 1e-9

# Node of the gauge graph: (manifold, J, irrep, copy). ``manifold`` is any
# hashable label; vectors of different manifolds live in different J spaces
# and are coupled only by the odd-rank (dipole) operator.
Node = Tuple[str, float, str, int]


def is_half_integer(J: float) -> bool:
    return abs(float(J) - round(float(J))) > 0.1


def _marker_operator(J: float) -> np.ndarray:
    """Rank-4 cubic operator on D^J, used to order the copies of an irrep."""
    from multitorch.angular.rac_generator import _d4h_operator_vector_complex

    return _build_coupling_operator(J, J, 4, _d4h_operator_vector_complex("TENDQ", 4))


def canonical_partner_sets(J: float, irrep: str) -> np.ndarray:
    """All partners of every copy of ``irrep`` in D^J, in a canonical copy order.

    Returns a ``(2J+1, multiplicity, dim_irrep)`` complex array indexed
    ``[m, copy, partner]``. Copies are the eigenbasis of the rank-4 cubic
    operator within the multiplicity space, ordered by eigenvalue, so the
    result does not depend on the host eigensolver's choice inside a degenerate
    projection subspace. The rank-4 operator is a point-group scalar, so it acts
    identically on every partner and the same copy mixing applies to all of
    them, which is what keeps the partners a basis of one irrep.
    """
    dim = OH_DOUBLE_IRREP_DIM[irrep]
    B = _complex_subduction_matrix_half_int(float(J), irrep)
    W = B.reshape(B.shape[0], B.shape[1] // dim, dim)    # columns are copy-major
    if W.shape[1] <= 1:
        return np.ascontiguousarray(W)
    M = W[:, :, 0].conj().T @ _marker_operator(float(J)) @ W[:, :, 0]
    _, U = np.linalg.eigh(0.5 * (M + M.conj().T))
    return np.ascontiguousarray(np.einsum('mcp,cd->mdp', W, U))


def canonical_partner_vectors(J: float, irrep: str) -> np.ndarray:
    """Partner-0 vector of every copy of ``irrep`` in D^J, ``(2J+1, multiplicity)``."""
    return np.ascontiguousarray(canonical_partner_sets(J, irrep)[:, :, 0])


def gauge_factor(Jb: float, Jk: float, k: int) -> complex:
    """i^(J_k − J_b) for even rank, one factor of i more for odd rank.

    The integer-J emitters apply the same rotation as the real sign
    (−1)^⌊(J_k−J_b)/2⌋ for even rank and ``_dipole_gauge_sign`` for the dipole;
    here it is kept complex because the vectors are complex.
    """
    return 1j ** int(round(Jk - Jb + (1 if k % 2 else 0)))


@dataclass
class GaugedBasis:
    """Gauge-fixed partner vectors of one calculation, keyed by :data:`Node`.

    ``partners[node]`` is ``(2J+1, dim_irrep)``, column p the p-th partner;
    ``vectors[node]`` is its partner-0 column, which is all the scalar
    operators need.
    """

    partners: Dict[Node, np.ndarray]
    nodes: List[Node] = field(default_factory=list)
    residual: float = 0.0          # largest relative |Im| found by the verification
    # phase per (Oh irrep, D4h irrep) subduction route; empty unless d4h=True
    route_phase: Dict[Tuple[str, str], complex] = field(default_factory=dict)
    vectors: Dict[Node, np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        self.vectors = {n: v[:, 0] for n, v in self.partners.items()}

    def target_partners(self, node: Node, target_irrep: str, copy: int = 0, *,
                        target_group: str = "D4h") -> np.ndarray:
        """Partner vectors of one Oh node, subduced onto *target_group*.

        The general form of :meth:`d4h_partners` (WP-B5-5b step 3), which is
        what lets one half-integer operator emitter serve Oh as well as D4h.
        With Oh as its own target the subduction is the identity and the route
        phase is 1, so this returns ``self.partners[node]`` -- whose column 0 is
        :attr:`vectors`, which is exactly what the retained
        ``_make_oh_op_adds_half`` projects onto through
        :meth:`coefficient`. Measured bit-identical: 54 cases over both
        manifolds, 868 ADD entries, max difference 0.0.

        **No route phase is needed for the Oh target and none is demanded.**
        An Oh calculation builds its basis with ``d4h=False``, so
        ``route_phase`` is empty; requiring it here would make the Oh route
        unreachable. The requirement is specific to D4h, where DT and DS are
        D4h scalars but not Oh ones (residual 31), and it stays on that branch.
        """
        if target_group == "Oh":
            return self.partners[node]
        if target_group != "D4h":
            raise ValueError(f"unknown subduction target {target_group!r}; "
                             f"expected 'Oh' or 'D4h'")
        if not self.route_phase:
            raise RuntimeError("this basis was built without d4h=True, so the subduction "
                               "routes carry no pinned phase and the D4h crystal field "
                               "would not be real")
        sub = oh_to_d4h_double_subduction(node[2])[target_irrep]
        dim = D4H_DOUBLE_IRREP_DIM[target_irrep]
        cols = sub[:, copy * dim:(copy + 1) * dim] * self.route_phase[(node[2], target_irrep)]
        return np.ascontiguousarray(self.partners[node] @ cols)

    def d4h_partners(self, node: Node, d4h_irrep: str, copy: int = 0) -> np.ndarray:
        """:meth:`target_partners` with the D4h target. Kept for its callers."""
        return self.target_partners(node, d4h_irrep, copy, target_group="D4h")

    def copies(self, manifold: str, J: float, irrep: str) -> int:
        return sum(1 for n in self.nodes if n[0] == manifold and n[1] == J and n[2] == irrep)

    def coefficient(self, a: Node, b: Node, k: int, op_vec: np.ndarray) -> float:
        """Real ⟨v_a|O^(k)|v_b⟩ including the i^(ΔJ) gauge; the imaginary part must vanish."""
        O = _build_coupling_operator(a[1], b[1], k, op_vec)
        z = gauge_factor(a[1], b[1], k) * (self.vectors[a].conj() @ O @ self.vectors[b])
        if abs(z.imag) > REALITY_TOL * max(abs(z), 1.0):
            raise RuntimeError(f"coupling {a} -> {b} rank {k} is not real in the fixed gauge: {z!r}")
        return float(z.real)

    def operator_tensor(self, a: Node, b: Node, k: int, op_vecs: Sequence[np.ndarray]) -> np.ndarray:
        """⟨partner p_a|O^(k)_q|partner p_b⟩ over every partner triple, with the i^(ΔJ) gauge.

        Shape ``(dim_a, len(op_vecs), dim_b)``. Unlike :meth:`coefficient` the
        entries are complex one by one: the partners of one pair pick different
        phases, and only the invariant projection of
        :func:`real_invariant_coefficients` is real.
        """
        g = gauge_factor(a[1], b[1], k)
        Wa, Wb = self.partners[a], self.partners[b]
        out = np.empty((Wa.shape[1], len(op_vecs), Wb.shape[1]), dtype=np.complex128)
        for q, op_vec in enumerate(op_vecs):
            out[:, q, :] = g * (Wa.conj().T @ _build_coupling_operator(a[1], b[1], k, op_vec) @ Wb)
        return out


@lru_cache(maxsize=None)
def _d4_double_irrep_matrices() -> Dict[str, Tuple[np.ndarray, ...]]:
    """The two D4 double-group irreps over the 8 D4_z rotations inside Oh.

    Both are 2-dimensional and both are already available as restrictions of
    the Oh double-group irreps, which is the cheapest correct realization and
    keeps one convention across the two groups: E1/2 is Γ6 restricted and E3/2
    is Γ7 (Oh's E5/2) restricted. Checked here to be irreducible and
    inequivalent over the double group, where each unbarred element pairs with
    a barred one of opposite character, so ⟨χ|χ⟩ = 2 Σ_8 |χ|² / 16.
    """
    from multitorch.angular.point_group import _oh_double_irrep_matrices, octahedral_rotations
    from multitorch.angular.symmetry import _is_d4z_rotation

    idx = [i for i, R in enumerate(octahedral_rotations()) if _is_d4z_rotation(R)]
    if len(idx) != 8:
        raise RuntimeError(f"expected 8 D4_z rotations inside Oh, got {len(idx)}")
    oh = _oh_double_irrep_matrices()
    out = {"E1/2": tuple(oh["E1/2"][i] for i in idx), "E3/2": tuple(oh["E5/2"][i] for i in idx)}
    chars = {k: np.array([np.trace(m) for m in v]) for k, v in out.items()}
    for k, chi in chars.items():
        if abs(2 * np.vdot(chi, chi).real / 16.0 - 1.0) > 1e-9:
            raise RuntimeError(f"D4 double irrep {k} is not irreducible")
    if abs(2 * np.vdot(chars["E1/2"], chars["E3/2"]) / 16.0) > 1e-9:
        raise RuntimeError("the two D4 double irreps are equivalent")
    return out


@lru_cache(maxsize=None)
def oh_to_d4h_double_subduction(oh_irrep: str) -> Dict[str, np.ndarray]:
    """Partner columns of each D4h double irrep inside one Oh double irrep.

    Returns ``{d4h_irrep: (dim_oh, mult × 2) complex array}``, partner-major
    per copy, the double-group counterpart of
    :func:`~multitorch.angular.symmetry.oh_to_d4h_subduction_matrix`: partner 0
    from the projector P_00 = (2/8) Σ_R Γ*_00(R) D(R) and partner 1 from the
    transfer operator P_10, so every route lands on one fixed realization of
    the D4h irrep. The branching is Γ6 → E1/2, Γ7 → E3/2, Γ8 → E1/2 ⊕ E3/2.

    **The overall phase of each route's partner 0 is not fixed here** — it is
    whatever the eigensolver returned. Nothing in the group theory fixes it
    (it is a Condon–Shortley-style convention), but it multiplies every reduced
    element whose bra and ket reach a D4h irrep by different routes, so the
    crystal field is real only once it is pinned; :func:`gauged_basis` does that
    with ``d4h=True``.
    """
    from multitorch.angular.point_group import _oh_double_irrep_matrices

    dim_oh = OH_DOUBLE_IRREP_DIM[oh_irrep]
    from multitorch.angular.point_group import octahedral_rotations
    from multitorch.angular.symmetry import _is_d4z_rotation
    idx = [i for i, R in enumerate(octahedral_rotations()) if _is_d4z_rotation(R)]
    mats = [_oh_double_irrep_matrices()[oh_irrep][i] for i in idx]

    out: Dict[str, np.ndarray] = {}
    for name, gamma in _d4_double_irrep_matrices().items():
        dim = D4H_DOUBLE_IRREP_DIM[name]

        def transfer(mu: int, nu: int) -> np.ndarray:
            P = np.zeros((dim_oh, dim_oh), dtype=np.complex128)
            for i in range(8):
                P += np.conj(gamma[i][mu, nu]) * mats[i]
            return P * (dim / 8.0)

        P00 = transfer(0, 0)
        eigvals, eigvecs = np.linalg.eigh(0.5 * (P00 + P00.conj().T))
        first = eigvecs[:, eigvals > 0.5]
        if first.shape[1] == 0:
            continue
        cols = []
        for c in range(first.shape[1]):
            for mu in range(dim):
                v = first[:, c] if mu == 0 else transfer(mu, 0) @ first[:, c]
                norm = np.linalg.norm(v)
                if norm < 1e-10:
                    raise RuntimeError(f"transfer operator gave a null partner {mu} for "
                                       f"{oh_irrep} -> {name}")
                cols.append(v / norm)
        out[name] = np.ascontiguousarray(np.stack(cols, axis=1))
    return out


def d4h_double_layout(j_sizes: Dict[float, int], parity: str = "g") -> Dict[str, List[Tuple]]:
    """Per-D4h-double-irrep basis layout, the half-integer
    :func:`~multitorch.angular.symmetry.d4h_basis_layout`.

    Returns ``{d4h_irrep + parity: [(J, oh_irrep, oh_copy, d4h_partner, n_states), ...]}``
    in the same shape the integer dispatcher consumes, with ``oh_irrep`` the
    double-group route (E1/2, E5/2, G3/2) the entry arrives by. Each Oh irrep
    contains each of its D4h irreps once, so there is no D4h copy index.
    """
    if parity not in ("g", "u"):
        raise ValueError(f"parity must be 'g' or 'u', got {parity!r}")
    layout: Dict[str, List[Tuple]] = {}
    for J in sorted(j_sizes):
        if not is_half_integer(J):
            raise ValueError(f"d4h_double_layout got integer J={J}; use d4h_basis_layout")
        for oh in DOUBLE_IRREPS:
            mult = oh_branching(J).get(oh, 0)
            if mult == 0:
                continue
            for d4h, sub in sorted(oh_to_d4h_double_subduction(oh).items()):
                if sub.shape[1] != D4H_DOUBLE_IRREP_DIM[d4h]:
                    raise RuntimeError(f"{oh} -> {d4h} has multiplicity "
                                       f"{sub.shape[1] // D4H_DOUBLE_IRREP_DIM[d4h]}, expected 1")
                for copy in range(mult):
                    for partner in range(D4H_DOUBLE_IRREP_DIM[d4h]):
                        layout.setdefault(d4h + parity, []).append(
                            (J, oh, copy, partner, j_sizes[J]))
    return layout


def real_invariant_coefficients(tensors: Sequence[np.ndarray]) -> np.ndarray:
    """Split one triad's partner matrix elements over its invariant channels.

    ``tensors`` are the partner tensors M_P of one (Γ_bra, operator, Γ_ket)
    triad, one per (J_bra copy, J_ket copy) pair P, all of the same shape
    (:meth:`GaugedBasis.operator_tensor`). Returns a real ``(len(tensors),
    n_channels)`` array whose row P holds the reduced matrix elements r_α(P) of

        M_P = Σ_α r_α(P) T^α

    over an orthonormal basis {T^α} of the invariant tensors of the triad
    (group Wigner–Eckart: every M_P lies in that space). ``n_channels`` is the
    rank of the pairs' Gram matrix — one for most triads, two for Γ8 ⊗ T1u ⊗ Γ8,
    the repeated coupling ttrac writes twice in the .ban.

    The basis is not built from the irrep matrices but *chosen* to make every
    r_α real, which the RAC's real coefficients require: the Gram matrix
    G(P,P') = ⟨M_P, M_P'⟩ is real in the gauge of :func:`gauged_basis` (checked
    here), and its real eigendecomposition G = R Rᵀ gives r_α(P) = R[P, α].
    Any orthogonal rotation of the channels is an equally good choice — it
    leaves Σ_α |⟨f|T^α|g⟩|², the intensity the assembler sums over the .ban's
    repeated triads, unchanged — so the channels are not individually
    comparable with ttrac's, only their sum is.
    """
    if not tensors:
        return np.zeros((0, 0))
    A = np.array([np.asarray(T).ravel() for T in tensors])
    G = A.conj() @ A.T
    scale = max(float(np.abs(G).max()), 1e-300)
    if float(np.abs(G.imag).max()) > REALITY_TOL * scale:
        raise RuntimeError(f"the triad's Gram matrix is not real in the fixed gauge "
                           f"({float(np.abs(G.imag).max()) / scale:.2e} relative); no real set of "
                           f"reduced matrix elements exists for it")
    ev, V = np.linalg.eigh(G.real)
    keep = np.argsort(-ev)[: int((ev > 1e-10 * max(float(ev.max()), 1e-300)).sum())]
    return V[:, keep] * np.sqrt(ev[keep])


def _seed_operators(ranks: Iterable[int]) -> Dict[int, List[np.ndarray]]:
    """Operator components per rank, spanning what the emitters will ask for.

    Every component matters: the gauge has to make all of them real at once, so
    the dipole contributes all three T1 partners, not just one.
    """
    from multitorch.angular.point_group import _c2r_unitary, _real_subduction_matrix
    from multitorch.angular.rac_generator import _d4h_operator_vector_complex

    out: Dict[int, List[np.ndarray]] = {}
    for k in ranks:
        if k == 0:
            out[0] = [np.array([1.0 + 0j])]
        elif k == 1:
            U1, B = _c2r_unitary(1), _real_subduction_matrix(1, "T1")
            out[1] = [U1.conj().T @ np.ascontiguousarray(B[:, p]).astype(np.complex128) for p in range(3)]
        elif k == 2:
            out[2] = [_d4h_operator_vector_complex("DS", 2)]
        elif k == 4:
            out[4] = [_d4h_operator_vector_complex("TENDQ", 4)]
        else:
            raise ValueError(f"no seed operator for rank {k}")
    return out


def gauged_basis(manifolds: Sequence[Tuple[str, Iterable[float]]],
                 d4h: bool = False) -> GaugedBasis:
    """Build the gauge-fixed partner basis of a calculation.

    With ``d4h=True`` the subduction routes into the D4h double irreps are
    phased too (:func:`_pin_d4h_routes`) and the D4h crystal-field couplings are
    verified real, which the Oh emitters do not need and which costs a pass over
    every node pair.

    ``manifolds`` pairs a label with the J values of that manifold, e.g.
    ``[("g", gs_dims), ("u", ex_dims)]``. The spanning tree runs over the whole
    coupling graph — scalar couplings inside an irrep and dipole couplings
    across the manifolds — because the dipole edges are what tie the irreps to
    one another; without them the scalar couplings of different irreps are
    fixed independently and do not all come out real. Only the scalar
    couplings are then verified to be real, which is what the Hamiltonian and
    crystal-field emitters need; the dipole's own reality check is the Gram
    matrix of :func:`real_invariant_coefficients`.

    Raises if a scalar coupling cannot be made real, or if a node stays
    unreachable, in which case its phase would be arbitrary.
    """
    seeds = _seed_operators((0, 1, 2, 4))
    scalar = (0, 2, 4)

    raw_partners: Dict[Node, np.ndarray] = {}
    nodes: List[Node] = []
    for label, js in manifolds:
        for J in sorted(float(j) for j in js):
            if not is_half_integer(J):
                raise ValueError(f"{label} manifold has integer J={J}; use the real emitters")
            for irrep in DOUBLE_IRREPS:
                V = canonical_partner_sets(J, irrep)
                for c in range(V.shape[1]):
                    node = (label, J, irrep, c)
                    nodes.append(node)
                    raw_partners[node] = V[:, c, :]
    if not nodes:
        return GaugedBasis(partners={}, nodes=[])

    def allowed(a: Node, b: Node) -> Tuple[int, ...]:
        if a[0] == b[0]:
            return scalar if a[2] == b[2] else ()
        return (1,)

    def raw(a: Node, b: Node, k: int, component: int) -> complex:
        """Coupling of one pair through one operator component.

        The component is part of the gauge: different partners of the dipole
        pick different phases for the same pair (the quaternionic structure of
        the module docstring), so the walk takes the first component that
        couples the pair and keeps to that order everywhere. Components beyond
        the first are only ever reached where the first gives nothing, which is
        what happens in a sparse coupling graph (d⁹: partner 0 of T1u leaves
        two nodes unreachable). Whether the resulting gauge is consistent is
        not assumed — it is what the reality check below tests.
        """
        g = gauge_factor(a[1], b[1], k)
        return g * (raw_partners[a][:, 0].conj()
                    @ _build_coupling_operator(a[1], b[1], k, seeds[k][component])
                    @ raw_partners[b][:, 0])

    phase: Dict[Node, complex] = {nodes[0]: 1.0 + 0j}
    queue = collections.deque([nodes[0]])
    while queue:
        a = queue.popleft()
        for b in nodes:
            if b in phase:
                continue
            for k in allowed(a, b):
                if abs(a[1] - b[1]) > k or a[1] + b[1] < k:
                    continue
                for component in range(len(seeds[k])):
                    z = np.conj(phase[a]) * raw(a, b, k, component)   # the bra phase enters conjugated
                    if abs(z) > 1e-6:
                        phase[b] = np.exp(-1j * np.angle(z))
                        queue.append(b)
                        break
                if b in phase:
                    break
    unreachable = [n for n in nodes if n not in phase]
    if unreachable:
        raise RuntimeError(f"no coupling path to {unreachable[:3]} (+{max(0, len(unreachable) - 3)} more): "
                           f"their phases would be arbitrary")

    basis = GaugedBasis(partners={n: phase[n] * raw_partners[n] for n in nodes}, nodes=nodes)
    worst, where = 0.0, None
    for a in nodes:
        for b in nodes:
            if a[0] != b[0] or a[2] != b[2]:
                continue
            for k in scalar:
                if abs(a[1] - b[1]) > k or a[1] + b[1] < k:
                    continue
                for op in seeds[k]:
                    z = gauge_factor(a[1], b[1], k) * (basis.vectors[a].conj()
                                                       @ _build_coupling_operator(a[1], b[1], k, op)
                                                       @ basis.vectors[b])
                    if abs(z) < 1e-8:
                        continue
                    r = abs(z.imag) / abs(z)
                    if r > worst:
                        worst, where = r, (a, b, k)
    if worst > REALITY_TOL:
        raise RuntimeError(f"gauge fixing left a complex scalar coupling ({worst:.2e} relative) at {where}; "
                           f"the pseudo-real gauge argument fails for this system")
    basis.residual = worst
    if d4h:
        basis.route_phase = _pin_d4h_routes(basis)
        basis.residual = max(basis.residual, _check_d4h_reality(basis))
    return basis


def _d4h_cf_operators() -> Dict[str, Tuple[int, np.ndarray]]:
    """The D4h-A1g crystal-field operators the generator emits, as (rank, vector)."""
    from multitorch.angular.rac_generator import _d4h_operator_vector_complex

    return {"TENDQ": (4, _d4h_operator_vector_complex("TENDQ", 4)),
            "DT": (4, _d4h_operator_vector_complex("DT", 4)),
            "DS": (2, _d4h_operator_vector_complex("DS", 2))}


def _d4h_reduced_elements(basis: "GaugedBasis", a: Node, b: Node, d4h_irrep: str,
                          route_phase: Optional[Dict[Tuple[str, str], complex]] = None):
    """Crystal-field reduced elements ⟨w_a|O|w_b⟩ of one node pair in one D4h irrep.

    Yields ``(operator name, value)``. A D4h-scalar operator is partner-diagonal
    by Schur's lemma, so partner 0 carries the whole reduced element.
    """
    phase = route_phase if route_phase is not None else basis.route_phase
    for name, (k, op_vec) in _d4h_cf_operators().items():
        if abs(a[1] - b[1]) > k or a[1] + b[1] < k:
            continue
        sub_a = oh_to_d4h_double_subduction(a[2]).get(d4h_irrep)
        sub_b = oh_to_d4h_double_subduction(b[2]).get(d4h_irrep)
        if sub_a is None or sub_b is None:
            continue
        M = basis.operator_tensor(a, b, k, [op_vec])[:, 0, :]
        z = (sub_a[:, 0].conj() @ M @ sub_b[:, 0]) * (np.conj(phase[(a[2], d4h_irrep)])
                                                      * phase[(b[2], d4h_irrep)])
        yield name, z


def _pin_d4h_routes(basis: "GaugedBasis") -> Dict[Tuple[str, str], complex]:
    """One phase per (Oh irrep → D4h irrep) subduction route, fixed by reality.

    Within a D4h irrep the routes form their own little gauge graph: the phase
    of each route's partner 0 is free (see
    :func:`oh_to_d4h_double_subduction`), and it enters every reduced element
    whose bra and ket arrive by different routes. The walk here is the same one
    :func:`gauged_basis` runs over the nodes, one level up: pick a route, then
    give each other route the phase that makes one cross-route element real.
    Whether *all* of them then come out real is not assumed —
    :func:`_check_d4h_reality` tests it.
    """
    out: Dict[Tuple[str, str], complex] = {}
    for d4h_irrep in D4H_DOUBLE_IRREPS:
        routes = [oh for oh in DOUBLE_IRREPS
                  if d4h_irrep in oh_to_d4h_double_subduction(oh)
                  and any(n[2] == oh for n in basis.nodes)]
        if not routes:
            continue
        fixed = {routes[0]: 1.0 + 0j}
        trivial = {(oh, d4h_irrep): 1.0 + 0j for oh in routes}
        for oh in routes[1:]:
            for a in basis.nodes:
                if a[2] not in fixed:
                    continue
                for b in basis.nodes:
                    if b[0] != a[0] or b[2] != oh:
                        continue
                    for _, z in _d4h_reduced_elements(basis, a, b, d4h_irrep, trivial):
                        if abs(z) > 1e-6:
                            fixed[oh] = np.conj(fixed[a[2]]) ** -1 * np.exp(-1j * np.angle(z))
                            break
                    if oh in fixed:
                        break
                if oh in fixed:
                    break
            if oh not in fixed:
                raise RuntimeError(f"no cross-route coupling reaches {oh} -> {d4h_irrep}; "
                                   f"its phase would be arbitrary")
        out.update({(oh, d4h_irrep): fixed[oh] for oh in routes})
    return out


def _check_d4h_reality(basis: "GaugedBasis") -> float:
    """Largest relative |Im| over every D4h crystal-field reduced element; raises if real fails."""
    worst, where = 0.0, None
    for d4h_irrep in D4H_DOUBLE_IRREPS:
        for a in basis.nodes:
            for b in basis.nodes:
                if a[0] != b[0]:
                    continue
                for name, z in _d4h_reduced_elements(basis, a, b, d4h_irrep):
                    if abs(z) < 1e-8:
                        continue
                    r = abs(z.imag) / abs(z)
                    if r > worst:
                        worst, where = r, (name, d4h_irrep, a, b)
    if worst > REALITY_TOL:
        raise RuntimeError(f"the pinned subduction routes left a complex D4h crystal-field "
                           f"coupling ({worst:.2e} relative) at {where}")
    return worst
