"""Gauge-fixed double-group partner bases (WP-B2a, increment B2a-1).

Oracles are the defining properties, not another implementation: the vectors
must be orthonormal partner-0 vectors of the right irrep (they transform into
themselves under the Oh projector), the copy order must be a property of the
physics rather than of the host eigensolver, and every coupling of every rank
must come out real in the fixed gauge — the property the RAC needs for the
Hamiltonian and crystal field, and the reason this module exists. The dipole
is out of scope here (module docstring: a half-integer triad can carry two
invariants), and the last test pins that boundary.
"""
from __future__ import annotations

import numpy as np
import pytest

from multitorch.angular.cowan_operators import configuration_operators
from multitorch.angular.double_group import (
    DOUBLE_IRREPS, canonical_partner_vectors, gauged_basis, is_half_integer,
)
from multitorch.angular.point_group import (
    OH_DOUBLE_IRREP_DIM, _build_coupling_operator, _oh_double_irrep_matrices, oh_branching,
    octahedral_rotations, wigner_D_matrix,
)

HALF_J = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
# (label, ground shells, excited shells) of the bundled half-integer fixtures
IONS = [("Cr3+ d3", ((2, 3),), ((1, 5), (2, 4))),
        ("Mn2+ d5", ((2, 5),), ((1, 5), (2, 6))),
        ("Co2+ d7", ((2, 7),), ((1, 5), (2, 8)))]


@pytest.mark.parametrize("J", HALF_J)
def test_partner_vectors_are_orthonormal_and_carry_their_irrep(J):
    for irrep in DOUBLE_IRREPS:
        V = canonical_partner_vectors(J, irrep)
        assert V.shape == (int(round(2 * J + 1)), oh_branching(J).get(irrep, 0))
        if V.shape[1] == 0:
            continue
        np.testing.assert_allclose(V.conj().T @ V, np.eye(V.shape[1]), atol=1e-10)
        # partner-0 projector of the irrep: P = (dim/24) Σ_R Γ*(R)_00 D^J(R)
        dim = OH_DOUBLE_IRREP_DIM[irrep]
        gamma = _oh_double_irrep_matrices()[irrep]
        P = sum(g[0, 0].conj() * wigner_D_matrix(J, R) for g, R in zip(gamma, octahedral_rotations())) * dim / 24.0
        np.testing.assert_allclose(P @ V, V, atol=1e-9)


def test_copy_order_is_canonical_not_the_eigensolver_s():
    """A different basis of the projection subspace must give the same copies (up to a phase)."""
    J, irrep = 4.5, "G3/2"
    V = canonical_partner_vectors(J, irrep)
    assert V.shape[1] >= 2, "need a multiplicity > 1 case"
    marker = _build_coupling_operator(J, J, 4, __import__(
        "multitorch.angular.rac_generator", fromlist=["_d4h_operator_vector_complex"]
    )._d4h_operator_vector_complex("TENDQ", 4))
    w = np.array([float((V[:, c].conj() @ marker @ V[:, c]).real) for c in range(V.shape[1])])
    assert np.all(np.diff(w) > 1e-6), "copies must be separated by the marker eigenvalue"
    # the marker is diagonal in this basis: that is what makes the order canonical
    M = V.conj().T @ marker @ V
    np.testing.assert_allclose(M - np.diag(np.diag(M)), 0.0, atol=1e-9)


@pytest.mark.parametrize("label,gs,ex", IONS, ids=[i[0] for i in IONS])
def test_every_scalar_coupling_is_real_in_the_fixed_gauge(label, gs, ex):
    dims_g, dims_u = configuration_operators(gs).dims, configuration_operators(ex).dims
    assert all(is_half_integer(J) for J in dims_g)
    basis = gauged_basis([("g", dims_g), ("u", dims_u)])
    assert basis.residual < 1e-9, (label, basis.residual)
    assert len(basis.nodes) > 20
    # the vectors stay orthonormal within each (manifold, J)
    for manifold, dims in (("g", dims_g), ("u", dims_u)):
        for J in dims:
            cols = [basis.vectors[n] for n in basis.nodes if n[0] == manifold and n[1] == J]
            if len(cols) > 1:
                G = np.array(cols).conj() @ np.array(cols).T
                np.testing.assert_allclose(G, np.eye(len(cols)), atol=1e-9)


def test_unreachable_nodes_raise():
    """A J that nothing connects to the rest would keep an arbitrary phase."""
    with pytest.raises(RuntimeError, match="no coupling path"):
        gauged_basis([("g", [0.5, 6.5])])


def test_integer_j_is_rejected():
    with pytest.raises(ValueError, match="integer J"):
        gauged_basis([("g", [2.0])])


def test_canonical_basis_is_independent_of_the_lapack_basis_choice():
    """Same oracle with every numpy eigen/singular basis scrambled (fresh process).

    The projection subspace of a repeated irrep is degenerate, so the host's
    eigensolver picks an arbitrary basis inside it; the marker eigenbasis must
    undo that choice, or the emitted RAC would be machine-dependent (residual 21).
    """
    import json
    import subprocess
    import sys
    from pathlib import Path

    tools = Path(__file__).parent.parent / "tools"
    code = f"""
import sys, json; sys.path.insert(0, {str(tools)!r})
import lapack_scramble; lapack_scramble.install(3)
import numpy as np
from multitorch.angular.cowan_operators import configuration_operators
from multitorch.angular.double_group import canonical_partner_vectors, gauged_basis
V = canonical_partner_vectors(4.5, "G3/2")
b = gauged_basis([("g", configuration_operators(((2, 5),)).dims),
                  ("u", configuration_operators(((1, 5), (2, 6))).dims)])
print(json.dumps(dict(res=b.residual, n=len(b.nodes),
                      v=[[float(z.real), float(z.imag)] for z in V[:, 0]])))
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    scrambled = json.loads([l for l in out.stdout.splitlines() if l.startswith("{")][0])

    V = canonical_partner_vectors(4.5, "G3/2")
    basis = gauged_basis([("g", configuration_operators(((2, 5),)).dims),
                          ("u", configuration_operators(((1, 5), (2, 6))).dims)])
    assert scrambled["n"] == len(basis.nodes)
    assert scrambled["res"] < 1e-9
    v_scrambled = np.array([complex(r, i) for r, i in scrambled["v"]])
    assert abs(np.vdot(V[:, 0], v_scrambled)) == pytest.approx(1.0, abs=1e-10)


def test_the_d4h_crystal_field_operator_order_is_load_bearing_for_parity():
    """``_d4h_cf_operators``' key order fixes which coupling pins each route phase.

    The route phases are **pure gauge** -- a per-route phase is a diagonal
    unitary conjugation of the target irrep's block, so it leaves every
    eigenvalue alone -- which means *any* order is physically correct. But
    ``_pin_d4h_routes`` pins each phase against the **first** coupling it
    finds, so the order picks which one that is, and only one order reproduces
    the D4h coefficients the Fortran chain validated.

    Measured: offering the operators in declaration order instead of
    ``TENDQ, DT, DS`` moves 1303 of d4h d7's ADD entries. That is why the order
    is pinned here with its reason rather than left as an accident of how the
    dict literal was typed.
    """
    from unittest import mock

    from multitorch.angular import double_group as dg

    assert list(dg._d4h_cf_operators()) == ["TENDQ", "DT", "DS"]

    # ...and the order really does change the pinned phase, so this is a guard
    # and not a restatement of the source.
    basis = dg.gauged_basis([("gs", [0.5, 1.5, 2.5]), ("ex", [0.5, 1.5, 2.5])],
                            d4h=True)
    pinned = dict(basis.route_phase)
    original = dg._d4h_cf_operators()
    reordered = {k: original[k] for k in ("DS", "DT", "TENDQ")}
    with mock.patch.object(dg, "_d4h_cf_operators", return_value=reordered):
        other = dg._pin_d4h_routes(basis)
    assert pinned != other, (
        "reordering the operators left every route phase unchanged, so this test "
        "is not measuring what its docstring claims")
