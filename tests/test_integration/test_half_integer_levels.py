"""Half-integer J from scratch: energy levels vs the Fortran chain (WP-B2a, increment B2a-2).

The Hamiltonian and crystal-field blocks of an odd-electron ion are emitted
from the gauge-fixed double-group basis (:mod:`multitorch.angular.double_group`).
Those two operators fix every energy level, so the levels are the oracle here:
both manifolds of Cr³⁺, Mn²⁺, Fe³⁺ and Co²⁺ must match the Fortran fixture at
the fixture's own atomic parameters, with charge transfer switched off.

Intensities are *not* checked here: the dipole is increment B2a-3, which
rebuilt the half-integer TRANSI blocks by invariant projection and has its own
oracles. ``api.calc`` accepts these ions.
"""
from __future__ import annotations

import pytest
import torch

from multitorch.angular.rac_generator import generate_ledge_template
from multitorch.api.calc import _build_ban_from_rac, _cache_ban, preload_fixture
from multitorch.hamiltonian.assemble import assemble_and_diagonalize_in_memory
from multitorch.hamiltonian.parametric import rebuild_hamiltonian_store

RY = 13.605693122994
# ttrac spells the double-group irreps S0/S1/^S0, the generator 1/2, 3/2, ^1/2
# (multitorch.angular.point_group.BUTLER_LABEL); both mean E1/2, G3/2, E5/2.
FORTRAN_TO_OURS = {"S0": "1/2", "S1": "3/2", "^S0": "^1/2"}
# fixture, element, valence, d electrons, 10Dq
IONS = [("cr3_d3_oh", "Cr", "iii", 3, 1.8),
        ("mn2_d5_oh", "Mn", "ii", 5, 1.0),
        ("fe3_d5_oh", "Fe", "iii", 5, 1.2),
        ("co2_d7_oh", "Co", "ii", 7, 1.0)]


def _levels_by_irrep(result, attr, window=50.0):
    """Levels of each irrep, relative to the manifold's lowest, keeping the ionic ones.

    The fixture manifolds also carry the ligand-hole configuration, decoupled
    at delta = 100 eV; ``window`` drops it so both paths list the same states.
    """
    per_irrep = {}
    for t in result.triads:
        key = t.gs_sym if attr == "Eg" else t.fs_sym
        stem, parity = key[:-1], key[-1]
        per_irrep.setdefault(FORTRAN_TO_OURS.get(stem, stem) + parity, getattr(t, attr))
    if not per_irrep:
        return {}
    base = min(float(E.min()) for E in per_irrep.values())
    return {k: torch.sort(E)[0][torch.sort(E)[0] - base < window] - base for k, E in per_irrep.items()}


def _from_scratch(fx, n, tendq):
    rac, store, dec = generate_ledge_template(2, n, sym="oh")
    for label, (section, kind) in (("gs", (2, "GROUND")), ("ex", (3, "GROUND"))):
        ours, theirs = dec.by_label(label), fx.decomposition.config(section, kind)
        values = theirs.parameter_values(0.8, 1.0)
        ours.reference = {name: float(values[name]) for name in ours.operators if name in values}
    cowan = rebuild_hamiltonian_store(store, dec, slater=1.0, soc=1.0)
    return assemble_and_diagonalize_in_memory(cowan, rac, _build_ban_from_rac(rac, tendq=tendq, sym="oh"))


@pytest.mark.parametrize("name,element,valence,n,tendq", IONS, ids=[i[0] for i in IONS])
def test_half_integer_levels_match_fortran(name, element, valence, n, tendq):
    fx = preload_fixture(element, valence, "oh")
    fortran = assemble_and_diagonalize_in_memory(
        rebuild_hamiltonian_store(fx.cowan_template, fx.decomposition, slater=0.8, soc=1.0),
        fx.rac, _cache_ban(fx, {"tendq": tendq}, 100.0, None, 0.0, None))
    scratch = _from_scratch(fx, n, tendq)

    for attr, tol in (("Eg", 2e-5), ("Ef", 2e-5)):
        f, s = _levels_by_irrep(fortran, attr), _levels_by_irrep(scratch, attr)
        shared = sorted(set(f) & set(s))
        assert len(shared) >= 3, (attr, sorted(f), sorted(s))
        for irrep in shared:
            a, b = f[irrep], s[irrep]
            assert a.shape == b.shape, (attr, irrep, a.shape, b.shape)
            assert float((a - b).abs().max()) < tol, (attr, irrep, a[:6], b[:6])
