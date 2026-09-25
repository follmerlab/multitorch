"""Charge transfer from scratch against the Fortran chain (WP-C, increment C4).

Oracle: the bundled pyctm/ttmult two-configuration fixtures run through the
fixture path (store rebuilt at its own Slater reduction, i.e. the printed
ttrcg store) with charge transfer switched on: Δ = 3 eV, U_pd − U_dd = 1 eV,
hybridisation in every channel, a crystal field, and in D4h a tetragonal one.
The from-scratch template gets the same per-configuration atomic parameters,
read off the fixture HAMILTONIAN blocks, and the same Δ, u, V(Γ) and
10Dq/Dt/Ds. Ground and final levels, stick intensities and the broadened
spectrum must agree.

The four odd-electron fixtures (Cr³⁺ d³, Mn²⁺ d⁵, Fe³⁺ d⁵, Co²⁺ d⁷) are
themselves two-configuration CT calculations, so they are a real Fortran oracle
for the half-integer charge-transfer path too (WP-B2a-5), on the same footing as
V³⁺ d², Fe²⁺ d⁶ and Ni²⁺ d⁸.

Final levels of nid8ct agree to 2e-4 eV only: its store prints the 2p⁵ blocks
around E_av = 860 eV with fewer decimals, and its exactly degenerate 2p⁵ L̲
levels scatter by ±9e-5 eV in the Fortran store itself (1e-6 in the Oh stores,
E_av = 20 eV).

D4h with charge transfer at half-integer J has no Fortran reference at all
(residual 32): its oracles are the Oh limit dt = ds = 0 against the Oh fixture
below, the dipole sum rule, and the LAPACK scramble harness.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from multitorch._constants import DTYPE
from multitorch.angular.ct_generator import build_ct_ban, generate_ct_ledge_template, hybridisation_operator_vectors
from multitorch.api.calc import _cache_ban, preload_fixture
from multitorch.hamiltonian.assemble import assemble_and_diagonalize_in_memory
from multitorch.hamiltonian.parametric import rebuild_hamiltonian_store
from multitorch.spectrum.broaden import pseudo_voigt
from multitorch.spectrum.parity import spectral_parity
from multitorch.spectrum.sticks import get_sticks_from_banresult

POOL = dict(T=80.0, max_gs=40)
DELTA, U = 3.0, 1.0
CASES = [
    # element, valence, sym, n, crystal field, hybridisation, final-level tolerance (eV)
    ("V", "iii", "oh", 2, {"tendq": 1.2}, {"eg": 2.0, "t2g": 1.0}, 2e-5),
    ("Fe", "ii", "oh", 6, {"tendq": 1.0}, {"eg": 2.2, "t2g": 1.1}, 2e-5),
    ("Ni", "ii", "oh", 8, {"tendq": 1.1}, {"eg": 2.0, "t2g": 1.0}, 2e-5),
    ("Ni", "ii", "d4h", 8, {"tendq": 1.2, "dt": 0.05, "ds": 0.1}, {"b1": 2.0, "a1": 1.6, "b2": 1.0, "e": 0.8}, 2e-4),
    # odd electron counts (WP-B2a-5): the four bundled half-integer fixtures are
    # two-configuration CT calculations, so ttrcg/ttrac is the oracle here too
    ("Cr", "iii", "oh", 3, {"tendq": 1.2}, {"eg": 2.0, "t2g": 1.0}, 2e-5),
    ("Mn", "ii", "oh", 5, {"tendq": 1.0}, {"eg": 2.2, "t2g": 1.1}, 2e-5),
    ("Fe", "iii", "oh", 5, {"tendq": 1.1}, {"eg": 2.0, "t2g": 1.0}, 2e-5),
    ("Co", "ii", "oh", 7, {"tendq": 1.0}, {"eg": 1.8, "t2g": 0.9}, 2e-5),
]
IDS = [f"{c[0]}{c[1]}-{c[2]}" for c in CASES]
LABELS = (("gs", (2, "GROUND")), ("gs_lh", (2, "EXCITE")), ("ex", (3, "GROUND")), ("ex_lh", (3, "EXCITE")))


def _run(element, valence, sym, n, cf, hyb, fortran=None):
    """Fortran fixture path and from-scratch template for one ion.

    ``fortran`` overrides the Fortran side as ``(sym, cf, hybridisation)``; the
    from-scratch side still takes that fixture's atomic parameters. Used for the
    D4h Oh limit, where the reference is the Oh fixture.
    """
    f_sym, f_cf, f_hyb = fortran if fortran is not None else (sym, cf, hyb)
    fx = preload_fixture(element, valence, f_sym)
    fortran_result = assemble_and_diagonalize_in_memory(
        rebuild_hamiltonian_store(fx.cowan_template, fx.decomposition, slater=0.8, soc=1.0),
        fx.rac, _cache_ban(fx, f_cf, DELTA, U, f_hyb, None))
    rac, store, dec = generate_ct_ledge_template(n, sym)
    for label, (section, kind) in LABELS:
        ours, theirs = dec.by_label(label), fx.decomposition.config(section, kind)
        values = theirs.parameter_values(0.8, 1.0)
        if ours.shells == theirs.shells:
            ours.reference = {name: float(values[name]) for name in ours.operators if name in values}
        else:   # nid8ct couples 3d before 2p in 2p^5 3d^9
            ours.reference = {ours.aliases()[alias]: float(values[name]) for alias, name in theirs.aliases().items()}
    scratch = assemble_and_diagonalize_in_memory(
        rebuild_hamiltonian_store(store, dec, slater=1.0, soc=1.0), rac,
        build_ct_ban(rac, sym, cf=cf, delta=DELTA, u=U, hybridisation=hyb))
    return fortran_result, scratch, fx.rac, rac


def _levels(result, attr):
    E = torch.sort(torch.cat([getattr(t, attr) for t in {t.fs_sym if attr == "Ef" else t.gs_sym: t for t in result.triads}.values()]))[0]
    return E - E[0]


def _levels_by_state(result, rac, attr):
    """Levels with each one repeated by the degeneracy of its irrep.

    Comparing two *different* point groups needs this: an Oh Γ8 level is one
    entry of a 4-dimensional irrep, and in D4h the same level is two entries
    (E1/2 and E3/2) of 2-dimensional ones. Counting states makes the two
    listings the same length and the same multiset.
    """
    dim = {i.name: i.dim for i in rac.irreps}
    key = "fs_sym" if attr == "Ef" else "gs_sym"
    per_irrep = {getattr(t, key): t for t in result.triads}
    E = torch.sort(torch.cat([getattr(t, attr).repeat_interleave(dim[sym])
                              for sym, t in per_irrep.items()]))[0]
    return E - E[0]


def _spectrum(E, M, hi):
    x = torch.linspace(-3.0, hi, 6000, dtype=DTYPE)
    return x, pseudo_voigt(x, E, M, fwhm_g=0.2, fwhm_l=0.2, fwhm_l2=0.4, med_energy=0.5 * hi, mode="legacy")


@pytest.mark.parametrize("element,valence,sym,n,cf,hyb,tol", CASES, ids=IDS)
def test_charge_transfer_from_scratch_matches_fortran(element, valence, sym, n, cf, hyb, tol):
    fortran, scratch, _, _ = _run(element, valence, sym, n, cf, hyb)

    gf, gs = _levels(fortran, "Eg"), _levels(scratch, "Eg")
    assert gf.shape == gs.shape
    assert float((gf - gs).abs().max()) < 1e-5
    ff, fs = _levels(fortran, "Ef"), _levels(scratch, "Ef")
    assert ff.shape == fs.shape
    assert float((ff - fs).abs().max()) < tol

    Ef, Mf = get_sticks_from_banresult(fortran, **POOL)[:2]
    Es, Ms = get_sticks_from_banresult(scratch, **POOL)[:2]
    # V3+: 3T1g levels within kT at 80 K, so Boltzmann weights magnify the 1e-6 eV level noise
    assert float(Ms.sum()) == pytest.approx(float(Mf.sum()), rel=1e-4)
    hi = float(max(Ef.max() - Ef.min(), Es.max() - Es.min())) + 3.0
    p = spectral_parity(*_spectrum(Es - Es.min(), Ms, hi), *_spectrum(Ef - Ef.min(), Mf, hi))
    assert p.cosine > 0.99999, p


HALF_D4H = [("Cr", "iii", 3, 1.2, {"eg": 2.0, "t2g": 1.0}),
            ("Fe", "iii", 5, 1.1, {"eg": 2.0, "t2g": 1.0}),
            ("Co", "ii", 7, 1.0, {"eg": 1.8, "t2g": 0.9})]
HALF_D4H_IDS = [f"{c[0]}{c[1]}-d4h-ct" for c in HALF_D4H]
# residual 15 again: the Oh d5 ions' six ground components are split by µeV, and
# the Oh and D4h labelings of the same manifold do not round them into the same
# lowest "level", so a max_gs=1 pool takes different components on the two sides
# (Fe3+ measured at cosine 0.99969 on 2026-09-24). Cr3+ and Co2+ are exact.
RES15 = pytest.mark.xfail(strict=True, reason="plan residual 15: max_gs counts exactly equal levels only")
HALF_D4H_SPECTRUM = [pytest.param(*c, marks=RES15) if (c[0], c[1]) == ("Fe", "iii") else c
                     for c in HALF_D4H]


def _oh_limit(element, valence, n, tendq, hyb):
    """From-scratch D4h + CT at dt = ds = 0 against the same ion's Oh fixture with CT on.

    D4h + CT at half-integer J has no Fortran reference (residual 32), so this
    limit is its parity oracle. The channel sums make it exact rather than
    approximate: B1HYBR + A1HYBR is EGHYBR and B2HYBR + EHYBR is T2GHYBR term by
    term (:data:`~multitorch.angular.ct_generator.HYBR_ROUTES`), so equal V
    within each pair is exactly the Oh hybridisation.
    """
    d4h_hyb = {"b1": hyb["eg"], "a1": hyb["eg"], "b2": hyb["t2g"], "e": hyb["t2g"]}
    return _run(element, valence, "d4h", n, {"tendq": tendq, "dt": 0.0, "ds": 0.0}, d4h_hyb,
                fortran=("oh", {"tendq": tendq}, hyb))


@pytest.mark.parametrize("element,valence,n,tendq,hyb", HALF_D4H, ids=HALF_D4H_IDS)
def test_half_integer_d4h_charge_transfer_reproduces_the_oh_limit_levels(element, valence, n, tendq, hyb):
    """Ground and final levels of the Oh limit, counted by state (:func:`_levels_by_state`)."""
    fortran, scratch, f_rac, s_rac = _oh_limit(element, valence, n, tendq, hyb)

    gf, gs = _levels_by_state(fortran, f_rac, "Eg"), _levels_by_state(scratch, s_rac, "Eg")
    assert gf.shape == gs.shape
    assert float((gf - gs).abs().max()) < 1e-5
    ff = _levels_by_state(fortran, f_rac, "Ef")
    fs = _levels_by_state(scratch, s_rac, "Ef")
    assert ff.shape == fs.shape
    assert float((ff - fs).abs().max()) < 2e-5


@pytest.mark.parametrize("element,valence,n,tendq,hyb", HALF_D4H_SPECTRUM, ids=HALF_D4H_IDS)
def test_half_integer_d4h_charge_transfer_reproduces_the_oh_limit_spectrum(element, valence, n, tendq, hyb):
    """Spectral shape of the Oh limit, from the single lowest ground level.

    ``max_gs=1``, not the ``POOL`` of the Fortran-parity tests, because the pool
    is a truncation by *level* and D4h splits each Oh level in two: at
    ``max_gs=40`` the two sides hold different physical states and the cosine
    falls to ~0.70 for the long-validated integer ions as well (V³⁺ d²,
    measured 2026-09-24), which is residual 15/D-1 and not a statement about
    this path. Absolute intensity is deliberately not compared for the same
    reason — the Cr³⁺ Oh side counts two components where the D4h side counts
    one, an exact factor 2. The dipole sum rule pins the absolute scale instead.
    """
    fortran, scratch, _, _ = _oh_limit(element, valence, n, tendq, hyb)
    pool = dict(T=80.0, max_gs=1)
    Ef, Mf = get_sticks_from_banresult(fortran, **pool)[:2]
    Es, Ms = get_sticks_from_banresult(scratch, **pool)[:2]
    hi = float(max(Ef.max() - Ef.min(), Es.max() - Es.min())) + 3.0
    p = spectral_parity(*_spectrum(Es - Es.min(), Ms, hi), *_spectrum(Ef - Ef.min(), Mf, hi))
    assert p.cosine > 0.99999, p


@pytest.mark.parametrize("element,valence,n,tendq,hyb", HALF_D4H, ids=HALF_D4H_IDS)
def test_half_integer_d4h_charge_transfer_tetragonal_terms_act(element, valence, n, tendq, hyb):
    """Each tetragonal term moves the levels *on its own*.

    The Oh-limit test above runs at dt = ds = 0, where only the Oh-scalar TENDQ
    acts, so it would pass with DT, DS and the tetragonal hybridisation channels
    all dead. This test varies exactly one of the three at a time against a
    common dt = ds = 0, symmetric-V baseline, so none of them can hide behind
    another.

    An earlier version varied dt, ds *and* the hybridisation ratios in one step
    and was therefore vacuous for DT/DS: measured on 2026-09-25, the
    hybridisation change alone moves the levels 0.172 eV against a 0.02 eV
    threshold, so it passed with DT and DS completely inert. The thresholds below
    are each a comfortable fraction of the measured isolated effect (Cr³⁺:
    dt-only 0.416 eV, ds-only 0.421 eV, hybridisation-only 0.172 eV).

    The *magnitudes* of DT and DS are pinned analytically, not here:
    ``tests/test_hamiltonian/test_cf_one_electron.py`` compares the d¹ and d⁹
    half-integer levels with the Ballhausen one-particle energies, which is what
    catches a sign or scale error in either.
    """
    rac, store, dec = generate_ct_ledge_template(n, "d4h")
    fx = preload_fixture(element, valence, "oh")
    for label, (section, kind) in LABELS:
        ours, theirs = dec.by_label(label), fx.decomposition.config(section, kind)
        values = theirs.parameter_values(0.8, 1.0)
        ours.reference = {name: float(values[name]) for name in ours.operators if name in values}
    cowan = rebuild_hamiltonian_store(store, dec, slater=1.0, soc=1.0)

    symmetric = {"b1": hyb["eg"], "a1": hyb["eg"], "b2": hyb["t2g"], "e": hyb["t2g"]}
    asymmetric = {"b1": hyb["eg"], "a1": 0.6 * hyb["eg"], "b2": hyb["t2g"], "e": 0.7 * hyb["t2g"]}

    def levels(dt, ds, v):
        res = assemble_and_diagonalize_in_memory(
            cowan, rac, build_ct_ban(rac, "d4h", cf=dict(tendq=tendq, dt=dt, ds=ds),
                                     delta=DELTA, u=U, hybridisation=v))
        E = torch.sort(torch.cat([t.Eg for t in res.triads]))[0]
        return (E - E[0])[:20]

    baseline = levels(0.0, 0.0, symmetric)
    for name, dt, ds, v, floor in (("DT", 0.05, 0.0, symmetric, 0.1),
                                   ("DS", 0.0, 0.4, symmetric, 0.1),
                                   ("hybridisation channels", 0.0, 0.0, asymmetric, 0.05)):
        shift = float((baseline - levels(dt, ds, v)).abs().max())
        assert shift > floor, f"{name} alone moved the levels only {shift:.4f} eV"


def test_hybridisation_channels_project_onto_their_orbitals():
    """Analytic oracle for the route phases: on one d electron (unit reduced elements) each
    hybridisation actor is √5 times the projector onto its orbitals (b1 x²−y², a1 z², b2 xy,
    e xz/yz; Oh eg, t2g), and the channels sum to √5·1."""
    from multitorch.angular.rac_generator import _operator_real_matrix

    # real d harmonics in _c2r_unitary order m = -2..2: xy, yz, z², xz, x²−y²
    expected = {
        "d4h": {"B1HYBR": [0, 0, 0, 0, 1], "A1HYBR": [0, 0, 1, 0, 0], "B2HYBR": [1, 0, 0, 0, 0], "EHYBR": [0, 1, 0, 1, 0]},
        "oh": {"EGHYBR": [0, 0, 1, 0, 1], "T2GHYBR": [1, 1, 0, 1, 0]},
    }
    for sym, channels in expected.items():
        total = np.zeros((5, 5))
        for channel, diag in channels.items():
            M = sum(_operator_real_matrix(2, 2, k, v) for k, v in hybridisation_operator_vectors(channel, sym).items())
            np.testing.assert_allclose(M, math.sqrt(5.0) * np.diag(diag), atol=1e-12, err_msg=f"{sym} {channel}")
            total += M
        np.testing.assert_allclose(total, math.sqrt(5.0) * np.eye(5), atol=1e-12)


@pytest.mark.parametrize("seed", [2, 5])
def test_charge_transfer_from_scratch_is_independent_of_the_lapack_basis_choice(seed):
    """The CT oracles with every numpy eigen/singular basis scrambled (fresh process).

    Oh Fe²⁺ and D4h nid8ct for integer J; Oh Cr³⁺ and the D4h Cr³⁺ Oh limit for
    half-integer J, where far more is a LAPACK output — the canonical copy
    order, the spanning-tree node phases, the pinned D4h subduction routes and
    the invariant channel basis are all eigendecompositions.
    """
    import subprocess
    import sys
    from pathlib import Path

    tools = Path(__file__).parent.parent / "tools"
    code = f"""
import sys; sys.path.insert(0, {str(tools)!r}); sys.path.insert(0, {str(Path(__file__).parent)!r})
import lapack_scramble; lapack_scramble.install({seed})
import test_ct_from_scratch_fortran_parity as t
for case in (t.CASES[1], t.CASES[3], t.CASES[4]):
    t.test_charge_transfer_from_scratch_matches_fortran(*case)
t.test_half_integer_d4h_charge_transfer_reproduces_the_oh_limit_levels(*t.HALF_D4H[0])
t.test_half_integer_d4h_charge_transfer_reproduces_the_oh_limit_spectrum(*t.HALF_D4H[0])
print("OK")
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=1200)
    assert out.returncode == 0 and "OK" in out.stdout, out.stdout[-2000:] + out.stderr[-2000:]


# ─────────────────────────────────────────────────────────────
# Public API: preload_from_scratch(charge_transfer=True) + calcXAS_cached (C5)
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("case", [CASES[1], CASES[3]], ids=[IDS[1], IDS[3]])
def test_cached_charge_transfer_from_scratch_with_fortran_atomic_overrides(case):
    """Same oracle through the public cached API; slater = soc = 0, every integral an ``atomic`` override."""
    from multitorch.api.calc import calcXAS_cached, preload_from_scratch

    element, valence, sym, n, cf, hyb, tol = case
    fortran = _run(element, valence, sym, n, cf, hyb)[0]
    Ef, Mf = get_sticks_from_banresult(fortran, **POOL)[:2]

    cache = preload_from_scratch(element, valence, sym, charge_transfer=True)
    fx = preload_fixture(element, valence, sym)
    atomic = {}
    for label, (section, kind) in LABELS:
        ours, theirs = cache.decomposition.by_label(label), fx.decomposition.config(section, kind)
        values = theirs.parameter_values(0.8, 1.0)
        if ours.shells == theirs.shells:
            atomic[label] = {name: values[name] for name in ours.operators if name in values}
        else:
            atomic[label] = {alias: values[name] for alias, name in theirs.aliases().items()}
    _, _, sticks = calcXAS_cached(cache, cf=cf, slater=0.0, soc=0.0, delta=DELTA, u=U, lmct=hyb, atomic=atomic,
                                  return_sticks=True, **POOL)
    Es, Ms = sticks[:, 0], sticks[:, 1]
    keep = Mf > 1e-10 * Mf.max()
    assert float(Ms.sum()) == pytest.approx(float(Mf.sum()), rel=1e-4)
    hi = float(max(Ef.max() - Ef.min(), Es.max() - Es.min())) + 3.0
    p = spectral_parity(*_spectrum(Es - Es.min(), Ms, hi), *_spectrum(Ef[keep] - Ef.min(), Mf[keep], hi))
    assert p.cosine > 0.99999, p


@pytest.mark.parametrize("element,valence,sym", [("V", "iii", "oh"), ("Fe", "ii", "oh"), ("Ni", "ii", "d4h")])
def test_ligand_hole_hfs_parameters_follow_pyctm(element, valence, sym):
    """Ligand-hole configurations carry the HFS parameters of d^(n+1) (one more 3d electron, same Z).

    Oracle: the ratio ligand-hole / plain configuration in the Fortran decks, for
    the well-determined integrals F²dd, F²pd, G¹pd and ζ2p (the decks print ζ3d
    with three decimals; absolute HFS values differ from RCN31's by the known
    ~2 % HX residual, which cancels in the ratio).
    """
    from multitorch.api.calc import preload_from_scratch

    ours = preload_from_scratch(element, valence, sym, charge_transfer=True).decomposition
    theirs = preload_fixture(element, valence, sym).decomposition

    def value(cfg, values, alias):
        """By physical alias, or for the two-d-shell ligand-hole ground configuration by metal-first name."""
        aliases = cfg.aliases()
        name = aliases.get(alias, {"F2dd": "F2_11"}.get(alias))
        return float(values[name]) if name in cfg.operators and name in values else None

    pairs = [("gs", "gs_lh", (2, "GROUND"), (2, "EXCITE"), ["F2dd"]),
             ("ex", "ex_lh", (3, "GROUND"), (3, "EXCITE"), ["F2pd", "G1pd", "zeta_p"])]
    checked = 0
    for plain, lh, sec_plain, sec_lh, names in pairs:
        cp, cl = ours.by_label(plain), ours.by_label(lh)
        fp, fl = theirs.config(*sec_plain), theirs.config(*sec_lh)
        for alias in names:
            if len(cl.shells) == 2 and plain == "ex" and alias != "zeta_p":
                assert set(cl.reference) == {"zeta_1"}   # 2p^5 3d^10 L: the ligand hole has no integrals
                continue
            a, b = value(cp, cp.reference, alias), value(cl, cl.reference, alias)
            fa, fb = value(fp, fp.params, alias), value(fl, fl.params, alias)
            if None in (a, b, fa, fb) or min(abs(fa), abs(fb)) < 1e-3:   # absent, or an identically zero operator (d^9)
                continue
            assert b / a == pytest.approx(fb / fa, abs=0.012), (plain, alias, b / a, fb / fa)
            checked += 1
    assert checked >= 1   # Ni d8: only zeta_2p (d^9 L has no F^2; 2p^5 3d^10 L no p-d integrals)
