"""End-to-end parity of the from-scratch pipeline against the Fortran chain.

Oracle: the bundled pyctm/ttmult Oh fixtures, run through the fixture path at
their own Slater reduction (which reproduces the Fortran store exactly) with
the ligand-hole configuration decoupled (lmct = 0, Δ = 100 eV). The
from-scratch generator gets the *same* atomic parameters, read off the
fixture's HAMILTONIAN blocks by the S1a decomposition (each equals the ttrcg
input deck value to ≤ 1.3e-5 eV). Stick energies (relative to the lowest bright
stick), absolute stick intensities and the broadened spectrum must agree.

D4h at dt = ds = 0 is the Oh limit and is checked against the same Oh fixture;
its degenerate ground components sit in several D4h irreps, so both sides use
a T = 80 K Boltzmann pool (Known residual 15: max_gs=1 is ill-posed there).

History: cosine 0.47 before 2026-09-10 (audit S3/S4/S5/S7); exact for Ni d⁸
only until 2026-09-13, when the excited Hamiltonian was put in the valence-term
gauge of the excited CF/MULTIPOLE blocks (V³⁺ was at 0.853, Fe²⁺ at 0.991).
Half-integer J (Cr³⁺, Mn²⁺, Fe³⁺, Co²⁺) is WP-B.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from multitorch._constants import DTYPE
from multitorch.angular.rac_generator import generate_ledge_rac
from multitorch.api.calc import _build_ban_from_rac, _run_phase5_pipeline
from multitorch.hamiltonian.assemble import assemble_and_diagonalize_in_memory
from multitorch.hamiltonian.build_cowan import load_hamiltonian_decomposition
from multitorch.spectrum.broaden import pseudo_voigt
from multitorch.spectrum.parity import spectral_parity
from multitorch.spectrum.sticks import get_sticks_from_banresult
from pathlib import Path

RY = 13.605693122994
REFDATA = Path(__file__).parent.parent / "reference_data"
POOL = dict(T=80.0, max_gs=40)

# (fixture, element, valence, d electrons, fixture 10Dq)
IONS = [
    ("v3_d2_oh", "V", "iii", 2, 1.8),
    ("fe2_d6_oh", "Fe", "ii", 6, 1.0),
    ("ni2_d8_oh", "Ni", "ii", 8, 1.0),
]


def _fixture_parameters(name):
    """Fortran store parameters (eV) → generate_ledge_rac keyword arguments (Ry)."""
    dec = load_hamiltonian_decomposition(REFDATA / name / f"{name}.rme_rcg")
    g = dec.config(2, "GROUND").params           # d^n
    e = dec.config(3, "GROUND").params           # 2p^5 d^(n+1), core first
    return dict(
        raw_slater_gs_ry={"F2": g.get("F2_11", 0.0) / RY, "F4": g.get("F4_11", 0.0) / RY},
        raw_zeta_gs_ry=g["zeta_1"] / RY,
        raw_slater_ex_ry={"F2_pd": e["F2_12"] / RY, "G1_pd": e["G1_12"] / RY, "G3_pd": e["G3_12"] / RY,
                          "F2_dd": e.get("F2_22", 0.0) / RY, "F4_dd": e.get("F4_22", 0.0) / RY},
        raw_zeta_ex_ry={"p": e["zeta_1"] / RY, "d": e["zeta_2"] / RY},
    )


def _bright(E, M):
    keep = M > 1e-10 * M.max()
    return E[keep], M[keep]


def _spectrum(E, M, hi):
    x = torch.linspace(-3.0, hi, 6000, dtype=DTYPE)
    y = pseudo_voigt(x, E, M, fwhm_g=0.2, fwhm_l=0.2, fwhm_l2=0.4, med_energy=0.5 * hi, mode="legacy")
    return x, y


@pytest.mark.parametrize("sym", ["oh", "d4h"])
@pytest.mark.parametrize("name,element,valence,n,tendq", IONS, ids=[i[0] for i in IONS])
def test_from_scratch_matches_fortran_fixture(name, element, valence, n, tendq, sym):
    res_f = _run_phase5_pipeline(element, valence, "oh", "l", {"tendq": tendq},
                                 slater=0.8, soc=1.0, delta=100.0, lmct=0.0, mlct=None)
    Ef, Mf = _bright(*get_sticks_from_banresult(res_f, **POOL)[:2])

    rac, cowan = generate_ledge_rac(2, n, sym=sym, **_fixture_parameters(name))
    ban = _build_ban_from_rac(rac, tendq=tendq, dt=0.0, ds=0.0, sym=sym)
    Es, Ms = _bright(*get_sticks_from_banresult(assemble_and_diagonalize_in_memory(cowan, rac, ban), **POOL)[:2])

    ef, es = (Ef - Ef.min()).numpy(), (Es - Es.min()).numpy()
    mf, ms = Mf.numpy(), Ms.numpy()
    # absolute intensities (no normalisation); floor: the store prints 6 decimals
    assert ms.sum() == pytest.approx(mf.sum(), rel=2e-5)
    # every strong Fortran stick: same energy (±2e-5 eV) and summed intensity
    for e in ef[mf > 1e-4 * mf.sum()]:
        near_s, near_f = np.abs(es - e) < 2e-5, np.abs(ef - e) < 2e-5
        assert near_s.any(), e
        assert abs(ms[near_s].sum() - mf[near_f].sum()) < 2e-5 * mf.sum(), e
    hi = float(max(ef.max(), es.max())) + 3.0
    p = spectral_parity(*_spectrum(Es - Es.min(), Ms, hi), *_spectrum(Ef - Ef.min(), Mf, hi))
    assert p.cosine > 0.99999, p
    assert p.area_ratio == pytest.approx(1.0, abs=1e-5), p


@pytest.mark.parametrize("sym", ["oh", "d4h"])
@pytest.mark.parametrize("name,element,valence,n,tendq", IONS, ids=[i[0] for i in IONS])
def test_from_scratch_cache_with_fortran_atomic_overrides(name, element, valence, n, tendq, sym):
    """WP-A6: preload_from_scratch + calcXAS_cached, every integral an ``atomic`` override.

    slater = soc = 0 so nothing comes from HFS; the overrides are the fixture's
    ttrcg parameters under their shell-letter aliases. Same Fortran oracle and
    tolerances as above, through the public cached API.
    """
    from multitorch.api.calc import calcXAS_cached, preload_from_scratch

    res_f = _run_phase5_pipeline(element, valence, "oh", "l", {"tendq": tendq},
                                 slater=0.8, soc=1.0, delta=100.0, lmct=0.0, mlct=None)
    Ef, Mf = _bright(*get_sticks_from_banresult(res_f, **POOL)[:2])

    dec = load_hamiltonian_decomposition(REFDATA / name / f"{name}.rme_rcg")
    atomic = {}
    for label, (section, kind) in (("gs", (2, "GROUND")), ("ex", (3, "GROUND"))):
        cfg = dec.config(section, kind)
        atomic[label] = {alias: cfg.params[op] for alias, op in cfg.aliases().items()}
    cache = preload_from_scratch(element, valence, sym)
    _, _, sticks = calcXAS_cached(cache, cf={"tendq": tendq}, slater=0.0, soc=0.0, atomic=atomic,
                                  return_sticks=True, **POOL)
    Es, Ms = _bright(sticks[:, 0], sticks[:, 1])

    ef, es = (Ef - Ef.min()).numpy(), (Es - Es.min()).numpy()
    mf, ms = Mf.numpy(), Ms.numpy()
    assert ms.sum() == pytest.approx(mf.sum(), rel=2e-5)
    for e in ef[mf > 1e-4 * mf.sum()]:
        near_s, near_f = np.abs(es - e) < 2e-5, np.abs(ef - e) < 2e-5
        assert near_s.any(), e
        assert abs(ms[near_s].sum() - mf[near_f].sum()) < 2e-5 * mf.sum(), e


# ─────────────────────────────────────────────────────────────
# D4h away from the Oh limit (2026-09-14)
# ─────────────────────────────────────────────────────────────

D4H_CF = [(1.0, 0.05, 0.1), (1.5, -0.1, 0.25)]


def _nid8ct_from_scratch():
    """Ni d8 D4h generator with the Fortran nid8ct store's own atomic parameters."""
    from multitorch.api.calc import preload_fixture
    fx = preload_fixture("Ni", "ii", "d4h")
    g = fx.decomposition.config(2, "GROUND").params
    e = fx.decomposition.config(3, "GROUND").params      # valence-first: shell 1 = 3d, 2 = 2p
    rac, cowan = generate_ledge_rac(
        2, 8, sym="d4h",
        raw_slater_gs_ry={"F2": g["F2_11"] / RY, "F4": g["F4_11"] / RY}, raw_zeta_gs_ry=g["zeta_1"] / RY,
        raw_slater_ex_ry={"F2_pd": e["F2_12"] / RY, "G1_pd": e["G1_12"] / RY, "G3_pd": e["G3_12"] / RY,
                          "F2_dd": 0.0, "F4_dd": 0.0},
        raw_zeta_ex_ry={"p": e["zeta_2"] / RY, "d": e["zeta_1"] / RY},
    )
    return fx, rac, cowan


def _lowest_ground_levels(result, n=20):
    per_irrep = {t.gs_sym: t.Eg for t in result.triads}
    E = torch.sort(torch.cat(list(per_irrep.values())))[0]
    return (E - E[0])[:n]


@pytest.mark.parametrize("tendq,dt,ds", D4H_CF)
def test_d4h_from_scratch_matches_fortran_away_from_oh_limit(tendq, dt, ds):
    """Tetragonal field with multiplets: every earlier D4h test sat at the Oh limit or had no multiplets.

    Oracle: the Fortran nid8ct store (LMCT off) at the same 10Dq, Dt, Ds. Before
    the E-partner basis was pinned, this failed by 0.27-0.64 eV on machines whose
    LAPACK returned a rotated E basis.
    """
    from multitorch.api.calc import _cache_ban
    from multitorch.hamiltonian.parametric import rebuild_hamiltonian_store

    fx, rac, cowan = _nid8ct_from_scratch()
    cf = {"tendq": tendq, "dt": dt, "ds": ds}
    fortran = assemble_and_diagonalize_in_memory(
        rebuild_hamiltonian_store(fx.cowan_template, fx.decomposition, slater=0.8, soc=1.0),
        fx.rac, _cache_ban(fx, cf, 100.0, None, 0.0, None))
    scratch = assemble_and_diagonalize_in_memory(cowan, rac, _build_ban_from_rac(rac, tendq=tendq, dt=dt, ds=ds, sym="d4h"))
    ef, es = _lowest_ground_levels(fortran), _lowest_ground_levels(scratch)
    assert float((ef - es).abs().max()) < 2e-5, (ef[:8], es[:8])
    assert float(ef[2]) > 0.1  # the tetragonal splitting is really in play


def test_d4h_from_scratch_is_independent_of_the_lapack_basis_choice():
    """Same oracle with eigenvectors scrambled inside degenerate subspaces (fresh process).

    Scramble seed 2 reproduced the exxa failure (0.636 eV) before the fix.
    """
    import subprocess
    import sys
    tools = Path(__file__).parent.parent / "tools"
    code = f"""
import sys; sys.path.insert(0, {str(tools)!r}); sys.path.insert(0, {str(Path(__file__).parent)!r})
import lapack_scramble; lapack_scramble.install(2)
import test_from_scratch_fortran_parity as t
for cf in t.D4H_CF:
    t.test_d4h_from_scratch_matches_fortran_away_from_oh_limit(*cf)
print("OK")
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0 and "OK" in out.stdout, out.stdout[-2000:] + out.stderr[-2000:]
