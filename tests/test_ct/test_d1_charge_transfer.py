"""d^1 + charge transfer, and the carried per-configuration dimensions it needed.

Residual 33. d^1 has J = 3/2, 5/2 only, branching to G3/2 and E5/2 + G3/2 --
**no E1/2 (Gamma6) at all** -- while its ligand-hole partner d^2 L does carry
E1/2. So the `1/2+` ground irrep is **pure ligand hole**: it has states in the
second configuration only, and the two dipole triads out of it,
``1/2+ 1- 1/2-`` and ``1/2+ 1- 3/2-``, have blocks in configuration 2 alone.

Until 2026-10-25 the assembler inferred a triad's per-configuration dimensions
from the order and extents of its TRANSI blocks, which cannot express a
configuration contributing no states, so d^1 + CT was refused. The RAC now
carries the dimensions (``RACFileFull.config_dims``), because the generator
computes them to size its own blocks and only the file-format round trip loses
them. The inference is retained as the reference and is gated against the
carried values at exact equality wherever both exist (residual 45); a
``.rme_rac`` parsed from a file carries nothing, so the Fortran path is
unchanged **by construction** rather than by a tolerance.

**The inference is not merely incomplete for d^1, it is wrong in three places**,
and two of them are irreps that are otherwise fine -- see
:func:`test_the_inference_is_wrong_for_d1_and_poisons_two_healthy_irreps`. That
was latent rather than shipped, because the refusal fired before any RAC was
built.

**The oracle is the V -> 0 decoupling limit**, not another multitorch path: at
zero hybridisation the two configurations separate exactly, so the metal block
must *be* the single-configuration d^1 Hamiltonian -- and single-configuration
d^1 is validated analytically against Ballhausen's one-electron energies
(``test_from_scratch_fortran_parity``). There is no Fortran d^1 two-configuration
fixture and ttrac cannot be re-run, so this limit is what d^1 + CT has.
"""
from __future__ import annotations

import pytest
import torch

from multitorch.angular.ct_generator import build_ct_ban, generate_ct_ledge_template
from multitorch.angular.rac_generator import (RY_TO_EV_FLOAT, generate_ledge_rac,
                                              ledge_reference_ev)
from multitorch.api.calc import _build_ban_from_rac
from multitorch.hamiltonian.assemble import (_get_config_dims_from_transi,
                                             assemble_and_diagonalize_in_memory,
                                             assemble_ground_hamiltonians)
from multitorch.hamiltonian.parametric import rebuild_hamiltonian_store

# Arbitrary but fixed raw atomic parameters, in Rydberg. Both sides of every
# comparison below are built from exactly these, so the comparison is about
# structure and not about physical values.
SL_GS = {"F2": 0.60, "F4": 0.40}
Z_GS = 0.004
SL_EX = {"F2_pd": 0.50, "G1_pd": 0.35, "G3_pd": 0.20, "F2_dd": 0.0, "F4_dd": 0.0}
Z_EX = {"p": 0.26, "d": 0.004}
TENDQ = 1.0

#: Delta for the decoupling limit. At a physical Delta the 600 ligand-hole final
#: levels interleave with the 23 metal ones, so "the lowest 23" are not the metal
#: block and a sorted comparison measures 5.9 eV of nothing. Separating the
#: configurations is what makes the limit readable; 100 eV is ample and the
#: result is identical at 10 000.
DECOUPLED_DELTA = 100.0

#: The single-configuration d^1 manifold sizes, which are what the carried
#: config-1 dimensions must equal. Measured from the analytically-validated
#: single-configuration build, and asserted against it in the tests below rather
#: than trusted from here.
SINGLE_GS = {"3/2+": 2, "^1/2+": 1}
SINGLE_FS = {"1/2-": 23, "3/2-": 45, "^1/2-": 22}


@pytest.fixture(scope="module")
def single_config_d1():
    """The reference build: one configuration, no charge transfer."""
    rac, store = generate_ledge_rac(
        2, 1, sym="oh", raw_slater_gs_ry=SL_GS, raw_zeta_gs_ry=Z_GS,
        raw_slater_ex_ry=SL_EX, raw_zeta_ex_ry=Z_EX)
    ban = _build_ban_from_rac(rac, tendq=TENDQ, sym="oh")
    return rac, store, ban


@pytest.fixture(scope="module")
def d1_decoupled():
    """d^1 + CT at zero hybridisation, the configurations pushed apart."""
    gs_ref, ex_ref = ledge_reference_ev(2, SL_GS, Z_GS, SL_EX, Z_EX, RY_TO_EV_FLOAT)
    rac, store, dec = generate_ct_ledge_template(1, "oh")
    dec.by_label("gs").reference = dict(gs_ref)
    dec.by_label("ex").reference = dict(ex_ref)
    for label in ("gs_lh", "ex_lh"):
        cfg = dec.by_label(label)
        # irrelevant at V = 0: nothing couples to these blocks
        cfg.reference = {name: 0.5 for name in cfg.operators}
    store = rebuild_hamiltonian_store(store, dec, slater=1.0, soc=1.0)
    ban = build_ct_ban(rac, "oh", cf={"tendq": TENDQ}, delta=DECOUPLED_DELTA,
                       u=1.0, hybridisation=0.0)
    return rac, store, ban


def test_d1_charge_transfer_is_no_longer_refused():
    """Both the structural refusal and the 2 <= n range guard are gone."""
    rac, _store, _dec = generate_ct_ledge_template(1, "oh")
    assert rac.config_dims is not None
    with pytest.raises(ValueError, match="1 <= n <= 8"):
        generate_ct_ledge_template(0, "oh")


def test_the_pure_ligand_hole_irrep_has_no_metal_configuration(single_config_d1):
    """
    ``1/2+`` is absent from the single-configuration d^1 build entirely and
    present in the CT build with configuration 1 contributing zero states. Two
    independent builds agreeing on the same structural fact.
    """
    rac, _store, _dec = generate_ct_ledge_template(1, "oh")
    gs_dims, _fs_dims = rac.config_dims

    assert gs_dims["1/2+"][0] == 0, gs_dims["1/2+"]
    assert gs_dims["1/2+"][1] > 0, gs_dims["1/2+"]

    rac1, store1, ban1 = single_config_d1
    single = assemble_ground_hamiltonians(store1, rac1, ban1)
    assert "1/2+" not in single, sorted(single)
    assert {k: v[0].shape[0] for k, v in single.items()} == SINGLE_GS


def test_the_channel_count_divides_by_the_contributing_configurations():
    """
    ``build_ct_ban`` lists one triad per invariant channel, counting a triad's
    TRANSI blocks and dividing by the configurations that carry them. Dividing
    by *all* of them instead leaves a remainder of 1 on the two one-sided
    triads and refuses the whole BAN, so this is the assertion that names what
    broke rather than letting a fixture error say it.
    """
    rac, _store, _dec = generate_ct_ledge_template(1, "oh")
    ban = build_ct_ban(rac, "oh", cf={"tendq": TENDQ}, delta=3.0, u=1.0,
                       hybridisation={"eg": 2.0, "t2g": 1.0})
    counts: dict = {}
    for key in ban.triads:
        counts[key] = counts.get(key, 0) + 1
    # one channel each for the pure-ligand-hole triads, which exist in one
    # configuration only
    assert counts[("1/2+", "1-", "1/2-")] == 1
    assert counts[("1/2+", "1-", "3/2-")] == 1
    # and the repeated coupling is still repeated: Gamma8 x T1u x Gamma8
    assert counts[("3/2+", "1-", "3/2-")] == 2, counts
    assert len(ban.triads) == 8, ban.triads


def test_the_two_one_sided_triads_are_emitted_and_assemble(d1_decoupled):
    """The triads residual 33 names, and a manifold sized 0 + 37."""
    rac, store, ban = d1_decoupled
    assert ("1/2+", "1-", "1/2-") in ban.triads
    assert ("1/2+", "1-", "3/2-") in ban.triads

    result = assemble_and_diagonalize_in_memory(store, rac, ban)
    by_gs = {t.gs_sym: t for t in result.triads}
    pure = by_gs["1/2+"]
    assert list(pure.gs_conf_sizes) == rac.config_dims[0]["1/2+"]
    assert list(pure.gs_conf_sizes)[0] == 0
    assert pure.Eg.numel() == sum(pure.gs_conf_sizes)
    # every state of it belongs to configuration 2
    assert set(pure.gs_conf_labels.tolist()) == {2}


def test_at_zero_hybridisation_the_metal_block_is_single_config_ground(
        single_config_d1, d1_decoupled):
    """Exact, not approximate: the sub-block is the same matrix."""
    rac1, store1, ban1 = single_config_d1
    rac, store, ban = d1_decoupled
    single = assemble_ground_hamiltonians(store1, rac1, ban1)
    ct = assemble_ground_hamiltonians(store, rac, ban)

    assert set(single) <= set(ct), (sorted(single), sorted(ct))
    worst = 0.0
    for sym, (H_ref, _, _) in single.items():
        H_ct, _labels, sizes = ct[sym]
        d_metal = list(sizes)[0]
        assert d_metal == H_ref.shape[0], (sym, d_metal, H_ref.shape)
        ev_ref = torch.linalg.eigvalsh(H_ref)
        ev_ct = torch.linalg.eigvalsh(H_ct[:d_metal, :d_metal])
        worst = max(worst, float((ev_ref - ev_ct).abs().max()))
    assert worst == 0.0, worst


def test_at_zero_hybridisation_the_metal_block_is_single_config_final(
        single_config_d1, d1_decoupled):
    """
    The final side, which is where the inference went wrong. With the
    configurations separated the lowest ``d_metal`` levels of each CT final
    manifold are the single-configuration ones.
    """
    rac1, store1, ban1 = single_config_d1
    rac, store, ban = d1_decoupled
    ref = {t.fs_sym: torch.sort(t.Ef)[0]
           for t in assemble_and_diagonalize_in_memory(store1, rac1, ban1).triads}
    got = {t.fs_sym: (torch.sort(t.Ef)[0], list(t.fs_conf_sizes))
           for t in assemble_and_diagonalize_in_memory(store, rac, ban).triads}

    assert {k: v.numel() for k, v in ref.items()} == SINGLE_FS
    worst = 0.0
    for sym, levels in ref.items():
        ct_levels, sizes = got[sym]
        d_metal = sizes[0]
        assert d_metal == levels.numel(), (sym, d_metal, levels.numel())
        worst = max(worst, float((ct_levels[:d_metal] - levels).abs().max()))
    assert worst < 1e-12, worst


def test_the_inference_is_wrong_for_d1_and_poisons_two_healthy_irreps():
    """
    The failure the carried dimensions prevent, stated exactly.

    The inference does not merely omit the pure-ligand-hole irrep: the lone
    block's **ket** extent is appended as configuration 1's dimension for the
    two final irreps that triad reaches, and those are reached from ``3/2+``
    and ``^1/2+`` as well, where they carry real intensity. It was latent, not
    shipped, because the refusal fired before a RAC was ever built.
    """
    rac, _store, _dec = generate_ct_ledge_template(1, "oh")
    carried_gs, carried_fs = rac.config_dims
    inferred_gs, inferred_fs = _get_config_dims_from_transi(rac, 2)

    # the known case: one entry, and it loses the configuration count as well
    assert carried_gs["1/2+"] == [0, 37]
    assert inferred_gs["1/2+"] == [37]
    assert {k for k in set(carried_gs) | set(inferred_gs)
            if carried_gs.get(k) != inferred_gs.get(k)} == {"1/2+"}

    # the sibling the residual does not name: two *final* irreps corrupted
    poisoned = {k for k in set(carried_fs) | set(inferred_fs)
                if carried_fs.get(k) != inferred_fs.get(k)}
    assert poisoned == {"1/2-", "3/2-"}, poisoned
    for sym in poisoned:
        metal, ligand = carried_fs[sym]
        assert inferred_fs[sym] == [ligand, ligand], (sym, inferred_fs[sym])
        # and the carried value is the one the validated path agrees with
        assert metal == SINGLE_FS[sym]


@pytest.mark.parametrize("sym,n", [("oh", 8), ("oh", 7), ("d4h", 8), ("d4h", 7)])
def test_carried_dims_equal_the_inference_wherever_it_works(sym, n):
    """
    Residual 45's gate: the carried value must reproduce what it replaces,
    exactly, on cases that already work -- which is what keeps the validated
    n = 2..8 output from moving.

    Both groups and both parities. The even D4h case is ``n = 8`` rather than
    ``n = 2`` because the n = 2 and n = 6 templates cost 50 and 58 seconds
    against 14 for n = 8, and n = 8 is also the count the nid8ct fixture chain
    validates. All 14 reachable (sym, n) pairs were verified at exact equality
    when this landed; see residual 72.
    """
    rac, _store, _dec = generate_ct_ledge_template(n, sym)
    assert rac.config_dims is not None
    carried_gs, carried_fs = rac.config_dims
    inferred_gs, inferred_fs = _get_config_dims_from_transi(rac, 2)
    assert carried_gs == inferred_gs, (sym, n)
    assert carried_fs == inferred_fs, (sym, n)


def test_ti3_lmct_is_reachable_from_the_public_api_and_decouples_to_d1():
    """
    The deliverable residual 33 named: Ti3+/V4+ with LMCT, through
    ``calcXAS_from_scratch`` rather than through the internals. Both are d^1.

    The same V -> 0 limit, at the level a user sees: with the hybridisation off
    and the ligand-hole configuration pushed away, the charge-transfer spectrum
    IS the single-configuration one. Measured 2026-10-25 at cosine
    1.000000000000 and max |dy| 4.7e-15.

    **What this does not say:** nothing here validates V != 0.
    ``cowan_operators.hopping_blocks``' n = 1 blocks have no Fortran oracle --
    there is no bundled d^1 two-configuration fixture and ttrac cannot be
    re-run -- so d^1 charge transfer is *reachable*, not *validated*, away from
    this limit. Residual 72.
    """
    from multitorch.api.calc import calcXAS_from_scratch
    from multitorch.spectrum.parity import spectral_parity

    grid = dict(xmin=-5.0, xmax=25.0, nbins=1500, broaden_mode="correct")
    cf = {"tendq": 1.8}

    x0, y0 = calcXAS_from_scratch("Ti", "iii", cf=cf, sym="oh", **grid)
    x1, y1 = calcXAS_from_scratch("Ti", "iii", cf=cf, sym="oh", delta=200.0, u=1.0,
                                  lmct={"eg": 0.0, "t2g": 0.0}, **grid)
    p = spectral_parity(x0, y0, x1, y1)
    assert p.cosine > 1.0 - 1e-12, p
    assert p.area_ratio == pytest.approx(1.0, abs=1e-12), p
    assert float((y1 - y0).abs().max()) < 1e-12

    # and it runs at all with the hybridisation on, which is the capability
    _, y_ct = calcXAS_from_scratch("Ti", "iii", cf=cf, sym="oh", delta=3.0, u=1.0,
                                   lmct={"eg": 2.0, "t2g": 1.0}, **grid)
    assert float(y_ct.max()) > 0.0
    assert torch.isfinite(y_ct).all()


def test_a_file_parsed_rac_carries_no_dimensions():
    """
    What keeps the Fortran path unchanged by construction. A parsed `.rme_rac`
    has no such field, so the assembler falls back to the inference there and
    nothing about the fixture cases can move. Same shape as ``symmetry``.
    """
    from multitorch.io.read_rme import RACFileFull, read_rme_rac_full
    from multitorch.data import fixtures

    assert RACFileFull().config_dims is None
    path = next(iter(sorted(
        (p for p in __import__("pathlib").Path(fixtures.__file__).parent.rglob("*.rme_rac")),
        key=lambda p: p.stat().st_size)), None)
    assert path is not None, "no bundled .rme_rac to parse"
    parsed = read_rme_rac_full(str(path))
    assert parsed.config_dims is None, path
    assert parsed.symmetry is None, path
