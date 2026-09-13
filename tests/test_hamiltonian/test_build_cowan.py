"""
Fixture-path COWAN store builder (WP-S S1a): exact decomposition of every
HAMILTONIAN block onto the configuration operators, and rebuild at absolute
``slater`` / ``soc``.

Oracles (Fortran, not code):

* The bundled ``.rme_rcg`` stores themselves: every HAMILTONIAN block of every
  fixture must be reproduced by E_av + Σ F^k O + Σ G^k O + Σ ζ O to 1e-5
  (relative to max(1, |H|)).
* ``fortran_ops/oh8_rcg/<name>_s<red>.rcg``: the ttrcg input decks of the eight
  Oh fixtures regenerated at Slater reduction 0.8 and 1.0. Every fitted
  coefficient must equal the Fortran input parameter it multiplies.
* ``fortran_ops/oh8_s1.0_hamiltonian.npz``: eigenvalues of the section 2/3
  HAMILTONIAN blocks of the 1.0 runs. The 0.8 fixtures, rebuilt with the 1.0
  input parameters, must reproduce them; the public ``slater=1.0`` rebuild must
  agree within the Weyl bound of pyctm's 3-decimal input rounding.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from multitorch._constants import DTYPE, RY_TO_EV
from multitorch.angular.cowan_operators import configuration_operators
from multitorch.atomic.parameter_fixtures import read_rcn31_out_params
from multitorch.hamiltonian.assemble import (
    assemble_and_diagonalize,
    assemble_and_diagonalize_in_memory,
)
from multitorch.hamiltonian.build_cowan import (
    CowanBlockMeta,
    FIXTURE_SLATER_REDUCTION,
    build_cowan_store_in_memory,
    j_value,
    load_hamiltonian_decomposition,
    read_cowan_configurations,
    read_cowan_metadata,
)
from multitorch.hamiltonian.build_rac import build_rac_in_memory
from multitorch.io.read_ban import read_ban
from multitorch.io.read_rme import read_cowan_store

REFDATA = Path(__file__).parent.parent / "reference_data"
FORTRAN_OPS = REFDATA / "fortran_ops"
NID8CT = REFDATA / "nid8ct"
NID8CT_BAN = NID8CT / "nid8ct.ban"
NID8CT_RCG = NID8CT / "nid8ct.rme_rcg"
NID8CT_RAC = NID8CT / "nid8ct.rme_rac"

OH8 = ["ti4_d0_oh", "v3_d2_oh", "cr3_d3_oh", "mn2_d5_oh",
       "fe3_d5_oh", "fe2_d6_oh", "co2_d7_oh", "ni2_d8_oh"]
ALL_STORES = sorted(REFDATA.glob("*/*.rme_rcg"))


def _rcg(name: str) -> Path:
    return REFDATA / name / f"{name}.rme_rcg"


# ─────────────────────────────────────────────────────────────
# ttrcg input decks
# ─────────────────────────────────────────────────────────────

def read_rcg_input_states(path: Path):
    """``[(values, type_codes)]`` of every ``state`` parameter line, in file order.

    Cowan's deck: label (20 columns incl. the parameter count), then fields of
    10 columns, 5 on the first line and 7 per continuation line. Each field is
    a value with 3 decimals followed by one digit giving the parameter type:
    0 E_av, 1 F^k(l,l), 2 ζ, 3 F^k(l,l'), 4 G^k(l,l').
    """
    lines = path.read_text().splitlines()
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        if not line.strip().startswith("state"):
            i += 1
            continue
        count = int(line[:20].split()[-1])
        fields = [line[20 + 10 * k:30 + 10 * k] for k in range(5)]
        j = i + 1
        while sum(1 for f in fields if f.strip()) < count:
            fields += [lines[j][10 * k:10 * k + 10] for k in range(7)]
            j += 1
        fields = [f for f in fields if f.strip()][:count]
        out.append(([float(f[:-1]) for f in fields], [int(f[-1]) for f in fields]))
        i = j
    return out


def input_parameters(shells, values, codes):
    """Map a deck line onto operator names by Cowan type code, in shell order."""
    f_ll = [f"F{k}_{i + 1}{i + 1}" for i, (l, n) in enumerate(shells)
            if 1 < n < 4 * l + 1 for k in range(2, 2 * l + 1, 2)]
    zeta = [f"zeta_{i + 1}" for i in range(len(shells))]
    l1, l2 = (shells[0][0], shells[1][0]) if len(shells) > 1 else (0, 0)
    f_12 = [f"F{k}_12" for k in range(2, 2 * min(l1, l2) + 1, 2)]
    g_12 = [f"G{k}_12" for k in range(abs(l1 - l2), l1 + l2 + 1, 2)]
    queues = {0: iter(["E_av"]), 1: iter(f_ll), 2: iter(zeta), 3: iter(f_12), 4: iter(g_12)}
    return {next(queues[c]): v for v, c in zip(values, codes)}


OH8_CONFIGS = [(2, "GROUND"), (2, "EXCITE"), (3, "GROUND"), (3, "EXCITE")]


@pytest.fixture(scope="module")
def oh8_eigs():
    return np.load(FORTRAN_OPS / "oh8_s1.0_hamiltonian.npz")


# ─────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────

def test_configurations_core_first_pyctm():
    cfg = read_cowan_configurations(_rcg("fe2_d6_oh"))
    assert len(cfg) == 4
    assert cfg[2] == {"GROUND": ((2, 6),), "EXCITE": ((2, 7), (2, 9))}
    assert cfg[3] == {"GROUND": ((1, 5), (2, 7)), "EXCITE": ((1, 5), (2, 8), (2, 9))}


def test_configurations_valence_first_ttmult():
    cfg = read_cowan_configurations(NID8CT_RCG)
    assert cfg[0] == {"GROUND": ((2, 8),), "EXCITE": ((2, 9), (1, 5))}
    assert cfg[3] == {"GROUND": ((2, 9), (1, 5)), "EXCITE": ((1, 5), (2, 9))}


def test_configurations_closed_shell_ground():
    assert read_cowan_configurations(_rcg("ti4_d0_oh"))[2]["GROUND"] == ()


@pytest.mark.parametrize("sym,J", [("0+", 0.0), ("4-", 4.0), ("s0+", 0.5), ("s3-", 3.5), ("^2+", 2.0)])
def test_j_value(sym, J):
    assert j_value(sym) == J


def test_metadata_aligns_with_store():
    meta, store = read_cowan_metadata(NID8CT_RCG), read_cowan_store(NID8CT_RCG)
    assert [len(s) for s in meta] == [len(s) for s in store] == [22, 24, 167, 142]
    assert all(isinstance(m, CowanBlockMeta) for s in meta for m in s)


# ─────────────────────────────────────────────────────────────
# Exact decomposition of every bundled store
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rcg", ALL_STORES, ids=lambda p: p.stem)
def test_every_hamiltonian_block_decomposes_exactly(rcg):
    """E_av + Slater + SOC operators reproduce each Fortran block (raises otherwise)."""
    dec = load_hamiltonian_decomposition(rcg)
    meta = read_cowan_metadata(rcg)
    n_h = sum(m.operator == "HAMILTONIAN" for s in meta for m in s)
    assert sum(len(c.block_index) for c in dec.configs) == n_h
    assert max(c.max_residual for c in dec.configs) < 1e-5


@pytest.mark.parametrize("name", OH8)
@pytest.mark.parametrize("reduction", ["0.8", "1.0"])
def test_fitted_parameters_equal_fortran_input(name, reduction):
    """Each fitted coefficient is the ttrcg input parameter of its operator.

    At 0.8 the fit is on the bundled store; at 1.0 the check uses the 0.8 fit
    scaled by 1/0.8, which must match the 1.0 deck within pyctm's rounding
    (both decks carry 3 decimals: |Δ| ≤ 5e-4/0.8 + 5e-4).
    """
    dec = load_hamiltonian_decomposition(_rcg(name))
    states = read_rcg_input_states(FORTRAN_OPS / "oh8_rcg" / f"{name}_s{reduction}.rcg")
    assert len(states) == 4
    for (sec, kind), (values, codes) in zip(OH8_CONFIGS, states):
        cfg = dec.config(sec, kind)
        want = input_parameters(cfg.shells, values, codes)
        assert cfg.e_av == pytest.approx(want.pop("E_av"), abs=2e-5)
        for n in set(cfg.params) | set(want):
            got = cfg.params.get(n, 0.0)
            if reduction == "0.8":
                assert got == pytest.approx(want.get(n, 0.0), abs=2e-5), (sec, kind, n)
            elif not n.startswith("zeta"):
                assert abs(got / 0.8 - want.get(n, 0.0)) <= 5e-4 / 0.8 + 5e-4 + 2e-5, (sec, kind, n)
            else:
                assert got == pytest.approx(want.get(n, 0.0), abs=2e-5), (sec, kind, n)


@pytest.mark.parametrize("name", OH8)
def test_operators_reproduce_fortran_at_full_slater(name, oh8_eigs):
    """0.8 fixture + Σ (p_1.0 − p_fit)·O  ==  Fortran run at 1.0 (eigenvalues, eV).

    Floor: the stores print 6 decimals on elements up to ~100 eV; two Fortran
    runs at the same parameters differ by up to 2e-5 in the store.
    """
    dec = load_hamiltonian_decomposition(_rcg(name))
    states = read_rcg_input_states(FORTRAN_OPS / "oh8_rcg" / f"{name}_s1.0.rcg")
    worst = 0.0
    for (sec, kind), (values, codes) in zip(OH8_CONFIGS, states):
        cfg = dec.config(sec, kind)
        p10 = input_parameters(cfg.shells, values, codes)
        ops = configuration_operators(cfg.shells)
        for J in cfg.block_index:
            H = cfg.fixture[J].numpy() + sum(
                (p10.get(n, 0.0) - v) * ops.blocks[n][J] for n, v in cfg.params.items()
            )
            ref = oh8_eigs[f"{name}/{sec}/{kind}/{J}"]
            worst = max(worst, np.abs(np.linalg.eigvalsh(H) - ref).max() / math.sqrt(2 * J + 1))
    assert worst < 2e-5, worst


@pytest.mark.parametrize("name", OH8)
def test_public_rebuild_at_full_slater_within_rounding_bound(name, oh8_eigs):
    """``slater=1.0`` on the 0.8 fixture vs Fortran at 1.0, per block.

    The only difference in inputs is rounding: δp = p_fit/0.8 − p_1.0 per
    operator. Weyl: |Δλ| ≤ Σ |δp|·‖O‖₂ (+ the 2e-5 store floor).
    """
    rac_plan = _plan_for(name)
    store = build_cowan_store_in_memory(rac_plan, slater=1.0, soc=1.0, source_rcg_path=_rcg(name))
    dec = load_hamiltonian_decomposition(_rcg(name))
    states = read_rcg_input_states(FORTRAN_OPS / "oh8_rcg" / f"{name}_s1.0.rcg")
    for (sec, kind), (values, codes) in zip(OH8_CONFIGS, states):
        cfg = dec.config(sec, kind)
        p10 = input_parameters(cfg.shells, values, codes)
        ops = configuration_operators(cfg.shells)
        for J, j in cfg.block_index.items():
            w = math.sqrt(2 * J + 1)
            bound = sum(
                abs(v / 0.8 - p10.get(n, 0.0)) * np.linalg.norm(ops.blocks[n][J], 2)
                for n, v in cfg.params.items() if not n.startswith("zeta")
            ) / w + 2e-5
            lam = np.linalg.eigvalsh(store[sec][j].detach().numpy()) / w
            ref = oh8_eigs[f"{name}/{sec}/{kind}/{J}"] / w
            assert np.abs(lam - ref).max() <= bound, (sec, kind, J, np.abs(lam - ref).max(), bound)


def _plan_for(name):
    _, plan = build_rac_in_memory(
        read_ban(REFDATA / name / f"{name}.ban"),
        source_rac_path=REFDATA / name / f"{name}.rme_rac",
        source_rcg_path=_rcg(name),
    )
    return plan


# ─────────────────────────────────────────────────────────────
# Fixture metadata: slater_reduction
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", OH8)
def test_slater_reduction_pyctm(name):
    """Fitted F^k and G^k of the bundled store = recorded reduction × ttrcg input at 1.0.

    Both decks are rounded to 3 decimals: |fit − r·p_1.0| ≤ 5e-4·(1 + r).
    """
    r = FIXTURE_SLATER_REDUCTION[name]
    dec = load_hamiltonian_decomposition(_rcg(name))
    states = read_rcg_input_states(FORTRAN_OPS / "oh8_rcg" / f"{name}_s1.0.rcg")
    checked = 0
    for (sec, kind), (values, codes) in zip(OH8_CONFIGS, states):
        cfg = dec.config(sec, kind)
        full = input_parameters(cfg.shells, values, codes)
        for n, p in full.items():
            if n[0] in "FG" and p > 0:
                assert abs(cfg.params[n] - r * p) <= 5e-4 * (1 + r) + 2e-5, (sec, kind, n)
                checked += 1
    assert checked >= 5


@pytest.mark.parametrize("stem,dirname", [("nid8", "nid8"), ("nid8ct", "nid8ct"),
                                          ("als1ni2", "als1ni2"), ("nid8ct_ems", "nid8ct")])
def test_slater_reduction_ttmult_examples(stem, dirname):
    """Ground d⁸ F²: fitted / Hartree-Fock value of nid8.rcn31_out = recorded reduction."""
    hf = read_rcn31_out_params(REFDATA / "nid8" / "nid8.rcn31_out").ground.f("3D", "3D", 2) * RY_TO_EV
    dec = load_hamiltonian_decomposition(REFDATA / dirname / f"{stem}.rme_rcg")
    # the last d⁸ configuration carries the parameters (transition-only sections 0/1 have none)
    d8 = [c for c in dec.configs if c.shells == ((2, 8),)][-1]
    assert d8.params["F2_11"] / hf == pytest.approx(FIXTURE_SLATER_REDUCTION[stem], abs=2e-4)


def test_unknown_fixture_requires_explicit_reduction(tmp_path):
    rcg = tmp_path / "mystery.rme_rcg"
    rcg.write_text(_rcg("ni2_d8_oh").read_text())
    with pytest.raises(KeyError, match="slater_reduction"):
        load_hamiltonian_decomposition(rcg)
    assert load_hamiltonian_decomposition(rcg, slater_reduction=0.8).slater_reduction == 0.8


# ─────────────────────────────────────────────────────────────
# Store layout and parity at the fixture's reduction
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def nid8ct_plan():
    return build_rac_in_memory(
        read_ban(NID8CT_BAN), source_rac_path=NID8CT_RAC, source_rcg_path=NID8CT_RCG,
    )[1]


def test_store_equals_fixture_at_own_reduction(nid8ct_plan):
    built = build_cowan_store_in_memory(nid8ct_plan, slater=0.8, soc=1.0, source_rcg_path=NID8CT_RCG)
    parsed = read_cowan_store(NID8CT_RCG)
    assert [len(s) for s in built] == [len(s) for s in parsed]
    for bs, ps in zip(built, parsed):
        for b, p in zip(bs, ps):
            assert b.shape == p.shape
            assert torch.equal(b, p)


def test_only_hamiltonian_blocks_change(nid8ct_plan):
    built = build_cowan_store_in_memory(nid8ct_plan, slater=0.6, soc=0.9, source_rcg_path=NID8CT_RCG)
    parsed = read_cowan_store(NID8CT_RCG)
    meta = read_cowan_metadata(NID8CT_RCG)
    for s, (bs, ps) in enumerate(zip(built, parsed)):
        for m, b, p in zip(meta[s], bs, ps):
            if m.operator == "HAMILTONIAN":
                continue
            assert torch.equal(b, p), (s, m)
    # the excited core-hole manifold (section 3) is rescaled too
    sec3 = [b for m, b, p in zip(meta[3], built[3], parsed[3]) if m.operator == "HAMILTONIAN"]
    ref3 = [p for m, p in zip(meta[3], parsed[3]) if m.operator == "HAMILTONIAN"]
    assert any((b - p).abs().max() > 1e-2 for b, p in zip(sec3, ref3))


def test_end_to_end_assembler_parity(nid8ct_plan):
    ban = read_ban(NID8CT_BAN)
    rac, plan = build_rac_in_memory(ban, source_rac_path=NID8CT_RAC, source_rcg_path=NID8CT_RCG)
    cowan = build_cowan_store_in_memory(plan, source_rcg_path=NID8CT_RCG)
    mem = assemble_and_diagonalize_in_memory(cowan, rac, ban)
    disk = assemble_and_diagonalize(NID8CT_RCG, NID8CT_RAC, NID8CT_BAN)
    assert len(mem.triads) == len(disk.triads)
    for a, b in zip(mem.triads, disk.triads):
        assert torch.allclose(a.Eg, b.Eg, atol=1e-12)
        assert torch.allclose(a.Ef, b.Ef, atol=1e-12)


# ─────────────────────────────────────────────────────────────
# Parameter semantics and gradients
# ─────────────────────────────────────────────────────────────

def test_rescale_is_linear_in_the_fitted_parts(nid8ct_plan):
    """H(slater, soc) − H(0.8, 1) = (slater/0.8 − 1)·S + (soc − 1)·Z, block by block."""
    dec = load_hamiltonian_decomposition(NID8CT_RCG)
    built = build_cowan_store_in_memory(nid8ct_plan, slater=0.5, soc=1.3, source_rcg_path=NID8CT_RCG)
    for cfg in dec.configs:
        for J, j in cfg.block_index.items():
            want = cfg.fixture[J] + (0.5 / 0.8 - 1) * cfg.slater_part[J] + 0.3 * cfg.soc_part[J]
            assert torch.allclose(built[cfg.section][j], want, atol=1e-12)


def test_zero_slater_leaves_configuration_average_and_soc(nid8ct_plan):
    """slater=0 removes every F^k/G^k: the ground d⁸ block is E_av + ζ·V only (Fortran oracle npz)."""
    oracle = np.load(FORTRAN_OPS / "ni2_oh_sc_hamiltonian.npz")
    dec = load_hamiltonian_decomposition(_rcg("ni2_d8_oh"))
    built = build_cowan_store_in_memory(_plan_for("ni2_d8_oh"), slater=0.0, soc=1.0,
                                        source_rcg_path=_rcg("ni2_d8_oh"))
    cfg = dec.config(2, "GROUND")
    for J, j in cfg.block_index.items():
        # oracle run is at 100%/100%: its no-Slater block is the same E_av + ζ·V
        want = oracle[f"gs_H_noslater_J{int(J)}"]
        np.testing.assert_allclose(built[2][j].numpy(), want, atol=3e-3)


def test_gradients_are_the_operator_parts(nid8ct_plan):
    slater = torch.tensor(0.8, dtype=DTYPE, requires_grad=True)
    soc = torch.tensor(1.0, dtype=DTYPE, requires_grad=True)
    built = build_cowan_store_in_memory(nid8ct_plan, slater=slater, soc=soc, source_rcg_path=NID8CT_RCG)
    dec = load_hamiltonian_decomposition(NID8CT_RCG)
    cfg = dec.config(3, "GROUND")
    J, j = next(iter(cfg.block_index.items()))
    g_sl, g_soc = torch.autograd.grad(built[3][j].sum(), [slater, soc])
    assert g_sl.item() == pytest.approx(cfg.slater_part[J].sum().item() / 0.8, rel=1e-12)
    assert g_soc.item() == pytest.approx(cfg.soc_part[J].sum().item(), rel=1e-12)


def test_gradient_isolation_on_a_soc_only_configuration(nid8ct_plan):
    """Ligand-hole d⁹L̲ ground: only ζ3d is nonzero, so ∂H/∂slater = 0 exactly."""
    slater = torch.tensor(0.8, dtype=DTYPE, requires_grad=True)
    soc = torch.tensor(1.0, dtype=DTYPE, requires_grad=True)
    built = build_cowan_store_in_memory(nid8ct_plan, slater=slater, soc=soc, source_rcg_path=NID8CT_RCG)
    cfg = load_hamiltonian_decomposition(NID8CT_RCG).config(2, "EXCITE")
    assert cfg.shells == ((2, 9), (2, 9))
    loss = sum((built[2][j] ** 2).sum() for j in cfg.block_index.values())
    g_sl, g_soc = torch.autograd.grad(loss, [slater, soc])
    assert abs(g_sl.item()) < 1e-10
    assert abs(g_soc.item()) > 1e-6
