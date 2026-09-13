"""WP-A1: the parameter-linear contraction shared by the fixture and from-scratch builders.

The physics oracles live elsewhere (operators vs ttrcg in
test_hamiltonian_operators.py / test_build_cowan.py; from-scratch spectra vs
Fortran in test_from_scratch_fortran_parity.py). These tests pin the contract of
the seam itself: linearity, overrides, naming, batching, and that both
builders go through it.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from multitorch._constants import DTYPE
from multitorch.angular.rac_generator import generate_ledge_rac, generate_ledge_template, ledge_reference_ev
from multitorch.hamiltonian.build_cowan import load_hamiltonian_decomposition
from multitorch.hamiltonian.parametric import physical_name, rebuild_hamiltonian_store

REFDATA = Path(__file__).parent.parent / "reference_data"
RY = dict(
    raw_slater_gs_ry={"F2": 0.73, "F4": 0.46}, raw_zeta_gs_ry=0.0061,
    raw_slater_ex_ry={"F2_dd": 0.78, "F4_dd": 0.49, "F2_pd": 0.41, "G1_pd": 0.30, "G3_pd": 0.17},
    raw_zeta_ex_ry={"p": 0.85, "d": 0.0075},
)


@pytest.fixture(scope="module")
def fe2_template():
    rac, store, dec = generate_ledge_template(2, 6, sym="oh")
    gs, ex = ledge_reference_ev(2, *RY.values())
    dec.by_label("gs").reference, dec.by_label("ex").reference = gs, ex
    return rac, store, dec


def test_physical_names():
    core_first, valence_first = ((1, 5), (2, 7)), ((2, 9), (1, 5))
    assert physical_name("F2_12", core_first) == "F2pd"
    assert physical_name("G3_12", valence_first) == "G3pd"
    assert physical_name("F4_22", core_first) == "F4dd"
    assert physical_name("zeta_1", core_first) == "zeta_p"
    assert physical_name("zeta_1", valence_first) == "zeta_d"


def test_from_scratch_aliases(fe2_template):
    _, _, dec = fe2_template
    assert set(dec.by_label("gs").aliases()) == {"F2dd", "F4dd", "zeta_d"}
    assert set(dec.by_label("ex").aliases()) == {"F2dd", "F4dd", "F2pd", "G1pd", "G3pd", "zeta_p", "zeta_d"}


def test_ambiguous_alias_is_not_offered():
    """Fixture ligand-hole d⁹ d⁹ configuration: two ζ_d, so only operator names address them."""
    cfg = load_hamiltonian_decomposition(REFDATA / "nid8ct" / "nid8ct.rme_rcg").config(2, "EXCITE")
    assert cfg.shells == ((2, 9), (2, 9))
    assert "zeta_d" not in cfg.aliases()
    with pytest.raises(KeyError):
        cfg.parameter_values(0.8, 1.0, {"zeta_d": 0.1})


def test_generate_ledge_rac_is_the_template_contraction(fe2_template):
    _, store, dec = fe2_template
    _, direct = generate_ledge_rac(2, 6, sym="oh", **RY)
    rebuilt = rebuild_hamiltonian_store(store, dec, slater=1.0, soc=1.0)
    assert all(torch.equal(a, b) for a, b in zip(direct[0], rebuilt[0]))


def test_overrides_at_the_scaled_values_change_nothing(fe2_template):
    _, store, dec = fe2_template
    base = rebuild_hamiltonian_store(store, dec, slater=0.7, soc=0.9)
    atomic = {}
    for label in ("gs", "ex"):
        cfg = dec.by_label(label)
        values = cfg.parameter_values(0.7, 0.9)
        atomic[label] = {alias: values[name] for alias, name in cfg.aliases().items()}
    over = rebuild_hamiltonian_store(store, dec, slater=0.1, soc=3.0, atomic=atomic)
    for a, b in zip(base[0], over[0]):
        assert torch.allclose(a, b, rtol=0, atol=1e-12)


def test_override_is_linear_in_its_operator(fe2_template):
    _, store, dec = fe2_template
    ex = dec.by_label("ex")
    g1 = ex.parameter_values(0.8, 1.0)["G1_12"]
    base = rebuild_hamiltonian_store(store, dec, slater=0.8, soc=1.0)
    moved = rebuild_hamiltonian_store(store, dec, slater=0.8, soc=1.0, atomic={"ex": {"G1pd": g1 + 0.5}})
    for J, j in ex.block_index.items():
        assert torch.allclose(moved[0][j] - base[0][j], 0.5 * ex.operators["G1_12"][J], atol=1e-12)
    others = set(range(len(store[0]))) - set(ex.block_index.values())
    assert all(torch.equal(moved[0][j], base[0][j]) for j in others)


def test_unknown_names_raise(fe2_template):
    _, store, dec = fe2_template
    with pytest.raises(KeyError):
        rebuild_hamiltonian_store(store, dec, slater=0.8, soc=1.0, atomic={"excited": {"G1pd": 1.0}})
    with pytest.raises(KeyError):
        rebuild_hamiltonian_store(store, dec, slater=0.8, soc=1.0, atomic={"gs": {"G1pd": 1.0}})


def test_batched_leaves_give_batched_blocks(fe2_template):
    _, store, dec = fe2_template
    s = torch.tensor([0.6, 0.8, 1.0], dtype=DTYPE)
    z = torch.tensor([1.0, 0.5, 0.0], dtype=DTYPE)
    batch = rebuild_hamiltonian_store(store, dec, slater=s, soc=z)
    for i in range(3):
        single = rebuild_hamiltonian_store(store, dec, slater=float(s[i]), soc=float(z[i]))
        for cfg in dec.configs:
            for j in cfg.block_index.values():
                assert batch[0][j].shape[0] == 3
                assert torch.allclose(batch[0][j][i], single[0][j], atol=1e-12)
