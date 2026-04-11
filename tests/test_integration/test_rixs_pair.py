"""
Tests for the bootstrap RIXS pipeline:
  multitorch.io.read_oba_pair  + multitorch.api.calc.calcRIXS

These tests use synthetic ``BanOutput`` / ``RIXSStore`` fixtures because the
repository does not yet ship a paired absorption + emission ``.ban_out``
fixture (the existing ``nid8ct.ban_out`` is absorption-only, with ground
syms ``[0+, 1+, 2+, ^0+, ^2+]`` and final syms ``[0-, 1-, 2-, ^0-, ^2-]``,
so it does not self-pair).

When a real paired fixture is committed (e.g. ``nid8ct.abs.ban_out`` +
``nid8ct.ems.ban_out`` plus a pyctm-generated ``nid8ct.rixs.npz`` reference),
add a ``test_calcRIXS_matches_pyctm`` here that asserts cosine similarity
≥ 0.99 between ``calcRIXS(...)`` and the reference 2D map.
"""
from __future__ import annotations

import math

import pytest
import torch

from multitorch._constants import DTYPE
from multitorch.api.calc import calcRIXS
from multitorch.api.plot import getRIXS
from multitorch.io.read_oba import BanOutput, TriadData
from multitorch.io.read_oba_pair import (
    RIXSChannel,
    RIXSStore,
    _group_by_triad,
    _stack_triad_group,
    build_rixs_store,
)
from multitorch.spectrum.rixs import kramers_heisenberg


# ---------------------------------------------------------------------------
# Synthetic-data factories
# ---------------------------------------------------------------------------


def _td(
    g_sym: str,
    op_sym: str,
    f_sym: str,
    Eg: list[float],
    Ef: list[float],
    M: list[list[float]],
    actor: str = "MULTIPOLE",
) -> TriadData:
    return TriadData(
        ground_sym=g_sym,
        op_sym=op_sym,
        final_sym=f_sym,
        actor=actor,
        Eg=torch.tensor(Eg, dtype=DTYPE),
        Ef=torch.tensor(Ef, dtype=DTYPE),
        M=torch.tensor(M, dtype=DTYPE),
    )


def _toy_pair() -> tuple[BanOutput, BanOutput]:
    """Build a minimal absorption / emission pair with a single channel.

    One ground sym 'g+', one intermediate sym 'i-', one final sym 'f+'.
    Absorption: 1 ground state → 4 intermediate states.
    Emission:   4 intermediate states → 3 final states.
    """
    abs_bo = BanOutput()
    abs_bo.triad_list.append(
        _td(
            g_sym="g+", op_sym="1-", f_sym="i-",
            Eg=[-100.0],                                  # Ry
            Ef=[850.0, 851.5, 853.0, 854.5],              # eV
            M=[[1.0, 0.5, 0.25, 0.1]],
        )
    )
    ems_bo = BanOutput()
    ems_bo.triad_list.append(
        _td(
            g_sym="i-", op_sym="1-", f_sym="f+",
            Eg=[850.1, 851.6, 853.1, 854.6],              # Ry — discarded
            Ef=[1.0, 2.0, 3.0],                           # eV
            M=[
                [0.9, 0.1, 0.05],
                [0.4, 0.6, 0.05],
                [0.2, 0.3, 0.5],
                [0.05, 0.1, 0.85],
            ],
        )
    )
    return abs_bo, ems_bo


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def test_group_by_triad_keys_on_full_triad_and_actor():
    bo = BanOutput()
    t1 = _td("g+", "1-", "i-", [-1.0], [850.0], [[1.0]])
    t2 = _td("g+", "1-", "i-", [-1.0], [850.0], [[2.0]])  # same key
    t3 = _td("g+", "1-", "j-", [-1.0], [851.0], [[3.0]])  # different f
    bo.triad_list.extend([t1, t2, t3])

    groups = _group_by_triad(bo)

    assert ("g+", "1-", "i-", "MULTIPOLE") in groups
    assert ("g+", "1-", "j-", "MULTIPOLE") in groups
    assert len(groups[("g+", "1-", "i-", "MULTIPOLE")]) == 2
    assert len(groups[("g+", "1-", "j-", "MULTIPOLE")]) == 1


def test_stack_triad_group_concatenates_bra_axis():
    t1 = _td("g+", "1-", "i-", [-1.0],       [850.0, 851.0], [[1.0, 2.0]])
    t2 = _td("g+", "1-", "i-", [-0.9, -0.8], [850.0, 851.0], [[3.0, 4.0], [5.0, 6.0]])

    merged = _stack_triad_group([t1, t2])

    assert merged.Eg.shape == (3,)
    assert merged.M.shape == (3, 2)
    # Ef is shared (must match) — taken from the first entry.
    assert torch.allclose(merged.Ef, t1.Ef)
    expected_M = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=DTYPE)
    assert torch.allclose(merged.M, expected_M)


def test_stack_triad_group_rejects_mismatched_kets():
    t1 = _td("g+", "1-", "i-", [-1.0], [850.0, 851.0],     [[1.0, 2.0]])
    t2 = _td("g+", "1-", "i-", [-0.9], [850.0, 852.0],     [[3.0, 4.0]])

    with pytest.raises(ValueError, match="ket grids differ"):
        _stack_triad_group([t1, t2])


# ---------------------------------------------------------------------------
# build_rixs_store pairing logic
# ---------------------------------------------------------------------------


def test_build_rixs_store_pairs_one_channel():
    abs_bo, ems_bo = _toy_pair()
    store = build_rixs_store(abs_bo, ems_bo)

    assert len(store.channels) == 1
    ch = store.channels[0]
    assert ch.key == ("g+", "1-", "i-", "1-", "f+")
    assert ch.Eg.shape == (1,)
    assert ch.TA.shape == (1, 4)
    assert ch.Ei.shape == (4,)
    assert ch.TE.shape == (4, 3)
    assert ch.Ef.shape == (3,)
    # Intermediate energies come from the absorption file's ket grid (eV).
    assert torch.allclose(ch.Ei, torch.tensor([850.0, 851.5, 853.0, 854.5], dtype=DTYPE))
    # All tensors are float64.
    for t in (ch.Eg, ch.TA, ch.Ei, ch.TE, ch.Ef):
        assert t.dtype == DTYPE


def test_build_rixs_store_skips_unmatched_intermediate():
    """Absorption triad whose final_sym has no emission match is dropped."""
    abs_bo, ems_bo = _toy_pair()
    abs_bo.triad_list.append(
        _td("g+", "1-", "i_orphan-", [-100.0], [870.0, 871.0], [[1.0, 1.0]])
    )
    store = build_rixs_store(abs_bo, ems_bo)
    # Still exactly one channel — the orphan absorption triad is skipped.
    assert len(store.channels) == 1
    assert store.channels[0].key[2] == "i-"


def test_build_rixs_store_truncates_to_smaller_intermediate_count():
    """When n_i_abs != n_i_ems, both sides are trimmed to the minimum."""
    abs_bo = BanOutput()
    abs_bo.triad_list.append(
        _td(
            "g+", "1-", "i-",
            Eg=[-1.0],
            Ef=[850.0, 851.0, 852.0, 853.0, 854.0],   # 5 intermediates
            M=[[1.0, 1.0, 1.0, 1.0, 1.0]],
        )
    )
    ems_bo = BanOutput()
    ems_bo.triad_list.append(
        _td(
            "i-", "1-", "f+",
            Eg=[850.0, 851.0, 852.0],                 # 3 intermediates
            Ef=[1.0, 2.0],
            M=[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
        )
    )
    store = build_rixs_store(abs_bo, ems_bo)
    ch = store.channels[0]
    assert ch.TA.shape == (1, 3)
    assert ch.TE.shape == (3, 2)
    assert ch.Ei.shape == (3,)


def test_build_rixs_store_returns_empty_when_no_pairs():
    """Two BanOutputs that share no intermediate symmetries → empty channels."""
    abs_bo = BanOutput()
    abs_bo.triad_list.append(_td("g+", "1-", "i-", [-1.0], [850.0], [[1.0]]))
    ems_bo = BanOutput()
    ems_bo.triad_list.append(_td("xx-", "1-", "f+", [1.0], [2.0], [[1.0]]))

    store = build_rixs_store(abs_bo, ems_bo)
    assert store.channels == []


# ---------------------------------------------------------------------------
# RIXSStore properties
# ---------------------------------------------------------------------------


def test_rixs_store_min_gs_picks_lowest_across_channels():
    abs_bo = BanOutput()
    abs_bo.triad_list.append(_td("g+", "1-", "i-", [-1.0], [850.0], [[1.0]]))
    abs_bo.triad_list.append(_td("h+", "1-", "i-", [-1.5], [850.0], [[1.0]]))
    ems_bo = BanOutput()
    ems_bo.triad_list.append(_td("i-", "1-", "f+", [850.0], [1.0], [[1.0]]))

    store = build_rixs_store(abs_bo, ems_bo)
    assert store.min_gs == pytest.approx(-1.5)


def test_rixs_store_channels_for_ground_sym_filters():
    abs_bo = BanOutput()
    abs_bo.triad_list.append(_td("g+", "1-", "i-", [-1.0], [850.0], [[1.0]]))
    abs_bo.triad_list.append(_td("h+", "1-", "i-", [-1.0], [850.0], [[1.0]]))
    ems_bo = BanOutput()
    ems_bo.triad_list.append(_td("i-", "1-", "f+", [850.0], [1.0], [[1.0]]))

    store = build_rixs_store(abs_bo, ems_bo)
    g_channels = store.channels_for_ground_sym("g+")
    assert len(g_channels) == 1
    assert g_channels[0].key[0] == "g+"


# ---------------------------------------------------------------------------
# calcRIXS end-to-end via monkey-patched parser
# ---------------------------------------------------------------------------


def _patch_parser(monkeypatch, abs_bo: BanOutput, ems_bo: BanOutput) -> None:
    """Make read_ban_output return our synthetic BanOutputs in calc.py."""
    def fake_read(path):
        return abs_bo if "abs" in str(path) else ems_bo
    # calcRIXS imports read_abs_ems_pair which imports read_ban_output;
    # patch the symbol on the module that read_abs_ems_pair uses.
    import multitorch.io.read_oba_pair as pair_mod
    monkeypatch.setattr(pair_mod, "read_ban_output", fake_read)


def test_calcRIXS_returns_2d_map_with_correct_shape(monkeypatch):
    abs_bo, ems_bo = _toy_pair()
    _patch_parser(monkeypatch, abs_bo, ems_bo)

    Einc, Efin, intensity = calcRIXS(
        ban_abs_path="fake_abs.ban_out",
        ban_ems_path="fake_ems.ban_out",
        n_Einc=64, n_Efin=48,
        Gamma_i=0.5, Gamma_f=0.3,
        T=0.0,
    )

    assert Einc.shape == (64,)
    assert Efin.shape == (48,)
    assert intensity.shape == (64, 48)
    assert Einc.dtype == DTYPE
    assert Efin.dtype == DTYPE
    assert intensity.dtype == DTYPE
    # Map is finite, real, non-negative.
    assert torch.isfinite(intensity).all()
    assert (intensity >= 0).all()
    # Some signal exists somewhere on the map.
    assert intensity.max() > 0.0


def test_calcRIXS_explicit_grids_are_respected(monkeypatch):
    abs_bo, ems_bo = _toy_pair()
    _patch_parser(monkeypatch, abs_bo, ems_bo)

    user_Einc = torch.linspace(848.0, 856.0, 32, dtype=DTYPE)
    user_Efin = torch.linspace(845.0, 856.0, 24, dtype=DTYPE)

    Einc, Efin, intensity = calcRIXS(
        ban_abs_path="fake_abs.ban_out",
        ban_ems_path="fake_ems.ban_out",
        Einc=user_Einc, Efin=user_Efin,
        T=0.0,
    )

    assert torch.allclose(Einc, user_Einc)
    assert torch.allclose(Efin, user_Efin)
    assert intensity.shape == (32, 24)


def test_calcRIXS_peaks_at_resonant_intermediate(monkeypatch):
    """Single absorption peak at Ei=850 eV → max signal near Einc=850.

    The kernel resonance condition is Eg + Einc - Ei = 0, so with Eg=0
    we expect the peak at Einc = Ei = 850 eV.
    """
    abs_bo = BanOutput()
    abs_bo.triad_list.append(
        _td(
            "g+", "1-", "i-",
            Eg=[0.0],
            Ef=[840.0, 845.0, 850.0, 855.0, 860.0],
            M=[[1e-6, 1e-6, 1.0, 1e-6, 1e-6]],   # all weight on i=850 eV
        )
    )
    ems_bo = BanOutput()
    ems_bo.triad_list.append(
        _td(
            "i-", "1-", "f+",
            Eg=[840.0, 845.0, 850.0, 855.0, 860.0],
            Ef=[1.0],
            M=[[1.0], [1.0], [1.0], [1.0], [1.0]],
        )
    )
    _patch_parser(monkeypatch, abs_bo, ems_bo)

    Einc, Efin, intensity = calcRIXS(
        ban_abs_path="fake_abs.ban_out",
        ban_ems_path="fake_ems.ban_out",
        n_Einc=200, n_Efin=200,
        Gamma_i=0.4, Gamma_f=0.2,
        T=0.0,
    )

    # The strongest column of the 2D map should sit near Einc = 850 eV.
    col_max = intensity.sum(dim=1)             # (n_Einc,)
    peak_idx = int(torch.argmax(col_max))
    peak_E = float(Einc[peak_idx])
    assert abs(peak_E - 850.0) < 1.0


def test_calcRIXS_raises_when_no_pair_paths():
    """Phase 5 stub still raises until tracks land."""
    with pytest.raises(NotImplementedError, match="Phase 5"):
        calcRIXS(element="Ni", valence="ii", sym="oh", edge="l", cf={})


def test_calcRIXS_raises_when_no_overlap(monkeypatch):
    abs_bo = BanOutput()
    abs_bo.triad_list.append(_td("g+", "1-", "i-", [-1.0], [850.0], [[1.0]]))
    ems_bo = BanOutput()
    ems_bo.triad_list.append(_td("xx-", "1-", "f+", [1.0], [2.0], [[1.0]]))
    _patch_parser(monkeypatch, abs_bo, ems_bo)

    with pytest.raises(ValueError, match="No matching"):
        calcRIXS(
            ban_abs_path="fake_abs.ban_out",
            ban_ems_path="fake_ems.ban_out",
        )


def test_getRIXS_wrapper_dispatches(monkeypatch):
    abs_bo, ems_bo = _toy_pair()
    _patch_parser(monkeypatch, abs_bo, ems_bo)

    Einc, Efin, intensity = getRIXS(
        ban_abs_path="fake_abs.ban_out",
        ban_ems_path="fake_ems.ban_out",
        n_Einc=32, n_Efin=32,
        T=0.0,
    )
    assert intensity.shape == (32, 32)
    assert torch.isfinite(intensity).all()


# ---------------------------------------------------------------------------
# Autograd: gradient flows back through the kernel to a tunable parameter
# ---------------------------------------------------------------------------


def test_kramers_heisenberg_is_autograd_safe():
    """Backprop a scalar loss through the kernel to verify gradient flow."""
    Eg = torch.tensor([-100.0], dtype=DTYPE)
    TA_scale = torch.tensor(1.0, dtype=DTYPE, requires_grad=True)
    TA = TA_scale * torch.tensor([[1.0, 0.5, 0.25]], dtype=DTYPE)
    Ei = torch.tensor([850.0, 852.0, 854.0], dtype=DTYPE)
    TE = torch.tensor(
        [[0.9, 0.1], [0.4, 0.6], [0.2, 0.8]], dtype=DTYPE
    )
    Ef = torch.tensor([1.0, 2.0], dtype=DTYPE)
    Einc = torch.linspace(848.0, 856.0, 64, dtype=DTYPE)
    Efin = torch.linspace(848.0, 856.0, 64, dtype=DTYPE)

    rm = kramers_heisenberg(
        Eg=Eg, TA=TA, Ei=Ei, TE=TE, Ef=Ef,
        Einc=Einc, Efin=Efin,
        Gamma_i=0.4, Gamma_f=0.2,
        min_gs=-100.0, T=0.0,
    )
    loss = rm.sum()
    loss.backward()

    assert TA_scale.grad is not None
    assert torch.isfinite(TA_scale.grad).all()
    # Loss is quadratic in TA → gradient at TA_scale=1 is non-zero.
    assert TA_scale.grad.abs().item() > 0.0
