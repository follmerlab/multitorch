"""
Track C2 — Tests for ``build_ban.modify_ban_params``.

Validates that the template-based BanData modifier correctly overrides
crystal-field, charge-transfer, and hybridization parameters while
preserving structural fields (triads, nconf, tran).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from multitorch.io.read_ban import read_ban, BanData
from multitorch.hamiltonian.build_ban import modify_ban_params

REFDATA = Path(__file__).parent.parent / "reference_data"


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def nid8ct_ban():
    """D4h fixture (4 XHAM values, 4 XMIX values)."""
    return read_ban(REFDATA / "nid8ct" / "nid8ct.ban")


@pytest.fixture
def ni2_oh_ban():
    """Oh fixture (2 XHAM values, 2 XMIX values)."""
    return read_ban(REFDATA / "ni2_d8_oh" / "ni2_d8_oh.ban")


# ─────────────────────────────────────────────────────────────
# Identity: no overrides → identical to template
# ─────────────────────────────────────────────────────────────

def test_no_overrides_preserves_all(nid8ct_ban):
    """modify_ban_params with no kwargs returns an equivalent BanData."""
    out = modify_ban_params(nid8ct_ban)
    assert out.xham[0].values == nid8ct_ban.xham[0].values
    assert out.xmix[0].values == nid8ct_ban.xmix[0].values
    assert out.eg == nid8ct_ban.eg
    assert out.ef == nid8ct_ban.ef
    assert out.triads == nid8ct_ban.triads
    assert out.tran == nid8ct_ban.tran
    assert out.nconf_gs == nid8ct_ban.nconf_gs
    assert out.nconf_fs == nid8ct_ban.nconf_fs


def test_no_overrides_is_a_copy(nid8ct_ban):
    """Returned BanData is a copy — mutating it doesn't affect the original."""
    out = modify_ban_params(nid8ct_ban)
    out.xham[0].values[1] = 999.0
    assert nid8ct_ban.xham[0].values[1] != 999.0


# ─────────────────────────────────────────────────────────────
# Crystal field overrides
# ─────────────────────────────────────────────────────────────

def test_cf_override_d4h(nid8ct_ban):
    """D4h: cf dict maps to xham[0].values = [1.0, tendq, dt, ds]."""
    out = modify_ban_params(nid8ct_ban, cf={'tendq': 2.5, 'dt': 0.05, 'ds': -0.1})
    assert out.xham[0].values == [1.0, 2.5, 0.05, -0.1]


def test_cf_override_d4h_partial(nid8ct_ban):
    """D4h: only tendq overridden; dt/ds remain from template."""
    out = modify_ban_params(nid8ct_ban, cf={'tendq': 1.5})
    assert out.xham[0].values[0] == 1.0
    assert out.xham[0].values[1] == 1.5
    # dt and ds remain from template (0.0 and 0.1 for nid8ct)
    assert out.xham[0].values[2] == nid8ct_ban.xham[0].values[2]
    assert out.xham[0].values[3] == nid8ct_ban.xham[0].values[3]


def test_cf_override_oh(ni2_oh_ban):
    """Oh: cf dict maps to xham[0].values = [1.0, tendq]."""
    out = modify_ban_params(ni2_oh_ban, cf={'tendq': 0.8})
    assert out.xham[0].values == [1.0, 0.8]


def test_cf_preserves_structure(nid8ct_ban):
    """CF override does not change triads, nconf, tran."""
    out = modify_ban_params(nid8ct_ban, cf={'tendq': 2.0})
    assert out.triads == nid8ct_ban.triads
    assert out.tran == nid8ct_ban.tran
    assert out.nconf_gs == nid8ct_ban.nconf_gs


# ─────────────────────────────────────────────────────────────
# Delta overrides
# ─────────────────────────────────────────────────────────────

def test_delta_float(nid8ct_ban):
    """Float delta sets eg[2]."""
    out = modify_ban_params(nid8ct_ban, delta=6.0)
    assert out.eg[2] == 6.0
    # ef should be unchanged from template
    assert out.ef == nid8ct_ban.ef


def test_delta_dict(nid8ct_ban):
    """Dict delta sets eg2 and ef2 independently."""
    out = modify_ban_params(nid8ct_ban, delta={'eg2': 7.0, 'ef2': 3.0})
    assert out.eg[2] == 7.0
    assert out.ef[2] == 3.0


def test_delta_dict_partial(nid8ct_ban):
    """Dict delta with only ef2 leaves eg unchanged."""
    out = modify_ban_params(nid8ct_ban, delta={'ef2': 2.5})
    assert out.eg == nid8ct_ban.eg
    assert out.ef[2] == 2.5


# ─────────────────────────────────────────────────────────────
# Hybridization overrides
# ─────────────────────────────────────────────────────────────

def test_lmct_float(nid8ct_ban):
    """Float lmct fills all XMIX channels."""
    out = modify_ban_params(nid8ct_ban, lmct=3.0)
    assert out.xmix[0].values == [3.0, 3.0, 3.0, 3.0]


def test_lmct_list(nid8ct_ban):
    """List lmct sets XMIX channels directly."""
    out = modify_ban_params(nid8ct_ban, lmct=[1.0, 2.0, 3.0, 4.0])
    assert out.xmix[0].values == [1.0, 2.0, 3.0, 4.0]


def test_lmct_list_wrong_length(nid8ct_ban):
    """List lmct with wrong length raises ValueError."""
    with pytest.raises(ValueError, match="lmct has 2 values"):
        modify_ban_params(nid8ct_ban, lmct=[1.0, 2.0])


def test_lmct_oh(ni2_oh_ban):
    """Oh: float lmct fills both XMIX channels."""
    out = modify_ban_params(ni2_oh_ban, lmct=1.5)
    assert out.xmix[0].values == [1.5, 1.5]


# ─────────────────────────────────────────────────────────────
# Combined overrides
# ─────────────────────────────────────────────────────────────

def test_combined_overrides(nid8ct_ban):
    """All three overrides can be applied simultaneously."""
    out = modify_ban_params(
        nid8ct_ban,
        cf={'tendq': 2.0, 'dt': 0.03, 'ds': -0.05},
        delta=6.5,
        lmct=2.5,
    )
    assert out.xham[0].values == [1.0, 2.0, 0.03, -0.05]
    assert out.eg[2] == 6.5
    assert out.xmix[0].values == [2.5, 2.5, 2.5, 2.5]
    # Structure unchanged
    assert out.triads == nid8ct_ban.triads
    assert out.tran == nid8ct_ban.tran


# ─────────────────────────────────────────────────────────────
# End-to-end: modified BanData through assembler
# ─────────────────────────────────────────────────────────────

def test_modified_ban_assembles(nid8ct_ban):
    """Modified BanData can be consumed by assemble_and_diagonalize_in_memory."""
    from multitorch.hamiltonian.assemble import assemble_and_diagonalize_in_memory
    from multitorch.hamiltonian.build_rac import build_rac_in_memory
    from multitorch.io.read_rme import read_cowan_store

    rcg_path = REFDATA / "nid8ct" / "nid8ct.rme_rcg"
    rac_path = REFDATA / "nid8ct" / "nid8ct.rme_rac"

    # Modify CF to a different value
    out = modify_ban_params(nid8ct_ban, cf={'tendq': 1.5, 'dt': 0.0, 'ds': 0.05})

    cowan = read_cowan_store(rcg_path)
    rac, plan = build_rac_in_memory(out, source_rac_path=rac_path, source_rcg_path=rcg_path)
    result = assemble_and_diagonalize_in_memory(cowan, rac, out)

    # Should produce valid triads
    assert len(result.triads) == len(nid8ct_ban.triads)
    for triad in result.triads:
        assert triad.Eg.shape[0] > 0
        assert triad.Ef.shape[0] > 0


def test_template_vs_modified_eigenvalues_differ(nid8ct_ban):
    """Changing CF parameters should produce different eigenvalues."""
    import torch
    from multitorch.hamiltonian.assemble import (
        assemble_and_diagonalize,
        assemble_and_diagonalize_in_memory,
    )
    from multitorch.hamiltonian.build_rac import build_rac_in_memory
    from multitorch.io.read_rme import read_cowan_store

    rcg_path = REFDATA / "nid8ct" / "nid8ct.rme_rcg"
    rac_path = REFDATA / "nid8ct" / "nid8ct.rme_rac"

    # Template (original) result
    ref = assemble_and_diagonalize(rcg_path, rac_path, REFDATA / "nid8ct" / "nid8ct.ban")

    # Modified CF
    mod = modify_ban_params(nid8ct_ban, cf={'tendq': 2.0, 'dt': 0.0, 'ds': 0.2})
    cowan = read_cowan_store(rcg_path)
    rac, plan = build_rac_in_memory(mod, source_rac_path=rac_path, source_rcg_path=rcg_path)
    result = assemble_and_diagonalize_in_memory(cowan, rac, mod)

    # Eigenvalues should differ (different CF)
    assert not torch.allclose(result.triads[0].Eg, ref.triads[0].Eg, atol=1e-6)
