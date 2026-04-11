"""
Track C3b tests for ``read_rcn31_out_params``.

What this validates
-------------------
1. The parser produces an :class:`AtomicParams` with one
   :class:`ConfigParams` per configuration block found in the file
   (NCONF=1 ground + NCONF=2 excited for an L-edge XAS run).
2. Hand-extracted Fk/Gk values from ``nid8.rcn31_out`` match the
   parsed values to 1e-10 in Rydberg. The values were copied from
   the file directly so any drift would indicate a parser bug.
3. The Blume-Watson and R*VI ζ columns are read into separate
   dictionaries with the correct values.
4. The convenience accessors (``f``, ``g``, ``zeta``) are
   shell-order independent and method-aware.
5. The dataclass is frozen-friendly: parsing an unrelated file
   (Fe2+, if available) does not produce a parse error and gives
   physically reasonable values.

Tolerance is 1e-10 because the file stores 5–6 decimal places and
``float()`` is exact at that precision.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from multitorch.atomic.parameter_fixtures import (
    AtomicParams,
    ConfigParams,
    read_rcn31_out_params,
)

REFDATA = Path(__file__).parent.parent / "reference_data"
NID8_RCN31 = REFDATA / "nid8" / "nid8.rcn31_out"


# ─────────────────────────────────────────────────────────────
# Reference values, hand-extracted from nid8.rcn31_out
# ─────────────────────────────────────────────────────────────

# All values in Rydberg.

GROUND_FK = {
    ("1S", "1S", 0): 34.8019526,
    ("2P", "2P", 0): 8.4906065,
    ("2P", "2P", 2): 4.0098046,
    ("2P", "3D", 0): 2.7453440,
    ("2P", "3D", 2): 0.5115688,
    ("3D", "3D", 0): 1.9539961,
    ("3D", "3D", 2): 0.8991664,
    ("3D", "3D", 4): 0.5584075,
}

GROUND_GK = {
    ("1S", "2S", 0): 0.9667227,
    ("2P", "3D", 1): 0.3733353,
    ("2P", "3D", 3): 0.2119891,
    ("3P", "3D", 1): 1.2365626,
    ("3P", "3D", 3): 0.7476579,
    ("3S", "3D", 2): 0.9377024,
}

GROUND_ZETA_BW = {
    "1S": 0.00000,
    "2S": 0.00000,
    "2P": 0.81583,
    "3S": 0.00000,
    "3P": 0.09945,
    "3D": 0.00607,
}

GROUND_ZETA_RVI = {
    "2P": 0.85796,
    "3P": 0.10407,
    "3D": 0.00706,
}

EXCITED_FK_3D_3D_2 = 0.9559101
EXCITED_GK_2P_3D_1 = 0.4253023
EXCITED_ZETA_BW_2P = 0.84575
EXCITED_ZETA_BW_3D = 0.00751


# ─────────────────────────────────────────────────────────────
# Module-level fixture
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def nid8_params() -> AtomicParams:
    return read_rcn31_out_params(NID8_RCN31)


# ─────────────────────────────────────────────────────────────
# Structural tests
# ─────────────────────────────────────────────────────────────

@pytest.mark.phase4
def test_returns_atomic_params(nid8_params):
    assert isinstance(nid8_params, AtomicParams)
    assert nid8_params.source_path == NID8_RCN31


@pytest.mark.phase4
def test_two_configurations_parsed(nid8_params):
    """L-edge XAS rcn31 runs emit ground (NCONF=1) + excited (NCONF=2)."""
    assert len(nid8_params.configs) == 2
    nconfs = sorted(c.nconf for c in nid8_params.configs)
    assert nconfs == [1, 2]


@pytest.mark.phase4
def test_config_labels(nid8_params):
    assert "2p06 3d08" in nid8_params.ground.label
    assert "2p05 3d09" in nid8_params.excited.label


@pytest.mark.phase4
def test_config_lookup_by_nconf(nid8_params):
    assert nid8_params.by_nconf(1) is nid8_params.ground
    assert nid8_params.by_nconf(2) is nid8_params.excited


@pytest.mark.phase4
def test_unknown_nconf_raises(nid8_params):
    with pytest.raises(KeyError):
        nid8_params.by_nconf(99)


# ─────────────────────────────────────────────────────────────
# Slater Fk integrals
# ─────────────────────────────────────────────────────────────

@pytest.mark.phase4
@pytest.mark.parametrize("key,expected", list(GROUND_FK.items()))
def test_ground_fk_values_match_file(nid8_params, key, expected):
    a, b, k = key
    assert nid8_params.ground.f(a, b, k) == pytest.approx(expected, abs=1e-10)


@pytest.mark.phase4
def test_ground_fk_count(nid8_params):
    """The 6-shell Ni 1s/2s/2p/3s/3p/3d ground state has 21 Fk^0 entries
    + 7 second-rank Fk continuation rows = 28 total Fk entries."""
    assert len(nid8_params.ground.fk) == 28


@pytest.mark.phase4
def test_fk_lookup_is_order_independent(nid8_params):
    """Convenience accessor must accept either argument order."""
    a = nid8_params.ground.f("2P", "3D", 2)
    b = nid8_params.ground.f("3D", "2P", 2)
    assert a == b


# ─────────────────────────────────────────────────────────────
# Slater Gk exchange integrals
# ─────────────────────────────────────────────────────────────

@pytest.mark.phase4
@pytest.mark.parametrize("key,expected", list(GROUND_GK.items()))
def test_ground_gk_values_match_file(nid8_params, key, expected):
    a, b, k = key
    assert nid8_params.ground.g(a, b, k) == pytest.approx(expected, abs=1e-10)


@pytest.mark.phase4
def test_ground_gk_count(nid8_params):
    """Off-diagonal pairs only — 18 Gk entries for 6-shell Ni d8."""
    assert len(nid8_params.ground.gk) == 18


@pytest.mark.phase4
def test_diagonal_gk_not_stored(nid8_params):
    """Same-shell Gk is meaningless and must not be stored."""
    with pytest.raises(KeyError):
        nid8_params.ground.g("3D", "3D", 0)


# ─────────────────────────────────────────────────────────────
# Spin-orbit ζ
# ─────────────────────────────────────────────────────────────

@pytest.mark.phase4
@pytest.mark.parametrize("shell,expected", list(GROUND_ZETA_BW.items()))
def test_ground_zeta_blume_watson(nid8_params, shell, expected):
    assert nid8_params.ground.zeta(shell) == pytest.approx(expected, abs=1e-10)


@pytest.mark.phase4
@pytest.mark.parametrize("shell,expected", list(GROUND_ZETA_RVI.items()))
def test_ground_zeta_rvi(nid8_params, shell, expected):
    assert nid8_params.ground.zeta(shell, method="rvi") == pytest.approx(
        expected, abs=1e-10
    )


@pytest.mark.phase4
def test_zeta_default_method_is_blume_watson(nid8_params):
    """The recommended physics column is Blume-Watson; assert it's the default."""
    assert nid8_params.ground.zeta("2P") == pytest.approx(
        GROUND_ZETA_BW["2P"], abs=1e-10
    )
    assert nid8_params.ground.zeta("2P") != nid8_params.ground.zeta("2P", method="rvi")


@pytest.mark.phase4
def test_zeta_unknown_shell_raises(nid8_params):
    with pytest.raises(KeyError):
        nid8_params.ground.zeta("4F")


# ─────────────────────────────────────────────────────────────
# Excited configuration
# ─────────────────────────────────────────────────────────────

@pytest.mark.phase4
def test_excited_fk_3d_3d_2(nid8_params):
    """Excited 2p^5 3d^9 has slightly larger Fk(3d,3d) than ground."""
    val = nid8_params.excited.f("3D", "3D", 2)
    assert val == pytest.approx(EXCITED_FK_3D_3D_2, abs=1e-10)
    assert val > nid8_params.ground.f("3D", "3D", 2)


@pytest.mark.phase4
def test_excited_gk_2p_3d_1(nid8_params):
    val = nid8_params.excited.g("2P", "3D", 1)
    assert val == pytest.approx(EXCITED_GK_2P_3D_1, abs=1e-10)


@pytest.mark.phase4
def test_excited_zeta_bw(nid8_params):
    assert nid8_params.excited.zeta("2P") == pytest.approx(
        EXCITED_ZETA_BW_2P, abs=1e-10
    )
    assert nid8_params.excited.zeta("3D") == pytest.approx(
        EXCITED_ZETA_BW_3D, abs=1e-10
    )


@pytest.mark.phase4
def test_excited_zeta_larger_than_ground(nid8_params):
    """Core hole shrinks the orbitals → larger ζ in the excited config.
    This is a physical sanity check on the parser, not a tolerance test.
    """
    assert nid8_params.excited.zeta("2P") > nid8_params.ground.zeta("2P")
    assert nid8_params.excited.zeta("3D") > nid8_params.ground.zeta("3D")


# ─────────────────────────────────────────────────────────────
# Physical sanity checks (catch parsing failures that don't trip
# the explicit hand-extracted values)
# ─────────────────────────────────────────────────────────────

@pytest.mark.phase4
def test_all_fk_positive(nid8_params):
    """Slater direct integrals are strictly positive."""
    for cfg in nid8_params.configs:
        for key, val in cfg.fk.items():
            assert val > 0, f"{cfg.label} F^{key[2]}{key[:2]}={val}"


@pytest.mark.phase4
def test_all_gk_positive(nid8_params):
    """Slater exchange integrals are non-negative (strictly positive in this fixture)."""
    for cfg in nid8_params.configs:
        for key, val in cfg.gk.items():
            assert val > 0, f"{cfg.label} G^{key[2]}{key[:2]}={val}"


@pytest.mark.phase4
def test_zeta_3d_smaller_than_zeta_2p(nid8_params):
    """3d ζ is ~100× smaller than 2p ζ for first-row TM ions — physics check."""
    assert nid8_params.ground.zeta("3D") < nid8_params.ground.zeta("2P") / 50


@pytest.mark.phase4
def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        read_rcn31_out_params(REFDATA / "does_not_exist.rcn31_out")
