"""
pytest fixtures for multitorch tests.

All reference data is loaded from tests/reference_data/ — pre-committed
Fortran outputs that are immutable. Tests never need to run Fortran binaries.
"""
from pathlib import Path
import pytest
import torch

REFDATA = Path(__file__).parent / "reference_data"


# ─────────────────────────────────────────────────────────────
# HFS / Slater parameter fixtures (Phase 4 reference)
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def nid8_hfs():
    """HFS orbital energies for Ni2+ from rcn31."""
    from multitorch.io.read_rcf import read_rcn31_out
    return read_rcn31_out(REFDATA / "nid8" / "nid8.rcn31_out")


@pytest.fixture(scope="session")
def nid8_slater():
    """Slater integrals for Ni2+ from rcn2."""
    from multitorch.io.read_rcf import read_rcn2_out
    return read_rcn2_out(REFDATA / "nid8" / "nid8.rcn2_out")


# ─────────────────────────────────────────────────────────────
# RME fixtures (Phase 3 reference)
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def nid8_rme_rcg():
    """Reduced matrix elements from ttrcg for Ni2+ single-config."""
    from multitorch.io.read_rme import read_rme_rcg
    return read_rme_rcg(REFDATA / "nid8" / "nid8.rme_rcg")


@pytest.fixture(scope="session")
def nid8_rme_rac():
    """Reduced matrix elements from ttrac (D4h symmetry) for Ni2+."""
    from multitorch.io.read_rme import read_rme_rac
    return read_rme_rac(REFDATA / "nid8" / "nid8.rme_rac")


@pytest.fixture(scope="session")
def nid8ct_rme_rcg():
    """RME from ttrcg for Ni2+ charge-transfer (2 configs)."""
    from multitorch.io.read_rme import read_rme_rcg
    return read_rme_rcg(REFDATA / "nid8ct" / "nid8ct.rme_rcg")


@pytest.fixture(scope="session")
def nid8ct_rme_rac():
    """RME from ttrac for Ni2+ charge-transfer (D4h symmetry)."""
    from multitorch.io.read_rme import read_rme_rac
    return read_rme_rac(REFDATA / "nid8ct" / "nid8ct.rme_rac")


@pytest.fixture(scope="session")
def als1ni2_rme_rcg():
    """RME from ttrcg for als1ni2 complex."""
    from multitorch.io.read_rme import read_rme_rcg
    return read_rme_rcg(REFDATA / "als1ni2" / "als1ni2.rme_rcg")


@pytest.fixture(scope="session")
def als1ni2_rme_rac():
    """RME from ttrac for als1ni2 complex."""
    from multitorch.io.read_rme import read_rme_rac
    return read_rme_rac(REFDATA / "als1ni2" / "als1ni2.rme_rac")


# ─────────────────────────────────────────────────────────────
# Ban output fixtures (Phase 2 reference)
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def nid8ct_ban():
    """ttban output (energies + transition matrices) for nid8ct."""
    from multitorch.io.read_oba import read_ban_output
    return read_ban_output(REFDATA / "nid8ct" / "nid8ct.ban_out")


# ─────────────────────────────────────────────────────────────
# Parametrize helper
# ─────────────────────────────────────────────────────────────

ALL_EXAMPLES = ["nid8", "nid8ct", "als1ni2"]

@pytest.fixture(params=ALL_EXAMPLES)
def example_name(request):
    """Parametrize over all three reference examples."""
    return request.param
