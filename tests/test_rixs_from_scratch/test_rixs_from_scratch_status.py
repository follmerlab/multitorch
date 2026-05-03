"""Status sentinel for the RIXS from-scratch port (plan Phase 4).

Each test is xfail until the corresponding piece of the RIXS port
lands. See `~/.claude/plans/lexical-toasting-kahan.md`.
"""
from __future__ import annotations

import pytest


@pytest.mark.xfail(reason="Phase 4: calcRIXS_from_scratch not yet implemented")
def test_calcRIXS_from_scratch_exists():
    """multitorch.api.calc should export calcRIXS_from_scratch."""
    from multitorch.api import calc
    assert hasattr(calc, "calcRIXS_from_scratch")


@pytest.mark.xfail(reason="Phase 4: RIXS parity test against als1ni2 pending")
def test_rixs_from_scratch_matches_als1ni2_fixture():
    """RIXS from-scratch should match the bundled als1ni2 fixture."""
    pytest.skip("requires Phase 4 implementation")
