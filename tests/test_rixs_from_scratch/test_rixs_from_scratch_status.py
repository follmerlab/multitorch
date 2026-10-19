"""Status sentinel for the RIXS from-scratch port, which genuinely has not landed.

Unlike its charge-transfer sibling, this one is accurate: there is no
``calcRIXS_from_scratch``. The plan file these used to cite,
``~/.claude/plans/lexical-toasting-kahan.md``, no longer exists; the current
plan is ``docs/DEVELOPMENT_PLAN_2026-09.md`` (RIXS from scratch is WP-E).
"""
from __future__ import annotations

import pytest


@pytest.mark.xfail(reason="Phase 4: calcRIXS_from_scratch not yet implemented")
def test_calcRIXS_from_scratch_exists():
    """multitorch.api.calc should export calcRIXS_from_scratch."""
    from multitorch.api import calc
    assert hasattr(calc, "calcRIXS_from_scratch")


@pytest.mark.skip(reason="no calcRIXS_from_scratch to compare against yet (WP-E); "
                         "written out so the oracle is ready rather than left implicit")
def test_rixs_from_scratch_matches_als1ni2_fixture():
    """RIXS from-scratch should match the bundled als1ni2 fixture.

    Was ``@pytest.mark.xfail`` wrapping a body that did nothing but
    ``pytest.skip``, so it could never test *or* fail anything -- it only
    inflated the suite's skip count. Marked skip outright, which is what it
    always was, until there is an implementation to point it at.
    """
