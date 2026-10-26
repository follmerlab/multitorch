"""The ADD coefficients of four generator cases, frozen against silent movement.

**What this is, and what it is not.** The golden files under
``tests/golden/rac/`` are a snapshot of *this package's own output*. They are
**not an oracle** and they cannot establish that any coefficient is right.
What they establish is that a coefficient has not **changed** -- which is a
different and, for one of these four cases, uncovered question.

Why it is uncovered. ``tests/test_angular/test_generator_refactor_parity.py``
already renders every ADD entry of all 18 ``(sym, n)`` cases through
:func:`~multitorch.io.rac_text.rac_fingerprint` and compares bit-exact, but it
compares ``generate_ledge_template`` against the retained
``_generate_ledge_template_legacy``: two paths *agreeing*, not output being
*unchanged*. Anything both sides consume moves both sides together and the gate
stays green. ``_pin_d4h_routes`` is exactly such a thing, and
``test_double_group.py``'s order guard pins only the one cause that was known --
the crystal-field operator iteration order.

Measured 2026-10-25, the gap is real. Flipping the sign of the phase
``_pin_d4h_routes`` pins -- making each pinned cross-route coupling real
*negative* instead of real positive, which is inside the gauge freedom and
therefore leaves every observable alone -- moves **488 of d4h d7's 1303 ADD
entries** while leaving oh7, oh8 and d4h8 bit-identical, and the **whole suite
stays green**. Residual 67 recorded the same class from the other direction:
individual stored coefficients move by up to 10.8 while every per-block
intensity sum stays bit-identical at 0.0.

Why d4h d7 is the case that matters. There is no Fortran output for a
tetragonal odd-electron ion (ttrac cannot be re-run), so away from the Oh limit
nothing validates these numbers:

  * ``oh7`` / ``oh8``   -- Fortran fixtures (``co2_d7_oh``, ``ni2_d8_oh``).
  * ``d4h8``            -- the ``nid8`` fixture chain.
  * ``d4h7``            -- Fortran-validated **only at dt = ds = 0**, where D4h
    *is* Oh (``test_half_integer_d4h_reproduces_the_oh_limit``), plus the
    one-electron and one-hole analytic Ballhausen cases. The many-electron
    tetragonal half-integer coefficients themselves have no reference.

So this file freezes what we have and makes a change loud. **When it fires, the
coefficients moved -- not necessarily wrongly.** The remedy is never to re-bless
reflexively:

  1. Check the gauge-invariant quantities still hold -- the Oh-limit fixture
     parity, the descent limit, the dipole sum rule, the LAPACK scramble
     harness. If any of those moved, the change is a bug, not a re-gauge.
  2. Decide whether the movement was intended.
  3. Only then re-bless, deliberately:
     ``python tests/tools/bless_rac_golden.py`` -- and say in the commit
     message which coefficients moved and why.
"""
from __future__ import annotations

import difflib
from pathlib import Path

import pytest

GOLDEN = Path(__file__).parent.parent / "golden" / "rac"

#: The generator arguments every golden case is built with. Both this file and
#: ``tests/tools/bless_rac_golden.py`` read the spec from here, so the snapshot
#: and the check cannot drift apart.
TEMPLATE_ARGS = dict(l_val=2, l_core=1, n_core_gs=6, cf_rank=4)

#: ``(sym, n, expected ADD-entry count)``. The counts are measured, and they are
#: here so that a golden file regenerated empty -- or a generator that silently
#: stopped emitting a block -- cannot pass by agreeing with itself.
CASES = [
    ("oh", 7, 275),
    ("oh", 8, 124),
    ("d4h", 7, 1303),
    ("d4h", 8, 530),
]

CASE_IDS = [f"{sym}{n}" for sym, n, _ in CASES]


def build_fingerprint(sym: str, n: int) -> str:
    """The canonical text rendering of one case's RAC, coefficients and all."""
    from multitorch.angular.rac_generator import generate_ledge_template
    from multitorch.io.rac_text import rac_fingerprint

    rac, _store, _dec = generate_ledge_template(
        TEMPLATE_ARGS["l_val"], n, TEMPLATE_ARGS["l_core"],
        TEMPLATE_ARGS["n_core_gs"], cf_rank=TEMPLATE_ARGS["cf_rank"], sym=sym,
    )
    return rac_fingerprint(rac)


def golden_path(sym: str, n: int) -> Path:
    return GOLDEN / f"{sym}{n}.fp"


def _add_lines(text: str) -> list[str]:
    return [l for l in text.splitlines() if l.lstrip().startswith("ADD ")]


@pytest.mark.parametrize("sym,n,n_add", CASES, ids=CASE_IDS)
def test_the_add_coefficients_have_not_moved(sym, n, n_add):
    """Bit-exact, because a change that only moves code has no licence to move a float."""
    path = golden_path(sym, n)
    assert path.exists(), (
        f"{path} is missing; regenerate with tests/tools/bless_rac_golden.py "
        "and read this module's docstring before you do")
    want = path.read_text()
    got = build_fingerprint(sym, n)

    if got == want:
        return

    want_adds, got_adds = _add_lines(want), _add_lines(got)
    moved = sum(1 for a, b in zip(want_adds, got_adds) if a != b)
    diff = "\n".join(list(difflib.unified_diff(
        want.splitlines(), got.splitlines(),
        fromfile=f"golden/{sym}{n}", tofile=f"generated/{sym}{n}",
        lineterm="", n=0))[:40])
    pytest.fail(
        f"{sym}{n}: the generated RAC no longer matches the golden snapshot. "
        f"{moved} of {len(want_adds)} ADD entries differ "
        f"({len(got_adds)} entries generated, {len(want_adds)} in the golden). "
        f"This says the coefficients MOVED, not that they are wrong -- see this "
        f"module's docstring for what to check before re-blessing.\n{diff}")


@pytest.mark.parametrize("sym,n,n_add", CASES, ids=CASE_IDS)
def test_each_golden_file_holds_the_entry_count_it_should(sym, n, n_add):
    """A golden file that lost its contents would otherwise pass the test above."""
    adds = _add_lines(golden_path(sym, n).read_text())
    assert len(adds) == n_add, (
        f"{sym}{n}: golden file holds {len(adds)} ADD entries, expected {n_add}")


def test_the_four_cases_cover_both_groups_and_both_parities():
    """
    The point of the set is the half-integer tetragonal case, which only exists
    because the other three pin the pieces it is built from. Losing a case to an
    edit would otherwise go unnoticed.
    """
    assert {(sym, n) for sym, n, _ in CASES} == {
        ("oh", 7), ("oh", 8), ("d4h", 7), ("d4h", 8)}
    halves = {(sym, n) for sym, n, _ in CASES if n % 2 == 1}
    assert halves == {("oh", 7), ("d4h", 7)}, halves
    assert sum(c for _, _, c in CASES) == 2232, "the 2232 entries of residual 69"


def test_no_golden_file_is_unaccounted_for():
    """A stale file left behind after a case is renamed is a trap for the next reader."""
    on_disk = {p.name for p in GOLDEN.glob("*.fp")}
    expected = {golden_path(sym, n).name for sym, n, _ in CASES}
    assert on_disk == expected, (on_disk ^ expected)
