"""Input validation added after FABLE_HANDOFF_2026-09-15 (S6 follow-ups, N4, N5, HFS).

Each of these inputs used to run and return a number: a wrong one (T = 0 with
max_gs > 1, half-integer from scratch), a meaningless one (soc = -1, a
string slater, section-0 overrides), or one computed from an unconverged SCF.
"""
from __future__ import annotations

import functools
import re

import pytest
import torch

import multitorch
from multitorch.api.calc import (
    HFS_STALL_TOL, _checked, calcXAS_cached, preload_fixture, preload_from_scratch,
)


@functools.lru_cache(maxsize=None)
def _ni(sym="d4h"):
    return preload_fixture("Ni", "ii", sym)


def _run(cache, **kw):
    return calcXAS_cached(cache, return_sticks=True, nbins=200, **kw)


@pytest.mark.parametrize("T,max_gs", [(0.0, 10), (80.0, 0), (80.0, -1), (-5.0, 1), (80.0, 1.5), (80.0, True)])
def test_boltzmann_pool_is_validated(T, max_gs):
    with pytest.raises(ValueError):
        _run(_ni(), T=T, max_gs=max_gs)


def test_zero_temperature_with_one_level_still_works():
    _, _, sticks = _run(_ni(), T=0.0, max_gs=1)
    assert float(sticks[:, 1].sum()) > 0


@pytest.mark.parametrize("name,value,error", [
    ("soc", -1.0, ValueError), ("slater", float("nan"), ValueError), ("slater", "0.8", TypeError),
    ("soc", True, TypeError), ("slater", torch.tensor([0.8, -0.1]), ValueError),
])
def test_reductions_are_validated(name, value, error):
    with pytest.raises(error):
        _run(_ni(), **{name: value})


def test_tensor_reductions_keep_their_gradient():
    slater = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    _, y = calcXAS_cached(_ni("oh"), slater=slater, nbins=200)
    (g,) = torch.autograd.grad(y.sum(), slater)
    assert torch.isfinite(g) and abs(float(g)) > 0


@pytest.mark.parametrize("sym,cf", [("d4h", {"tdq": 1.0}), ("oh", {"tendq": 1.0, "dt": 0.1}), ("oh", {"ds": 0.1})])
def test_fixture_crystal_field_keys_are_validated(sym, cf):
    with pytest.raises(ValueError, match="crystal-field"):
        _run(_ni(sym), cf=cf)


def test_fixture_overrides_of_unused_sections_raise():
    cache = _ni("oh")
    with pytest.raises(ValueError, match="sections 0/1"):
        _run(cache, atomic={"0.GROUND": {"zeta_1": 0.1}})
    _run(cache, atomic={"2.GROUND": {"zeta_1": 0.1}})   # the ones the assembler reads still work


@pytest.mark.parametrize("element,valence,sym,error", [
    ("Ti", "iv", "oh", ValueError), ("Zn", "ii", "oh", ValueError), ("Ni", "ii", "ohh", ValueError),
])
def test_from_scratch_requests_are_rejected_before_hfs(monkeypatch, element, valence, sym, error):
    import multitorch.api.calc as calc

    def no_hfs(*args, **kwargs):
        raise AssertionError("HFS ran before the request was validated")
    monkeypatch.setattr(calc, "_hfs_to_slater_params", no_hfs)
    with pytest.raises(error):
        preload_from_scratch(element, valence, sym)


@pytest.mark.parametrize("sym", ["oh", "d4h"])
def test_half_integer_is_accepted_in_both_groups(sym):
    """WP-B2a-3 opened Oh to odd electron counts and B2a-4 D4h; nothing refuses
    an electron count now. Spectra are checked against Fortran in
    tests/test_integration/test_from_scratch_fortran_parity.py.
    """
    from multitorch.angular.rac_generator import generate_ledge_template

    rac, store, dec = generate_ledge_template(2, 5, sym=sym)
    assert {c.label for c in dec.configs} == {"gs", "ex"}
    assert any(b.geometry == "10DQ" and b.add_entries for b in rac.blocks)
    assert preload_from_scratch("Fe", "iii", sym).sym == sym


def test_unconverged_hfs_raises_but_a_stalled_limit_cycle_is_accepted():
    from multitorch.atomic.hfs import HFSResult
    ok = HFSResult(orbitals=[], r=torch.zeros(1), IDB=0, Z=28, ION=1, MESH=1, converged=False, n_iter=130, delta=3.7e-6)
    assert _checked(ok, 28, {}) is ok
    bad = HFSResult(orbitals=[], r=torch.zeros(1), IDB=0, Z=28, ION=1, MESH=1, converged=False, n_iter=130,
                    delta=10 * HFS_STALL_TOL)
    with pytest.raises(RuntimeError, match="did not converge"):
        _checked(bad, 28, {})


def test_preload_from_scratch_is_exported():
    assert multitorch.preload_from_scratch is preload_from_scratch


# ── FABLE_HANDOFF A1, A2, N7 ──

def test_calcxas_rejects_misspelled_keywords():
    from multitorch.api.calc import calcXAS
    with pytest.raises(TypeError):
        calcXAS("Ni", "ii", "oh", "l", {"tendq": 1.0}, tempreature=300)


def test_calcdoc_and_calcrixs_reject_misspelled_keywords():
    from multitorch.api.calc import calcDOC, calcRIXS
    with pytest.raises(TypeError, match="unexpected"):
        calcDOC("Ni", "ii", "oh", "l", {"tendq": 1.0}, slatter=0.7)
    with pytest.raises(TypeError, match="unexpected"):
        calcRIXS("Ni", "ii", "oh", "l", {"tendq": 1.0}, slatter=0.7)


@pytest.mark.parametrize("edge", ["k", "m23", "", None])
def test_only_the_l_edge_is_accepted(edge):
    from multitorch.api.calc import calcXAS
    with pytest.raises(NotImplementedError, match="L2,3"):
        calcXAS("Ni", "ii", "oh", edge, {"tendq": 1.0}, nbins=200)
    with pytest.raises(NotImplementedError, match="L2,3"):
        preload_fixture("Ni", "ii", "oh", edge=edge)


def test_ban_output_mode_ignores_edge():
    from pathlib import Path
    from multitorch.api.calc import calcXAS
    ban_out = Path(__file__).parent.parent / "reference_data" / "ni2_d8_oh" / "ni2_d8_oh.ban_out"
    x, y = calcXAS("", "", "", "", {}, ban_output_path=str(ban_out), nbins=200)
    assert float(y.max()) > 0


def test_partial_d4h_crystal_field_zeroes_the_omitted_tetragonal_terms():
    """nid8ct ships Ds = 0.1: cf={'tendq': 1} must mean Dt = Ds = 0, not the template's Ds."""
    from multitorch.api.calc import _cache_ban, fixture_defaults
    cache = _ni("d4h")
    partial = _cache_ban(cache, {"tendq": 1.0}, None, None, None, None).xham[0].values
    explicit = _cache_ban(cache, {"tendq": 1.0, "dt": 0.0, "ds": 0.0}, None, None, None, None).xham[0].values
    assert [float(v) for v in partial] == [float(v) for v in explicit]
    template = _cache_ban(cache, {}, None, None, None, None).xham[0].values
    defaults = fixture_defaults("Ni", "ii", "d4h")
    assert defaults["fixture"] == "nid8ct" and defaults["cf"]["ds"] == pytest.approx(0.1)
    shipped = _cache_ban(cache, defaults["cf"], None, None, None, None).xham[0].values
    assert [float(v) for v in shipped] == pytest.approx([float(v) for v in template])


def test_fixture_defaults_reproduce_the_shipped_calculation():
    from multitorch.api.calc import fixture_defaults
    d = fixture_defaults("Ni", "ii", "oh")
    assert d["fixture"] == "ni2_d8_oh" and set(d["lmct"]) == {"eg", "t2g"}
    cache = _ni("oh")
    _, _, a = calcXAS_cached(cache, return_sticks=True, nbins=200)
    _, _, b = calcXAS_cached(cache, cf=d["cf"], delta=d["delta"], u=d["u"], lmct=d["lmct"], return_sticks=True, nbins=200)
    assert torch.allclose(a, b)


def _sym_parameter_entry(doc: str):
    """The numpydoc ``sym :`` entry of *doc*, or None if it documents no parameters.

    The entry, not the whole docstring: a reader looking up what ``sym``
    accepts reads this block, and it is the block that drifted. Mere presence
    of a symmetry name somewhere in the docstring is not enough -- measured,
    an injection that reverted both the prose line and this entry still left
    ``'td'`` in the ``cf`` entry and passed a whole-docstring check.
    """
    lines = doc.splitlines()
    for n, line in enumerate(lines):
        m = re.match(r"^(\s*)sym :", line)
        if not m:
            continue
        indent = len(m.group(1))
        out = [line]
        for later in lines[n + 1:]:
            if re.match(r"^\s{%d}\S" % indent, later) and not later.startswith(" " * (indent + 1)):
                break
            out.append(later)
        return "\n".join(out)
    return None


def test_supported_symmetries_are_documented_everywhere():
    """Every ``sym`` the package dispatches on is named where a reader looks it up.

    ``SYM_TO_GROUP`` is the registry ``plan_for_sym`` dispatches on, so it is
    the source of truth, and the docstrings fell behind it: ``'td'`` landed in
    WP-B5-9 and four docstrings went on saying "'oh' or 'd4h'" -- including the
    Parameters entry of the one entry point that supports it. The failure mode
    is a user not knowing a working symmetry exists, so the guard is keyed to
    the registry rather than to any list in prose, and to the ``sym`` parameter
    entry rather than to the docstring as a whole.

    ``calcXAS`` is included deliberately even though it cannot *reach* 'td':
    it is the fixture path, and what its reader needs is the pointer to the
    path that can.
    """
    from multitorch.angular.symmetry_plan import SYM_TO_GROUP, plan_for_sym
    from multitorch.api.calc import calcXAS, calcXAS_from_scratch

    docs = {"calcXAS": calcXAS.__doc__,
            "calcXAS_from_scratch": calcXAS_from_scratch.__doc__,
            "preload_from_scratch": preload_from_scratch.__doc__,
            "plan_for_sym": plan_for_sym.__doc__}
    # Where a ``sym`` Parameters entry exists it is the authoritative text; the
    # two without one (no Parameters section at all) are checked whole.
    scoped = {name: _sym_parameter_entry(doc) or doc for name, doc in docs.items()}
    assert _sym_parameter_entry(docs["calcXAS_from_scratch"]) is not None
    assert _sym_parameter_entry(docs["plan_for_sym"]) is None
    missing = {(name, sym) for name, text in scoped.items()
               for sym in SYM_TO_GROUP if f"'{sym}'" not in text}
    assert not missing, f"undocumented symmetries: {sorted(missing)}"

    # And where the type line *enumerates* the accepted set, the enumeration
    # must be the registry. Presence alone is too weak -- measured: reverting
    # the ``sym : {...}`` line to its two-group form left ``'td'`` in the
    # entry's own description and passed the check above. An enumeration that
    # contradicts the prose beside it is worse than an omission, because it
    # reads as the authoritative answer.
    for name, doc in docs.items():
        entry = _sym_parameter_entry(doc)
        if entry is None:
            continue
        m = re.match(r"^\s*sym : \{(.+?)\}", entry)
        if m is None:
            continue
        assert set(re.findall(r"'([^']+)'", m.group(1))) == set(SYM_TO_GROUP), (
            f"{name}'s sym enumeration disagrees with SYM_TO_GROUP: {m.group(1)}")


def test_the_supported_symmetry_set_is_the_registry_in_every_validator():
    """The three validators agree with ``SYM_TO_GROUP``, tested by behaviour.

    The set is written out as a literal in ``_validate_from_scratch`` and in
    ``generate_ledge_template``, which is why the docstrings could drift at
    all. Asserted by calling them rather than by reading their source, since a
    source-text assertion is not evidence the line runs. The charge-transfer
    generator is a deliberate *subset* -- CT beyond Oh/D4h is carried-forward
    work -- so it is checked to be one, not to be equal.
    """
    from multitorch.angular.ct_generator import generate_ct_ledge_template
    from multitorch.angular.rac_generator import generate_ledge_template
    from multitorch.angular.symmetry_plan import SYM_TO_GROUP
    from multitorch.api.calc import _validate_from_scratch

    for sym in SYM_TO_GROUP:
        _validate_from_scratch("Ni", "ii", 8, sym)                 # must not raise
    # 'c4v' is registered as group theory and has no store convention, which is
    # the state 'd3d' was in until it got one -- so it is the standing negative
    # case. (It was 'd3d' here; that is exactly the drift this test exists for.)
    with pytest.raises(ValueError, match="unsupported symmetry"):
        _validate_from_scratch("Ni", "ii", 8, "c4v")
    with pytest.raises(ValueError, match="[Uu]nsupported symmetry"):
        generate_ledge_template(2, 8, sym="c4v")                   # raises before any work

    # A supported symmetry can still restrict the *electron count*, which is a
    # different question from whether the group is registered: D3d is
    # even-only (residual 68), and that refusal names its reason.
    _validate_from_scratch("Ni", "ii", 8, "d3d")                   # even: fine
    with pytest.raises(NotImplementedError, match="Frobenius-Schur"):
        _validate_from_scratch("Co", "ii", 7, "d3d")

    ct_supported = set()
    for sym in SYM_TO_GROUP:
        try:
            generate_ct_ledge_template(2, sym=sym)
        except ValueError as exc:
            assert "unsupported symmetry" in str(exc), exc
        else:
            ct_supported.add(sym)
    assert ct_supported and ct_supported < set(SYM_TO_GROUP), ct_supported


def test_the_fixture_path_points_at_the_from_scratch_one_for_a_bundled_symmetry_gap():
    """``calcXAS(sym='td')`` fails on a missing fixture; the message must say where to go.

    Td is supported and has no Fortran reference, so the fixture lookup cannot
    find a directory for it. Before this the user got a bare "No fixture
    directory found" plus a listing of unrelated directories, which reads as
    "Td is unsupported" -- the opposite of the truth.
    """
    from multitorch.api.calc import _find_fixture_dir

    for sym in ("td", "d3d"):
        with pytest.raises(FileNotFoundError, match="calcXAS_from_scratch"):
            _find_fixture_dir("Ni", "ii", sym)
    # ...while a group with no store convention at all gets no such pointer,
    # because there is nowhere to point. ('d3d' was this case until it got one.)
    with pytest.raises(FileNotFoundError) as exc:
        _find_fixture_dir("Ni", "ii", "c4v")
    assert "calcXAS_from_scratch" not in str(exc.value), "c4v is supported nowhere"
