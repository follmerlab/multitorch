"""WP-B5-3: the net under the generator rewrite.

B5-5 through B5-7 replace the generator's two parallel emission paths -- the
``if sym == 'd4h'`` blocks and the ``legacy_*_irreps = ... if sym == 'oh' else
[]`` loops -- with one plan-driven loop. The plan's gate for every one of those
increments is "byte-equal RAC output vs today's Oh and D4h paths", so the net
has to exist and be trusted *before* anything moves. That is what this file is,
and it is green trivially today: :func:`generate_ledge_template` currently
delegates to
:func:`~multitorch.angular.rac_generator._generate_ledge_template_legacy`, so
the two sides are the same code. The point is not that it passes now; the point
is that it fails the moment they diverge, which the second half of this file
tests by making them diverge.

**The fingerprint on its own is not enough, and that is the one design decision
here.** :func:`~multitorch.io.rac_text.rac_fingerprint` renders every block
field and every ADD entry, including each ADD's ``matrix_idx`` -- but a
``matrix_idx`` is an *index into the COWAN store*, not a value. A refactor that
reordered the store while leaving the indices alone would emit a byte-identical
RAC and a different Hamiltonian; ``test_a_reordered_cowan_store_fails_the_gate``
below constructs exactly that and checks the fingerprint really does stay
identical through it. ``generate_ledge_template`` returns ``(rac, cowan_store,
decomposition)``, so all three are compared: the RAC by fingerprint, the store
tensor by tensor with ``torch.equal``, and every field of the decomposition's
configurations -- above all ``block_index``, which is the other place store
positions are recorded. Bit-exact rather than a tolerance throughout: a
refactor that only moves code has no licence to change a float.

**No golden files here**, deliberately: the 18 full fingerprints run to 1.35
MiB, and a stored copy of the generator's output is a second thing to keep in
step. The retained legacy function is the reference instead, exactly as WP-B3
kept ``_d4h_operator_vector_butler`` as the Butler-route oracle.

**But that decision does not cover everything, and since 2026-10-25 four cases
do have golden files** -- ``tests/test_angular/test_rac_golden_coefficients.py``
and ``tests/golden/rac/``. The gate in *this* file compares two code paths
*agreeing*; it cannot see a change both paths consume, because both move
together. ``_pin_d4h_routes`` is such a thing: flipping the sign of the phase
it pins moves 488 of d4h d7's 1303 ADD entries, and the whole suite -- this
file included -- stays green. The golden snapshot covers that class for
``oh7``, ``oh8``, ``d4h7`` and ``d4h8``, at 192 KiB. See residual 71.

All 18 (sym, n) cases run by default. The plan sized B5-3's gate at a 6-case
subset (oh/d4h x n = 2, 5, 8) on the assumption that 18 would be too slow;
measured, one build of all 18 takes 3.9 s and this harness's two builds take
7.8 s, so the subset buys nothing and the later increments' "all 18" gate is
available from the start.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from multitorch.angular.rac_generator import (_generate_ledge_template_legacy,
                                              generate_ledge_template)
from multitorch.io.rac_text import rac_fingerprint, rac_fingerprint_diff

#: Every electron count the generator accepts, in both groups: the 18 cases.
CASES = [(sym, n) for sym in ("oh", "d4h") for n in range(1, 10)]
CASE_IDS = [f"{sym}-d{n}" for sym, n in CASES]


def _build(sym, n, generator):
    return generator(2, n, 1, 6, cf_rank=4, sym=sym)


def _assert_tensors_equal(new, ref, where):
    assert new.shape == ref.shape, f"{where}: shape {tuple(new.shape)} vs {tuple(ref.shape)}"
    assert torch.equal(new, ref), (
        f"{where}: differs, max |delta| = {(new - ref).abs().max().item():.3e}")


def _assert_stores_equal(new_store, ref_store, label):
    assert len(new_store) == len(ref_store), f"{label}: store section count"
    for section, (new_sec, ref_sec) in enumerate(zip(new_store, ref_store)):
        assert len(new_sec) == len(ref_sec), (
            f"{label}: store section {section} holds {len(new_sec)} matrices, the "
            f"reference holds {len(ref_sec)} -- every ADD entry's matrix_idx indexes "
            f"into this list, so a length change silently re-points the blocks after it")
        for i, (new_mat, ref_mat) in enumerate(zip(new_sec, ref_sec)):
            _assert_tensors_equal(new_mat, ref_mat, f"{label}: store section {section} matrix {i}")


def _assert_decompositions_equal(new_dec, ref_dec, label):
    assert [c.label for c in new_dec.configs] == [c.label for c in ref_dec.configs], \
        f"{label}: configuration labels"
    assert new_dec.slater_reduction == ref_dec.slater_reduction, f"{label}: slater_reduction"
    assert new_dec.soc_reduction == ref_dec.soc_reduction, f"{label}: soc_reduction"

    for new_c, ref_c in zip(new_dec.configs, ref_dec.configs):
        where = f"{label}: configuration {ref_c.label!r}"
        assert new_c.section == ref_c.section, f"{where}: section"
        assert new_c.block_type == ref_c.block_type, f"{where}: block_type"
        assert new_c.shells == ref_c.shells, f"{where}: shells"
        # the store positions of this configuration's J blocks: the second
        # place, after the ADD entries, where a reordered store would show up
        assert new_c.block_index == ref_c.block_index, f"{where}: block_index (store positions)"
        assert new_c.e_av == ref_c.e_av, f"{where}: e_av"
        assert new_c.anchor_params == ref_c.anchor_params, f"{where}: anchor_params"
        assert new_c.reference == ref_c.reference, f"{where}: reference"
        assert new_c.reference_slater == ref_c.reference_slater, f"{where}: reference_slater"
        assert new_c.reference_soc == ref_c.reference_soc, f"{where}: reference_soc"
        assert new_c.states == ref_c.states, f"{where}: basis states"

        assert sorted(new_c.operators) == sorted(ref_c.operators), f"{where}: operator names"
        for name in ref_c.operators:
            new_op, ref_op = new_c.operators[name], ref_c.operators[name]
            assert sorted(new_op) == sorted(ref_op), f"{where}: operator {name!r} J sectors"
            for J in ref_op:
                _assert_tensors_equal(new_op[J], ref_op[J], f"{where}: operator {name!r} J={J}")

        assert sorted(new_c.anchor) == sorted(ref_c.anchor), f"{where}: anchor J sectors"
        for J in ref_c.anchor:
            _assert_tensors_equal(new_c.anchor[J], ref_c.anchor[J], f"{where}: anchor J={J}")


def _run_gate(sym, n, generator):
    """Compare *generator* against the retained legacy path. The gate itself."""
    new_rac, new_store, new_dec = _build(sym, n, generator)
    ref_rac, ref_store, ref_dec = _build(sym, n, _generate_ledge_template_legacy)

    label = f"{sym} d{n}"
    if rac_fingerprint(new_rac) != rac_fingerprint(ref_rac):
        raise AssertionError(
            f"{label}: RAC fingerprints differ\n" + rac_fingerprint_diff(ref_rac, new_rac))
    _assert_stores_equal(new_store, ref_store, label)
    _assert_decompositions_equal(new_dec, ref_dec, label)


@pytest.mark.parametrize("sym,n", CASES, ids=CASE_IDS)
def test_the_generator_still_emits_what_the_legacy_path_emits(sym, n):
    """The B5 gate: RAC, COWAN store and decomposition all unchanged."""
    _run_gate(sym, n, generate_ledge_template)


def test_the_harness_covers_every_electron_count_in_both_groups():
    """18 cases, not a subset: d1-d9 in Oh and D4h."""
    assert len(CASES) == 18
    assert {n for _sym, n in CASES} == set(range(1, 10))
    assert {sym for sym, _n in CASES} == {"oh", "d4h"}


def test_the_two_sides_of_the_gate_are_different_code():
    """The gate must not be comparing the new emission against itself.

    **This is the failure mode that actually happened.** B5-5's first attempt
    replaced the three emission loops *in place*, inside
    ``_generate_ledge_template_legacy`` -- which the public entry point then
    delegated to. Both sides of every comparison above changed together, all
    25 tests passed, and the harness proved nothing. A 188-line restructure of
    the Oh and D4h emission passing byte equality on the first try with no
    iteration is what gave it away, not the suite.

    So the gate needs a gate. ``_emit_operator_blocks`` (B5-5) and
    ``_emit_transi_blocks`` (B5-6) are reached only by the plan emission;
    perturbing either must make the comparison **fail**. If someone again
    collapses the two sides into one body, the legacy path starts going through
    them too, both sides move together, the comparison goes quiet -- and this
    test fails instead.

    Every plan-side emitter needs a line here. An emitter the legacy path also
    reaches is one whose increment was never really gated, so a new emitter
    added without an entry below is the same mistake wearing a different name.
    """
    import multitorch.angular.rac_generator as rg

    for emitter in ("_emit_operator_blocks", "_emit_transi_blocks"):
        real = getattr(rg, emitter)

        def drops_the_last_block(blocks, *args, _real=real, **kwargs):
            _real(blocks, *args, **kwargs)
            if blocks:
                blocks.pop()

        setattr(rg, emitter, drops_the_last_block)
        try:
            for sym in ("oh", "d4h"):
                with pytest.raises(AssertionError):
                    _run_gate(sym, 8, generate_ledge_template)
        finally:
            setattr(rg, emitter, real)

    _run_gate("oh", 8, generate_ledge_template)      # and clean again afterwards
    _run_gate("d4h", 8, generate_ledge_template)


def test_every_plan_side_emitter_is_live_and_the_legacy_path_avoids_it():
    """Each plan emitter must run on the plan path and *not* on the legacy one.

    The perturbation test above shows a difference exists; this shows where it
    is, and in the direction that matters -- a plan emitter the legacy path also
    called would make its increment's gate vacuous without any test going red.
    """
    import multitorch.angular.rac_generator as rg

    for emitter in ("_emit_operator_blocks", "_emit_transi_blocks"):
        real = getattr(rg, emitter)
        seen = {"plan": 0, "legacy": 0}
        which = {"side": None}

        def counting(*args, _real=real, **kwargs):
            seen[which["side"]] += 1
            return _real(*args, **kwargs)

        setattr(rg, emitter, counting)
        try:
            for sym in ("oh", "d4h"):
                which["side"] = "plan"
                _build(sym, 8, generate_ledge_template)
                which["side"] = "legacy"
                _build(sym, 8, _generate_ledge_template_legacy)
        finally:
            setattr(rg, emitter, real)

        assert seen["plan"] > 0, f"{emitter} is never reached by the plan emission"
        assert seen["legacy"] == 0, f"{emitter} is reached by the legacy emission too"


def test_the_legacy_emission_is_retained_and_selectable():
    """Both emissions still exist and the entry points pick different ones."""
    import inspect

    from multitorch.angular.rac_generator import (_generate_ledge_template_impl,
                                                  _generate_ledge_template_legacy)

    assert "emission='plan'" in inspect.getsource(generate_ledge_template)
    assert "emission='legacy'" in inspect.getsource(_generate_ledge_template_legacy)
    body = inspect.getsource(_generate_ledge_template_impl)
    assert "def _emission_legacy()" in body and "def _emission_plan()" in body
    with pytest.raises(ValueError, match="emission must be"):
        _generate_ledge_template_impl(2, 8, sym="oh", emission="nonsense")


# ─────────────────────────────────────────────────────────────
# The net itself: make the two sides diverge and watch it fail
# ─────────────────────────────────────────────────────────────

def _wrap(mutate):
    """A stand-in generator returning the legacy output, mutated."""
    def generator(*args, **kwargs):
        return mutate(*_generate_ledge_template_legacy(*args, **kwargs))
    return generator


def test_a_changed_add_coefficient_fails_the_gate():
    """One ADD coefficient, perturbed in its last bits."""
    def mutate(rac, store, dec):
        i = next(i for i, b in enumerate(rac.blocks) if b.add_entries)
        block = rac.blocks[i]
        adds = list(block.add_entries)
        adds[0] = dataclasses.replace(adds[0], coeff=adds[0].coeff * (1.0 + 1e-15))
        blocks = list(rac.blocks)
        blocks[i] = dataclasses.replace(block, add_entries=adds)
        return dataclasses.replace(rac, blocks=blocks), store, dec

    with pytest.raises(AssertionError, match="fingerprints differ"):
        _run_gate("oh", 8, _wrap(mutate))


def test_a_dropped_block_fails_the_gate():
    """A whole emitted block missing -- the shape a half-done loop rewrite takes."""
    def mutate(rac, store, dec):
        i = next(i for i, b in enumerate(rac.blocks) if b.add_entries)
        blocks = [b for k, b in enumerate(rac.blocks) if k != i]
        return dataclasses.replace(rac, blocks=blocks), store, dec

    with pytest.raises(AssertionError, match="fingerprints differ"):
        _run_gate("d4h", 8, _wrap(mutate))


def _swap_two_store_matrices(store):
    """Swap two same-shaped, unequal matrices of store section 0."""
    section = list(store[0])
    by_shape = {}
    for i, mat in enumerate(section):
        by_shape.setdefault(tuple(mat.shape), []).append(i)
    for idxs in by_shape.values():
        for a in idxs:
            for b in idxs:
                if a < b and not torch.equal(section[a], section[b]):
                    section[a], section[b] = section[b], section[a]
                    return [section] + list(store[1:])
    raise AssertionError("no two same-shaped unequal store matrices to swap")


def test_a_reordered_cowan_store_fails_the_gate():
    """The failure a fingerprint comparison alone cannot see.

    Two store matrices of the same shape are swapped and the RAC is left
    exactly as it was, so every ADD ``matrix_idx`` still points at the slot it
    pointed at before. The fingerprint is therefore byte-identical -- asserted
    here, so the test is really exercising the blind spot it claims to -- while
    the Hamiltonian those indices now build is different. This is why the
    harness compares the store and not only the RAC.
    """
    def mutate(rac, store, dec):
        return rac, _swap_two_store_matrices(store), dec

    mutated_rac, mutated_store, _dec = _build("oh", 8, _wrap(mutate))
    plain_rac, plain_store, _ = _build("oh", 8, _generate_ledge_template_legacy)
    assert rac_fingerprint(mutated_rac) == rac_fingerprint(plain_rac), \
        "the mutation must leave the RAC untouched, or it is not testing the blind spot"
    assert any(not torch.equal(a, b) for a, b in zip(mutated_store[0], plain_store[0])), \
        "the mutation must actually change the store"

    with pytest.raises(AssertionError, match="store section 0 matrix"):
        _run_gate("oh", 8, _wrap(mutate))


def test_a_changed_decomposition_store_position_fails_the_gate():
    """``block_index`` is compared, not just the operator tensors."""
    def mutate(rac, store, dec):
        cfg = dec.configs[0]
        cfg.block_index = {J: p + 1 for J, p in cfg.block_index.items()}
        return rac, store, dec

    with pytest.raises(AssertionError, match="block_index"):
        _run_gate("oh", 8, _wrap(mutate))


def test_a_changed_operator_tensor_fails_the_gate():
    """A parameter-free operator block perturbed in its last bits."""
    def mutate(rac, store, dec):
        cfg = dec.configs[0]
        name = sorted(cfg.operators)[0]
        J = sorted(cfg.operators[name])[0]
        cfg.operators[name][J] = cfg.operators[name][J] * (1.0 + 1e-15)
        return rac, store, dec

    with pytest.raises(AssertionError, match="operator"):
        _run_gate("oh", 8, _wrap(mutate))


# ─────────────────────────────────────────────────────────────
# WP-B5-5b step 1: one operator emitter for Oh and D4h
# ─────────────────────────────────────────────────────────────
#
# B5-5b routes Oh through the dispatcher with **Oh as its own target group**,
# so the subduction step is the identity and the per-Oh-irrep emitters retire.
# The plan flagged the risk: it changes *which code computes the coefficients*,
# so byte equality would become the claim rather than the check.
#
# Measured before touching the generator, which is what answers that (residual
# 48's lesson): with the subduction replaced by the identity the dispatcher
# reproduces ``_make_oh_op_adds`` **exactly** -- 80 cases, 608 ADD entries,
# largest coefficient 1.095, largest difference 0.0. So it is a relabelling,
# not a recomputation, and the equality below is a *licence* for step 2 rather
# than a physics oracle. The physics oracle for step 2 is the Oh Fortran
# fixtures at cosine > 0.99999, which no amount of this would replace.

#: Oh single-group irreps in the order ``irrep_j_ordering`` walks them.
_OH_IRREPS_FOR_PARITY = ("A1", "A2", "E", "T1", "T2")


def _oh_case(n_val, oh_label, parity):
    """``(j_order, entries, index maps)`` for one (electron count, irrep, parity).

    ``j_order`` is the retired emitter's ``(J, copy, n)``; ``entries`` is the
    dispatcher's ``(J, oh_label, copy, partner, n)``. The index maps are
    synthetic but shared by both sides, so they cancel.
    """
    import multitorch.angular.rac_generator as rg
    from multitorch.angular.point_group import oh_branching

    j_sizes = rg._j_sector_sizes(rg._get_terms(2, n_val))
    if any(abs(J - round(J)) > 1e-9 for J in j_sizes):
        return None                                  # half-integer: step 3
    js = sorted(j_sizes)
    ham_idx = {J: i + 1 for i, J in enumerate(js)}
    pairs = [(a, b) for a in js for b in js]
    cf_idx = {p: i + 1 for i, p in enumerate(pairs)}
    j_order = [(J, copy, j_sizes[J]) for J in js
               for copy in range(oh_branching(int(round(J))).get(oh_label, 0))]
    if not j_order:
        return None
    entries = [(J, oh_label, copy, 0, n) for (J, copy, n) in j_order]
    return j_order, entries, ham_idx, cf_idx


B55B_CASES = [(n, irrep, parity)
              for n in range(1, 10)
              for irrep in _OH_IRREPS_FOR_PARITY
              for parity in ("g", "u")]


#: ``(operator name, non-empty cases, ADD entries)`` -- the last two
#: measured, so a version of this test that compared nothing would fail rather
#: than pass. 128 + 480 = the 608 entries the pre-implementation measurement
#: reported.
B55B_OPERATORS = [("HAMILTONIAN", 40, 128), ("TENDQ", 40, 480)]


@pytest.mark.parametrize("operator,n_cases,n_entries", B55B_OPERATORS)
def test_the_one_emitter_reproduces_the_oh_emitter_exactly(operator, n_cases, n_entries):
    """``_make_op_adds(target_group='Oh')`` == ``_make_oh_op_adds``, bit for bit.

    Over every integer electron count, both manifold parities and all five Oh
    irreps. Counted rather than merely asserted: a version of this that compared
    nothing would pass, so the test fails if fewer than 200 ADD entries or fewer
    than the measured number of non-empty cases were actually reached.
    """
    import multitorch.angular.rac_generator as rg

    compared = entries_seen = 0
    worst = biggest = 0.0
    for n_val, oh_label, parity in B55B_CASES:
        case = _oh_case(n_val, oh_label, parity)
        if case is None:
            continue
        j_order, entries, ham_idx, cf_idx = case
        want = rg._make_oh_op_adds(oh_label, j_order, operator, ham_idx, cf_idx)
        got = rg._make_op_adds(
            oh_label + parity, entries, operator,
            ham_idx_map=ham_idx, cf_idx_map=cf_idx, cf_idx_map_rank2=cf_idx,
            target_group="Oh", rank=4,
        )
        key = lambda a: (a.matrix_idx, a.bra, a.ket, a.nbra, a.nket)  # noqa: E731
        assert {key(a) for a in want} == {key(a) for a in got}, (n_val, oh_label, parity)
        by_key = {key(a): a.coeff for a in got}
        for add in want:
            worst = max(worst, abs(add.coeff - by_key[key(add)]))
            biggest = max(biggest, abs(add.coeff))
        if want:
            compared += 1
            entries_seen += len(want)
    assert (compared, entries_seen) == (n_cases, n_entries)
    assert worst == 0.0, f"max |difference| {worst:.3e} over {entries_seen} entries"
    assert biggest > 0.5, biggest


def test_the_identity_subduction_is_what_makes_that_equality_hold():
    """Inject a non-identity "identity" and watch the equality above fail.

    The substitution is a partner permutation inside the Oh irrep, which keeps
    the subduction unitary and the block dimensions right -- so nothing
    structural catches it and only the coefficients move. That is the mistake
    a hand-written identity table would actually make.
    """
    import numpy as np

    import multitorch.angular.rac_generator as rg

    real = rg.oh_to_target_subduction_matrix

    def permuted(oh_irrep, target_group):
        out = real(oh_irrep, target_group)
        if target_group == "Oh":
            return {k: np.roll(v, 1, axis=1) for k, v in out.items()}
        return out

    rg.oh_to_target_subduction_matrix = permuted
    try:
        with pytest.raises(AssertionError):
            test_the_one_emitter_reproduces_the_oh_emitter_exactly("TENDQ", 40, 480)
    finally:
        rg.oh_to_target_subduction_matrix = real


def test_an_operator_with_no_known_rank_is_refused_rather_than_guessed():
    """The rank now comes from the plan; a name the fallback cannot place errors.

    This is what stops D3d's ``DSIGMA``/``DTAU`` silently inheriting rank 4 from
    a table of the three D4h names when they reach this emitter.
    """
    import multitorch.angular.rac_generator as rg

    case = _oh_case(8, "E", "g")
    _j_order, entries, ham_idx, cf_idx = case
    with pytest.raises(ValueError, match="no known rank"):
        rg._make_op_adds("Eg", entries, "DSIGMA", ham_idx_map=ham_idx,
                         cf_idx_map=cf_idx, cf_idx_map_rank2=cf_idx,
                         target_group="Oh")


# ─────────────────────────────────────────────────────────────
# WP-B5-5b step 2: the Oh integer route moves to the one emitter
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("emitter,n_val", [("_make_oh_op_adds", 8),
                                           ("_make_oh_op_adds_half", 7)])
def test_the_retired_oh_operator_emitter_is_off_the_production_path(emitter, n_val):
    """``_make_oh_op_adds`` is the oracle now, so nothing production may call it.

    Residual 45's gate-for-the-gate, one level out: the equality test above
    compares the shared emitter against ``_make_oh_op_adds``, and that
    comparison is worth nothing if the production path still *is*
    ``_make_oh_op_adds``. Both call sites are covered -- the generator's plan
    emission and ``ct_generator``'s, which the plan's "the Oh ADD makers retire"
    did not mention (residual 54). Both emitters are covered, at an even
    electron count for the integer one and an odd one for the half-integer.

    It must stay reachable from the **legacy** emission, which is what keeps it
    exercised rather than merely present; residual 38 is the cost of a dead
    oracle.
    """
    import multitorch.angular.ct_generator as cg
    import multitorch.angular.rac_generator as rg
    from multitorch.angular.ct_generator import generate_ct_ledge_template

    seen = {"plan": 0, "legacy": 0, "ct": 0}
    which = {"side": None}
    real = getattr(rg, emitter)

    def counting(*args, **kwargs):
        seen[which["side"]] += 1
        return real(*args, **kwargs)

    setattr(rg, emitter, counting)
    setattr(cg, emitter, counting)
    try:
        for sym in ("oh", "td"):
            which["side"] = "plan"
            _build(sym, n_val, generate_ledge_template)
        which["side"] = "legacy"
        _build("oh", n_val, _generate_ledge_template_legacy)
        which["side"] = "ct"
        generate_ct_ledge_template(n_val, sym="oh")
    finally:
        setattr(rg, emitter, real)
        setattr(cg, emitter, real)

    assert seen["plan"] == 0, f"the plan emission still calls {emitter}"
    assert seen["ct"] == 0, f"ct_generator still calls {emitter}"
    assert seen["legacy"] > 0, f"the legacy emission no longer exercises {emitter}"


def test_the_shared_emitter_is_what_the_oh_plan_path_now_calls():
    """And the positive half: the plan path reaches the shared emitters for Oh.

    Asserted by recording what the function is *handed* rather than by reading
    the source, which the previous handoff's lesson 3 is about: a source-text
    assertion passed against an injection that replaced the enclosing ``if``
    with ``if False:``. Every Oh call must carry ``target_group='Oh'``, which is
    the thing that makes the subduction the identity.
    """
    import multitorch.angular.rac_generator as rg

    calls = []
    real = rg._make_op_adds

    def recording(*args, **kwargs):
        calls.append(kwargs.get("target_group"))
        return real(*args, **kwargs)

    real_half = rg._make_op_adds_half

    def recording_half(*args, **kwargs):
        calls.append(kwargs.get("target_group"))
        return real_half(*args, **kwargs)

    rg._make_op_adds = recording
    rg._make_op_adds_half = recording_half
    try:
        _build("oh", 8, generate_ledge_template)      # integer J
        _build("oh", 7, generate_ledge_template)      # half-integer J
    finally:
        rg._make_op_adds = real
        rg._make_op_adds_half = real_half

    assert calls, "the shared emitters are never reached by the Oh plan emission"
    assert set(calls) == {"Oh"}, calls


# ─────────────────────────────────────────────────────────────
# WP-B5-5b step 3: the half-integer route too
# ─────────────────────────────────────────────────────────────
#
# Measured before implementing, as step 2 was, and this one needed it more:
# the two half-integer emitters are *not* obviously the same arithmetic.
# ``_make_oh_op_adds_half`` goes through ``GaugedBasis.coefficient``, while
# ``_make_d4h_op_adds_half`` fetched D4h partner vectors and applied
# ``gauge_factor`` itself. Reading them suggests they agree; measuring them
# shows it -- 54 cases over both manifolds, 868 ADD entries, max difference 0.0.
#
# One thing the reading did turn up and the measurement confirmed: the D4h
# partner lookup *demanded* a pinned route phase, which an Oh calculation never
# has (it builds its basis with ``d4h=False``). That requirement is specific to
# D4h -- DT and DS are D4h scalars but not Oh ones, residual 31 -- so
# ``target_partners`` keeps it on the D4h branch instead of in the shared path.

#: ``(operator, non-empty cases, ADD entries)``, both counts measured.
#: 27 + 25 = the 52 non-empty cases of the aggregate measurement, and
#: 131 + 737 = its 868 entries. The two operators differ in case count because
#: two irreps carry a HAMILTONIAN block and no rank-4 ADD at all.
B55B_HALF_OPERATORS = [("HAMILTONIAN", 27, 131), ("TENDQ", 25, 737)]


def _half_basis_and_sizes(n_val):
    """``(basis, {manifold: j_sizes})`` for one odd electron count, or None."""
    from multitorch.angular import double_group
    import multitorch.angular.rac_generator as rg

    gs_j = rg._j_sector_sizes(rg._get_terms(2, n_val))
    if all(abs(J - round(J)) < 1e-9 for J in gs_j):
        return None
    ex_j = rg._get_excited_j_sizes(2, n_val, 1, 6)
    basis = double_group.gauged_basis([('gs', gs_j), ('ex', ex_j)], d4h=False)
    return basis, {'gs': gs_j, 'ex': ex_j}


@pytest.mark.parametrize("operator,n_cases,n_entries", B55B_HALF_OPERATORS)
def test_the_one_half_integer_emitter_reproduces_the_oh_one_exactly(operator, n_cases,
                                                                    n_entries):
    """``_make_op_adds_half(target_group='Oh')`` == ``_make_oh_op_adds_half``.

    Both manifold parities, every odd electron count, all three Oh double
    irreps. Counts asserted so a version that compared nothing fails.
    """
    from multitorch.angular.double_group import DOUBLE_IRREPS
    import multitorch.angular.rac_generator as rg

    compared = entries_seen = 0
    worst = biggest = 0.0
    for n_val in (1, 3, 5, 7, 9):
        prepared = _half_basis_and_sizes(n_val)
        if prepared is None:
            continue
        basis, sizes = prepared
        for manifold, parity in (('gs', 'g'), ('ex', 'u')):
            js = sorted(sizes[manifold])
            ham_idx = {J: i + 1 for i, J in enumerate(js)}
            cf_idx = {p: i + 1 for i, p in enumerate([(a, b) for a in js for b in js])}
            for irrep in DOUBLE_IRREPS:
                j_order = [(J, c, sizes[manifold][J]) for J in js
                           for c in range(basis.copies(manifold, J, irrep))]
                if not j_order:
                    continue
                want = rg._make_oh_op_adds_half(irrep, manifold, j_order, operator,
                                                basis, ham_idx, cf_idx)
                got = rg._make_op_adds_half(
                    irrep + parity, manifold,
                    [(J, irrep, c, 0, n) for (J, c, n) in j_order],
                    operator, basis, ham_idx_map=ham_idx, cf_idx_map=cf_idx,
                    cf_idx_map_rank2=cf_idx, target_group='Oh', rank=4)
                key = lambda a: (a.matrix_idx, a.bra, a.ket, a.nbra, a.nket)  # noqa: E731
                assert {key(a) for a in want} == {key(a) for a in got}, (
                    n_val, manifold, irrep)
                by_key = {key(a): a.coeff for a in got}
                for add in want:
                    worst = max(worst, abs(add.coeff - by_key[key(add)]))
                    biggest = max(biggest, abs(add.coeff))
                if want:
                    compared += 1
                    entries_seen += len(want)
    assert (compared, entries_seen) == (n_cases, n_entries)
    assert worst == 0.0, f"max |difference| {worst:.3e} over {entries_seen} entries"
    assert biggest > 0.5, biggest


def test_the_oh_target_needs_no_pinned_route_phase_and_the_d4h_one_still_does():
    """The asymmetry that would have made the Oh half-integer route unreachable.

    An Oh calculation builds its basis with ``d4h=False``, so ``route_phase`` is
    empty. The D4h partner lookup raises on that, correctly -- without the
    pinned phases DT and DS would not be real (residual 31). Had the shared path
    kept that check, every Oh half-integer call would have raised; had it dropped
    the check, D4h would have lost a real guard. Both halves asserted here.
    """
    prepared = _half_basis_and_sizes(7)
    assert prepared is not None
    basis, sizes = prepared
    node = next(n for n in basis.nodes if n[0] == 'gs' and n[2] == 'G3/2')

    assert not basis.route_phase, "an Oh basis should carry no route phases"
    import numpy as np
    assert np.array_equal(basis.target_partners(node, 'G3/2', target_group='Oh'),
                          basis.partners[node])
    with pytest.raises(RuntimeError, match="without d4h=True"):
        basis.target_partners(node, 'E1/2', target_group='D4h')
    with pytest.raises(ValueError, match="unknown subduction target"):
        basis.target_partners(node, 'G3/2', target_group='D3d')


# ── Is the half-integer path dimension-general? (residual 62) ──
#
# D3's double group has two **one-dimensional** spinor irreps, while every D4h
# double irrep is 2-D, so the half-integer path has only ever been exercised at
# dimension 2. The worry was a hardcoded 2 somewhere in it. Measured, there is
# none: every downstream use of a double-irrep dimension uses it *as a
# variable* -- ``sqrt(dim / (2J + 1))`` in the operator coefficient,
# ``sub[:, copy * dim:(copy + 1) * dim]`` in the partner slice, ``range(dim)``
# in the layout. What was D4h-specific was the *lookup*, and that is
# ``target_double_irrep_dim``.
#
# These two tests pin both halves of that: the lookup now knows the trigonal
# dimensions, and the arithmetic responds to a dimension of 1 the way the
# formula says it should rather than ignoring it.

def test_target_double_irrep_dim_knows_the_trigonal_dimensions():
    """D3d's {1, 1, 2}, with Oh's and D4h's tables untouched."""
    from multitorch.angular.double_group import target_double_irrep_dim

    assert target_double_irrep_dim('G3/2', 'Oh') == 4
    assert target_double_irrep_dim('E1/2', 'Oh') == 2
    assert target_double_irrep_dim('E1/2', 'D4h') == 2
    assert target_double_irrep_dim('E3/2', 'D4h') == 2

    assert target_double_irrep_dim('E1/2', 'D3d') == 2
    assert target_double_irrep_dim('1E3/2', 'D3d') == 1
    assert target_double_irrep_dim('2E3/2', 'D3d') == 1

    # D4h's E3/2 is not a D3d irrep, and saying so beats returning 2
    with pytest.raises(ValueError, match="no double-group irrep"):
        target_double_irrep_dim('E3/2', 'D3d')


def test_the_half_integer_operator_coefficients_scale_as_sqrt_of_the_dimension():
    """Force the double-irrep dimension to 1 and watch the coefficients move.

    The probe is artificial -- D4h has no 1-D double irrep -- and that is the
    point: it drives the one quantity a trigonal target would change and shows
    the arithmetic is driven by it rather than by a literal 2. Every operator
    coefficient scales by exactly ``sqrt(1/2)``, uniformly across HAMILTONIAN,
    10DQ, DT and DS, because the dimension enters only as ``sqrt(dim)``.

    **The dipole blocks do not move at all**, which is a caveat rather than a
    reassurance: the dipole emitter never consults the double-irrep dimension,
    so this measurement says nothing about whether it is right for a 1-D block.
    Residual 57 declined to unify the dipole emitters and residual 62 leaves
    that question open; it is the thing to check when D3d's half-integer path
    is built, not something these numbers settle.
    """
    import math

    from multitorch.angular import double_group
    from multitorch.angular.rac_generator import generate_ledge_template

    def coefficients(sym, n):
        rac, _store, _dec = generate_ledge_template(2, n, sym=sym)
        return {(b.bra_sym, b.geometry): tuple(a.coeff for a in b.add_entries)
                for b in rac.blocks if b.add_entries}

    # d7 puts half-integer J in both manifolds, so every block here is a
    # double-group one and the probe reaches all of them
    real = coefficients('d4h', 7)

    original = double_group.target_double_irrep_dim
    double_group.target_double_irrep_dim = (
        lambda stem, target: 1 if target == 'D4h' else original(stem, target))
    try:
        forced = coefficients('d4h', 7)
    finally:
        double_group.target_double_irrep_dim = original

    assert set(real) == set(forced), "forcing the dimension changed the block set"

    ratios: dict = {}
    compared = 0
    for key in real:
        assert len(real[key]) == len(forced[key]), key
        for before, after in zip(real[key], forced[key]):
            if abs(before) > 1e-12:
                ratios.setdefault(key[1], set()).add(round(after / before, 10))
                compared += 1

    assert compared > 500, f"only {compared} coefficients compared; too few to mean much"

    expected = round(1.0 / math.sqrt(2.0), 10)
    for operator in ('HAMILTONIAN', '10DQ', 'DT', 'DS'):
        assert ratios[operator] == {expected}, (
            f"{operator}: ratios {sorted(ratios[operator])}, expected "
            f"the single value {expected} = sqrt(1/2)")

    for operator in ('PARA', 'PERP'):
        assert ratios[operator] == {1.0}, (
            f"{operator}: the dipole emitter should not consult the "
            f"double-irrep dimension, but ratios were {sorted(ratios[operator])}")
