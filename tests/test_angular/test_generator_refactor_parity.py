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

**No golden files**, deliberately: the 18 full fingerprints run to 1.35 MiB,
and a stored copy of the generator's output is a second thing to keep in step.
The retained legacy function is the reference instead, exactly as WP-B3 kept
``_d4h_operator_vector_butler`` as the Butler-route oracle.

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


def test_the_public_entry_point_delegates_to_the_retained_legacy_path():
    """Today the two sides are the same code, and the file says so openly.

    When a later increment makes them different this assertion is the one to
    delete -- not the gate above, which is the whole point of the file.
    """
    import inspect

    src = inspect.getsource(generate_ledge_template)
    assert "_generate_ledge_template_legacy(" in src
    assert "if sym == 'd4h'" not in src, "the branchy body should live in the legacy function"


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
