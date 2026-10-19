"""Canonical RAC rendering — WP-B5's "byte-equal RAC output" oracle.

The plan names that oracle for B5, and nothing could produce such bytes before:
``io.write_inputs.write_rac`` writes a ttmult *input* file, not the
:class:`RACFileFull` the generator returns. These tests establish that the
rendering is faithful enough to serve as one, which means two things —
it must be **deterministic** (same structure, same string) and
**discriminating** (any change to any field or coefficient shows up). The
second is the one worth testing hard: a fingerprint that misses a changed
coefficient would make every later B5 increment look correct.
"""
from __future__ import annotations

import dataclasses

import pytest

from multitorch.angular.rac_generator import generate_ledge_template
from multitorch.io.rac_text import (COEFF_FORMAT, rac_digest, rac_fingerprint,
                                    rac_fingerprint_diff)


@pytest.fixture(scope="module")
def oh_rac():
    rac, _store, _dec = generate_ledge_template(2, 8, 1, 6, cf_rank=4, sym="oh")
    return rac


@pytest.fixture(scope="module")
def d4h_rac():
    rac, _store, _dec = generate_ledge_template(2, 8, 1, 6, cf_rank=4, sym="d4h")
    return rac


def test_the_fingerprint_is_deterministic(oh_rac):
    assert rac_fingerprint(oh_rac) == rac_fingerprint(oh_rac)
    assert rac_digest(oh_rac) == rac_digest(oh_rac)


def test_regenerating_the_same_rac_gives_the_same_fingerprint():
    """Two independent builds of the same input must be byte-identical.

    This is the property the whole B5 oracle rests on: if the generator were
    not reproducible run to run, no byte comparison could mean anything.
    """
    first, _s, _d = generate_ledge_template(2, 6, 1, 6, cf_rank=4, sym="d4h")
    second, _s2, _d2 = generate_ledge_template(2, 6, 1, 6, cf_rank=4, sym="d4h")
    assert rac_fingerprint(first) == rac_fingerprint(second)


def test_oh_and_d4h_fingerprints_differ(oh_rac, d4h_rac):
    assert rac_digest(oh_rac) != rac_digest(d4h_rac)


def test_the_fingerprint_covers_every_block_and_add_entry(oh_rac):
    text = rac_fingerprint(oh_rac)
    assert f"BLOCKS {len(oh_rac.blocks)}" in text
    assert f"IRREPS {len(oh_rac.irreps)}" in text
    for index, block in enumerate(oh_rac.blocks):
        assert f"BLOCK {index} kind={block.kind}" in text
    total_adds = sum(len(b.add_entries) for b in oh_rac.blocks)
    assert text.count("    ADD ") == total_adds


def test_coefficients_are_rendered_at_full_double_precision():
    """``%.17g`` round-trips an IEEE double, so the last bits cannot hide."""
    nearly = 0.1 + 0.2                  # 0.30000000000000004
    assert float(COEFF_FORMAT % nearly) == nearly
    assert (COEFF_FORMAT % nearly) != "0.3"


# ─────────────────────────────────────────────────────────────
# Injected changes: the fingerprint must notice every one
# ─────────────────────────────────────────────────────────────

def _first_block_with_adds(rac):
    return next(i for i, b in enumerate(rac.blocks) if b.add_entries)


def test_a_changed_coefficient_changes_the_fingerprint(oh_rac):
    """One coefficient, perturbed in the last bits."""
    i = _first_block_with_adds(oh_rac)
    block = oh_rac.blocks[i]
    add = block.add_entries[0]
    nudged = dataclasses.replace(add, coeff=add.coeff * (1.0 + 1e-15))
    blocks = list(oh_rac.blocks)
    blocks[i] = dataclasses.replace(block, add_entries=[nudged] + list(block.add_entries[1:]))
    changed = dataclasses.replace(oh_rac, blocks=blocks)

    assert rac_digest(changed) != rac_digest(oh_rac)
    assert "coeff=" in rac_fingerprint_diff(oh_rac, changed)


@pytest.mark.parametrize("field,value", [
    ("kind", "EXCITE"), ("bra_sym", "ZZ"), ("op_sym", "ZZ"), ("ket_sym", "ZZ"),
    ("geometry", "NOPE"), ("n_bra", 999), ("n_ket", 999),
])
def test_a_changed_block_field_changes_the_fingerprint(oh_rac, field, value):
    i = _first_block_with_adds(oh_rac)
    blocks = list(oh_rac.blocks)
    blocks[i] = dataclasses.replace(oh_rac.blocks[i], **{field: value})
    assert rac_digest(dataclasses.replace(oh_rac, blocks=blocks)) != rac_digest(oh_rac)


@pytest.mark.parametrize("field,value", [
    ("matrix_idx", 999), ("bra", 999), ("ket", 999), ("nbra", 999), ("nket", 999),
])
def test_a_changed_add_field_changes_the_fingerprint(oh_rac, field, value):
    i = _first_block_with_adds(oh_rac)
    block = oh_rac.blocks[i]
    adds = list(block.add_entries)
    adds[0] = dataclasses.replace(adds[0], **{field: value})
    blocks = list(oh_rac.blocks)
    blocks[i] = dataclasses.replace(block, add_entries=adds)
    assert rac_digest(dataclasses.replace(oh_rac, blocks=blocks)) != rac_digest(oh_rac)


def test_reordering_blocks_changes_the_fingerprint(oh_rac):
    """Block order is part of the RAC file, so the oracle must not sort it away."""
    blocks = list(oh_rac.blocks)
    blocks[0], blocks[1] = blocks[1], blocks[0]
    assert rac_digest(dataclasses.replace(oh_rac, blocks=blocks)) != rac_digest(oh_rac)


def test_reordering_add_entries_changes_the_fingerprint(oh_rac):
    i = next(i for i, b in enumerate(oh_rac.blocks) if len(b.add_entries) > 1)
    block = oh_rac.blocks[i]
    adds = list(block.add_entries)
    adds[0], adds[1] = adds[1], adds[0]
    blocks = list(oh_rac.blocks)
    blocks[i] = dataclasses.replace(block, add_entries=adds)
    assert rac_digest(dataclasses.replace(oh_rac, blocks=blocks)) != rac_digest(oh_rac)


def test_the_diff_reports_identical_for_equal_structures(oh_rac):
    assert rac_fingerprint_diff(oh_rac, oh_rac) == "<identical>"
