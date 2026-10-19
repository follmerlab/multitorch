"""Canonical text rendering of a parsed RAC structure (WP-B5's oracle).

WP-B5 rewrites the RAC generator as a loop over a symmetry plan, and the oracle
the plan names for it is "byte-equal RAC output vs today's Oh and D4h paths".
Nothing in the package could produce such bytes: ``io.write_inputs.write_rac``
writes a ttmult *input* file, not the :class:`RACFileFull` the generator
returns. This module supplies the missing rendering.

:func:`rac_fingerprint` writes every field of every block and every ADD entry,
with coefficients at full double precision (``repr``-grade, 17 significant
digits), in the order the generator emitted them. Block order is *preserved*,
not sorted: the order is part of the RAC file and so part of what must not
change. Two generators agree byte for byte exactly when their fingerprints are
equal, which makes a refactor's correctness a string comparison rather than a
spectrum comparison — much sharper, since a spectrum can absorb a sign error
that cancels while a coefficient cannot.

It is also useful on its own for reading a generated RAC, or diffing two of
them when they *do* differ, which is why it lives here rather than in the tests.
"""
from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:  # pragma: no cover
    from multitorch.io.read_rme import RACFileFull

#: Format for a coefficient: full double precision, so a fingerprint cannot
#: hide a difference in the last bits. ``%.17g`` round-trips an IEEE double.
COEFF_FORMAT = "%.17g"


def rac_fingerprint(rac: "RACFileFull") -> str:
    """Canonical, fully-precise text rendering of *rac*.

    Deterministic and order-preserving. Every ADD coefficient appears at 17
    significant digits, so equality of two fingerprints is equality of the two
    RAC structures down to the last bit of every coefficient.
    """
    lines: List[str] = []

    lines.append(f"IRREPS {len(rac.irreps)}")
    for irrep in rac.irreps:
        lines.append(f"  IRREP name={irrep.name} kind={irrep.kind} "
                     f"mult={irrep.multiplicity} dim={irrep.dim}")

    lines.append(f"BLOCKS {len(rac.blocks)}")
    for index, block in enumerate(rac.blocks):
        lines.append(
            f"  BLOCK {index} kind={block.kind} bra={block.bra_sym} op={block.op_sym} "
            f"ket={block.ket_sym} geometry={block.geometry} "
            f"n_bra={block.n_bra} n_ket={block.n_ket} adds={len(block.add_entries)}")
        for add_index, add in enumerate(block.add_entries):
            coeff = COEFF_FORMAT % add.coeff
            lines.append(
                f"    ADD {add_index} matrix_idx={add.matrix_idx} bra={add.bra} "
                f"ket={add.ket} nbra={add.nbra} nket={add.nket} coeff={coeff}")
    return "\n".join(lines) + "\n"


def rac_digest(rac: "RACFileFull") -> str:
    """SHA-256 of :func:`rac_fingerprint`, for a compact golden file."""
    return hashlib.sha256(rac_fingerprint(rac).encode("utf-8")).hexdigest()


def rac_fingerprint_diff(left: "RACFileFull", right: "RACFileFull",
                         max_lines: int = 20) -> str:
    """First differing fingerprint lines, for a failure message worth reading."""
    a = rac_fingerprint(left).splitlines()
    b = rac_fingerprint(right).splitlines()
    out: List[str] = []
    for i in range(max(len(a), len(b))):
        la = a[i] if i < len(a) else "<missing>"
        lb = b[i] if i < len(b) else "<missing>"
        if la != lb:
            out.append(f"line {i}:\n  -{la}\n  +{lb}")
            if len(out) >= max_lines:
                out.append("...")
                break
    return "\n".join(out) if out else "<identical>"
