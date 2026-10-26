"""Regenerate the ADD-coefficient golden snapshots under ``tests/golden/rac/``.

    /opt/anaconda3/envs/multi/bin/python tests/tools/bless_rac_golden.py

Read ``tests/test_angular/test_rac_golden_coefficients.py``'s module docstring
first. These files are a snapshot of this package's own output, not an oracle:
re-blessing records a change, it does not justify one. Before running this,
confirm the gauge-invariant quantities have not moved -- the Oh-limit Fortran
parity, the descent limit, the dipole sum rule, the LAPACK scramble harness --
and say in the commit message which coefficients moved and why.

Prints the per-case entry count and digest, and for an existing file how many
ADD entries the re-blessing moves, so the number goes in the commit message
rather than being discovered later.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "test_angular"))

from test_rac_golden_coefficients import (  # noqa: E402
    CASES, GOLDEN, build_fingerprint, golden_path, _add_lines,
)


def main() -> int:
    import hashlib

    GOLDEN.mkdir(parents=True, exist_ok=True)
    changed = 0
    for sym, n, n_add in CASES:
        path = golden_path(sym, n)
        new = build_fingerprint(sym, n)
        adds = _add_lines(new)
        if len(adds) != n_add:
            print(f"{sym}{n}: REFUSED -- {len(adds)} ADD entries, but CASES says "
                  f"{n_add}. Update CASES deliberately if the count really changed.")
            return 1
        moved = ""
        if path.exists():
            old = path.read_text()
            if old == new:
                print(f"{sym}{n}: unchanged ({len(adds)} entries)")
                continue
            old_adds = _add_lines(old)
            n_moved = sum(1 for a, b in zip(old_adds, adds) if a != b)
            moved = f"  <-- {n_moved} of {len(old_adds)} ADD entries moved"
            changed += 1
        else:
            moved = "  <-- new file"
            changed += 1
        path.write_text(new)
        digest = hashlib.sha256(new.encode("utf-8")).hexdigest()[:16]
        print(f"{sym}{n}: {len(adds):5d} entries  {digest}{moved}")
    print(f"\n{changed} file(s) written to {GOLDEN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
