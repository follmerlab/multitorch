# `tests/golden/` — snapshots of this package's own output

**These are not reference data and not oracles.** They cannot establish that a
number is correct. They establish that a number has not *changed*.

That is a different thing from `tests/reference_data/`, which holds **Fortran**
output and *is* an oracle — and which must never be modified. A golden file
here, by contrast, is expected to be rewritten when a change to our own code is
intended, by its blessing script and never by hand.

| directory | what it snapshots | blessed by |
|---|---|---|
| `rac/` | every ADD coefficient of the generator's `oh7`, `oh8`, `d4h7`, `d4h8` cases, at 17 significant digits | `tests/tools/bless_rac_golden.py` |

Before re-blessing anything here, read the module docstring of the test that
consumes it. For `rac/` that is
`tests/test_angular/test_rac_golden_coefficients.py`, which explains why the
`d4h7` case is the one that matters: it is the only one of the four with no
Fortran reference away from the Oh limit, so a snapshot is the *only* thing
standing between an unintended coefficient change and a green suite.
