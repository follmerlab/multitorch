# Port notes — bugs found during standalone-parity work

Running log of pre-existing bugs surfaced while porting Fortran
ttmult features into multitorch's standalone path. Each entry: where
the bug was, runtime impact, and how it was fixed.

The phased plan that motivated this work is at
`~/.claude/plans/lexical-toasting-kahan.md`.

---

## BUG-001 — D4H_SHELL_BRANCHES had wrong values and insufficient schema

**File:** `multitorch/angular/symmetry.py`
**Found:** 2026-05-02 (Phase 1a of standalone-parity port)
**Runtime impact:** None — the table had zero consumers in-repo. No
shipped spectra were affected. But Phase 1c would have silently
produced wrong D4h spectra if the table had been wired in as-is.

### Symptom

Three Butler branch coefficients in the dormant scaffolding table:

```python
D4H_SHELL_BRANCHES = {
    ('2+', '0+', '0+'): -math.sqrt(70.0) / 10.0,    # ds → A1g
    ('4+', '0+', '0+'): 3.0 * math.sqrt(30.0) / 5.0,    # tendq → A1g
    ('4+', '2+', '2+'): -5.0 * math.sqrt(42.0) / (2.0 * 7.0),   # dt → Eg
}
```

Two of three values disagreed with `pyctm/pyctm/write_RAC.py:gen_X400_X420_X220`:

| Entry | Was | Should be | Off by |
|---|---|---|---|
| dt → Eg `('4+','2+','2+')` | `-5*sqrt(42)/14` | `-5*sqrt(42)/2` | 7× |
| ds → A1g `('2+','0+','0+')` | `-sqrt(70)/10` | `-sqrt(70)` | 10× |
| tendq → A1g `('4+','0+','0+')` | `3*sqrt(30)/5` | `6*sqrt(30)/10` | ✓ correct |

Plus a missing entry: dt also contributes to A1g via `-7*sqrt(30)/2 * dt`
(the cross term in pyctm's `X400 = ... - (7/2)*sqrt(30)*dt`).

The schema itself was also broken: with key `(O3_irrep, Oh_irrep, D4h_irrep)`
the tendq → A1g and dt → A1g contributions both map to `('4+','0+','0+')`,
so they cannot coexist in a single dict.

### Root cause

The table comment said *"These match the exact BRANCH entries in
write_RAC.py"* and even listed the right numbers in a sub-comment
(`-0.8367 = -sqrt(70)/10`), but the values stored in the dict were a
factor of 10 smaller than those numbers. Likely a units / normalization
confusion at scaffolding time that was never caught because nothing
consumed the table.

### Fix

Replaced with `D4H_BRANCHES_BY_OPERATOR`, a per-operator dict of unit
branch coefficients matching the pyctm formulas verbatim:

```python
D4H_BRANCHES_BY_OPERATOR = {
    'TENDQ': {('4+', '0+', '0+'): 6.0 * sqrt(30.0) / 10.0},
    'DT': {
        ('4+', '0+', '0+'): -7.0 / 2.0 * sqrt(30.0),
        ('4+', '2+', '2+'): -5.0 / 2.0 * sqrt(42.0),
    },
    'DS': {('2+', '0+', '0+'): -sqrt(70.0)},
}
```

The legacy `D4H_SHELL_BRANCHES` is kept as a deprecated alias populated
from `D4H_BRANCHES_BY_OPERATOR['TENDQ']` (the only single-branch
operator) so any future readers get the correct tendq value.

### Verification

`tests/test_d4h/test_d4h_from_scratch_status.py::test_d4h_butler_coefficients_match_pyctm`
asserts each entry verbatim against the pyctm formulas. Test passes.

### Update during Phase 1b implementation

Even after replacing values, the **key triplets themselves** were
wrong vs pyctm `BRANCH` lines. The third slot of the Butler key is
the D4h target irrep; pyctm always has `'0+'` (= D4h A1g) there
because CF operators are scalar (totally symmetric, A1g of D4h —
they're invariant under the symmetry operations even though they
*split* basis states with different angular weights). The middle
slot (Oh route) was also wrong for DS: should be `'2+'` (= Oh E),
not `'0+'` (= Oh A1).

Final corrected keys:

| Operator | Key | pyctm BRANCH line |
|---|---|---|
| TENDQ | `('4+', '0+', '0+')` | `BRANCH 4+ > 0 0+ > 0+` |
| DT (rank-4 A1) | `('4+', '0+', '0+')` | `BRANCH 4+ > 0 0+ > 0+` |
| DT (rank-4 E) | `('4+', '2+', '0+')` | `BRANCH 4+ > 0 2+ > 0+` |
| DS | `('2+', '2+', '0+')` | `BRANCH 2+ > 0 2+ > 0+` |

This was caught by `test_d4h_cf_operator_recipe_ds` failing because
`d4h_cf_operator_recipe('DS')` returned `oh='A1'` when pyctm says it
should go via Oh E.

### Follow-up gap: DS block silently empty

After the D4h dispatcher landed (commit pending), DS blocks emit
correctly-shaped RACBlockFull entries but with **zero ADD entries**
because `oh_coupling_coefficients_full(J_max, k=2, l=2)` returns
no couplings — rank-2 in Oh is E + T2, not A1, and the existing
helper only computes A1-projected (scalar) couplings.

DT works partially because rank-4 has both A1 and E components in
Oh, so `oh_coupling_coefficients_full(J_max, k=4)` returns A1
couplings; the E path is also questionable but at least produces
some output.

The proper fix requires computing matrix elements of the rank-2
operator within the Oh E irrep and projecting via Butler subduction
to D4h A1g. This is the kind of work the deferred ttrac.c port
handles; multitorch's current angular helpers don't expose it
directly. A `multitorch/angular/non_scalar_oh_couplings.py` helper
that computes per-irrep matrix elements for rank-K operators that
aren't scalar in Oh would close this gap.

Verified: `ds=0` and `ds=0.2` give the same spectrum, confirming
DS doesn't propagate. `dt=0.2` does change the spectrum (2.12
max diff) because rank-4 has an A1 component the existing helper
can compute.

### Lesson

Scaffolding-without-consumers is high-risk: when a table or function
exists but isn't called, drift accumulates silently. Whenever this
port adds a new module-level constant, add a parity test that
references it immediately, even if no production code path uses it
yet. The xfail-sentinel pattern in `tests/test_d4h/`,
`tests/test_ct/`, `tests/test_rixs_from_scratch/` is the structural
fix.
