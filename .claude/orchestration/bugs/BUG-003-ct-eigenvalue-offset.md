# BUG-003: CT Eigenvalue Offset ~1 Ry for Non-Ni Cases

**Status:** RESOLVED (2026-04-10)
**Severity:** HIGH
**Related investigation:** INV-001

## Root cause

**Multi-line COWAN matrix row parsing bug in `multitorch/io/read_rme.py:_parse_rme_block`.**

The .rme_rcg file stores sparse matrices with one row per line:
```
    1    5        ,   1     -1.848579,   3     -0.008981,   5     -0.008981,
   6      0.026031,   7     -0.028402 ;
```

When a row has many non-zero entries, it wraps to continuation lines. The continuation line starts with `col val` pairs (no row/nnz header). The parser incorrectly treated the first number on continuation lines as a new row index, silently dropping ~50% of matrix data for large blocks.

Small matrices (Ni2+ d8 with 2-12 states per block) fit on single lines and parsed correctly. Larger matrices (V3+ d2 with up to 29 states per block) wrapped and lost data.

## Fix

Modified `_parse_rme_block` to distinguish new row headers (`row nnz , ...` where nnz is an integer) from continuation lines (`col val , ...` where val is a float). Continuation lines now correctly append data to the current row.

## Verification

- 141/141 existing tests pass
- 7/8 fresh Fortran comparison cases pass (eigenvalues match to <6e-5 Ry across all triads)
- Ti4+ d0 eigenvalues match (3.7e-7 Ry) but spectrum cosine = 0.978 (minor broadening difference, not a Hamiltonian issue)
- Max eigenvalue error across 69 triads: 5.5e-5 Ry
