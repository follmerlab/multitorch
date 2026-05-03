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

---

## BUG-002 — D4h Fe fixture bootstrap blocked by commented disktrnsform

**File (upstream):** `ttmult/src/ttrac.c` line 7027
**Found:** 2026-05-03 (during Phase 0-bootstrap attempt)
**Runtime impact:** Cannot generate new D4h fixtures via the Fortran
toolchain. The bundled `nid8`/`nid8ct` fixtures must have been
generated with a different ttrac build that had this call enabled.

### Symptom

Running `pyctm.calcXAS(element='Fe', valence='ii', sym='d4h', ...)`
or invoking `pyttmult.runrac(filename)` directly produces all the
expected files (`.rac`, `.rcg`, `.m14`, `.ora`, `.rcn31_out`, etc.)
EXCEPT the critical `.m15` file (= `rme_rac` in multitorch's
fixture naming convention).

### Root cause

In `ttmult/src/ttrac.c:7027`:
```c
printtrnsform(a);
/* disktrnsform(a); */    ← commented out
```

`disktrnsform()` is the function that calls `open_out_disk()` →
`fopen("rme_out.dat", "w")`. With it commented out, ttrac never
creates `rme_out.dat`, so `pyttmult.runrac()`'s
`if exists('rme_out.dat'): move(..., '.m15')` is a no-op.

### Workarounds

1. **Recompile ttrac.c** with line 7027 uncommented and rebuild
   the `ttrac` binary in `ttmult/bin/`. Out of scope this session.
2. **Use the in-multitorch D4h dispatcher** (Phase 1c interim) for
   D4h work — accepts the from-scratch-path accuracy ceiling
   (~0.97 cos vs 0.99999) and the documented DS-block gap.
3. **Fit in Oh approximation** for D4h experimental data using the
   bundled Oh Fe fixtures (`fe2_d6_oh`, `fe3_d5_oh`) as a v0 step
   while the proper D4h path matures.

### Status

Fixture-bootstrap pathway deferred. Phase 6 fitting proceeds with
option (3) — Oh approximation v0.

---

## Perf-001 — Fe(III) calcXAS_cached eigh bottleneck (2026-05-03)

**File:** `multitorch/hamiltonian/diagonalize.py:25` `safe_eigh`
**Found:** 2026-05-03 (during Phase 6 v0 fitter diagnostics)
**Runtime impact:** Fe(III) d5 calcXAS_cached takes 25.4 s/forward
vs Fe(II)'s 0.32 s — 80× slowdown. Makes 200-step Adam fits ~3 hours
per Fe(III) spectrum.

### Root cause

cProfile (single no_grad call): 89 % of wall time is inside one
`torch.linalg.eigh` call (LAPACK `dsyevd`) on dense float64 Hermitian
blocks. 16 calls × avg 2.92 s each. Largest block 190×190 (Fe(III) d5
high-spin has 12 LS terms vs d6's 5, producing larger Hamiltonian
blocks per J-sector after symmetry reduction).

NOT a Python overhead, NOT autograd graph size, NOT Hamiltonian
assembly, NOT deepcopy of cache.ban. Just dense CPU eigendecomposition.

### Refactor plan

Full review in `multitorch/CODE_REVIEW.md` (gitignored, local artifact).
Headline priority order:

1. **P0: GPU eigh** — confirmed 10–50× speedup achievable via
   torch.linalg.eigh on user's NVIDIA workstation. 1–2 hours, zero
   physics risk. Single change unblocks all Fe(III) fitting work.
2. **P1: Drop `torch.allclose` Hermiticity check** in `safe_eigh` —
   1.68 s of pure overhead per Fe(III) forward. 15-min change.
3. **P3: Lanczos partial eigh** for Boltzmann-bounded eigenpairs —
   5–20× additional speedup for blocks where N >> k_relevant.
   1–2 days, requires custom autograd.Function.

### De-risking (do BEFORE any refactor)

- Add Fe(III) regression bench cell to `bench/` (record 10 lowest
  stick energies + L3/L2 ratio + wall time)
- Add autograd parity test (finite-diff vs autograd grad on slater
  for Fe(III) at the from-scratch level — currently only Fe(II) and
  Ni(II) are tested per `tests/test_atomic/test_scaled_params.py:250`)
- Confirm P0 on the user's workstation with a 5-line `device='cuda'`
  smoke test before committing to the refactor

### Workaround until P0 lands

v0 fitter (`fits/fe_xas_fit.py`) demonstrates the differentiable
fitting machinery on Fe(II)Pc only (~30 sec/200-step fit). Fe(III)
fits deferred. Documented in `fits/results/v0_diagnostics.md`.

---

## Update — narrow ttrac port assessment (2026-05-03)

`CODE_REVIEW.md` Addendum 2 scopes a NARROW port to close BUG-001's
follow-up DS gap and the Oh-vs-D4h labeling gap.

**Headline**: ttrac.c is NOT a viable port target for these gaps because:

1. ttrac.c uses Butler-tabulated 3jm/6j symbols loaded from data files
   (F6/Oh, F3/SO3O, F6/D4, F3/OD4). Those tables aren't in the
   workspace; the binary embeds them.
2. multitorch's existing `point_group.py` (1886 LOC) computes Oh
   subduction from scratch via SO(3) Wigner D-matrices + character
   theory. This is mathematically equivalent to Butler's approach but
   doesn't depend on tabulated data.
3. The DS gap is a *one-line* generalization: `_a1_vector(k)` (line 924)
   only works for operator irreps containing an A1 component (k=0, k=4).
   Generalizing to `_irrep_vector(k, op_irrep)` for non-scalar
   operators (k=2 → E or T2) is the focused fix.

**Recommended path**: Path B — extend multitorch from-scratch
infrastructure with `_irrep_vector` + `oh_coupling_coefficients_for_op`.
~310 LOC, ~4.5 days. Pure Python (no autograd implications since ADD
coefficients are constants per Track C scoping).

**Gap #2 (D4h labels)**: separate ~150 LOC, 2-3 days. Needs a new
`oh_to_d4h_subduction_matrix` helper to rotate basis vectors from
Oh-irrep blocks to D4h-irrep blocks. Independent of Gap #1; can be
done after.

**Total to close both gaps**: ~1 week, all in multitorch (no upstream
ttrac dependency, no Butler-table dependency).

The Track C "defer ttrac" scoping decision still holds for the
WHOLESALE port (~7000 LOC). It does NOT preclude this narrow extension
because the foundational angular-algebra infrastructure already exists.

---

## Update — DS gap deeper than expected (2026-05-03)

Started Path B implementation (CODE_REVIEW.md Addendum 2 plan):
landed `_irrep_vector(k, op_irrep)` and skeleton
`oh_coupling_coefficients_for_op` in
`multitorch/angular/point_group.py`. Internal A1-consistency check
passes for 21/25 entries (sign discrepancies due to BFS phase
determination not being applied in the new function — minor).

**But empirical test reveals a deeper issue**: even with a valid
non-zero `_irrep_vector(2, 'E')`, the resulting
`oh_coupling_coefficients_for_op(k=2, op_irrep='E')` returns **0
entries** for all basis irreps including E and T2 (which the
selection rule says SHOULD be non-zero).

Root cause: the trace `Tr(B_bra^† O B_ket) / dim_g` formulation
captures coupling **strength** correctly only for **scalar (A1)
operators**, where matrix elements within an irrep are independent
of the partner index μ (Wigner-Eckart in trivial form). For
non-scalar operators, the right quantity is

    < Γ_b, μ | O^Γ_op_q | Γ_a, ν > = < Γ_b || O^{Γ_op} || Γ_a > × CG^Oh(Γ_a, ν; Γ_op, q; Γ_b, μ)

where the **Butler-style CG^Oh coefficients** are exactly what
ttrac's tabulated F3/SO3O and F3/OD4 tables provide. The trace
formulation averages over μ=ν and the off-diagonal partners cancel
for non-scalar ops, leaving a near-zero result.

**Implication for the Path B plan**: closing the DS gap properly
needs implementing Butler 3jm symbols from scratch via Oh subduction
matrices — the SO(3) Wigner machinery alone isn't sufficient. This
is more substantial work than the senior-code-reviewer ~4.5 day
estimate suggested. Realistic estimate: 5–7 days.

Three options going forward:

1. **Implement Butler 3jm via Oh subduction** (~5–7 days). Use
   `multitorch/angular/point_group.py:oh_subduction_matrix` to
   project basis states + operator-component states into Oh-irrep
   blocks; extract CG^Oh coefficients from the projection. Uses
   only existing primitives in multitorch — no Butler tables needed.

2. **Use ttrac's BRANCH-coefficient lookup approach as a partial
   workaround**: for D4h CF operators specifically (TENDQ, DT, DS,
   plus any future trigonal CF), pre-tabulate the Butler-derived
   matrix-element coefficients per (operator, basis_irrep, J_b, J_k)
   tuple by running ttrac once and parsing its output. This avoids
   implementing Butler 3jm from scratch but requires the workaround
   for BUG-002 (recompiled ttrac binary that writes rme_out.dat).

3. **Defer Phase 1c DS work entirely**: ship the v0 fitter with
   only Fe(II) Oh-approximation (working today). Document the DS
   gap as future work. Continue Phase 1c labeling fix (Gap #2) and
   GPU eigh confirmation (Perf-001 P0) independently.

Recommendation: **option 3 short-term, option 1 medium-term**.
The DS gap is real but doesn't block the v0 fitter narrative. Option
1 is the right long-term approach; option 2 is messier and reintroduces
the Fortran dependency we want to remove.

Code committed (point_group.py changes + 1 new test) provides the
foundation for the eventual Butler-3jm implementation: `_irrep_vector`
gives the right operator-component vectors, the `oh_coupling_coefficients_for_op`
skeleton has the right loop structure, and the test infrastructure is in
place. The missing piece is the Butler-3jm computation that replaces
the trace formulation.

---

## Update — D4h labeling gap is bigger than expected too (2026-05-03)

Started Thread 3 (D4h labels per Phase 1c gap #2). Landed:

- `D4H_TO_BUTLER` mapping (`A1g→'0+'`, `A2g→'^0+'`, `Eg→'1+'`,
  `B1g→'2+'`, `B2g→'^2+'`)
- `d4h_butler_label(d4h_irrep)` accessor
- `d4h_irreps_for_J(J)` returning the per-J D4h irrep multiplicities
  (composes oh_branching with OH_TO_D4H)

These are the foundation for relabeling. But: **the proper relabeling
needs an `oh_to_d4h_subduction_matrix(oh_irrep)` helper that
multitorch doesn't have yet.** Required because:

- Each Oh irrep splits into 1 or more D4h irreps per OH_TO_D4H:
  Oh Eg → D4h A1g + B1g (rotation in 2D Eg space)
  Oh T1g → D4h A2g + Eg (rotation in 3D T1g space)
  Oh T2g → D4h B2g + Eg (rotation in 3D T2g space)
- The subduction matrices specify HOW the Oh-irrep partner basis
  rotates into D4h-irrep partners. multitorch's existing
  `oh_subduction_matrix(J)` projects D^J → Oh; we need the next step
  Oh → D4h.

Implementing requires:

- D4h irrep matrices (for the 8 rotations in D4h's rotation subgroup)
- D4h character tables
- Character-projector eigenvectors per D4h irrep within each Oh irrep

Scope estimate: ~3–5 days alone, on top of the Butler-3jm work for
Gap #1.

**Workaround note**: the current dispatcher uses
`butler_label('A1', '+') = '0+'` etc., which gives **Oh-Butler labels**
not D4h-Butler labels. The two coincide only for the 1D irreps
(A1g/'0+'). For Oh Eg the dispatcher emits label '2+' which the
nid8 fixture uses for D4h B1g — so a downstream consumer reading
the RAC structure would see semantically wrong labels.

**This doesn't break the spectrum** (assembler is symmetry-agnostic)
but it does break parse-time validation against fixtures like nid8.

**Decision**: defer Thread 3 step 2+ (subduction matrices + dispatcher
relabeling) to a future session. Pivoting to Thread B (Butler 3jm
implementation) per user's A→B→C ordering.
