# D4h dispatcher refactor — V2 implementation plan
**Status as of:** 2026-05-03 (V2 supersedes V1)
**V1 reference:** `docs/D4H_DISPATCHER_PLAN.md` (kept for history; do not consult during implementation)
**Glossary:** `UBIQUITOUS_LANGUAGE.md` — read this BEFORE the plan. The terms **manifold parity**, **operator parity**, **MULT**, **partner-summed entry count**, **D4h-Butler vs Oh-Butler label** are load-bearing throughout.
**Audit trail:** `CODE_REVIEW_D4H_WIRING.md` (the review that informed V2; flagged 3 must-fix bugs in V1 plus an unreachable gate test).

This document is self-contained. You should not need to consult V1 or
the review file once you start implementing.

---

## 0. Overview

The Phase 1c goal is to replace the broken `if sym == 'd4h':` branch in
`generate_ledge_rac` (currently producing **Oh-Butler-labeled** GROUND
and EXCITE blocks with empty DS ADDs) with a per-D4h-irrep dispatcher
that emits **D4h-Butler-labeled** GROUND, EXCITE, and TRANSI blocks
matching the bundled `nid8.rme_rac` fixture exactly (up to per-entry
sign). This closes BUG-001 (DS angular gap) and BUG #2 (label
collision) — the unified Threads 2+3 finding from Session 2.

**Foundation already on `origin/main`:** Butler-coefficient table,
Oh→D4h subduction matrices, D4h partner-basis builder, layout function,
Butler-label maps. See V1 plan §"What's already landed" if you want the
file:line table; not needed for implementation.

**Helpers already drafted in Session 2 (UNCOMMITTED in
`multitorch/angular/rac_generator.py:642-840`):** `_d4h_partner_vector`,
`_d4h_op_routes`, `_d4h_operator_vector_complex`, `_operator_real_matrix`,
`_make_d4h_op_adds`. **They contain three latent bugs that V2 fixes
in commit 1.**

## 1. The three latent bugs in the Session-2 helpers

These MUST be fixed before any wiring.

### Bug A — `parity` kwarg is wrong by construction
`_d4h_operator_vector_complex` and `_make_d4h_op_adds` both accept a
`parity` kwarg. V1's plan instructed callers to pass `parity='u'` for
EXCITE; this **conflates manifold parity with operator parity**.
CF operators are intrinsically gerade. With `parity='u'`, the helper
calls `oh_to_d4h_subduction_matrix('A1u')['A1g']`, which `KeyError`s
(empirically verified — A1u subduces only to A1u keys).

### Bug B — block matrix dimension formula
The Session-2 helper iterates `entries × entries`, where `entries` is a
list from `d4h_basis_layout`. For a multi-D D4h irrep (Eg, Eu) the
layout emits **one entry per partner**, so entries for Eg in Ni d8
contain 12 tuples that aggregate to a **partner-summed entry count** of
20. The matrix the assembler expects is sized by **MULT** (= 10 for
Eg). Iterating all 12 entries produces a 20×20 matrix; the assembler
splices it into a 10×10 slot, corrupting indices.

### Bug C — partner duplication in iteration
Same root cause as B. For Eg, layout entries with `partner_idx=0` and
`partner_idx=1` carry the same `n_states`. The CF operator is
partner-symmetric (Schur on a D4h-A1g operator), so iterating both
partners is double-counting. The cross-partner matrix elements are
zero by symmetry, but the **diagonal-in-partner** block from
`partner_idx=1` still emits ADDs at out-of-bounds positions.

**Resolution (commit 1):**

```python
def _make_d4h_op_adds(
    d4h_irrep: str,
    entries: List[Tuple[float, str, int, int, int]],
    operator: str,
    ham_idx_map: Dict[float, int],
    cf_idx_map: Dict[Tuple[float, float], int],
    cf_idx_map_rank2: Dict[Tuple[float, float], int],
    *,
    target_d4h_irrep: str = 'A1g',
) -> List[ADDEntry]:
    """Build ADD entries for one (D4h irrep, operator) pair.

    SCOPE LIMITATION: assumes the operator targets D4h-A1g (scalar
    CF operator). For non-scalar target irreps, partner_idx
    filtering below is invalid (partners would not be degenerate).
    Pass `target_d4h_irrep` to point at a non-A1g target only when
    you've audited the partner-symmetry assumption for that irrep.

    OPERATOR PARITY INVARIANT: the operator vector is built using
    gerade Oh subductions unconditionally. CF operators are
    intrinsically gerade; manifold parity is read from
    `d4h_irrep[-1]` inside `_d4h_partner_vector` and does not
    enter the operator vector.
    """
    # Filter to one partner per (J, oh, copy) — see SCOPE LIMITATION.
    entries = [e for e in entries if e[3] == 0]
    ...
```

`_d4h_operator_vector_complex` becomes:

```python
def _d4h_operator_vector_complex(
    operator: str, rank: int, target_d4h_irrep: str = 'A1g',
) -> np.ndarray:
    """Build the operator's m-basis vector in D^k (complex basis).

    OPERATOR PARITY INVARIANT: gerade Oh subductions, hardcoded.
    The function previously took a `parity` kwarg that conflated
    operator parity with manifold parity; that kwarg is removed.
    """
    op_vec_real = np.zeros(2 * rank + 1, dtype=np.float64)
    for r_k, oh_label, coeff in _d4h_op_routes(operator, target_d4h_irrep):
        if r_k != rank:
            continue
        oh_irrep_full = oh_label + 'g'  # CF operators are gerade
        B_oh = _real_subduction_matrix(rank, oh_label)
        sub = oh_to_d4h_subduction_matrix(oh_irrep_full)[target_d4h_irrep]
        route_vec = (B_oh @ sub).flatten()
        op_vec_real += coeff * route_vec
    U_k = _c2r_unitary(rank)
    return U_k.conj().T @ op_vec_real.astype(np.complex128)
```

`_d4h_op_routes` is parameterized on the target:

```python
def _d4h_op_routes(operator: str, target_d4h_irrep: str = 'A1g') \
        -> List[Tuple[int, str, float]]:
    target_butler = D4H_TO_BUTLER[target_d4h_irrep]
    branches = D4H_BRANCHES_BY_OPERATOR[operator]
    routes = []
    for (o3_irr, oh_butler, d4h_butler), coeff in branches.items():
        if d4h_butler != target_butler:
            raise RuntimeError(
                f"{operator}: branch target {d4h_butler!r} != "
                f"{target_butler!r} (target {target_d4h_irrep}); "
                "this dispatcher assumes a single target D4h irrep "
                "per operator."
            )
        rank = _O3_TO_RANK[o3_irr]
        oh_label = _BUTLER_OH_TO_LABEL[oh_butler]
        routes.append((rank, oh_label, coeff))
    return routes
```

`_make_d4h_op_adds` no longer threads `parity` to
`_d4h_operator_vector_complex`. The `parity` kwarg is **removed** from
both functions.

## 2. Sequencing (feature branch `d4h-dispatcher-v2`)

Create the branch from the current main at `c061f86` (most recent
commit before Session-2 helpers). The Session-2 helpers are uncommitted;
include them in commit 1 along with the bug fixes.

```bash
git checkout -b d4h-dispatcher-v2
```

Six commits, squashed on merge:

| # | Title | Acceptance |
|---|---|---|
| 1 | `d4h: import Session-2 helpers, fix parity / partner-filter bugs` | `pytest tests/test_d4h tests/test_angular -q` → 116 + 1 xfail (baseline holds) |
| 2 | `d4h: per-D4h-irrep GROUND + EXCITE block emission` | `pytest tests/test_d4h/test_d4h_from_scratch_status.py::test_d4h_dispatcher_emits_nid8_irrep_set -q` flips xfail → passing |
| 3 | `d4h: per-D4h-irrep TRANSI block emission` | (no atomic test mid-flight; commit 4's deletion is when the suite goes green) |
| 4 | `d4h: delete OLD Oh-labeled emission` | `pytest tests/test_d4h tests/test_angular tests/test_integration -q` all green |
| 5 | `d4h: add gate, structural, and per-coefficient parity tests` | new tests pass; full `pytest tests/ -q` 477 + N green |
| 6 | `d4h: update PORT_NOTES.md (BUG-001/-002 closed)` | docs only; no test impact |

**Rollback plan:** if any commit fails to recover by squash time,
`git checkout main && git branch -D d4h-dispatcher-v2`. The OLD path
is untouched on main until the squash merge lands.

**Do not commit between commits 2 and 3 to main.** The interim state
between them has D4h-Butler GROUND/EXCITE labels but Oh-Butler TRANSI
labels — the assembler triad-matches by string, and label collisions
in this interim state will silently match wrong triads. The branch
allows arbitrary intermediate states; main does not.

## 3. Commit 1 — fix Session-2 helpers

Edit `multitorch/angular/rac_generator.py:642-840` (the helper block
between `generate_ground_state_rac` and `generate_ledge_rac`):

1. Drop `parity` kwarg from `_d4h_operator_vector_complex` (lines
   696-721). Hardcode `'g'` for the operator's Oh-route subduction.
2. Drop `parity` kwarg from `_make_d4h_op_adds` (lines 743-840). Stop
   forwarding it.
3. Add `target_d4h_irrep='A1g'` kwarg to both functions and
   `_d4h_op_routes`. Thread through. Relax the `'0+'` runtime check in
   `_d4h_op_routes` to compare against `D4H_TO_BUTLER[target_d4h_irrep]`.
4. Inside `_make_d4h_op_adds`, first line: filter
   `entries = [e for e in entries if e[3] == 0]` and update the
   docstring with the SCOPE LIMITATION block from §1.
5. Add an inline comment in the HAMILTONIAN branch:
   `# Cross-(oh, copy) matrix elements are zero by partner-basis`
   `# orthogonality; same_entry is necessary AND sufficient.`

**Validation:**
```bash
pytest tests/test_d4h tests/test_angular -q
# Expected: 116 passed, 1 xfailed (baseline unchanged — helpers still not wired in)
```

## 4. Commit 2 — Step 3 wiring (GROUND + EXCITE)

In `generate_ledge_rac`, replace the existing `if sym == 'd4h':`
GROUND/EXCITE emission (the per-Oh-irrep loops at lines 1143-1295)
with a per-D4h-irrep emission. **Do NOT touch the TRANSI loop yet
(lines 1079-1141).** The interim state will have the label collision;
this is expected for the lifetime of commit 2 → 3.

Add a half-integer J gate at the top of the d4h branch (per **Q5a** of
the grilling — V2 protects users from a confusing deep
`NotImplementedError`):

```python
if sym == 'd4h':
    is_half_gs = abs(J_max_gs - round(J_max_gs)) > 0.1
    is_half_ex = abs(J_max_ex - round(J_max_ex)) > 0.1
    if is_half_gs or is_half_ex:
        raise NotImplementedError(
            "sym='d4h' currently supports only even-electron-count "
            "configurations (integer J in both ground and excited "
            "manifolds). Got half-integer J — likely an odd "
            "electron count (Fe d5, Cu d9, etc.). Use sym='oh' "
            "or wait for D4h double-group support."
        )
    if cf_rank != 4:
        raise ValueError(
            f"sym='d4h' uses fixed rank-4 (TENDQ/DT) and rank-2 (DS) "
            f"operators; cf_rank={cf_rank} is ignored on this path. "
            f"Pass cf_rank=4 (default) to silence this."
        )
```

Then per-D4h-irrep emission for GROUND. The pattern (key formula:
`block_dim = sum(...) // D4H_IRREP_DIM[d4h_irrep]`):

```python
from multitorch.angular.symmetry import (
    D4H_TO_BUTLER, D4H_IRREP_DIM, d4h_basis_layout,
)

# --- GROUND blocks ---
layout_gs = d4h_basis_layout(gs_j_sizes, parity='g')
for d4h_irrep, entries in sorted(layout_gs.items()):
    if not entries:
        continue
    butler = D4H_TO_BUTLER[d4h_irrep]
    block_dim = sum(e[4] for e in entries) // D4H_IRREP_DIM[d4h_irrep]

    irrep_infos.append(IrrepInfo(
        name=butler, kind='GROUND',
        multiplicity=block_dim,            # = MULT, not partner-summed
        dim=D4H_IRREP_DIM[d4h_irrep],
    ))

    # 4 operator blocks per D4h irrep
    for op_name, geometry in (
        ('HAMILTONIAN', 'HAMILTONIAN'),
        ('TENDQ',       '10DQ'),           # TENDQ slot is named "10DQ"
        ('DT',          'DT'),
        ('DS',          'DS'),
    ):
        adds = _make_d4h_op_adds(
            d4h_irrep, entries, op_name,
            ham_idx_map=gs_ham_idx,
            cf_idx_map=gs_cf_idx,
            cf_idx_map_rank2=gs_cf_idx_rank2,
        )
        if not adds:
            continue
        blocks.append(RACBlockFull(
            kind='GROUND',
            bra_sym=butler,
            op_sym='0+',                   # operator targets D4h-A1g
            ket_sym=butler,
            geometry=geometry,
            n_bra=block_dim,
            n_ket=block_dim,
            add_entries=adds,
        ))
```

EXCITE follows identically with `parity='u'`, `ex_*` index maps, and
`kind='EXCITE'` in `IrrepInfo` but `kind='GROUND'` in `RACBlockFull`
(the assembler's config-1 convention; do NOT "fix" this asymmetry).

The OLD per-Oh-irrep GROUND/EXCITE emission (lines 1143-1295) is
deleted in commit 4; for now it lives alongside the new code, dead
because the d4h branch reaches the new code first. (If you prefer
parsimony, delete it now and roll commit 4 into commit 2 — but you'll
lose the visual diff against the OLD path that helps debug Q6-B
failures.)

**Validation after commit 2:**
```bash
pytest tests/test_d4h/test_d4h_from_scratch_status.py::test_d4h_dispatcher_emits_nid8_irrep_set -q
# Expected: PASSED (was xfail)

pytest tests/test_d4h tests/test_angular -q
# Expected: many tests still pass; the loose-tolerance test
# `test_d4h_ni_from_scratch_runs_and_matches_oh_baseline` MAY FAIL
# transiently because of the label collision. That's expected at this
# checkpoint. Do not commit fixes — proceed to commit 3.
```

## 5. Commit 3 — Step 4 wiring (TRANSI)

Replace the existing TRANSI loop in `generate_ledge_rac` (lines
1079-1141) with per-D4h-irrep TRANSI emission.

Reference data — `nid8.rme_rac` lines 13-197 list 13 TRANSI triads
(8 PERP, 5 PARA). Your dispatcher must reproduce this set exactly
(parsed at test time per **Q3-B**).

**The dipole prefactor:** start from the existing OLD path's
convention:
- `PERP_FACTOR = √(dim_Eu / dim_T1u) = √(2/3)`
- `PARA_FACTOR = √(dim_A2u / dim_T1u) = √(1/3)`

If Q6-B's per-coefficient parity test fails on TRANSI blocks, the
factor is wrong; the test diagnoses by how much. Implementation
recipe:

```python
# --- TRANSI (dipole) blocks ---
layout_gs = d4h_basis_layout(gs_j_sizes, parity='g')
layout_ex = d4h_basis_layout(ex_j_sizes, parity='u')

PERP_FACTOR = math.sqrt(2.0 / 3.0)   # √(dim_Eu / dim_T1u)
PARA_FACTOR = math.sqrt(1.0 / 3.0)   # √(dim_A2u / dim_T1u)

for d4h_gs, gs_entries in sorted(layout_gs.items()):
    gs_entries = [e for e in gs_entries if e[3] == 0]   # one partner per (J, oh, copy)
    if not gs_entries:
        continue
    n_bra = sum(e[4] for e in gs_entries)
    gs_butler = D4H_TO_BUTLER[d4h_gs]

    for d4h_ex, ex_entries in sorted(layout_ex.items()):
        ex_entries = [e for e in ex_entries if e[3] == 0]
        if not ex_entries:
            continue
        n_ket = sum(e[4] for e in ex_entries)
        ex_butler = D4H_TO_BUTLER[d4h_ex]

        for op_d4h_target, op_butler, geometry, factor in (
            ('Eu',  '1-',  'PERP', PERP_FACTOR),
            ('A2u', '^0-', 'PARA', PARA_FACTOR),
        ):
            adds = _make_d4h_dipole_adds(
                d4h_gs=d4h_gs, gs_entries=gs_entries,
                d4h_ex=d4h_ex, ex_entries=ex_entries,
                op_d4h_target=op_d4h_target,
                multipole_idx=multipole_idx,
                factor=factor,
            )
            if adds:
                blocks.append(RACBlockFull(
                    kind='TRANSI',
                    bra_sym=gs_butler,
                    op_sym=op_butler,
                    ket_sym=ex_butler,
                    geometry=geometry,
                    n_bra=n_bra,
                    n_ket=n_ket,
                    add_entries=adds,
                ))
```

You will need a new helper `_make_d4h_dipole_adds` that mirrors
`_make_d4h_op_adds`'s structure but for the dipole operator
(rank-1, target Eu OR A2u, derived from T1u via
`oh_to_d4h_subduction_matrix('T1u')[op_d4h_target]`). Pseudocode:

```python
def _make_d4h_dipole_adds(
    d4h_gs, gs_entries, d4h_ex, ex_entries,
    op_d4h_target, multipole_idx, factor,
):
    # Build dipole operator vector for the target D4h sub-irrep of T1u.
    # T1u = D^1_real (real-SH dipole basis); subduce to op_d4h_target,
    # pick partner 0 (partners are degenerate in spectrum due to BAN's
    # standard scaling).
    sub = oh_to_d4h_subduction_matrix('T1u')[op_d4h_target]
    op_vec_real = sub[:, 0]                   # 3-dim real-SH dipole vector
    U_1 = _c2r_unitary(1)
    op_vec_complex = U_1.conj().T @ op_vec_real.astype(np.complex128)

    adds = []
    bra_pos = 1
    for (Jb, oh_b, cb, _, nb) in gs_entries:
        ket_pos = 1
        for (Jk, oh_k, ck, _, nk) in ex_entries:
            if abs(Jb - Jk) > 1 or Jb + Jk < 1:
                ket_pos += nk
                continue
            if (Jb, Jk) not in multipole_idx:
                ket_pos += nk
                continue
            v_b = _d4h_partner_vector(Jb, oh_b, cb, d4h_gs, 0)
            v_k = _d4h_partner_vector(Jk, oh_k, ck, d4h_ex, 0)
            O_real = _operator_real_matrix(Jb, Jk, 1, op_vec_complex)
            me = float(v_b @ O_real @ v_k)
            if abs(me) < 1e-13:
                ket_pos += nk
                continue
            coeff = factor * me     # PERP / PARA scaling
            adds.append(ADDEntry(
                matrix_idx=multipole_idx[(Jb, Jk)],
                bra=bra_pos, ket=ket_pos,
                nbra=nb, nket=nk,
                coeff=coeff,
            ))
            ket_pos += nk
        bra_pos += nb
    return adds
```

The OLD TRANSI loop (lines 1079-1141) is deleted in commit 4.

## 6. Commit 4 — delete OLD d4h emission

Remove from `generate_ledge_rac`:
- The OLD per-Oh-irrep TRANSI loop (lines 1079-1141).
- The OLD per-Oh-irrep GROUND/EXCITE loops (lines 1143-1295) restricted
  to the `if sym == 'd4h':` branches and their `dt_a1_scale` /
  `ds_e_scale` magic.
- The Oh-only path stays untouched (it does not touch
  `D4H_BRANCHES_BY_OPERATOR` etc.).

After this commit, the file should have a clean
`if sym == 'd4h': <new dispatcher>` else `<Oh dispatcher>` structure.

**Validation after commit 4:**
```bash
pytest tests/ -q
# Expected: 477 passed, 0 xfail
# (the previous xfail flipped in commit 2; no new tests added yet)
```

If any unrelated test fails, debug — do not proceed.

## 7. Commit 5 — tests

Add the eight new tests below, delete one obsolete test, and verify
six existing tests still pass. Final coverage: **15 d4h-related tests**.

### 7.1. New tests (8)

#### `test_d4h_collapses_to_oh_when_dt_ds_zero` (Q3-A)
Numerical parity: in the `ds=dt=0` limit the d4h dispatcher must
collapse to the Oh dispatcher to machine precision. Catches numerical
regressions independent of the HFS Slater / F2_pd inaccuracy.

```python
def test_d4h_collapses_to_oh_when_dt_ds_zero():
    """When ds=dt=0, sym='d4h' must give the same spectrum as sym='oh'.

    This is a strict numerical-parity test that's independent of
    HFS Slater accuracy (both paths use the same Slater integrals).
    """
    from multitorch.api.calc import calcXAS_from_scratch
    x_oh, y_oh = calcXAS_from_scratch(
        "Ni", "ii", sym="oh", cf={"10dq": 1.0},
    )
    x_d4h, y_d4h = calcXAS_from_scratch(
        "Ni", "ii", sym="d4h",
        cf={"tendq": 1.0, "dt": 0.0, "ds": 0.0},
    )
    import numpy as np
    cosine = float((y_oh @ y_d4h) /
                   (np.linalg.norm(y_oh) * np.linalg.norm(y_d4h)))
    assert cosine >= 0.99999, f"d4h(ds=dt=0) ↛ oh: cosine = {cosine:.6f}"
```

#### `test_d4h_dispatcher_block_set_matches_nid8` (Q3-B, tightened)
Structural parity: every block emitted by the dispatcher must match
nid8 by (kind, bra_sym, op_sym, ket_sym, geometry, n_bra, n_ket).
Catches label collisions, missing TRANSI triads, and dimension-formula
errors.

```python
def test_d4h_dispatcher_block_set_matches_nid8():
    from multitorch.angular.rac_generator import generate_ledge_rac
    from multitorch.io.read_rme import read_rme_rac_full

    rac, _ = generate_ledge_rac(
        l_val=2, n_val_gs=8, l_core=1, n_core_gs=6, sym='d4h',
    )
    fixture = read_rme_rac_full(
        'multitorch/data/fixtures/nid8/nid8.rme_rac'
    )

    def block_tuple(b):
        return (b.kind, b.bra_sym, b.op_sym, b.ket_sym,
                b.geometry, b.n_bra, b.n_ket)

    disp = sorted(block_tuple(b) for b in rac.blocks)
    fix = sorted(block_tuple(b) for b in fixture.blocks)
    assert disp == fix, f"Block set diverges:\n  only-disp={set(disp)-set(fix)}\n  only-fix={set(fix)-set(disp)}"

    irrep_set_disp = sorted((i.name, i.kind, i.multiplicity, i.dim)
                            for i in rac.irreps)
    irrep_set_fix = sorted((i.name, i.kind, i.multiplicity, i.dim)
                           for i in fixture.irreps)
    assert irrep_set_disp == irrep_set_fix
```

#### `test_d4h_dispatcher_add_coefficients_match_nid8` (Q6-B)
Per-coefficient magnitude parity for **all four operators**, modulo
sign (eigvec sign convention from LAPACK can flip overall signs).

```python
def test_d4h_dispatcher_add_coefficients_match_nid8():
    from multitorch.angular.rac_generator import generate_ledge_rac
    from multitorch.io.read_rme import read_rme_rac_full

    rac, _ = generate_ledge_rac(
        l_val=2, n_val_gs=8, l_core=1, n_core_gs=6, sym='d4h',
    )
    fixture = read_rme_rac_full(
        'multitorch/data/fixtures/nid8/nid8.rme_rac'
    )

    def block_key(b):
        return (b.kind, b.bra_sym, b.op_sym, b.ket_sym, b.geometry)
    def add_key(a):
        return (a.matrix_idx, a.bra, a.ket, a.nbra, a.nket)

    disp_blocks = {block_key(b): b for b in rac.blocks}
    fix_blocks = {block_key(b): b for b in fixture.blocks}

    # Same set (already asserted by structural test, but defensive)
    assert set(disp_blocks) == set(fix_blocks)

    for key in sorted(disp_blocks):
        d_adds = sorted(disp_blocks[key].add_entries, key=add_key)
        f_adds = sorted(fix_blocks[key].add_entries, key=add_key)
        assert len(d_adds) == len(f_adds), key
        for d, f in zip(d_adds, f_adds):
            assert add_key(d) == add_key(f), (key, d, f)
            assert abs(abs(d.coeff) - abs(f.coeff)) < 1e-6, (
                f"{key} entry {add_key(d)}: |disp|={abs(d.coeff)} "
                f"|fix|={abs(f.coeff)}"
            )
```

#### `test_d4h_ds_perturbation_matches_cached_delta` (Q6-D)
Delta-isolation: compare `Δ(from-scratch, ds=0.1) − Δ(from-scratch, ds=0.0)`
against `Δ(cached, ds=0.1) − Δ(cached, ds=0.0)`. Isolates DS angular
contribution from HFS/F2_pd noise.

```python
def test_d4h_ds_perturbation_matches_cached_delta():
    import numpy as np
    from multitorch.api.calc import (
        calcXAS_from_scratch, calcXAS_cached, preload_fixture,
    )

    cf0 = {"tendq": 1.0, "dt": 0.0, "ds": 0.0}
    cf1 = {"tendq": 1.0, "dt": 0.0, "ds": 0.1}
    _, y_fs0 = calcXAS_from_scratch("Ni", "ii", sym="d4h", cf=cf0)
    _, y_fs1 = calcXAS_from_scratch("Ni", "ii", sym="d4h", cf=cf1)
    cache = preload_fixture("Ni", "ii", "d4h")
    _, y_c0 = calcXAS_cached(cache, cf=cf0)
    _, y_c1 = calcXAS_cached(cache, cf=cf1)

    delta_fs = (y_fs1 - y_fs0).detach().numpy()
    delta_c  = (y_c1 - y_c0).detach().numpy()
    cosine = float((delta_fs @ delta_c) /
                   (np.linalg.norm(delta_fs) * np.linalg.norm(delta_c)))
    assert cosine >= 0.99, (
        f"DS angular structure mismatch between dispatcher and "
        f"cached fixture: cosine(Δ_fs, Δ_cache) = {cosine:.4f}"
    )
```

#### `test_d4h_raises_on_half_integer_J` (Q5a)
Confirms the `NotImplementedError` gate from §4.

```python
def test_d4h_raises_on_half_integer_J():
    from multitorch.angular.rac_generator import generate_ledge_rac
    import pytest
    with pytest.raises(NotImplementedError, match="half-integer"):
        # Fe d5 has half-integer J in both manifolds
        generate_ledge_rac(
            l_val=2, n_val_gs=5, l_core=1, n_core_gs=6, sym='d4h',
        )
```

#### `test_make_d4h_op_adds_DS_eg_block_nonzero` (Q7d)
Helper-level unit test. Independent of full calc roundtrip.

```python
def test_make_d4h_op_adds_DS_eg_block_nonzero():
    from multitorch.angular.rac_generator import (
        _get_terms, _j_sector_sizes, _make_d4h_op_adds,
    )
    from multitorch.angular.symmetry import d4h_basis_layout

    gs_terms = _get_terms(2, 8)
    j_sizes = _j_sector_sizes(gs_terms)
    layout = d4h_basis_layout(j_sizes, parity='g')

    # Provide synthetic idx maps with an entry for every (J, J) pair
    cf_idx_rank2 = {(2.0, 2.0): 1, (2.0, 4.0): 2, (4.0, 2.0): 3,
                    (4.0, 4.0): 4}
    ham_idx = {J: 0 for J in j_sizes}
    cf_idx = {}

    adds = _make_d4h_op_adds(
        'Eg', layout['Eg'], 'DS',
        ham_idx_map=ham_idx, cf_idx_map=cf_idx,
        cf_idx_map_rank2=cf_idx_rank2,
    )
    assert adds, "DS Eg block must have nonzero ADDs (BUG-001 regression)"
    assert all(abs(a.coeff) > 0 for a in adds)
```

#### `test_make_d4h_dipole_adds_perp_para_factor` (Q7d-bis)
Helper-level unit test for the dipole helper. Pins the
`PERP_FACTOR = √(2/3)` and `PARA_FACTOR = √(1/3)` prefactors against
a synthetic input. Faster failure signal than Q6-B for TRANSI bugs.

```python
def test_make_d4h_dipole_adds_perp_para_factor():
    """Pin the dipole PERP/PARA prefactor with a minimal synthetic
    layout. If this fails, the dipole helper's scaling is wrong;
    Q6-B would also fail but with a less specific signal.
    """
    import numpy as np
    from multitorch.angular.rac_generator import (
        _get_terms, _j_sector_sizes, _get_excited_j_sizes,
        _make_d4h_dipole_adds,
    )
    from multitorch.angular.symmetry import d4h_basis_layout

    gs_terms = _get_terms(2, 8)
    gs_layout = d4h_basis_layout(_j_sector_sizes(gs_terms), parity='g')
    ex_layout = d4h_basis_layout(
        _get_excited_j_sizes(2, 8, 1, 6), parity='u',
    )
    # All J pairs map to a synthetic matrix index; entry 1 in COWAN store
    multipole_idx = {(Jb, Jk): 1 for Jb in {0,1,2,3,4} for Jk in {0,1,2,3,4,5}}
    perp_adds = _make_d4h_dipole_adds(
        d4h_gs='A1g', gs_entries=[e for e in gs_layout['A1g'] if e[3] == 0],
        d4h_ex='Eu',  ex_entries=[e for e in ex_layout['Eu']  if e[3] == 0],
        op_d4h_target='Eu', multipole_idx=multipole_idx,
        factor=np.sqrt(2.0/3.0),
    )
    para_adds = _make_d4h_dipole_adds(
        d4h_gs='A1g', gs_entries=[e for e in gs_layout['A1g'] if e[3] == 0],
        d4h_ex='A2u', ex_entries=[e for e in ex_layout['A2u'] if e[3] == 0],
        op_d4h_target='A2u', multipole_idx=multipole_idx,
        factor=np.sqrt(1.0/3.0),
    )
    assert perp_adds, "PERP A1g→Eu must produce nonzero ADDs"
    assert para_adds, "PARA A1g→A2u must produce nonzero ADDs"
    # Magnitudes scale as factor; pin via the ratio
    perp_max = max(abs(a.coeff) for a in perp_adds)
    para_max = max(abs(a.coeff) for a in para_adds)
    ratio = perp_max / para_max
    assert abs(ratio - np.sqrt(2.0)) < 0.05, (
        f"PERP/PARA magnitude ratio = {ratio:.4f}, expected √2 ≈ 1.414"
    )
```

#### `test_d4h_dispatcher_no_cross_copy_hamiltonian_adds` (Q5b)
Pins the `same_entry`-only HAMILTONIAN convention.

```python
def test_d4h_dispatcher_no_cross_copy_hamiltonian_adds():
    """HAMILTONIAN ADDs must be diagonal in (oh, copy, partner)
    within a J-sector. Cross-(oh, copy) couplings are zero by
    partner-basis orthogonality. Pinning this against silent
    'refactors'."""
    from multitorch.angular.rac_generator import generate_ledge_rac
    rac, _ = generate_ledge_rac(
        l_val=2, n_val_gs=8, l_core=1, n_core_gs=6, sym='d4h',
    )
    for b in rac.blocks:
        if b.geometry != 'HAMILTONIAN':
            continue
        # Every ADD entry must reference a single matrix_idx
        # corresponding to one (Jb, Jk=Jb) pair, AND bra==ket
        # (since the HAMILTONIAN block is diagonal in entry index).
        for a in b.add_entries:
            assert a.bra == a.ket, (
                f"HAMILTONIAN cross-entry coupling: {b.bra_sym} "
                f"matrix_idx={a.matrix_idx} bra={a.bra} ket={a.ket}"
            )
```

### 7.2. Existing tests to delete (1)

`test_d4h_dispatcher_emits_per_operator_blocks_appropriately`
(`tests/test_d4h/test_d4h_from_scratch_status.py:458`) — uses
Oh-Butler reasoning ("DS only in E irrep") that's incoherent
post-refactor. Subsumed by the structural test above.

### 7.3. Existing tests that should still pass

| Test (in `tests/test_d4h/test_d4h_from_scratch_status.py`) | Notes |
|---|---|
| `test_generate_ledge_rac_accepts_sym_oh_default` | Oh path untouched |
| `test_generate_ledge_rac_d4h_runs_end_to_end` | unchanged |
| `test_generate_ledge_rac_unknown_sym_raises` | unchanged |
| `test_calcXAS_from_scratch_accepts_sym_d4h` | unchanged |
| `test_d4h_butler_coefficients_match_pyctm` | unchanged |
| `test_d4h_branching_J0_J2_J4` | unchanged |
| `test_d4h_basis_layout_matches_nid8` | layout convention preserved |
| `test_d4h_basis_layout_matches_nid8_excited` | layout convention preserved |
| `test_oh_to_d4h_partners_canonical_signs` | unchanged |
| `test_ds_operator_is_diagonal_in_d4h_basis` | helper-level; unchanged |
| `test_d4h_ni_from_scratch_runs_and_matches_oh_baseline` | should TIGHTEN (cosine bumps from ~0.95 toward ~0.97-0.98) but the test's `≥ 0.95` threshold should still pass |

### 7.4. Existing xfail to flip

`test_d4h_dispatcher_emits_nid8_irrep_set` (line 425) — flips from
xfail to passing in commit 2.

## 8. Commit 6 — PORT_NOTES.md update

Add an entry documenting BUG-001 closure (DS angular gap) and BUG #2
closure (D4h labeling collision). Brief — these are referenced as
the "Threads 2+3 unification" finding from Session 2.

## 9. Definition of Done

- [ ] Commits 1-6 landed on the feature branch.
- [ ] Full suite green: `pytest tests/ -q` shows 477 + 8 new passes,
      0 xfails (the prior xfail flipped), 1 deletion accounted for.
- [ ] All seven new tests in §7.1 pass at their stated tolerances.
- [ ] `test_d4h_dispatcher_emits_nid8_irrep_set` flipped from xfail
      to passing.
- [ ] OLD `if sym == 'd4h':` emission code removed from
      `generate_ledge_rac`.
- [ ] PORT_NOTES.md updated.
- [ ] Branch squashed and merged to main (one commit).
- [ ] User explicit approval to push.

## 10. Risks (mitigated)

- **LAPACK eigvec sign convention:** Q6-B compares ADD coefficients
  modulo sign; Q3-A compares spectra (squared magnitudes); Q3-B
  compares structure only. Sign-flips don't break any test.
- **HFS Slater / F2_pd noise:** Q3-A uses ds=dt=0 limit (Slater is the
  same on both paths in that limit); Q6-D uses delta isolation
  (Slater-induced bias cancels in the difference). Neither test is
  hostage to the orthogonal Slater accuracy issue.
- **Multi-copy J-sectors (Fe d5+, Co d6+, Cu d8+):** the new helper's
  CF-operator branch iterates `entries × entries` and projects each
  `<v_b | O | v_k>` directly, so multi-copy is handled correctly.
  The Q5b test pins this for HAMILTONIAN. Ni d8 happens not to
  exercise the multi-copy paths, so the Q3-A / Q3-B / Q6-B tests
  cover the structure but not the multi-copy numerics — for that,
  the foundation test `test_ds_operator_is_diagonal_in_d4h_basis`
  and the helper-level Q7d test are your guards.
- **Half-integer J:** explicit `NotImplementedError` at the top of
  the d4h branch. Q5a's test pins this.
- **Strength prefactor "double counting":** false alarm in V1 plan.
  The Butler coefficient absorbs the strength factor by
  construction (TENDQ Butler = 6√30/10 = √(54/5) = k=4 strength;
  for DS, Butler = -√70 IS the canonical pyctm coefficient — see
  `D4H_BRANCHES_BY_OPERATOR` docstring). Q6-B (per-coefficient
  parity vs nid8) catches any deviation if this assumption is wrong.

## 11. Glossary anchors

If any term in this plan is ambiguous to you, look it up in
`UBIQUITOUS_LANGUAGE.md`. The load-bearing terms:

- **manifold parity** vs **operator parity** (§ Operators / State manifolds)
- **MULT** vs **partner-summed entry count** (§ Dimensions and counts)
- **D4h-Butler label** vs **Oh-Butler label** (§ Labels and naming conventions)
- **layout entry** structure (§ Code structures)
- **partner** and **multiplicity copy** (§ Symmetry and basis)

If a future author can't follow this plan without re-reading V1 or
the review file, the glossary is missing a term — add it.
