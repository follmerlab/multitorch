# D4h dispatcher refactor — implementation plan

**Status as of:** 2026-05-03 — Helpers landed (uncommitted), TRANSI scope expanded
**Next session pickup:** read this file's "Session 2 progress" first, then jump to "Step 4 — TRANSI per-D4h-irrep emission" below.
**Context bridge:** `~/.claude/projects/.../memory/project_d4h_dispatcher.md`

---

## Session 2 progress (2026-05-03, uncommitted)

The four-operator dispatcher helpers were implemented in
`multitorch/angular/rac_generator.py` (changes uncommitted; baseline
`pytest tests/test_d4h tests/test_angular -q` still passes 116 + 1 xfail
because the helpers aren't wired into `generate_ledge_rac` yet).
Specifically added (between `generate_ground_state_rac` and
`generate_ledge_rac`):

- `_d4h_partner_vector(J, oh_irrep_label, copy_idx, d4h_irrep, partner_idx)`
  composes `_real_subduction_matrix` with `oh_to_d4h_subduction_matrix`.
- `_d4h_op_routes` and `_d4h_operator_vector_complex` decode
  `D4H_BRANCHES_BY_OPERATOR` and build the rank-K m-basis vector
  with Butler coefficients folded in.
- `_operator_real_matrix(J_b, J_k, k, op_vec_complex)` handles the
  parity-rule real/imag extraction.
- `_make_d4h_op_adds(d4h_irrep, entries, operator, ham_idx_map,
  cf_idx_map, cf_idx_map_rank2, parity)` emits ADD entries with
  `coeff = sqrt(dim_d4h / (2J_b+1)) × <v_b | O_real | v_k>`.

**Hand-verified numbers:** TENDQ at (D4h-A1g, J_b=0, J_k=4)
reproduces the existing Oh-A1 path's `1.095445`. HAMILTONIAN at
(D4h-A1g, J=4, Oh-A1 route) gives `0.333` matching `sqrt(1/9)`.
Operator-vector L2 norms equal the Butler coefficients
(TENDQ → `sqrt(54/5) ≈ 3.286`, DS → `sqrt(70) ≈ 8.367`,
DT → `sqrt(630) ≈ 25.10`).

**SCOPE EXPANSION discovered in Session 2:** The plan's claim that
"TRANSI/MULTIPOLE blocks are already partially D4h-aware (PERP/PARA
splitting). Verify no regression but probably no change needed" is
**wrong**. Inspection of the current `sym='d4h'` output and the bundled
`nid8` fixture shows:

- The existing TRANSI emission uses **Oh-Butler labels** for `bra_sym`
  and `ket_sym` (e.g., `'2+'` = Oh-Eg dim 2 mult 5,
  `'1+'` = Oh-T1g dim 3 mult 4).
- The `nid8.rme_rac` fixture has **D4h-Butler labels** with the SAME
  string (`'2+'` = D4h-B1g dim 1 mult 6, `'1+'` = D4h-Eg dim 2 mult 10).
- This collision is why the existing `sym='d4h'` from-scratch path only
  achieves cosine ≈ 0.95 against `nid8` — the labels are wrong AND the
  matrix elements aren't projected onto D4h-partner basis on either
  bra (GS) or ket (EX) side.
- The assembler matches triads `(gs_sym, act_sym, fs_sym)` by symbol;
  for the dispatcher refactor to be coherent, GROUND, EXCITE, AND
  TRANSI must all use D4h-Butler labels with consistent matrix elements.

So the gate test `test_calcXAS_from_scratch_d4h_ni_matches_nid8_strict`
(cosine ≥ 0.9999) requires both pieces:
1. Per-D4h-irrep GROUND/EXCITE blocks (the helpers are ready, see
   "Wiring Step 3" below).
2. Per-D4h-irrep TRANSI blocks (NOT yet implemented; see Step 4).

This is the unified Threads 2+3 work — it closes BOTH the DS coupling
gap (BUG-001 follow-up, "DS gap deeper than expected") AND the D4h
labeling gap (BUG #2). They turned out to be the same problem; a single
basis rotation closes both.

---

## What's already landed (verified, foundation)

All on `origin/main`, validated:

| Helper | File:line | Validation |
|---|---|---|
| `D4H_BRANCHES_BY_OPERATOR` (TENDQ/DT/DS) | `multitorch/angular/symmetry.py:97` | `test_d4h_butler_coefficients_match_pyctm` passes verbatim against `pyctm/write_RAC.py:gen_X400_X420_X220` |
| `d4h_cf_operator_recipe(operator)` | `multitorch/angular/symmetry.py:201` | `test_d4h_cf_operator_recipe_{tendq,dt,ds,unknown_raises}` |
| `_irrep_vector(k, op_irrep)` | `multitorch/angular/point_group.py:949` | A1-reduction matches `_a1_vector` for k=4 |
| `oh_coupling_coefficients_for_op` (broken trace approach) | `multitorch/angular/point_group.py:1513` | Returns 0 for non-scalar ops — superseded by basis rotation |
| `oh_to_d4h_subduction_matrix(oh_irrep)` | `multitorch/angular/symmetry.py:227` | All 5 g-irreps tested; orthonormality + dimensions + determinism |
| `d4h_partner_basis_per_J(J, d4h_irrep)` | `multitorch/angular/symmetry.py:329` | Completeness for J=0..4 (sum = 2J+1) |
| `d4h_basis_layout(j_sizes, parity)` | `multitorch/angular/symmetry.py:419` | **Matches nid8 fixture exactly** (g + u sides) |
| `D4H_TO_BUTLER`, `D4H_IRREP_DIM`, `d4h_butler_label` | `multitorch/angular/symmetry.py:141`–170 | Round-trip test |

Selection-rule diagonalization proof (DS → diagonal in D4h basis):
`tests/test_d4h/test_d4h_from_scratch_status.py:test_ds_operator_is_diagonal_in_d4h_basis`.
Got `<A1g|DS|A1g> = -0.535`, `<B1g|DS|B1g> = +0.535`,
`<Eg|DS|Eg> = -0.267 × I`, all cross-irrep off-diags = 0 within 1e-9.

Audit: `CODE_REVIEW.md` Addendum 3 (gitignored local artifact).
Trust HIGH on all foundations; magnitude difference vs pyctm is a
documented convention difference, not an error.

## What's left: the dispatcher itself

The `if sym == 'd4h':` branch in
`multitorch/angular/rac_generator.py:generate_ledge_rac` (the function
starting at line 619) currently emits **Oh-labeled blocks with broken
DS** (DS blocks have empty ADD entries because the trace-only approach
fails for non-scalar ops — see PORT_NOTES.md "DS gap deeper than
expected" 2026-05-03 entry).

The refactor replaces that branch with a **per-D4h-irrep** dispatcher
that emits blocks labeled per `D4H_TO_BUTLER` and computes ADD
coefficients via the rotated basis.

---

## Step-by-step recipe

### Step 1 — `_make_d4h_op_adds` helper (~80 LOC)

Add to `multitorch/angular/rac_generator.py` (next to `_make_cf_adds`):

```python
def _make_d4h_op_adds(
    d4h_irrep: str,
    entries: List[Tuple[float, str, int, int, int]],   # from d4h_basis_layout
    operator: str,                                       # 'HAMILTONIAN'/'10DQ'/'DT'/'DS'
    cf_idx_map: Dict[Tuple[float, float], int],         # rank-4 COWAN matrix indices
    cf_idx_map_rank2: Dict[Tuple[float, float], int],   # rank-2 COWAN (for DS)
    ham_idx_map: Dict[float, int],                      # k=0 HAMILTONIAN matrix indices
    l_val: int,
) -> List[ADDEntry]:
    """Build ADD entries for one (D4h irrep, operator) pair.

    Iterates entries × entries pairs; for each (entry_b, entry_k):
      - Builds the D4h-partner basis vectors v_b, v_k via composition
        of _real_subduction_matrix + oh_to_d4h_subduction_matrix
      - Computes the operator's matrix at (J_b, J_k) in the m-basis
      - Projects: coeff = <v_b | O | v_k>
      - Emits an ADDEntry pointing at the COWAN matrix index for
        (rank, J_b, J_k)
    """
```

Implementation notes:

- For **HAMILTONIAN** (k=0, A1g): the matrix is identity per J;
  matrix element is `<v_b | I_per_J | v_k> = δ(J_b, J_k) × v_b · v_k`.
  Within a D4h-irrep block, partners across different (oh, copy) pairs
  are orthogonal (they're columns of an orthonormal basis); so this
  reduces to δ(entry_b == entry_k for matched J).
  
  Strictly: emit one ADD per entry (diagonal only) referencing
  `ham_idx_map[J]` with coeff = 1.0.

- For **TENDQ** (k=4, A1g of D4h, comes from Oh-A1 only):
  - op_vec_complex = `_a1_vector(4)` (existing rank-4 A1 vector)
  - For each (J_b, J_k) with selection rule: build `O_real = U_J_b
    @ _build_coupling_operator(J_b, J_k, 4, op_vec_complex) @ U_J_k.conj().T`
    in real basis. (`U_J = _c2r_unitary(J)`).
  - For each entry pair: `coeff = v_b @ O_real @ v_k`.
  - Skip if |coeff| < 1e-15.
  - matrix_idx = `cf_idx_map[(J_b, J_k)]`.

- For **DT** (k=4, D4h-A1g, comes from Oh-A1 + Oh-E paths):
  Both contribute — the operator vector for DT is a SUM of:
  - Oh-A1 path: `D4H_BRANCHES_BY_OPERATOR['DT'][('4+','0+','0+')] /
    D4H_BRANCHES_BY_OPERATOR['TENDQ'][('4+','0+','0+')] × _a1_vector(4)`
    (rank-4 A1 with DT/TENDQ Butler ratio)
  - Oh-E path: `D4H_BRANCHES_BY_OPERATOR['DT'][('4+','2+','0+')]`
    × `_real_subduction_matrix(4, 'E')` projected to D4h-A1g

  Concrete: `op_vec_dt_real = (B_oh_a1 @ sub_a1g_to_a1g) × ratio_A1
                          + (B_oh_e  @ sub_eg_to_a1g) × ratio_E`
  where the "ratio" terms are the Butler coefficients normalized
  against the per-Oh-irrep strength prefactor that
  `_build_coupling_operator` will apply.

  TODO during implementation: verify the relative scaling between
  these two paths empirically by comparing to the ds=0 case (DT alone
  should change the spectrum predictably as dt grows).

- For **DS** (k=2, D4h-A1g, comes from Oh-E only):
  - op_vec_real = `B_oh_eg @ sub_eg_to_a1g` (5-dim)
  - op_vec_complex = `U_2.conj().T @ op_vec_real`
  - matrix_idx = `cf_idx_map_rank2[(J_b, J_k)]`

  This is the case validated by
  `test_ds_operator_is_diagonal_in_d4h_basis`.

- The Butler-factor scaling (the absolute value the user supplies for
  `dt`, `ds`, `tendq`) is applied via the BAN's XHAM at assemble time,
  NOT here. The ADD coefficients we emit are unit-parameter values
  (= the Butler coefficient × angular projection).

### Step 2 — `_d4h_partner_vector` helper (~30 LOC)

Convenience helper to build the D4h-partner basis vector for a single
layout entry:

```python
def _d4h_partner_vector(
    J: float, oh_irrep_label: str, copy_idx: int,
    d4h_irrep: str, partner_idx: int,
) -> np.ndarray:
    """Return one D4h-partner basis vector (real-SH, dim 2J+1).

    Composes _real_subduction_matrix and oh_to_d4h_subduction_matrix
    to find the specific partner basis vector keyed by
    (J, oh_irrep, copy_idx, d4h_irrep, partner_idx).
    """
```

This is called inside `_make_d4h_op_adds` for each entry's bra/ket
position. Cache the per-(J, d4h_irrep) results to avoid recomputation.

### Step 3 — Replace the `if sym == 'd4h':` branch (~120 LOC)

Find this block in `generate_ledge_rac` (around line 919–1062):

```python
# --- GROUND blocks (gerade, d^n) ---
for irrep in oh_irreps_gs:    # ← existing per-Oh-irrep loop
    ...
```

Replace with per-D4h-irrep emission:

```python
if sym == 'd4h':
    layout_gs = d4h_basis_layout(gs_j_sizes, parity='g')
    for d4h_irrep, entries in sorted(layout_gs.items()):
        if not entries:
            continue
        block_dim = sum(e[4] for e in entries)
        butler = D4H_TO_BUTLER[d4h_irrep]
        irrep_infos.append(IrrepInfo(
            name=butler, kind='GROUND',
            multiplicity=block_dim,
            dim=D4H_IRREP_DIM[d4h_irrep],
        ))

        # 4 operator blocks per D4h irrep
        for op in ('HAMILTONIAN', '10DQ', 'DT', 'DS'):
            adds = _make_d4h_op_adds(
                d4h_irrep, entries, op,
                cf_idx_map=gs_cf_idx,
                cf_idx_map_rank2=gs_cf_idx_rank2,
                ham_idx_map=gs_ham_idx,
                l_val=l_val,
            )
            if adds:
                blocks.append(RACBlockFull(
                    kind='GROUND',
                    bra_sym=butler, op_sym='0+',  # A1g operator
                    ket_sym=butler, geometry=op,
                    n_bra=block_dim, n_ket=block_dim,
                    add_entries=adds,
                ))
else:
    # Original Oh-only path stays (used for sym='oh')
    ...
```

Same shape for EXCITE (line ~999), with parity='u' and `_ex_*` indices.

**Caveat (Session 2 finding):** the helper signature actually
implemented in `rac_generator.py` is

```python
_make_d4h_op_adds(
    d4h_irrep, entries, operator,
    ham_idx_map, cf_idx_map, cf_idx_map_rank2,
    parity,                      # 'g' for GROUND, 'u' for EXCITE
)
```

i.e., positional order differs slightly from the plan's draft, and the
`l_val` argument was removed (`_d4h_operator_vector_complex` builds the
operator vector directly from `D4H_BRANCHES_BY_OPERATOR` so does not
need `l`). Use the actual signature when wiring.

### Step 4 — TRANSI per-D4h-irrep emission (~150 LOC, NEW in Session 2)

The existing `if sym == 'd4h':` TRANSI loop (lines ~855–917) does
PERP/PARA splitting BUT keeps Oh-Butler labels for `bra_sym`/`ket_sym`,
which collide with D4h-Butler labels in the `nid8` fixture. Replace
with per-D4h-irrep TRANSI emission so the assembler matches triads
correctly.

Recipe:

```python
if sym == 'd4h':
    layout_gs = d4h_basis_layout(gs_j_sizes, parity='g')
    layout_ex = d4h_basis_layout(ex_j_sizes, parity='u')
    for d4h_gs, gs_entries in sorted(layout_gs.items()):
        if not gs_entries:
            continue
        n_bra = sum(e[4] for e in gs_entries)
        for d4h_ex, ex_entries in sorted(layout_ex.items()):
            if not ex_entries:
                continue
            n_ket = sum(e[4] for e in ex_entries)
            for op_name, op_d4h_irrep, op_butler, geom in [
                ('PERP', 'Eu', '1-', 'PERP'),
                ('PARA', 'A2u', '^0-', 'PARA'),
            ]:
                # Compute per (gs_entry, ex_entry) the dipole matrix
                # element <v_b_d4h | T1u (D4h-component) | v_k_d4h>.
                # Use multipole_blocks (rank-1 transition coupling) as
                # the SHELL matrix; project both sides onto D4h-partner
                # basis. Skip if all couplings are below 1e-15.
                ...
```

Implementation notes for Step 4:

- The dipole `T1u` projects onto `Eu` (PERP, dim=2) and `A2u` (PARA,
  dim=1) of D4h. So the "operator vector" for PERP is the
  `oh_to_d4h_subduction_matrix('T1u')['Eu']` and for PARA is the
  `oh_to_d4h_subduction_matrix('T1u')['A2u']`. Pick one partner
  (the dipole magnitude is partner-symmetric within each D4h sub-irrep
  due to the BAN's standard scaling).
- The SHELL block is `multipole_idx[(J_gs, J_ex)]` (rank-1 transition
  matrix). Same store as the existing path.
- For each `(gs_entry, ex_entry)` pair, compute
  `<v_b_d4h_gs | O_dipole | v_k_d4h_ex>` and emit an ADD entry with
  `coeff = sqrt(dim_d4h_op / sqrt((2J_gs+1)(2J_ex+1))) × <…>` (verify
  the prefactor empirically — the existing PERP/PARA `sqrt(2/3)` and
  `sqrt(1/3)` factors map to `dim_Eu/dim_T1u` and `dim_A2u/dim_T1u`).

Validate Step 4 against the bundled `nid8.rme_rac` fixture: the
TRANSI block headers must match (`bra_sym`, `op_sym`, `ket_sym`,
`PERP/PARA`, `n_bra`, `n_ket`).

### Step 5 — Update `_build_ban_from_rac` (no change expected)

The 4-operator XHAM `[1.0, tendq, dt, ds]` is already correct (commit
`e3c39c4`). The triads are derived from TRANSI bra/op/ket strings, so
once Step 4 emits D4h-Butler labels, the BAN will naturally have D4h
triads. No code change in `_build_ban_from_rac` itself.

### Step 6 — Gate test (~50 LOC)

Add to `tests/test_d4h/test_d4h_from_scratch_status.py`:

```python
def test_calcXAS_from_scratch_d4h_ni_matches_nid8_strict():
    """Phase 1c gate test: D4h Ni from-scratch ≈ nid8 cached path.
    
    Must pass at strict tolerances before the dispatcher refactor merges:
      - cosine ≥ 0.9999
      - peak shift ≤ 0.05 eV
      - L3/L2 ratio within 1%
    """
    import sys
    sys.path.insert(0, '<workspace>/multitorch/bench')
    from bench.parity import compare, INTRA_COSINE_TOLERANCE, ...
    from multitorch.api.calc import calcXAS_cached, calcXAS_from_scratch, preload_fixture

    cf = {"tendq": 1.0, "ds": 0.0, "dt": 0.0}
    cache = preload_fixture("Ni", "ii", "d4h")
    x_ref, y_ref = calcXAS_cached(cache, cf=cf)
    x_new, y_new = calcXAS_from_scratch("Ni", "ii", cf=cf, sym="d4h")
    
    result = compare(
        x_new.detach().numpy(), y_new.detach().numpy(),
        x_ref.detach().numpy(), y_ref.detach().numpy(),
        calctype='xas',
    )
    ok, failures = result.passes(
        cosine_min=INTRA_COSINE_TOLERANCE,
        max_abs_diff_max=INTRA_MAX_ABS_DIFF_TOLERANCE,
        peak_pos_max_ev=INTRA_PEAK_POS_TOLERANCE_EV,
        l3l2_ratio_tol=INTRA_L3L2_RATIO_TOLERANCE,
    )
    assert ok, f"D4h dispatcher parity failures: {failures}"
```

Do NOT skip this test. If it fails, debug — don't loosen the tolerance.

### Step 7 — DS perturbation test (~30 LOC)

Add a regression test that ds≠0 changes the spectrum:

```python
def test_d4h_ds_perturbation_changes_spectrum():
    """Once the dispatcher emits non-empty DS ADD entries, ds=0.1 must
    produce a measurably different spectrum from ds=0.
    """
    from multitorch.api.calc import calcXAS_from_scratch
    
    x0, y0 = calcXAS_from_scratch("Ni", "ii", sym="d4h",
                                    cf={"tendq": 1.0, "dt": 0.0, "ds": 0.0})
    x_ds, y_ds = calcXAS_from_scratch("Ni", "ii", sym="d4h",
                                        cf={"tendq": 1.0, "dt": 0.0, "ds": 0.1})
    
    diff = float((y_ds - y0).abs().max())
    assert diff > 0.01, (
        f"ds=0.1 should produce a measurably different spectrum, "
        f"got max diff = {diff:.4e}"
    )
```

### Step 8 — Drop the deprecated v1 path

Once gate test passes, delete the OLD per-Oh-irrep `if sym == 'd4h':`
emission code (the broken DT/DS blocks, the `dt_a1_scale` /
`ds_e_scale` logic). Keep the helpers (`_irrep_vector`,
`oh_coupling_coefficients_for_op` skeleton) — they're foundation
pieces useful for future work even if not the primary path.

---

## Implementation order (Session 3+ pickup)

Steps 1–2 (helpers) are DONE in `multitorch/angular/rac_generator.py`
between `generate_ground_state_rac` and `generate_ledge_rac`
(Session 2, uncommitted). Verify with
`pytest tests/test_d4h tests/test_angular -q` → 116 pass + 1 xfail
baseline still holds; nothing is wired yet.

Remaining work, in order:

1. Read this file (especially "Session 2 progress") + the helpers in
   `rac_generator.py` (`_d4h_partner_vector`, `_d4h_op_routes`,
   `_d4h_operator_vector_complex`, `_operator_real_matrix`,
   `_make_d4h_op_adds`).
2. Wire Step 3 (GROUND/EXCITE per-D4h-irrep) into `generate_ledge_rac`
   under `if sym == 'd4h':`. Use the actual signature with the
   `parity` kwarg ('g' for GROUND, 'u' for EXCITE).
3. Wire Step 4 (TRANSI per-D4h-irrep) — this is the new chunk added in
   Session 2; expect ~150 LOC. Validate `bra_sym/op_sym/ket_sym/n_bra/
   n_ket` against `nid8.rme_rac` headers as you go.
4. Add tests from Steps 6 and 7 (gate + DS perturbation).
5. Flip the existing xfail `test_d4h_dispatcher_emits_nid8_irrep_set`
   to passing once the irrep set matches.
6. Drop the deprecated v1 D4h emission code (Step 8).
7. Run `pytest tests/ -q` — full 477-test suite must pass.
8. Update PORT_NOTES.md: BUG-001/-002 closed, Phase 1c gap closed.
9. Surface result to user; do not commit without explicit request.

## Risks and known gotchas

- **Eigvec sign convention**: `_oh_irrep_matrices_real_std` uses LAPACK
  default. Currently stable but if numpy/lapack updates flip a sign, the
  partner basis rotates. Pinned via `test_oh_to_d4h_partners_canonical_signs`.

- **Strength prefactor double-counting**: `_build_coupling_operator`
  doesn't include `strength = sqrt((2k+1)(2l+2)/(2l+1))`. The
  multitorch convention is that the strength factor is applied at
  ADD-emission time via `_make_cf_adds`'s `prefactor`. The new
  `_make_d4h_op_adds` must apply the same prefactor.

- **Multiplicity copies**: when an Oh irrep has mult > 1 in D^J (e.g.
  Eg appears once in J=2 but multiple times in higher J's), each copy
  contributes its own (oh, copy) entry to the layout. The operator
  matrix coupling between different copies of the same Oh irrep is
  generally NON-ZERO and must be computed explicitly.

- **Cross-J entries within a D4h-irrep block**: an A1g D4h-irrep in d8
  has contributions from J=0, J=2, J=4. The Hamiltonian (k=0)
  matrix elements ARE diagonal in J (selection rule), so the J=0 and
  J=2 entries don't couple. CF operators (k=4 or k=2) DO couple
  different J's where triangle inequality permits.

- **D4h labels vs Butler labels in `op_sym`**: the operator label for
  HAMILTONIAN/CF blocks is `'0+'` (D4h A1g, since these are scalar in
  D4h). The TRANSI block's `op_sym` for the dipole stays as `'1-'`
  (PERP) and `'^0-'` (PARA).

## Cross-reference / context

- `docs/PORT_NOTES.md` — BUG-001/002 + Threads 2/3 unification finding
- `CODE_REVIEW.md` — Addendum 3 (foundation audit)
- `bench/v0_fitter_results/v0_report.md` — manuscript-intake narrative;
  the 714 eV residual visible in all 5 fits IS the gap this dispatcher
  closes
- `bench/AGENT_HANDOFF_GPU_EIGH.md` — perf context; GPU is auto-routed
  for Fe d5 etc. via `device_utils.suggest_device_for_xas`
- `~/.claude/plans/lexical-toasting-kahan.md` — overall standalone-port
  plan (Phase 1 = D4h, this dispatcher closes Phase 1c)

## Definition of done

- [ ] Step 1–7 above implemented
- [ ] `test_calcXAS_from_scratch_d4h_ni_matches_nid8_strict` passes at
      cosine ≥ 0.9999, peak shift ≤ 0.05 eV, L3/L2 within 1%
- [ ] `test_d4h_ds_perturbation_changes_spectrum` passes
- [ ] `test_d4h_dispatcher_emits_nid8_irrep_set` (currently xfail) flips
      to passing — block irreps now match nid8 exactly
- [ ] Existing 116 angular + d4h tests still pass
- [ ] `pytest tests/test_integration` still passes (Oh from-scratch path
      regression check)
- [ ] PORT_NOTES.md updated: BUG-001 follow-up + Phase 1c gap #2 closed
- [ ] User reviews + approves the merge
