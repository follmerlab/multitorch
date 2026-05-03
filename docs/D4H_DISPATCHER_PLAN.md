# D4h dispatcher refactor — implementation plan

**Status as of:** 2026-05-03 (commit `9f45702`)
**Next session pickup:** read this file + `CODE_REVIEW.md` Addendum 3 first.
**Context bridge:** `~/.claude/projects/.../memory/project_d4h_dispatcher.md`

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

The TRANSI/MULTIPOLE blocks (lines 791–917) are already partially
D4h-aware (PERP/PARA splitting). Verify no regression but probably no
change needed — the PERP/PARA labels are valid D4h Butler labels.

### Step 4 — Update `_build_ban_from_rac` (no change expected)

The 4-operator XHAM `[1.0, tendq, dt, ds]` is already correct (commit
`e3c39c4`). No change.

### Step 5 — Gate test (~50 LOC)

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

### Step 6 — DS perturbation test (~30 LOC)

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

### Step 7 — Drop the deprecated v1 path

Once gate test passes, delete the OLD per-Oh-irrep `if sym == 'd4h':`
emission code (the broken DT/DS blocks, the `dt_a1_scale` /
`ds_e_scale` logic). Keep the helpers (`_irrep_vector`,
`oh_coupling_coefficients_for_op` skeleton) — they're foundation
pieces useful for future work even if not the primary path.

---

## Implementation order (in a fresh session)

1. Read this file + `CODE_REVIEW.md` Addendum 3
2. Run `pytest tests/test_d4h tests/test_angular -q` — confirm
   116 pass + 1 xfail baseline
3. Implement Step 1: `_make_d4h_op_adds` for HAMILTONIAN only
4. Test on Ni d8: emit one block, manually verify ADDs against nid8
5. Implement TENDQ — same approach
6. Implement DS (the previously-broken case) — verify diagonalization
   matches the existing `test_ds_operator_is_diagonal_in_d4h_basis`
7. Implement DT (most complex — multi-path operator vector)
8. Wire everything into the `if sym == 'd4h':` branch
9. Run gate test (Step 5). Fix bugs. Iterate.
10. Run DS perturbation test (Step 6).
11. Run autograd parity test (already exists for slater/soc; extend
    to dt/ds/tendq).
12. Verify the bench harness's parity for nid8/Fe d4h (if fixture
    bootstrap is unblocked) before merging.

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
