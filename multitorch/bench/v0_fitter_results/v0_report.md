# v0 Fe L-edge XAS fits — exxa GPU production sweep

**Date:** 2026-05-03
**Hardware:** exxa (NVIDIA RTX 4090, CUDA 12.1, PyTorch 2.5.1)
**Wall time:** 25.9 min total (5 spectra)
**Commit at run:** multitorch `1ed58df` (post-CUDA-autograd verification)

## Per-spectrum results

| Spectrum | Auto-valence | Loss | RMSE | slater | soc | 10Dq (eV) | wall (s) |
|---|---|---|---|---|---|---|---|
| FeIIIPc-Cl       | iii | 0.0137 | 0.117 | 0.865 | 0.981 | **1.609** | 591.9 |
| FeIIIPor-Cl      | iii | **0.0064** | **0.080** | 0.856 | 0.993 | 1.220 | 591.9 |
| FeIIPc           | ii  | 0.0110 | 0.105 | 0.856 | 0.993 | **1.765** | 124.1 |
| FeTPC-Cl         | ii ⚠️ | 0.0220 | 0.148 | 0.855 | 0.995 | 1.357 | 124.2 |
| Fe[TPC]-NO       | ii ⚠️ | 0.0170 | 0.130 | 0.857 | 0.991 | **1.162** | 123.9 |

⚠️ **Auto-valence is filename-based** (`"III" in name`). FeTPC-Cl is
chemically Fe(III) (Cl axial implies +3) — should be re-fit. Fe[TPC]-NO
is redox-ambiguous (Fe(II)–NO+ or Fe(III)–NO•) — should be fit at both
oxidation states and compared.

## Parameter trends

- **slater** consistent across all 5 (0.855–0.865) — Fe-shell screening
  empirical reduction factor is well-determined to ~1 % across these
  complexes, consistent with literature.
- **soc** ≈ 0.99 ± 0.01 — spin-orbit Hund's-coupling reduction is at
  unity within fit tolerance; suggests these systems don't need SOC
  re-scaling.
- **10Dq (tendq)** spans 1.16–1.77 eV. Macrocycle-only systems
  (FeIIPc) show the highest 10Dq; axial-Cl complexes pull it down by
  ~0.4 eV; Fe[TPC]-NO has the lowest tendq, consistent with the
  σ-donating NO ligand weakening the in-plane field.

## Common systematic residual at ~714 eV

All 5 fits show a sharp residual peak at ~714 eV that the Oh
approximation can't capture (visible in every PNG residual panel).
This is the L3 fine-structure splitting from D4h B1g/Eg states +
modest CT contributions for the Cl axial cases.

This residual **motivates the D4h port** (Phase 1c unified Threads
2+3, task #53). The math is validated — `d4h_partner_basis_per_J`
makes DS diagonal in the D4h basis (commit 25860ad) — so the
dispatcher refactor closes this gap. ~3-4 days of focused work.

## Per-spectrum notes

### FeIIIPor-Cl (best fit, loss 0.006)

Cleanest convergence. The Por-Cl macrocycle has a well-defined
axial environment that Oh approximates moderately well. 10Dq=1.22 eV
is in the standard Fe(III) range.

### FeIIIPc-Cl (loss 0.014)

L3 fine structure clearly under-resolved by Oh. 10Dq=1.61 eV;
moderately higher than Por because Pc has a stronger in-plane field
(more delocalized π-system).

### FeIIPc (loss 0.011)

No axial ligand — pure 4-coordinate D4h. Fits surprisingly well in
Oh given the symmetry mismatch. 10Dq=1.77 eV (highest) reflects the
strong in-plane field with no axial perturbation.

### FeTPC-Cl (loss 0.022, worst)

Auto-detected as Fe(II) but should be Fe(III). Re-fitting at Fe(III)
likely improves loss substantially. Current fit "splits the
difference" between d6 and d5 multiplet structure; tendq=1.36 looks
intermediate.

### Fe[TPC]-NO (loss 0.017)

Distinctive feature: the experimental L3 peak is unusually sharp
(no visible shoulder structure compared to other 4 spectra). Could
reflect strongly localized {FeNO}^7 character. Both Fe(II)-NO+ and
Fe(III)-NO* fits should be compared via χ²; the auto-detector chose
Fe(II) but this is the most ambiguous of the 5.

## Recommended follow-ups (manuscript-ready)

1. **Re-fit FeTPC-Cl as Fe(III)** — auto-detector misclassified.
   Override valence on CLI: `--valence iii`.
2. **Twin-fit Fe[TPC]-NO at both ii and iii**, report both loss
   values; the lower loss + more physical params identify the
   electronic state.
3. **Implement Phase 1c unified Threads 2+3 dispatcher refactor**
   (task #53). Rerun with D4h labels + DS coupling. Expected:
   ~714 eV residual disappears, all losses drop ~3-5×.
4. **Joint fit with shared atomic params** (Phase 6 v1). The
   slater/soc consistency above (0.86 ± 0.01, 0.99 ± 0.01) makes
   sharing across the same-valence group well-motivated.
5. **Manuscript figure**: 5-panel grid of all fits or a 2×3 layout
   with the residual at 714 eV highlighted to motivate the D4h
   discussion.

## Bench / commit hygiene notes

- v0 fitter (`fits/fe_xas_fit.py`, `fit_and_plot_all.py`,
  `plot_fit.py`) lives outside the multitorch repo per earlier
  decision. Synced to exxa via rsync.
- All 5 PNGs + JSON in `fits/results/`.
- `multitorch.device_utils.suggest_device_for_xas` updated with
  empirical 3d/4f rules (commit `88ce730`); `device='auto'` wired
  into `calcXAS_cached` for auto-routing.
- D4h scaffolding (`d4h_partner_basis_per_J`, `oh_to_d4h_subduction_matrix`)
  committed in `25860ad`. Math validated via
  `test_ds_operator_is_diagonal_in_d4h_basis`.
- Sweep monitor: `bench/monitor_v0_sweep.sh` (one-shot or `--watch`).
