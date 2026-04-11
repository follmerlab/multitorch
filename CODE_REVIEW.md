# multitorch Code Review
**Reviewer:** senior-code-reviewer
**Date:** 2026-04-06
**Status:** 138 / 141 tests passing; 3 HFS-SCF failures (helium, Ni²⁺ 3d, Ni²⁺ 2p)
**Scope:** all of `multitorch/multitorch/{atomic, angular, hamiltonian, spectrum, io, api}/*.py` (≈ 6 300 LoC)

---

## What it is

A PyTorch port of the Cowan/ttmult multiplet X-ray spectroscopy suite (Fortran 77 → torch float64). It is **feature-complete for the high-level pipeline** (read .rme_rcg / .rme_rac / .ban → assemble block Hamiltonian → diagonalize → build transition matrix → broaden) and is ~70 % complete on the from-scratch atomic-parameter pipeline (HFS SCF, Slater integrals, RME builder).

The package is GPU-agnostic (`torch.Tensor` with explicit `dtype=DTYPE=float64`) and the core math primitives (Wigner 3j/6j/9j, CFP, U^(k)) are validated to ≤ 1 e-6 against the Fortran reference outputs in `tests/reference_data/{nid8, nid8ct, als1ni2}/`.

---

## Architecture

| Layer | Module | LoC | Maturity | Validated against |
|---|---|---|---|---|
| Phase 1 — spectrum | `spectrum/{sticks,broaden,rixs,background}` | 590 | shipping | `*.xy`; broaden has documented legacy bug |
| Phase 2 — Hamiltonian | `hamiltonian/{assemble,crystal_field,charge_transfer,diagonalize,transitions}` | 1 040 | shipping | `*.ban_out` eigenvalues + intensities, 13/13 triads |
| Phase 3 — angular | `angular/{wigner,cfp,rme,symmetry}` | 1 770 | RME builder ~90 % | SHELL/SPIN ≤ 3 e-6; MULTIPOLE ≤ 1 e-6 (basis-invariant only) |
| Phase 4 — atomic | `atomic/{hfs,slater,radial_mesh,tables}` | 1 170 | **HFS broken** | Slater Fk/Gk OK; HFS energies wrong by 30–40 Ry |
| I/O | `io/{read_rcf,read_rme,read_oba,read_ban,write_inputs}` | 1 550 | shipping | parses every fixture in `tests/reference_data/` |
| API | `api/{calc,plot}` | 290 | thin wrapper | end-to-end smoke tests |

The dependency graph is acyclic and clean: `spectrum ← hamiltonian ← angular ← atomic`, with `io` and `api` as cross-cutting leaves. No circular imports, no global mutable state outside of `lru_cache`'d Wigner symbols.

### Critical bug fixes already in place
- **`angular/wigner.py:wigner9j`** — added as a 6j-summation following ttrcg.f::S9J. Validated implicitly by the SHELL block matches.
- **`angular/cfp.py:get_cfp_block` / `get_uk_for_shell`** — binary parser for `rcg_cfp72/73`; CFP signs and U^(k) tables match Fortran exactly.
- **`hamiltonian/assemble.py`** — the `IDIM` (irrep-dimension) scaling factor `1/√d` is applied to Hamiltonian blocks but **not** to the EG/EF energy offsets. This subtlety took time to find and reproduces all 13 reference triads.

---

## Current results

| Layer | Tests | Status | Max abs error vs reference |
|---|---|---|---|
| Wigner 3j/6j/9j | 14 | pass | ~1e-12 (analytical sum rules) |
| CFP / U^(k) tables | 12 | pass | < 1 e-4 |
| Single-shell SHELL/SPIN/ORBIT RME | 14 | pass | 3 e-6 / 8 e-7 / structural |
| MULTIPOLE p^6d^n → p^5d^(n+1) | 1 | pass | 1 e-6 (SVD/Frobenius — element-wise blocked on Phase B2) |
| Hamiltonian eigenvalues (3 CT fixtures) | 13 | pass | < 1 e-4 eV |
| Transition intensities | 4 | pass | within ttban tolerance |
| Spectrum broadening (legacy + correct PV) | 6 | pass | matches `*.xy` |
| File parsers (.rcn31_out / .rcn2_out / .rme_rcg / .rme_rac / .ban_out) | 53 | pass | byte-exact field parsing |
| HFS SCF | 3 | **fail** | He 1s = −2.0 vs −0.9 Ry; Ni²⁺ 3d = −43.6 vs −2.8 Ry; Ni²⁺ 2p = −98.0 vs −67.7 Ry |

### Porting trajectory
Phases 1–2 (spectrum + Hamiltonian assembly from existing RME data) shipped quickly because they are pure linear algebra over already-validated inputs. Phase 3 (angular momentum primitives + RME builder) was slow because the Fortran source uses dense common blocks and hand-rolled recoupling chains. Phase 4 (HFS SCF) is the deepest port and the only place where basic numerics still misbehave.

---

## Critical issues (must fix)

### 1. HFS SCF is fundamentally broken (Phase B1)

`atomic/hfs.py` has at least four independent bugs that conspire to produce the observed errors:

**(a) `hfs_scf` line 662 — wrong "total P" reconstruction.**
```python
P_total = density.sqrt() * r.clamp(min=1e-30)   # ← WRONG
RUEE = quad2_coulomb(P_total, r, IDB)
```
`quad2_coulomb` expects a single orbital P (so P² is one orbital's density). The code instead feeds it √(Σ_nl w_nl P_nl²) × r, then `quad2_coulomb` re-squares it and integrates. This is not the Y⁰ of the total density: cross-orbital interference terms are silently dropped and the resulting "potential" does not represent any real charge distribution. **Fix:** loop over orbitals, sum `Σ_nl w_nl × quad2_coulomb(P_nl, r, IDB)` (which is what `rcn31.f` does).

**(b) `hfs_scf` lines 671–675 — units mismatch in the potential build.**
```python
RU_new[i] = -2.0*Z + 2.0*float(RUEE[i]) - float(EXF)*float(RUEXCH[i])
V = RU_new / r.clamp(min=1e-30)
```
Cowan's RU is `V(r) × r` (carries one factor of r). But `quad2_coulomb` returns `XI = I_fwd/r + I_bwd_r`, which has units of potential, **not** potential × r. So `RUEE` is added directly into a quantity that is then divided by r, giving an extra 1/r the Coulomb part should not have. `RUEXCH` (line 246) correctly returns `V_ex × r`, so the exchange term is consistent — the Coulomb term is off by exactly one factor of r. **Fix:** either multiply `RUEE` by `r` before adding, or build `V` directly without going through the `RU = V × r` intermediate.

**(c) Helium = exactly −2.0 Ry is the smoking gun.** −2.0 Ry = −Z²/n² Ry for hydrogenic Z=2 n=1, which is the **bare nuclear** answer with no electron–electron repulsion at all. Combined with (a)+(b), the SCF loop is effectively running with V(r) = −2Z/r and the orbital energy collapses to the hydrogenic eigenvalue. The Ni²⁺ 3d energy −43.6 Ry is similarly suspicious: −Z_eff²/n² for Z_eff ≈ 19.8, n=3 gives −43.6 — i.e. the 3d orbital is seeing the bare Z=28 nucleus only modestly screened by the (broken) Slater initial guess.

**(d) Spin-orbit ζ silently set to zero.** Lines 696–703:
```python
orb.zeta_ry = 0.0
orb.zeta_ev = 0.0
```
despite a docstring claiming "Blume-Watson formula". The spectrum pipeline currently sources ζ from external `.rcn2_out` files, so this has not surfaced as a test failure, but anyone running the from-scratch atomic-parameter pipeline will get zero spin-orbit splitting and an L₂/L₃ ratio of 1:1.

**Risk / blast radius.** Currently isolated to the from-scratch pipeline (`hfs_scf` is not in any user-facing default path; the test fixtures get their atomic parameters from pre-committed `.rcn31_out` / `.rcn2_out` files). But it is a hard blocker for the stated project goal of a Fortran-free package.

### 2. MULTIPOLE two-shell J-basis ordering (Phase B2 — known)

`angular/rme.py:build_two_shell_j_basis` enumerates coupled (S_total, L_total, J) states in a hand-chosen order that **differs from the Fortran convention** used by `ttrcg.f::CALCFC` (lines 6715–7065), which iterates through pre-stored `NTRMK` / `NALSJP` records whose order is set during the JJ-coupling enumeration in `LOOPFC`. The physics is correct — singular values + Frobenius norms of every (J_bra, J_ket) MULTIPOLE block agree to ≤ 1 e-6 — but element-wise comparison fails because columns are permuted and some states pick up sign flips. The reference test uses the basis-invariant SVD route as a workaround.

**Fix sketch.** Trace `ttrcg.f::CALCFC → LOOPFC → upstream NTRMK/NALSJP construction` to extract the canonical basis ordering, mirror it in `build_two_shell_j_basis`, and verify SVD agreement collapses to elementwise. If sign conventions still differ (Condon–Shortley vs Wigner phase choice), apply a per-state phase from the recoupling phase.

**Risk.** Cosmetic for the spectrum pipeline (eigenvalues and intensities are basis-invariant), critical only if anyone wants to compare individual matrix elements with Fortran output. Documented in the test docstring.

### 3. Pseudo-Voigt "legacy" mode preserves a known pyctm bug (Phase B3)

`spectrum/broaden.py::_pv_params` line 118:
```python
n = 1.36603*ratio - 0.47719*ratio + 0.11116*ratio**3   # legacy: missing **2
```
This **intentionally** reproduces a typo in `pyctm/get_spectrum.py` so that broadened spectra match the historical reference `*.xy` files. The `mode='correct'` branch implements Thompson 1987 correctly. Both modes are exercised by the test suite.

**Risk.** None — the bug is loud, documented in three places, and gated behind a string flag. Worth reiterating in the README so that downstream users do not unwittingly inherit it.

---

## Non-critical observations

1. **`atomic/hfs.py` Slater screening (line 632) is wrong for K-shell** — uses 0.35 for same-n electrons and applies a `-1` self-exclusion that misbehaves for K-shell (n=1, occ=2). K shell should use 0.30. Will not matter once bugs (a)+(b) are fixed because the SCF iterations swamp the initial guess.
2. **`assemble.py` line 438** hard-codes the dipole geometry guess from `act_sym`. For non-cubic symmetries beyond D4h this will silently mislabel transitions; add an assertion.
3. **`assemble.py` line 458** does `T_eig = Ug.T @ T_raw @ Uf` — fine for the current real symmetric H, but if anyone ever flips to a complex Hamiltonian (e.g. magnetic field), this should be `Ug.conj().T @ T_raw @ Uf`. Cheap defensive change.
4. **`angular/rme.py` is 876 lines in one file.** Now that single-shell, two-shell, MULTIPOLE, ORBIT, SPIN are all there, splitting into `rme_singleshell.py` / `rme_twoshell.py` / `rme_multipole.py` would help future readers. Not urgent.
5. **No `__all__` anywhere.** Re-exports from `__init__.py` files are sparse; import paths leak implementation structure. Cosmetic.
6. **`tests/reference_data/` is committed binary data.** Regenerating it requires `make all` in `ttmult/src/` plus the `pyttmult` driver. README should call this out.

---

## Immediate priorities (ordered by impact)

1. **Fix HFS SCF (issues 1a–1d).** Without this, the from-scratch atomic-parameter pipeline cannot run. Largest single piece of remaining work.
2. **Implement Phase A3 parity sweep driver** so all eight layers are validated end-to-end against every fixture in one command.
3. **Fix MULTIPOLE basis ordering** (Phase B2). Cosmetic but needed for direct numerical comparison with Fortran output.
4. **Write README.md.** Other groups cannot install or run the package without it.
5. **Implement spin-orbit ζ from Blume-Watson** (issue 1d). Required for the from-scratch pipeline to match the Fortran spectrum.
6. **Split `angular/rme.py`** into per-operator submodules.

---

## Files reviewed in depth
- `multitorch/atomic/hfs.py` (715 LoC)
- `multitorch/spectrum/broaden.py` (201 LoC)
- `multitorch/hamiltonian/assemble.py` (473 LoC)
- `multitorch/angular/wigner.py` (273 LoC)
- `multitorch/angular/rme.py` (876 LoC, prior session)

## Files reviewed at the interface level
All other modules in `multitorch/multitorch/` (read class signatures, public functions, and any docstring claims about Fortran source-line correspondence). No code changes were made during this review.
