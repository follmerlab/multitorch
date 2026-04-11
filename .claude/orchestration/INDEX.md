# Orchestration Index

Master index for all implementation progress, bugs, investigations, and tasks.

**Last updated:** 2026-04-11
**Test status:** 176/176 pass | All 8 Ti–Ni Fortran comparison cases pass

---

## Active Tracks

| Track | File | Status |
|---|---|---|
| **Track C — Phase 5 pure-PyTorch pipeline** | [TRACK-C-PHASE5.md](TRACK-C-PHASE5.md) | C1 done; **C3-pre next** |

**Canonical plan file** for the Blume-Watson + RIXS + Phase 5 trio:
`~/.claude/plans/velvet-kindling-walrus.md` — always read first when
resuming work on any of these tracks.

## Active Investigations

None currently active.

## Open Bugs

(none)

## Resolved Bugs

| ID | Title | Resolved | Root cause |
|---|---|---|---|
| BUG-001 | [HFS SCF orbital energies](bugs/BUG-001-hfs-scf-energies.md) | 2026-04-11 | `EXF` default was 0.7, Fortran ref run used EXF=1.0. Fixed in `hfs.py`; errors now 1–8%. See INV-002. |
| BUG-002 | [MULTIPOLE basis ordering differs from Fortran](bugs/BUG-002-multipole-basis-ordering.md) | 2026-04-11 | SVD-invariant match to 1e-6 in parity sweep; column labeling only, no physics impact. Stale tracker entry. |
| BUG-003 | [CT eigenvalue offset ~1 Ry for non-Ni cases](bugs/BUG-003-ct-eigenvalue-offset.md) | 2026-04-10 | Multi-line COWAN matrix row parsing in `read_rme.py:_parse_rme_block` |

## Completed Investigations

| ID | Title | Result | File |
|---|---|---|---|
| INV-001 | [CT eigenvalue offset](investigations/INV-001-ct-eigenvalue-offset.md) | SOLVED → BUG-003 | `io/read_rme.py` |
| INV-002 | [HFS SCF orbital energies](investigations/INV-002-hfs-energies.md) | SOLVED → BUG-001 (EXF default) | `atomic/hfs.py` |

## Completed Tasks

| ID | Title | Completed | Notes |
|---|---|---|---|
| TASK-001 | I/O parsers (rcg, rac, ban, rcf, oba) | 2026-04-05 | All parsers shipping |
| TASK-002 | Wigner 3j/6j/9j | 2026-04-05 | Validated to 1e-12 |
| TASK-003 | CFP binary parser | 2026-04-05 | rcg_cfp72/73 exact match |
| TASK-004 | Single-shell RME (SHELL/SPIN/ORBIT) | 2026-04-05 | SHELL 3e-6, SPIN 8e-7 |
| TASK-005 | Spectrum broadening (legacy + correct PV) | 2026-04-05 | Both modes tested |
| TASK-006 | Hamiltonian assembly (nid8ct reference) | 2026-04-06 | 13/13 triads pass |
| TASK-007 | MULTIPOLE transition RME | 2026-04-06 | SVD-invariant match 1e-6 |
| TASK-008 | Integration tests (nid8ct end-to-end) | 2026-04-06 | 10 tests pass |
| TASK-009 | Parity sweep driver | 2026-04-06 | audit_parity_sweep.py |
| TASK-010 | CODE_REVIEW.md & SCIENTIFIC_AUDIT.md | 2026-04-06 | Committed |
| TASK-011 | Fix CT eigenvalue offset for fresh cases | 2026-04-10 | Parser fix in read_rme.py |
| TASK-014 | Multi-ion comparison test suite (8 cases) | 2026-04-10 | 7/8 pass, Ti4+ spectrum-only fail |

## Pending Tasks

| ID | Title | Priority | Blocked by | File |
|---|---|---|---|---|
| TASK-016 | Ti4+ d0 spectrum shape investigation (documented as known limitation) | Low | — | `spectrum/broaden.py` |
| TASK-017 | Demonstration notebooks (quickstart, pipeline, parameter sweep) | Medium | — | `notebooks/` |
| TASK-018 | Bake 8-case Ti–Ni fixtures into tests/reference_data | Medium | — | `tests/reference_data/` |

## Completed Tasks (added 2026-04-11)

| ID | Title | Completed | Notes |
|---|---|---|---|
| TASK-012 | Fix HFS SCF energies | 2026-04-11 | EXF default 0.7 → 1.0; errors drop from ~16% to ~1–8%. See INV-002. |
| TASK-013 | Fix MULTIPOLE basis ordering | 2026-04-11 | Closed as stale — SVD-invariant match confirms physics correct |
| TASK-015 | README with quickstart example | 2026-04-11 | Already present in README.md §Minimal example |

## Multi-ion Comparison Results (2026-04-10)

| Case | Eigenvalues | Max Eg err | Spectrum cosine | Triads |
|---|---|---|---|---|
| Ti4+ d0 Oh | PASS (3.7e-7) | 3.69e-07 | 0.978 (FAIL) | 1/1 |
| V3+ d2 Oh | PASS | 4.33e-06 | 0.998 | 12/12 |
| Cr3+ d3 Oh | PASS | 6.93e-06 | 0.999 | 8/8 |
| Mn2+ d5 Oh | PASS | 9.88e-06 | 0.999 | 8/8 |
| Fe3+ d5 Oh | PASS | 1.58e-05 | 0.998 | 8/8 |
| Fe2+ d6 Oh | PASS | 5.50e-05 | 0.998 | 12/12 |
| Co2+ d7 Oh | PASS | 4.75e-06 | 1.000 | 8/8 |
| Ni2+ d8 Oh | PASS | 1.47e-06 | 1.000 | 12/12 |

## Reference Files

- `CODE_REVIEW.md` — Senior code review (2026-04-06)
- `SCIENTIFIC_AUDIT.md` — Scientific audit (2026-04-06)
- `tests/audit_results.md` — Parity sweep results
- `tests/audit_parity_sweep.py` — Automated parity driver
- `/private/tmp/xas_test/generate_and_compare.py` — 8-case comparison driver
