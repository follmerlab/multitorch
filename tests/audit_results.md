# multitorch Parity Sweep Results
**Generated:** 2026-04-09 11:13:45
**Wall time:** 0.5 s
**Driver:** `tests/audit_parity_sweep.py`

Layer checks:
- `wigner_3j_orthogonality` — Σ_m₁m₂ (j₁j₂j;m₁m₂−m)² = 1/(2j+1) sum rule
- `io_parsers` — every reference file loads without exception
- `rme_shell_d8` — SHELL₁ d⁸ blocks vs `.rme_rcg` reference
- `multipole_ew` — MULTIPOLE d⁸→p⁵d⁹ blocks element-wise
- `eigenvalues` — Hamiltonian Eg vs `.ban_out` per triad
- `hfs_scf` — orbital energies vs `.rcn31_out`

| Fixture | Layer | Status | Max abs err | Tolerance | Note |
|---|---|---|---|---|---|
| nid8 | wigner_3j_6j | ✓ OK | 8.33e-17 | 1e-12 |  |
| nid8 | io_parsers | ✓ OK | 0.00e+00 | 0e+00 | all parsers loaded |
| nid8 | rme_shell_d8 | ✓ OK | 7.71e-07 | 1e-05 |  |
| nid8 | multipole_ew | ✓ OK | 7.55e-07 | 1e-05 | SVD match (reference has different phase convention) |
| nid8 | eigenvalues | · SKIP | — | — | missing .rme_rac or .ban |
| nid8 | hfs_scf | ✓ OK | 3.23e+00 | 5e+00 | 3d=-2.35 2p=-64.47 Ry |
| nid8ct | wigner_3j_6j | ✓ OK | 8.33e-17 | 1e-12 |  |
| nid8ct | io_parsers | ✓ OK | 0.00e+00 | 0e+00 | all parsers loaded |
| nid8ct | rme_shell_d8 | ✓ OK | 3.08e-06 | 1e-05 |  |
| nid8ct | multipole_ew | ✓ OK | 8.98e-07 | 1e-05 | element-wise match |
| nid8ct | eigenvalues | ✓ OK | 2.08e-04 | 1e-03 | 13 triads |
| nid8ct | hfs_scf | · SKIP | — | — | no .rcn31_out reference |
| als1ni2 | wigner_3j_6j | ✓ OK | 8.33e-17 | 1e-12 |  |
| als1ni2 | io_parsers | ✓ OK | 0.00e+00 | 0e+00 | all parsers loaded |
| als1ni2 | rme_shell_d8 | · SKIP | — | — | only nid8/nid8ct have d^8 SHELL blocks |
| als1ni2 | multipole_ew | · SKIP | — | — | only nid8/nid8ct have d^8 MULTIPOLE blocks |
| als1ni2 | eigenvalues | · SKIP | — | — | missing .rme_rac or .ban |
| als1ni2 | hfs_scf | · SKIP | — | — | no .rcn31_out reference |
