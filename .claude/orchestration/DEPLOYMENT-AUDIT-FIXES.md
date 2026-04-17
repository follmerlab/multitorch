# Deployment Audit Fixes — 2026-04-17

Tracking document for fixes identified by the 5-agent deployment audit.
Each fix is marked with status: PENDING → IN_PROGRESS → DONE.

## Priority 1: Critical Bugs

| ID | Issue | Location | Status |
|---|---|---|---|
| FIX-01 | `_hfs_to_slater_params` KeyError for ions missing pd integrals | `calc.py:652-656` | DONE |
| FIX-02 | `calcXES` silently drops T, gamma1, gamma2 params | `calc.py:860-861` | DONE |
| FIX-03 | DOC Phase 5 unit doubling (Ry→eV applied twice) | `calc.py:1340` | DONE |

## Priority 2: Autograd Stability

| ID | Issue | Location | Status |
|---|---|---|---|
| FIX-04 | eigh backward NaN at eigenvalue degeneracies | `hamiltonian/diagonalize.py` | DONE — `safe_eigh` with linear perturbation |
| FIX-05 | COWAN rebuild limited to section 2 only (excited-state gradients broken) | `hamiltonian/build_cowan.py` | DEFERRED (architectural, not a bug) |
| FIX-06 | Broadening params not differentiable (math.sqrt vs torch.sqrt) | `spectrum/broaden.py` | DONE — precomputed constants, tensor-compatible API |

## Priority 3: High-Priority Code Fixes

| ID | Issue | Location | Status |
|---|---|---|---|
| FIX-07 | sticks.py auto-squaring heuristic fragile | `spectrum/sticks.py:77-78` | DONE |
| FIX-08 | getXAS with no ban_output_path gives confusing error | `api/plot.py:13-28` | DONE |
| FIX-09 | Duplicate RY_TO_EV and kB constants (drift risk) | `calc.py:1307,1223` | DONE — centralized to `_constants.py` |
| FIX-10 | Global mutable caches in cfp.py not thread-safe | `angular/cfp.py:283,471` | DONE — double-checked locking |
| FIX-11 | F2_pd contribution silently skipped in rac_generator | `angular/rac_generator.py:301` | DONE — `warnings.warn()` |

## Priority 4: Missing Tests

| ID | Issue | Status |
|---|---|---|
| TEST-01 | Hermiticity of assembled Hamiltonians | DONE |
| TEST-02 | torch.autograd.gradcheck finite-difference validation | DONE |
| TEST-03 | Transition intensity sum rule | DONE |
| TEST-04 | d0/d10 edge cases | DONE |
| TEST-05 | Zero crystal field (free-ion limit) | DONE |
| TEST-06 | Degenerate eigenvalue handling in autograd | DONE |

## Priority 5: Brittle Test Rewrites

| ID | Issue | Status |
|---|---|---|
| REWRITE-01 | test_rme_rac_full_blocks — asserts exact block count=226 | DONE — behavioral checks |
| REWRITE-02 | test_cowan_store_sections — asserts exact matrix counts | DONE — behavioral checks |

## Test status after all fixes

460 passed, 0 failed, 21 warnings (2026-04-17)
