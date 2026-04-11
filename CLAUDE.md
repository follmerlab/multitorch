# multitorch — Project Instructions

## What this is

PyTorch port of the Fortran ttmult/Cowan multiplet X-ray spectroscopy suite for L-edge XAS of 3d transition metals. Must match Fortran numerically.

## Environment

- **Conda env:** `multi` (Python 3.11, PyTorch 2.5)
- **Python:** `/opt/anaconda3/envs/multi/bin/python`
- **Pytest:** `/opt/anaconda3/envs/multi/bin/pytest tests/ -q`
- **Working dir:** `/Users/afollmer/Follmer_UCD/Follmer_Lab/Code/multiplets/multitorch/`

## Codebase layout

```
multitorch/
  multitorch/           # package source
    angular/            # Wigner 3j/6j/9j, CFP, RME builder, symmetry
    atomic/             # HFS SCF, Slater integrals, radial mesh
    hamiltonian/        # assemble.py (main CT pipeline), diagonalize, transitions
    spectrum/           # sticks, broaden (Voigt/pseudo-Voigt), rixs, background
    io/                 # parsers: read_rme.py, read_ban.py, read_rcf.py, read_oba.py
    api/                # calc.py (calcXAS entry point), plot.py
  tests/
    reference_data/     # Fortran reference outputs: nid8/, nid8ct/, als1ni2/
    test_angular/       # Wigner, CFP, RME tests
    test_atomic/        # HFS, Slater, mesh tests
    test_hamiltonian/   # assemble, diagonalize, BAN parser tests
    test_integration/   # end-to-end nid8ct XAS tests
    test_spectrum/      # broadening, sticks tests
```

Fortran source lives at `../ttmult/src/` (rcn31.f, rcn2.f, ttrcg.f, ttrac.c, ttban_exact.f).
Fortran binaries at `../ttmult/bin/`.
Python wrapper at `../pyttmult/`.
Legacy Python API at `../pyctm/`.

## Test status

141/141 tests pass as of 2026-04-09.

## Orchestration

All implementation progress, bugs, investigation logs, and task tracking are in `.claude/orchestration/`. See `.claude/orchestration/INDEX.md` for the full index.

**Before starting any work:**
1. Read `.claude/orchestration/INDEX.md` to understand current state
2. Check if your task has an existing investigation log
3. Update the relevant files when you make progress or find something new

## Key conventions

- All tensors use `float64` (`DTYPE` from `_constants.py`)
- COWAN store sections are 0-indexed in Python; ADD entry `matrix_idx` is 1-based (subtract 1)
- Energy offsets (EG/EF) are NOT scaled by 1/sqrt(IDIM); Hamiltonian matrix elements ARE
- `.rme_rcg` sections delimited by FINISHED markers map 1:1 to Fortran PAIRIN calls
- `.rme_rac` operator blocks: "GROUND" = config 1, "EXCITE" = config 2 (even in ground state manifold)
- Hybridization blocks in `.rme_rac` are labeled TRANSI with geometry containing "HYBR"

## Known pitfalls

- The Fortran PAIRIN subroutine reads .rme_rcg/.rme_rac SEQUENTIALLY. Each call consumes one COWAN section. The Python code indexes by section number instead.
- PAIRIN overwrites mateg(1)/mateg(2) on each call. The FINAL call (ground mixing, section 2) is the one that matters for make_hamiltonian.
- Empty .rme_rac blocks (no ADD entries) must be skipped — they exist as placeholders from earlier PAIRIN calls.
- The Fortran `gausslegendre` with W=0, N=1 gives point=0, weight=1 (special case, line 375-378 of ttban_exact.f).
- `SUBANAx` parses `DEF EG2 = 4.000 UNITY` by finding the keyword, scanning backwards for the numeric coefficient, and multiplying by the keyword's value (UNITY=1.0, DEL=delta, UCV=Ucv, UVV=Uvv).

## Do not

- Do not modify reference data in `tests/reference_data/`
- Do not add features beyond what's needed to match Fortran output
- Do not refactor code that already works and passes tests
- Do not commit to git without explicit user request
