# INV-001: CT Eigenvalue Offset for Fresh Fortran Cases

**Status:** IN PROGRESS
**Severity:** HIGH — blocks multi-ion validation (TASK-014)
**Related bug:** BUG-003
**Primary file:** `multitorch/hamiltonian/assemble.py`

## Symptom

When running fresh Fortran CT calculations (not using pre-committed reference data), multitorch eigenvalues have a systematic offset of ~1 Ry compared to Fortran. Spectrum shapes still match (cosine similarity > 0.997).

| Ion | Edge | EG2 | multitorch Eg0 | Fortran Eg0 | Diff |
|---|---|---|---|---|---|
| Ni2+ d8 Oh | L | 5.0 | -1.7076 | -1.7074 | 0.0002 (PASS) |
| V3+ d2 Oh | L | 4.0 | -3.918 | -4.930 | 1.012 |
| Cr3+ d3 Oh | L | 4.0 | ~offset | ~offset | ~1 Ry |
| Mn2+ d5 Oh | L | 5.0 | ~offset | ~offset | ~1 Ry |
| Fe3+ d5 Oh | L | 4.0 | ~offset | ~offset | ~1 Ry |
| Fe2+ d6 Oh | L | 5.0 | ~offset | ~offset | ~1 Ry |
| Co2+ d7 Oh | L | 5.0 | ~offset | ~offset | ~1 Ry |

Ni2+ passes because it uses pre-committed reference data (nid8ct) with a "pre-merged" .rme_rac format.

## What the V3+ full eigenvalue comparison shows

55 eigenvalues for 0+ triad:
- Fortran trace = 211.063, multitorch trace = 193.136, difference = -17.93
- Non-uniform differences (range -2.08 to +1.01)
- NOT a simple constant shift or unit conversion

## Root cause investigation

### Ruled out

1. **COWAN section selection** — both use section 2 for ground state CT. Verified.
2. **ADD entry block matching** — both find blocks at line 915 of .rme_rac with matching data. Verified.
3. **COWAN matrix values** — parser matches raw file exactly. Verified.
4. **XHAM/XMIX scaling factors** — match between Fortran and multitorch. Verified.
5. **IDIM scaling** — both apply 1/sqrt(IDIM) to matrix, NOT to energy offset. Verified.
6. **Energy offset values** — both use EG1=0, EG2=4.0 (or 5.0 for Ni). Verified.
7. **Empty block filtering** — empty blocks at line 126 skipped via add_entries check. Verified.
8. **Subblock placement logic** — row/column indexing verified identical to Fortran fill_vector/fill_vector_x.
9. **Simple eV-to-Ry conversion of EG2** — gives -5.034, still off; trace diff doesn't match.
10. **DEF parsing (SUBANAx)** — `DEF EG2 = 4.000 UNITY` correctly parsed as 4.0 * 1.0 = 4.0 by both codes. No DEL/UCV/UVV modifiers in BAN files.

### Remaining hypotheses (2026-04-09)

**H1: "Pre-merged" vs "separate" .rme_rac format handling**
The nid8ct reference has a "pre-merged" .rme_rac where all operator blocks are consolidated. Fresh Fortran output has a "separate" format where:
- Set 1 (PAIRIN call 1): Config 1 transition + EMPTY ground/excited placeholders
- Set 2 (PAIRIN call 2): Config 2 transition + Config 2 ground/excited blocks WITH data
- Set 3 (PAIRIN call 3): Hybridization + Config 1 ground + Config 2 excited blocks WITH data

The Fortran PAIRIN call 3 OVERWRITES mateg(1) and mateg(2) with the operators from Set 3. But in a separate-format .rme_rac, Set 2 also has ground/excited blocks with data for config 2.

**Key question:** Does `_find_operator_blocks` correctly find the Set 3 blocks (the ones PAIRIN call 3 uses) and NOT the Set 2 blocks?

For config 2 (kind='EXCITE'):
- Set 2 has EXCITE blocks with ADD entries for config 2 — these reference COWAN section 1 matrices
- Set 3 has EXCITE blocks with ADD entries for config 2 — these reference COWAN section 2 matrices
- If `_find_operator_blocks` returns the FIRST non-empty EXCITE block (Set 2), but the assembler feeds it `cowan[2]` (section 2), the matrix indices would be WRONG because Set 2's ADD entries reference section 1 indices.

**THIS IS VERY LIKELY THE BUG.** The assembler hardcodes `gs_cowan_sec = 2` but the `.rme_rac` blocks it finds may have ADD entries that reference a different COWAN section.

**H2: Fortran reads .rme_rac sequentially; Python searches by name**
PAIRIN reads blocks in file order. Python `_find_operator_blocks` searches all blocks and returns the first match. For the "separate" format, the first EXCITE block with data is from Set 2 (PAIRIN call 2), not Set 3 (PAIRIN call 3). This would cause wrong COWAN matrix references.

## Next steps

1. Verify H1: Check what ADD entry indices are in the Set 2 vs Set 3 EXCITE blocks for V3+
2. If confirmed: fix `_find_operator_blocks` to use the LAST occurrence of non-empty blocks (or the occurrence corresponding to the ground mixing PAIRIN call)
3. Re-run the 8-case comparison after fix

## Session log

- **2026-04-09 session 1**: Investigated DEF parsing, XHAM/XMIX, COWAN sections, fill_vector. All correct.
- **2026-04-09 session 2**: Deep-dived into PAIRIN call sequence, NEWFIL behavior, ETR/sp_BUILD. Identified H1/H2 as likely root cause.
