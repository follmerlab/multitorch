# External scientific cross-checks — scoping report (2026-10-20)

Scope: assess MolSym, EDRIXS and `JacksonKaunismaa/group_theory` as candidate
external oracles, in particular as a cheaper source for the **names** than
recompiling `ttrac` (WP-B6). **Nothing is wired in.** No dependency added, no
code changed. Each claim below was measured against the package's own source,
not its README.

**Headline: none of the three can supply the names, and the thing that can was
already in the working tree.** `ttmult/src/disk0` is Butler (1981)'s tables in
ASCII, including the `O ⇒ D3` subduction. See §4, which is the part of this
report with consequences.

---

## 1. Verdicts

| candidate | can it settle the names? | useful for anything? | dependency? |
|---|---|---|---|
| **EDRIXS** | **No** — has no general trigonal crystal field | Yes, one targeted use (§2.3) | not yet; a decision for the user |
| **MolSym** | **No** — and it does not settle T either (§3) | Marginal | **recommend no** |
| **group_theory** | No | No | **no** — ruled out |

`JacksonKaunismaa/group_theory` is ruled out on inspection: it is a calculator
for *concrete* finite groups (permutation and relation-defined groups —
multiplication, cosets, centralizers, normality). It has no representation
theory, no characters, no point groups. 1 star, 47 commits, GPL-3.0. It is not
the weakest fit of three; it is the wrong category.

---

## 2. EDRIXS

Brookhaven CMPMSD + NSLS-II. GPL-3.0, conda-forge, ~678 commits. Published:
Wang, Fabbris, Dean & Kotliar, *Comput. Phys. Commun.* **243**, 151 (2019).
A real, citable, maintained ED/RIXS/XAS code — the only one of the three that
could in principle be a physics oracle.

### 2.1 Its crystal field does not reach our open groups

Measured from `edrixs/angular_momentum.py`. The exported crystal-field
functions are exactly four:

```
cf_cubic_d(ten_dq)          5×5 (×spin) full d,   Oh
cf_tetragonal_d(ten_dq, d1, d3)  full d,          D4h
cf_square_planar_d(ten_dq, ds)   full d,          the D4h limit, dt = 2/35·ten_dq
cf_trigonal_t2g(delta)      3×3 (×spin) t2g only
```

`cf_trigonal_t2g` is a **one-parameter, t2g-only** model: an equal off-diagonal
`delta` between the three t2g orbitals. It is the trigonal splitting of the t2g
manifold used in correlated-electron work, **not** the three-parameter
(10Dq, Dσ, Dτ) D3d field on the full d shell. There is no `cf_trigonal_d`.
There is nothing at all for D2h/D2/C2v, O, T, D4, C4v, D2d.

**So EDRIXS adds no group we do not already have, and cannot name Dσ or Dτ.**
That is the question we went to it with, and the answer is no.

### 2.2 Where it *does* agree, it agrees by construction — so it proves nothing

`cf_tetragonal_d` sets, with `dq = ten_dq/10`, `dt = (3d3 − 4d1)/35`,
`ds = (d3 + d1)/7`:

| orbital | EDRIXS | `tests/test_hamiltonian/test_cf_one_electron.py` |
|---|---|---|
| d(3z²−r²) | `6dq − 2ds − 6dt` | `6Dq − 2Ds − 6Dt` |
| d(x²−y²) | `6dq + 2ds − dt` | `6Dq + 2Ds − Dt` |
| d(xy) | `−4dq + 2ds − dt` | `−4Dq + 2Ds − Dt` |
| d(xz), d(yz) | `−4dq − ds + 4dt` | `−4Dq − Ds + 4Dt` |

**Identical, term for term.** This is a genuine convention pin and worth
recording: EDRIXS's D4h convention *is* ours, Ballhausen's, in the electron
picture. But the consequence cuts the other way from how it looks — because the
convention is the same table we already test against, **a D4h agreement test
against EDRIXS would be a restatement, not a check.** It would pass for the
same reason both files were typed from the same textbook.

This is residual 39's trap seen from the good side: the convention *can* be
pinned here, which is exactly why there is nothing left for the oracle to do.

### 2.3 The atomic parameters are Cowan's — EDRIXS is **not** independent there

`edrixs/utils.py:722`, in the `get_atom_data` docstring:

> "The Slater integrals and spin-orbit coupling are calculated by Cowan's code
> (https://github.com/mretegan/atomic-parameters) based on Hartree-Fock"

`edrixs/atom_data/*.json` holds per-configuration `slater` and `soc` lists in eV
(e.g. Ni `3d10_2p5`: `slater = [0.0, 12.234, 7.598, 0.0, 7.721, 5.787, 3.291]`,
`soc = [0.102, 11.507]`).

**This is the same Cowan lineage as ttmult.** Checking multitorch's Slater
integrals or ζ against EDRIXS would be Cowan-vs-Cowan — the "never
code-vs-code" rule in disguise, and it would come out green while proving
nothing about correctness. **EDRIXS must not be used as an independent oracle
for the atomic core**, which was its most tempting use (residual S3 is exactly
about from-scratch F^k/G^k being 2.5–2.8× off).

The legitimate framing: `atom_data/*.json` is a **convenient transcription of
Cowan output**, equal in provenance to our Fortran fixtures and *below* them
(we can regenerate ours; theirs we cannot audit). For residual S3 it is useful
**reference data** — a quick table of Cowan values across the 3d ions — not an
oracle. Use it that way or not at all, and say which in the ledger.

### 2.4 The one genuinely valuable use

EDRIXS's ED solver accepts an **arbitrary** one-body matrix `emat`. So we can
hand it *our* D3d (or any new-group) crystal-field matrix plus our Slater/SOC
values and compare the resulting L-edge spectrum against ours.

What that validates: the **many-body ED and the XAS transition machinery in a
symmetry where we have no Fortran reference.** Independently written, published
and used. That is a real rung and we do not otherwise have one for the nine.

What it does **not** validate, and must be stated wherever it is used:
- not the crystal-field construction (we supplied the matrix),
- not the atomic parameters (same Cowan source),
- not the names (EDRIXS has none for these groups).

Rank: **below** Fortran and analytic, above nothing else. It is a check on one
layer, with the other two layers held fixed on both sides.

### 2.5 Cost

conda-forge (mamba) on macOS/Linux; Docker on Windows. Compiled component
(CMake + a Fortran backend) but conda-forge ships binaries; there are also
`scipy_backend` and `dense_backend` pure-Python paths, so a single-atom L-edge
run should not need MPI. **Not installed here**; installing it is a user
decision and is not required to act on anything in this report.

---

## 3. MolSym

MIT, conda-forge/PyPI, 283 commits, 33 stars. Published: *J. Chem. Phys.*
(2024), doi 10.1063/5.0216738. Point-group detection, symmetrization, character
tables, SALCs.

It addresses **the half of the gate that is already closed** — Mulliken labels,
where we are conventional in 13 of 14 — and not the half that is open.
Measured from `molsym/symtext/character_table.py`:

- **Character tables are hard-coded per family** (cubic tables are literal
  arrays; Cn/Cnv/Cnh/Dn/Dnh/Dnd/Sn are generated).
- **No double groups, no spinor or half-integer representations** — grep for
  `double`/`spinor`/`half-integer` returns nothing. We need `E1/2`, `E3/2`,
  `G3/2`; B1 derives these already.
- **No subduction or correlation tables.** That is rung (6) of the ladder and
  MolSym does not have it.
- **No crystal field, no rank-2/rank-4 spherical tensor operators.**

**It does not settle T, which was the specific hope.** MolSym's T table is

```python
irreps = ["A", "E", "T"]
classes = ["E", "4C_3", "4C_3^2", "3C_2"]
```

— three irreps, with the `E` row carrying `cp3 = 2cos(2π/3)` in both C₃
columns. That is the **complex-conjugate pair merged into one real 2-D
representation**. The true T character table has four irreps (A, ¹E, ²E, T),
and our `labels_are_conventional = False` for T is precisely about labelling
that separated pair. MolSym does not label it; it avoids the question by
summing the pair. **On the one point where we wanted an independent opinion,
MolSym is less resolved than we are, not more.**

**Recommendation: no.** Not as a dependency and not as a cross-check. There is
no question it answers that B1 has not already answered better.

---

## 4. What this turned up instead, and it matters more than the three packages

Chasing "what strings would `ttrac` write" through the Fortran/C source gave a
different answer from the one WP-B6 assumes. Three things, all measured:

**(a) The operator name is free text, chosen by whoever writes the deck.**
`ttrac.c:5893`:

```c
} else if (in_eq("OPER")) {
  /* FORMAT:"OPER" NAME */
```

`ttrac` reads the name and never interprets it. `10DQ` appears **nowhere** in
`ttmult/` — not in `ttrac.c`, not in any deck. The decks in `ttmult/` say
`OPER MULTIPOLE`, `OPER SHELL2`, `OPER HAMILTONIAN`; `pyctm/write_RAC.py`
writes those same three. The `10DQ` in our fixtures' `REDUCEDMATRIX` tags came
from whatever deck generated them, not from `ttrac`.

**So "what would ttrac name the trigonal parameters" has no answer, because
ttrac does not name parameters.** Recompiling it would produce whatever name
we typed into the input we wrote for it. This is not a fact awaiting a lookup.

**(b) What is *not* free is the `BRANCH` structure, and Butler's tables for it
are already in the tree.** The deck format is `OPER <name>` followed by
`BRANCH <irrep path> <strength>` — a Butler branching path down a declared
chain, with a complex strength. `pyctm/write_RAC.py` shows the shape for D4h:
three routes (`4+ > 0 0+ > 0+`, `4+ > 0 2+ > 0+`, `2+ > 0 2+ > 0+`), which is
the two-rank-4-routes-plus-rank-2 structure residual 21 already describes.

`ttmult/src/disk0` is **271 KB of ASCII**, headed
`GROUPDATA D4 CREATED FOR_BUTLER(1981) 25_MAR_1980`. It holds `GROUPDATA` for
C1–C6, D00, D2, D3, D4, D5, D6, K, O, T (17 blocks) and **33 `BRANCHDATA`
blocks**, including:

```
BRANCHDATA O=>D3 CREATED FOR_BUTLER(1981) 28_MAR_1980
BRANCHINGS 0 +0 ; S0 +S0 ; 1 +^0 +1 ; S1 +S0 +S1 +-S1 ; 2 +1 ;
           ^1 +0 +1 ; ^S0 +S0 ; ^0 +^0 ; ENDBRANCHINGS
```

Every one of the 14 targets maps onto an internal group present here
(`ttrac.c:4148–4169`: Oh→O+split, Td→O, D4h→D4+split, D3d→D3+split, C3v→D3,
D2h→D2+split, C2v→D2, D2d→D4, C4v→D4, Th→T+split). **The branching tables,
triads and 3jm primitives for the nine unreachable groups are readable right
now, with no recompile and no new dependency** — and they are the *same
authority ttrac itself reads*, i.e. Butler (1981), a published book.

**(c) Which route is "Dσ" and which "Dτ" is a Ballhausen question, not a Butler
one, and it has the oracle we already use.** Butler's tables fix the routes and
their order; they do not assign crystal-field parameter names. The assignment is
the published one-electron energy table for D3d in (Dq, Dσ, Dτ) — the trigonal
counterpart of the D4h table in `test_cf_one_electron.py`, which §2.2 shows
EDRIXS independently encodes. That is a literature/analytic oracle of a kind
this repo already uses and trusts.

**Net effect on WP-B6.** It splits into three items of very different cost:

| was one item | actually | cost | blocked on? |
|---|---|---|---|
| branch routes + block order | read `disk0` | **available now** | nothing |
| Dσ/Dτ assignment | published D3d one-electron table | a literature lookup | nothing |
| the label strings | our choice | — | **not a fact** |

None of the three needs `ttrac` recompiled, and none needs an external package.

**Caveats, stated rather than skipped.** I have confirmed `disk0`'s contents,
provenance line and that `O=>D3` is present; I have **not** verified that our
reading of Butler's compressed label syntax (`0`, `^0`, `S0`, `1`, `S1`, `2`,
`^2`, the `+`/`-` signs, `CHOSEN`) is correct — that is its own task and wants
its own oracle, the obvious one being to reproduce the known D4h routes in
`pyctm/write_RAC.py` from the `O=>D4` block before trusting the `O=>D3` one.
Do that first; a parser for Butler's syntax validated only where we already
know the answer is the right shape, and reproducing D4h is precisely that test.
Whether the strengths in a `BRANCH` line are Butler-normalised or carry a
ttrac-side factor is also unverified.

---

## 5. Recommendation

1. **Drop `group_theory`** — wrong category.
2. **Drop MolSym** — answers only the closed half, and is less resolved than we
   are on T, the one case we wanted it for.
3. **Do not add EDRIXS as a dependency.** Keep it as an optional, one-off
   cross-check for §2.4 only (our CF matrix through its ED, in a group with no
   Fortran), run from a script and recorded in the ledger with its provenance:
   independent on ED/transitions, **not** independent on atomic parameters
   (Cowan both sides), silent on names.
4. **Pick up §4 instead.** Reading `disk0` is cheaper than all three, needs no
   install, and is the only one of these routes that touches the actual gate.
   Validate any Butler parser against `O=>D4` before using `O=>D3`.

Nothing here is a dependency decision taken; all three remain un-wired.

**User decisions (2026-10-20):** no external package becomes a dependency —
all three stay un-wired. The Butler `disk0` route (§4) is the one taken, gated
on reproducing the known D4h routes first.

---

## 6. Outcome of the Butler read — `scripts/derive_butler_routes.py`

Written and run the same session. Reads `disk0` read-only, writes nothing, no
new dependency. The gate always runs; no derived route is printed unless it
passes.

### 6.1 The gate passed, on order as well as content

Re-deriving the routes from `SO3=>O` and `O=>D4` reproduces what
`pyctm/write_RAC.py` hard-codes, **as an ordered sequence**:

```
[ok ] oh   hamiltonian  ['0+ > 0 0+']
[ok ] oh   shell        ['4+ > 0 0+']
[ok ] d4h  hamiltonian  ['0+ > 0 0+ > 0+']
[ok ] d4h  shell        ['4+ > 0 0+ > 0+', '4+ > 0 2+ > 0+', '2+ > 0 2+ > 0+']
```

Order is compared, not sorted away, because residual 49 names block order as
part of what is missing. The emitted order — rank 4 before rank 2, and within
rank 4 the Oh A1 route before the Oh E route — is pyctm's own.

### 6.2 The gate was shown to fail (`--selftest`)

Per the standing rule that a new oracle is not an oracle until the bug it is
meant to catch has been injected and seen to fail. Four perturbations, applied
to the **in-memory** tables only (`disk0` is read-only source data and is never
written, so there is no backup to restore):

| injection | rejected |
|---|---|
| `O=>D4` loses E → A1 (kills a rank-4 and the rank-2 route) | yes |
| `O=>D4` gains a spurious T1 → A1 (adds routes) | yes |
| `SO3=>O` drops A1 from rank 4 (kills the cubic invariant) | yes |
| `SO3=>O` swaps the rank-4 order (order only, counts unchanged) | yes |

The fourth matters most: it changes **no count and no set**, only the order, and
the gate still rejects it. That is the evidence that the order check is
load-bearing rather than incidental.

### 6.3 Butler agrees with our own derivation, 15 of 15, resolved by rank

Butler's tabulated branchings against `cf_invariant_vectors`, which B1 derives
by the Dixon class-sum method from generators. Two independently written
things, which is what makes this a check and not a restatement.

| | k=4 | k=2 | groups |
|---|---|---|---|
| cubic | 1 | 0 | Oh, O, Td, T, Th |
| tetragonal | 2 | 1 | D4h, D4, C4v, D2d |
| trigonal | 2 | 1 | D3d, D3, C3v |
| orthorhombic | 3 | 2 | D2h, D2, C2v |

**15/15 agree.** This is stronger than the B-status table, which carried only
the totals: the agreement is rank-resolved. It is a new rung on residual 49's
ladder and it cost no reference data.

### 6.4 The trigonal routes, which is what we went for

```
d3d -- chain SO3 -> O -> D3
  OPER HAMILTONIAN   BRANCH 0+ > 0 0+ > 0+
  OPER SHELL         BRANCH 4+ > 0 0+ > 0+      <- via Oh A1 (the cubic invariant)
  OPER SHELL         BRANCH 4+ > 0 ^1+ > 0+     <- via Oh T2
  OPER SHELL         BRANCH 2+ > 0 ^1+ > 0+     <- via Oh T2
```

The structure is the trigonal analogue of D4h's, with **T2 everywhere D4h has
E**: D4h's second rank-4 route and its rank-2 route both pass through Oh E
(`2+`); D3d's pass through Oh T2 (`^1+`). That is the group-theoretic content
of the trigonal field, and it is now pinned by a published table rather than
guessed.

D2h also comes out, through `O -> D4 -> D2` (Butler tabulates no `O=>D2`):
five routes, 3 at rank 4 and 2 at rank 2.

### 6.5 What this still does not settle

- **The label strings.** Still ours to choose; §4(a) stands — not a fact.
- **Which rank-4 trigonal route is `Dτ` and which rank-2 is `Dσ`.** Butler
  fixes the routes and their order, not the Ballhausen parameter names. That
  remains §4(c): the published D3d one-electron energy table, the trigonal
  counterpart of the D4h table `test_cf_one_electron.py` already uses.
- **The `BRANCH` strengths.** The derivation returns routes, not coefficients.
  Butler's phases are parsed and deliberately discarded, since using them would
  need its own oracle. The strengths are what turn two routes into a specific
  (Dq, Dσ, Dτ) parameterisation, so they are needed before D3d is reachable
  under conventional names — but *not* before it is reachable under neutral
  ones.
- **That the `O -> D4 -> D2` chain gives the same D2h as a direct descent.**
  Rung (6) of the ladder (subduction composition) is the check, and it has not
  been run here.
