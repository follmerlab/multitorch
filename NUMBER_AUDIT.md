# NUMBER_AUDIT.md — externally-sourced numerical values

Ledger for numbers that entered this package from outside it. One block per
value or coherent group of values. Tiers are T1–T7 as in the
`external-number-provenance` convention; T2 is a direct quote from a source's
body text with no uncertainty attached.

Numbers that are **not** here because they are not external in this sense: the
Fortran reference fixtures under `multitorch/data/fixtures/` (we generate those
and can regenerate them), the bundled CFP tables (same lineage), and every
quantity this package derives.

---

[NUMBER id: d3d_one_electron_Kaizaki_1.2.5]
VALUE      : The trigonal (D3/D3d) one-electron d-orbital energies,
             E(e(e_g))    = 6Dq + (7/3) Dtau
             E(e(t_2g))   = -4Dq + Dsigma + (2/3) Dtau
             E(a_1(t_2g)) = -4Dq - 2Dsigma - 6Dtau
             and, in the same paragraph, the splitting
             E(e(t_2g)) - E(a_1(t_2g)) = 3Dsigma + (20/3) Dtau
UNIT       : dimensionless coefficients of Dq, Dsigma, Dtau (each an energy)
TIER       : T2 — quoted from body text, exact rational coefficients, no
             uncertainty stated and none applicable (they are analytic).
SOURCE     : S. Kaizaki, "Ligand Field Theory", chapter 1 of *Instrumental
             Analysis of Coordination Compounds*, volume 1, eds. H. Oshio and
             G. N. Newton, Coordination Chemistry Fundamentals Series vol. 4,
             Royal Society of Chemistry, 2024, pp. 1-26.
             Chapter DOI 10.1039/9781837674978-00001;
             book DOI 10.1039/9781837674978.
LOCATION   : Section 1.2.5, "Ligand Field Parameters for Low Symmetry Field".
             Unnumbered display equations in the body text — **not** a numbered
             equation and **not** a table. Figure 1.13 is the accompanying
             splitting diagram.
METHOD     : Read from the publisher's HTML rendering of the chapter
             (books.rsc.org/books/edited-volume/2243/chapter/8221259) by
             automated text extraction, 2026-10-20. Two independent retrievals
             returned the same three formulas verbatim.
PRECISION  : exact rationals; no coarsening applies.
CONDITIONS : source: electron picture (not hole); Dq = 10Dq/10; the trigonal
             basis, i.e. the C3 axis as the quantisation axis; orbital labels
             a_1(t_2g), e(t_2g), e(e_g) after the parent Oh set.
             this package: the trigonal groups are registered with C3 along
             **[111]** (WP-B5-8, residual 40), so the source's frame is ours
             rotated. The rotation is applied explicitly by
             `crystal_field.axial_frame_rotation` and the comparison is made in
             the source's frame.
             differs: the frame only, and it is converted rather than ignored.
EXTRACTED  : 2026-10-20
USED IN    : `multitorch/angular/crystal_field.py` — `ONE_ELECTRON_PATTERNS["D3d"]`
             (as the *check* on `AXIAL_RANK_2` / `AXIAL_RANK_4`, which are
             inherited from D4h rather than transcribed from here) and the module
             docstring's table;
             `tests/test_hamiltonian/test_cf_one_electron.py` —
             `KAIZAKI_TRIGONAL`, `KAIZAKI_T2G_SPLITTING` and the five tests
             around them;
             `docs/DEVELOPMENT_PLAN_2026-09.md` residual 53.
HISTORY    : v1 (2026-10-20) first and only entry. No value has changed.

CAVEATS, stated rather than skipped:

1. **Secondary source.** This is a 2024 edited-volume chapter, not Ballhausen's
   *Introduction to Ligand Field Theory* (1962) or König & Kremer's *Ligand
   Field Energy Diagrams* (1977), which are where the convention originates and
   which were **not** consulted. A Quanty workshop slide set
   (quanty.org, Arrio, Heidelberg 2022) independently attributes the
   (10Dq, Dsigma, Dtau) scheme for D3d to König & Kremer, but its extracted
   text is garbled and it was not used as a source for any number.
2. **HTML rendering, not the printed chapter.** See METHOD. Nothing here was
   read off a figure; Rule 8's counter-check is satisfied — the formulas are in
   the running text of §1.2.5, which is where the extraction found them.
3. **The source gives no off-diagonal term** between e(t_2g) and e(e_g), and
   the general trigonal field has one. Our off-diagonal is **derived, not
   quoted**: it is forced because the diagonal map is injective within each
   rank, and it comes out proportional to `Dsigma - (5/3) Dtau` with
   `|block| = 2` and `10/3` per unit Dsigma and Dtau. Anywhere that number
   appears it is ours, not Kaizaki's.
4. **What the quoted table cannot do.** The three diagonal energies are
   traceless-constrained, so they carry two independent numbers, not three. The
   map from (10Dq, Dsigma, Dtau) to them has a one-dimensional kernel,
   `∝ (-35/9, -20/9, 1)`. The table defines the convention; it does not let a
   fit to level energies recover the parameters.

5. **Other normalisations of Dsigma/Dtau exist and are not interchangeable.**
   Fitted trigonal parameters in the literature are sometimes reported in a
   normalised-spherical-harmonic (Wybourne/Stevens) scheme and correspond to
   conventional Dq/Dsigma/Dtau only after conversion. The convention adopted
   here is the one above, which is the Ballhausen one this package already uses
   for D4h's Ds and Dt — so the two groups' parameters are on the same footing,
   which is the property the shared pattern tuples assert. A number taken from
   a paper using a different normalisation must be converted, not substituted.

6. **A second independent transcription was looked for and not found.** The
   nearest primary source is J. R. Perumareddi, "Trigonally quantized ligand
   field potentials, d-orbitals and d-orbital energies", *Polyhedron* (1988),
   doi 10.1016/S0277-5387(00)80401-5, which derives the trigonal potentials and
   d-orbital energies along the trigonal axis; only its abstract was reachable
   here, so **it was not consulted** and contributed nothing. Repeated searches
   returned the Kaizaki chapter again rather than a second table. So the
   external pin rests on **one** source, and the strength of the check comes
   from the next paragraph rather than from corroboration.

INDEPENDENT CORROBORATION (not a second source, but it is what makes a
transcription slip implausible): the operators were **not** fitted to this
table. They are D4h's `DS` and `DT` — the pure axial rank-2 and rank-4
operators, coefficients `sqrt(70)` and `-sqrt(630)`, measured from this
package's Fortran-validated D4h path — rotated onto the trigonal axis, with
zero free parameters. That construction reproduces all nine coefficients above
and the splitting relation exactly. A transcription error in the chapter would
have to coincide with an independent derivation anchored on Fortran output.

