"""Diagnostic script for debugging the two-shell exchange calculation.

Computes V^(k,1) double tensor, validates exchange angular coefficients
against the Ni²⁺ L-edge fixture HAMILTONIAN blocks.
"""
import math
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from multitorch.angular.wigner import wigner3j, wigner6j, wigner9j
from multitorch.angular.rme import (
    LSTerm, compute_uk_ls, build_two_shell_j_basis,
    compute_two_shell_shell_blocks, compute_two_shell_soc, uncpla,
)
from multitorch.angular.cfp import get_cfp_block


def compute_vk1_ls(
    l: int,
    n: int,
    k: int,
    terms, parent_terms, cfp_matrix,
) -> np.ndarray:
    """Compute V^(k,1) double tensor in the LS basis.

    V^(k,1) = u^(k) ⊗ σ^(1) acting on the Nth electron.

    Uses the same orbital recoupling as U^(k) (compute_uk_ls) but adds
    the spin recoupling factor. The convention matches the Fortran
    V tables: for N=1 (single electron), V^(k,1) = √(3/2) for all k.

    Formula:
    V^(k,1)_{ij} = N × Σ_p cfp_i × cfp_j
        × (-1)^{Lp+l+Lj+k} × √((2Li+1)(2Lj+1)) × {Li k Lj; l Lp l}
        × spin_phase × √((2Si+1)(2Sj+1)) × {Si 1 Sj; 1/2 Sp 1/2} × √(3/2)

    where spin_phase = (-1)^{round(Sp + Sj + 3/2)}
    """
    n_terms = len(terms)
    vk = np.zeros((n_terms, n_terms), dtype=np.float64)
    fl = float(l)
    sqrt32 = math.sqrt(1.5)  # √(3/2) = ⟨s=1/2 || σ^(1) || s=1/2⟩

    for i, ti in enumerate(terms):
        for j, tj in enumerate(terms):
            # Selection rules: |Si - Sj| ≤ 1 (spin rank 1)
            if abs(ti.S - tj.S) > 1.0 + 1e-10:
                continue
            # Orbital triangle: |Li - Lj| ≤ k ≤ Li + Lj
            if k > ti.L + tj.L or k < abs(ti.L - tj.L):
                continue

            val = 0.0
            for p, tp in enumerate(parent_terms):
                cfp_i = cfp_matrix[i, p] if i < cfp_matrix.shape[0] and p < cfp_matrix.shape[1] else 0.0
                cfp_j = cfp_matrix[j, p] if j < cfp_matrix.shape[0] and p < cfp_matrix.shape[1] else 0.0
                if abs(cfp_i) < 1e-15 or abs(cfp_j) < 1e-15:
                    continue

                # Orbital part (same as U^(k))
                sixj_orb = wigner6j(ti.L, k, tj.L, fl, tp.L, fl)
                if abs(sixj_orb) < 1e-15:
                    continue
                phase_orb = int(round(tp.L + fl + tj.L + k))

                # Spin part
                sixj_spin = wigner6j(ti.S, 1.0, tj.S, 0.5, tp.S, 0.5)
                if abs(sixj_spin) < 1e-15:
                    continue
                phase_spin = int(round(tp.S + tj.S + 1.5))

                phase = (-1.0) ** (phase_orb + phase_spin)

                val += cfp_i * cfp_j * phase * sixj_orb * sixj_spin

            # Factors: N × √((2Li+1)(2Lj+1)) × √((2Si+1)(2Sj+1)) × √(3/2)
            val *= n * math.sqrt(
                (2.0 * ti.L + 1.0) * (2.0 * tj.L + 1.0)
                * (2.0 * ti.S + 1.0) * (2.0 * tj.S + 1.0)
            ) * sqrt32

            vk[i, j] = val

    return vk


def get_terms_and_cfp(l, n):
    """Get terms, parent terms, and CFP matrix for l^n."""
    block = get_cfp_block(l, n)
    terms = [LSTerm(index=t.index, S=t.S, L=t.L,
                    seniority=t.seniority,
                    label=f"{int(2*t.S+1)}{t.L_label}")
             for t in block.terms]
    parent_block = get_cfp_block(l, n - 1)
    parent_terms = [LSTerm(index=t.index, S=t.S, L=t.L,
                          seniority=t.seniority,
                          label=f"{int(2*t.S+1)}{t.L_label}")
                   for t in parent_block.terms]
    cfp = block.cfp
    return terms, parent_terms, cfp


def test_vk1_single_electron():
    """V^(k,1) for single electron should equal √(3/2) for all k."""
    print("=== V^(k,1) for single-electron shells ===")

    for l_val, name in [(1, 'p^1'), (2, 'd^1')]:
        terms, parents, cfp = get_terms_and_cfp(l_val, 1)
        print(f"\n{name}: single term = {terms[0].label}")
        for k in range(0, 2 * l_val + 1):
            vk = compute_vk1_ls(l_val, 1, k, terms, parents, cfp)
            uk = compute_uk_ls(l_val, 1, k, terms, parents, cfp)
            print(f"  k={k}: V={vk[0,0]:+.6f}  U={uk[0,0]:+.6f}  "
                  f"(expect V=√(3/2)={math.sqrt(1.5):.6f})")


def test_vk1_almost_filled():
    """V^(k,1) for p^5 and d^9 (single-term, almost-filled shells)."""
    print("\n=== V^(k,1) for almost-filled shells ===")

    for l_val, n, name in [(1, 5, 'p^5'), (2, 9, 'd^9')]:
        terms, parents, cfp = get_terms_and_cfp(l_val, n)
        print(f"\n{name}: single term = {terms[0].label}")
        for k in range(0, 2 * l_val + 1):
            vk = compute_vk1_ls(l_val, n, k, terms, parents, cfp)
            uk = compute_uk_ls(l_val, n, k, terms, parents, cfp)
            print(f"  k={k}: V={vk[0,0]:+.6f}  U={uk[0,0]:+.6f}")


def compute_exchange_reij(
    l_core, n_core, l_val, n_val, k_exchange,
    core_terms, val_terms,
    U_core, V_core, U_val, V_val,
):
    """Compute exchange G^k angular coefficients using REIJ formula.

    For a two-shell system (core, valence), implements the Fortran REIJ
    algorithm from ttrcg.f lines 6519-6663.

    Parameters
    ----------
    k_exchange : int
        Exchange rank (1 or 3 for p-d).
    U_core, V_core : dict mapping R → np.ndarray
        Intra-shell U^(R) and V^(R,1) matrices for the core shell.
    U_val, V_val : dict mapping R → np.ndarray
        Intra-shell U^(R) and V^(R,1) matrices for the valence shell.

    Returns
    -------
    dict mapping J → np.ndarray
        Exchange angular coefficient matrix for each J sector.
    """
    two_shell_basis = build_two_shell_j_basis(core_terms, val_terms)

    # CIJKF = ⟨l_I || C^(k) || l_J⟩
    # For exchange: CIJ = CIJKF(l_I, l_J, k)^2
    # CIJKF(l1, l2, k) = (-1)^l1 × √((2l1+1)(2l2+1)) × 3j(l1 k l2; 0 0 0)
    lpl = l_core + l_val + k_exchange
    if lpl % 2 == 0:
        # Even parity: CIJ = (CIJKF)^2
        cijkf = ((-1.0) ** l_core * math.sqrt((2*l_core+1)*(2*l_val+1))
                 * wigner3j(l_core, k_exchange, l_val, 0, 0, 0))
        CIJ = cijkf ** 2
    else:
        # Odd parity: special formula
        CIJ = 2.0 * 0.5 * ((-1.0) ** k_exchange)

    # R range for the intermediate rank
    R_min = 0
    R_max = min(2 * l_core, 2 * l_val)

    # CAVE (average configuration energy, added to diagonal)
    if lpl % 2 == 0:
        s3j0sq = wigner3j(l_core, k_exchange, l_val, 0, 0, 0) ** 2
        CAVE = n_core * n_val * s3j0sq / 2.0
    else:
        CAVE = -((-1.0)**k_exchange * 0.5) * n_core * n_val / (
            (2*l_core+1) * (2*l_val+1))

    blocks = {}

    for J, states in two_shell_basis.items():
        n_states = len(states)
        mat = np.zeros((n_states, n_states), dtype=np.float64)

        for ib, sa in enumerate(states):
            for ik, sb in enumerate(states):
                t1_a = core_terms[sa.term1_idx]   # core term, bra
                t2_a = val_terms[sa.term2_idx]     # val term, bra
                t1_b = core_terms[sb.term1_idx]    # core term, ket
                t2_b = val_terms[sb.term2_idx]     # val term, ket

                S1_bra, L1_bra = t1_a.S, t1_a.L
                S2_bra, L2_bra = t2_a.S, t2_a.L
                S1_ket, L1_ket = t1_b.S, t1_b.L
                S2_ket, L2_ket = t2_b.S, t2_b.L
                S_tot_bra, L_tot_bra = sa.S_total, sa.L_total
                S_tot_ket, L_tot_ket = sb.S_total, sb.L_total

                # ---- AR (orbital phase for direct part) ----
                AR = (-1.0) ** int(round(L2_bra + L1_ket + L_tot_bra))

                # ---- A1R (spin-exchange factor) ----
                A1R = (-1.0) ** int(round(S2_bra + S1_ket + S_tot_bra))
                A1R *= wigner6j(S1_bra, S2_bra, S_tot_bra,
                                S2_ket, S1_ket, 1.0)

                # Delta check: AR = 0 unless same term on each shell
                # (the "direct" spin constraint)
                # Check S coupling for each shell
                if abs(S1_bra - S1_ket) > 1e-10 or abs(S2_bra - S2_ket) > 1e-10:
                    AR = 0.0
                # Also check L coupling (same shell, same L)
                # Actually the Fortran checks:
                # TSCS(1,M1) == TSCS(2,M1) for intermediate couplings
                # TS(1,I) == TS(2,I) and TS(1,J) == TS(2,J)
                # For 2-shell: S1_bra==S1_ket and S2_bra==S2_ket (already checked)
                # Plus L coupling: TSCL for intermediate = L1, so L1_bra==L1_ket
                # But the Fortran only checks S, not L... let me re-read.
                #
                # Actually, the Fortran lines 6586-6591:
                # do 431 M1=I,MX → M1=1,1:
                #   if (TSCS(1,1).NE.TSCS(2,1)) AR=0  → S_coupled_1_bra != S_coupled_1_ket
                # if (TS(1,I).NE.TS(2,I)) AR=0 → S_term_shell_I_bra != ket
                # if (TS(1,J).NE.TS(2,J)) AR=0 → S_term_shell_J_bra != ket
                #
                # TSCS(1,1) = S_coupled_up_to_shell_1_bra = S1_bra
                # TSCS(2,1) = S_coupled_up_to_shell_1_ket = S1_ket
                # TS(1,I=1) = S_term_of_shell_1_bra = S1_bra
                # TS(2,I=1) = S_term_of_shell_1_ket = S1_ket
                # TS(1,J=2) = S_term_of_shell_2_bra = S2_bra
                # TS(2,J=2) = S_term_of_shell_2_ket = S2_ket
                #
                # So the checks are: S1_bra==S1_ket AND S1_bra==S1_ket (redundant)
                # AND S2_bra==S2_ket. Already covered above.

                A1R *= AR  # Fortran line 6585

                # ---- RSUM ----
                RSUM = 0.0

                for R in range(R_min, R_max + 1):
                    # Orbital recoupling 6j
                    PR = wigner6j(L1_bra, L1_ket, float(R),
                                  L2_ket, L2_bra, L_tot_bra)

                    # P1R starts as 4.0 (Fortran line 6599)
                    P1R = 4.0

                    # For 2-shell system: no intermediate shells, no I>1 block
                    # Fortran lines 6600-6608 are skipped.

                    # Line 6609: P1R = P1R * PR * A1R
                    P1R = P1R * PR * A1R

                    # Line 6610: PR = PR * AR
                    PR = PR * AR

                    # Shell I (core) operators
                    if n_core == 1:
                        # N=1: P1R *= √(3/2), PR unchanged
                        P1R *= math.sqrt(1.5)
                    else:
                        # Multi-electron: use U and V tables
                        u_val_R = U_core[R][sa.term1_idx, sb.term1_idx]
                        v_val_R = V_core[R][sa.term1_idx, sb.term1_idx]
                        PR *= u_val_R
                        P1R *= v_val_R

                    # Shell J (valence) operators
                    if n_val == 1:
                        P1R *= math.sqrt(1.5)
                    else:
                        u_val_R = U_val[R][sa.term2_idx, sb.term2_idx]
                        v_val_R = V_val[R][sa.term2_idx, sb.term2_idx]
                        PR *= u_val_R
                        P1R *= v_val_R

                    # B factor
                    B = (2.0 * R + 1.0) * wigner6j(
                        float(l_core), float(l_core), float(R),
                        float(l_val), float(l_val), float(k_exchange))
                    if R % 2 == 1:
                        B = -B

                    RSUM += B * (PR + P1R)

                RSUM *= -0.5

                # Full matrix element
                element = CIJ * RSUM

                # Add CAVE for diagonal
                if ib == ik:
                    element += CAVE

                # Fortran two-shell state phase convention
                phase_conv = (-1) ** int(round(
                    sa.S_total + sa.L_total
                    + sb.S_total + sb.L_total))

                # J coupling: UNCPLA for scalar (k=0 in J)
                # Exchange is scalar in J, so the J-coupling factor is:
                # UNCPLA(L_total, S_total, J, 0, L_total', J)
                rc = uncpla(sa.L_total, sa.S_total, J,
                            0, sb.L_total, J)

                mat[ib, ik] = element * phase_conv * rc

        if np.any(np.abs(mat) > 1e-15):
            blocks[J] = mat

    return blocks


def test_exchange_ni2():
    """Test exchange for Ni2+ L-edge (p^5 d^9) against fixture."""
    import torch
    from multitorch.io.read_rme import read_cowan_store
    from multitorch.hamiltonian.build_cowan import read_cowan_metadata
    from multitorch._constants import RY_TO_EV_FLOAT

    print("\n=== Exchange test: Ni2+ L-edge (p^5 d^9) ===")

    l_core, n_core = 1, 5  # p^5
    l_val, n_val = 2, 9     # d^9

    # Get terms and CFP
    core_terms, core_parents, core_cfp = get_terms_and_cfp(l_core, n_core)
    val_terms, val_parents, val_cfp = get_terms_and_cfp(l_val, n_val)

    print(f"Core: {[t.label for t in core_terms]}")
    print(f"Valence: {[t.label for t in val_terms]}")

    # Compute U^(R) and V^(R,1) for both shells
    U_core = {}
    V_core = {}
    for R in range(0, 2 * l_core + 1):
        U_core[R] = compute_uk_ls(l_core, n_core, R, core_terms, core_parents, core_cfp)
        V_core[R] = compute_vk1_ls(l_core, n_core, R, core_terms, core_parents, core_cfp)
        print(f"  Core U^({R}) = {U_core[R][0,0]:+.6f}, V^({R},1) = {V_core[R][0,0]:+.6f}")

    U_val = {}
    V_val = {}
    for R in range(0, 2 * l_val + 1):
        U_val[R] = compute_uk_ls(l_val, n_val, R, val_terms, val_parents, val_cfp)
        V_val[R] = compute_vk1_ls(l_val, n_val, R, val_terms, val_parents, val_cfp)
        print(f"  Val U^({R}) = {U_val[R][0,0]:+.6f}, V^({R},1) = {V_val[R][0,0]:+.6f}")

    # Compute exchange for G^1 and G^3
    for k in [1, 3]:
        exchange = compute_exchange_reij(
            l_core, n_core, l_val, n_val, k,
            core_terms, val_terms,
            U_core, V_core, U_val, V_val,
        )
        print(f"\nExchange G^{k} angular coefficients:")
        for J in sorted(exchange.keys()):
            mat = exchange[J]
            print(f"  J={J:.0f}: diag={np.diag(mat)}")

    # Now assemble the full HAMILTONIAN and compare with fixture
    ref_dir = os.path.join(os.path.dirname(__file__), '..', 'reference_data')
    rcg_file = os.path.join(ref_dir, 'ni2_d8_oh', 'ni2_d8_oh.rme_rcg')

    if not os.path.exists(rcg_file):
        print("No fixture file, skipping comparison")
        return

    cowan = read_cowan_store(rcg_file)
    meta = read_cowan_metadata(rcg_file)

    # Section 3 has the excited-state blocks
    sec3_meta = meta[3]
    sec3_mats = cowan[3]

    # Find HAMILTONIAN blocks
    print("\n=== Fixture HAMILTONIAN eigenvalues (section 3) ===")
    for idx, m in enumerate(sec3_meta):
        if m.operator == "HAMILTONIAN" and m.block_type == "GROUND":
            mat = sec3_mats[idx]
            if mat.numel() > 0:
                J_sym = m.bra_sym
                evals = torch.linalg.eigvalsh(mat)
                print(f"  {J_sym}: shape={mat.shape}, evals={evals.tolist()}")

    # Assemble our HAMILTONIAN from components
    # H = Σ_k F^k(dd) × SHELL2_k + ζ_d × SOC_d + ζ_p × SOC_p
    #   + Σ_k G^k(pd) × EXCHANGE_k + Σ_k>0 F^k(pd) × DIRECT_k
    #   + F^2(pp) × pp_angular

    # Atomic parameters (from rcn31_out, in Rydberg)
    F2_dd = 0.8146951 * RY_TO_EV_FLOAT
    F4_dd = 0.5024169 * RY_TO_EV_FLOAT
    zeta_d = 0.00546 * RY_TO_EV_FLOAT  # Blume-Watson for d shell
    zeta_p = 0.81599 * RY_TO_EV_FLOAT  # p shell SOC
    G1_pd = 0.3359510 * RY_TO_EV_FLOAT
    G3_pd = 0.1906853 * RY_TO_EV_FLOAT
    F2_pd = 0.4656287 * RY_TO_EV_FLOAT
    F2_pp = 4.0099902 * RY_TO_EV_FLOAT

    print(f"\nParameters: F2_dd={F2_dd:.3f} F4_dd={F4_dd:.3f} "
          f"ζ_d={zeta_d:.4f} ζ_p={zeta_p:.3f}")
    print(f"  G1_pd={G1_pd:.3f} G3_pd={G3_pd:.3f} "
          f"F2_pd={F2_pd:.3f} F2_pp={F2_pp:.3f}")

    # Get the SHELL2 blocks (d-d Coulomb angular factors)
    shell2_k0 = compute_two_shell_shell_blocks(l_val, n_val, l_core, n_core, 0)
    shell2_k2 = compute_two_shell_shell_blocks(l_val, n_val, l_core, n_core, 2)
    shell2_k4 = compute_two_shell_shell_blocks(l_val, n_val, l_core, n_core, 4)

    # Get SOC blocks
    soc_d = compute_two_shell_soc(l_val, n_val, l_core, n_core, shell_idx=2)
    soc_p = compute_two_shell_soc(l_val, n_val, l_core, n_core, shell_idx=1)

    # Compute exchange
    exchange_k1 = compute_exchange_reij(
        l_core, n_core, l_val, n_val, 1,
        core_terms, val_terms,
        U_core, V_core, U_val, V_val)
    exchange_k3 = compute_exchange_reij(
        l_core, n_core, l_val, n_val, 3,
        core_terms, val_terms,
        U_core, V_core, U_val, V_val)

    # Compute direct inter-shell Coulomb F^2(pd)
    # This is a SHELL block for the (core, val) pair with cross-shell rank
    # For now, compute it using the same approach as SHELL2 blocks
    # TODO: implement properly

    print("\n=== Assembled HAMILTONIAN eigenvalues ===")
    two_shell_basis = build_two_shell_j_basis(core_terms, val_terms)

    for J in sorted(two_shell_basis.keys()):
        n_states = len(two_shell_basis[J])
        H = np.zeros((n_states, n_states), dtype=np.float64)

        # d-d Coulomb (F^k(dd), k>0)
        if J in shell2_k2:
            H += F2_dd * shell2_k2[J]
        if J in shell2_k4:
            H += F4_dd * shell2_k4[J]

        # SOC
        if J in soc_d:
            H += zeta_d * soc_d[J]
        if J in soc_p:
            H += zeta_p * soc_p[J]

        # Exchange
        if J in exchange_k1:
            H += G1_pd * exchange_k1[J]
        if J in exchange_k3:
            H += G3_pd * exchange_k3[J]

        # Make symmetric
        H = 0.5 * (H + H.T)

        evals = np.linalg.eigvalsh(H)
        print(f"  J={J:.0f}: partial H evals = {evals}")


if __name__ == '__main__':
    test_vk1_single_electron()
    test_vk1_almost_filled()
    test_exchange_ni2()
