"""
Blume-Watson multi-orbital spin-orbit parameter (rcn31.f::ZETABW port).

The central-field formula

    ζ_nl^CF = (α²/2) ∫ (1/r)(dV/dr) P_nl²(r) dr

(implemented in :func:`multitorch.atomic.hfs.hfs_scf`) reproduces Cowan's
"R*VI" column to ~5% but misses the multi-orbital exchange corrections that
come from the full Hartree-Fock potential. Blume & Watson (1962/63) added
those corrections; in the Fortran source they live in
``rcn31.f::ZETABW(M, RM3)`` (lines 3302-3421) with helper integrals
``SM`` (3424-3448), ``SN`` (3451-3475), ``ZK`` (3478-3526),
``VK`` (3529-3555) and ``DYK`` (3558-3624).

This module ports those routines to PyTorch on the same logarithmic mesh
that ``hfs_scf`` already uses, with explicit ``k`` arguments instead of the
Fortran ``COMMON/C4`` thread state.

All inputs are float64 ``torch.Tensor`` objects; outputs are float64 scalar
tensors in Rydbergs. Wavefunctions ``P(r)`` use the same convention as
``hfs.py`` (the rRᵤ orbital, not Rᵤ itself).

The angular factors share their primitives with
:mod:`multitorch.angular.wigner`:

    S6J(j1,j2,j3,j4,j5,j6) → ``wigner6j(j1,j2,j3,j4,j5,j6)``
    S3J0SQ(l1,k,l2)        → ``wigner3j(l1,k,l2,0,0,0) ** 2``

The radial helpers (SM/SN/VK) all carry the prefactor ``α²/2 ≈ 2.6626e-5``
(``ALPHA_SQ_HALF``) so that their outputs are in Rydbergs and combine
directly with the bare ``Z·⟨r⁻³⟩·α²`` seed.

Reference:
    M. Blume and R. E. Watson, Proc. Roy. Soc. London A270, 127 (1962);
    A271, 565 (1963).
    R. D. Cowan, *The Theory of Atomic Structure and Spectra* (1981), §7.5.
"""
from __future__ import annotations
from typing import List, Sequence
import math

import torch

from multitorch._constants import DTYPE, C_AU
from multitorch.angular.wigner import wigner3j, wigner6j


# α²/2 in Rydberg units. Matches Cowan's constant 2.6625666E-5 (rcn31.f:3446).
ALPHA_SQ_HALF: float = 0.5 / (float(C_AU) ** 2)


# ─────────────────────────────────────────────────────────────
# Low-level mesh helpers
# ─────────────────────────────────────────────────────────────


def _cum_trap(f: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Cumulative trapezoidal integral on a non-uniform mesh.

    Returns a tensor with the same shape as ``f`` whose i-th entry is
    ``∫_{r[0]}^{r[i]} f(r') dr'``. ``output[0] = 0`` by construction.
    """
    dr = r[1:] - r[:-1]
    slabs = 0.5 * (f[1:] + f[:-1]) * dr
    cum = torch.cat(
        [torch.zeros(1, dtype=f.dtype, device=f.device),
         torch.cumsum(slabs, dim=0)]
    )
    return cum


def _trap(f: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Scalar trapezoidal integral on a non-uniform mesh."""
    return torch.trapezoid(f, r)


def _mesh_derivative(P: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Centered second-order finite difference of ``P(r)`` on a non-uniform
    mesh. Endpoints copied from their nearest interior neighbour.

    Same scheme as ``hfs.py:706-708`` (used to differentiate ``V·r`` for the
    central-field ζ).
    """
    N = P.shape[0]
    dP = torch.zeros_like(P)
    h_l = r[1:-1] - r[:-2]
    h_r = r[2:] - r[1:-1]
    num = (P[2:] * h_l ** 2
           - P[:-2] * h_r ** 2
           + P[1:-1] * (h_r ** 2 - h_l ** 2))
    den = h_l * h_r * (h_l + h_r)
    dP[1:-1] = num / den
    dP[0] = dP[1]
    dP[-1] = dP[-2]
    return dP


# ─────────────────────────────────────────────────────────────
# SM, SN, VK building blocks
# ─────────────────────────────────────────────────────────────
#
# In the Fortran, ZK(M1,M2,M3,M4) sets
#
#     XJ(I) = P_{M1}(r) P_{M3}(r) · X1(r) / r^(K+3)
#
# where X1(r) = ∫_0^r r'^K · P_{M2}(r') P_{M4}(r') dr'  (the "inner
# integral"). QUAD5 then trapezoid-integrates XJ over the kept mesh.
#
# Substituting (M1,M2,M3,M4):
#
#     SM(M,MP) = ZK(M, MP, M,  MP)  → Pa = P_M·P_M ,  inner = P_MP·P_MP
#     SN(M,MP) = ZK(M, MP, MP, M)   → Pa = P_M·P_MP,  inner = P_MP·P_M
#
# both then multiplied by α²/2.
#
# VK(M,MP) is the difference of two derivative integrals
# (DYK(M,MP) − DYK(MP,M)) · α²/2 — see ``_dyk_integral``.


def compute_sm(
    P_M: torch.Tensor,
    P_MP: torch.Tensor,
    r: torch.Tensor,
    k: int,
    KKK: int | None = None,
) -> torch.Tensor:
    """Cowan ``SM(M, MP)`` at multipole rank ``k``.

    .. math::

        \\mathrm{SM}_k(M, M') = \\frac{\\alpha^2}{2}
            \\int_0^\\infty \\frac{P_M^2(r)}{r^{k+3}}\\,
            \\left[ \\int_0^r r'^{k}\\, P_{M'}^2(r')\\, dr' \\right] dr.

    Returns a 0-d torch tensor in Rydbergs.
    """
    if KKK is None:
        KKK = r.shape[0] - 1
    rr = r[: KKK + 1]
    pm = P_M[: KKK + 1]
    pmp = P_MP[: KKK + 1]

    inner_integrand = (rr ** k) * (pmp ** 2)
    inner = _cum_trap(inner_integrand, rr)         # X1(r) = ∫_0^r r'^k P_MP² dr'
    integrand = (pm ** 2) / (rr ** (k + 3)) * inner
    return ALPHA_SQ_HALF * _trap(integrand, rr)


def compute_sn(
    P_M: torch.Tensor,
    P_MP: torch.Tensor,
    r: torch.Tensor,
    k: int,
    KKK: int | None = None,
) -> torch.Tensor:
    """Cowan ``SN(M, MP)`` at multipole rank ``k``.

    .. math::

        \\mathrm{SN}_k(M, M') = \\frac{\\alpha^2}{2}
            \\int_0^\\infty \\frac{P_M(r) P_{M'}(r)}{r^{k+3}}\\,
            \\left[ \\int_0^r r'^{k}\\, P_M(r') P_{M'}(r')\\, dr' \\right] dr.

    The integrand is the same as :func:`compute_sm` but with the cross
    density ``P_M·P_MP`` substituted for ``P_M²`` (and ``P_MP·P_M`` for
    ``P_MP²``).
    """
    if KKK is None:
        KKK = r.shape[0] - 1
    rr = r[: KKK + 1]
    pm = P_M[: KKK + 1]
    pmp = P_MP[: KKK + 1]
    pmpm = pm * pmp

    inner_integrand = (rr ** k) * pmpm
    inner = _cum_trap(inner_integrand, rr)
    integrand = pmpm / (rr ** (k + 3)) * inner
    return ALPHA_SQ_HALF * _trap(integrand, rr)


def _dyk_integral(
    P_M: torch.Tensor,
    P_MP: torch.Tensor,
    r: torch.Tensor,
    k: int,
    KKK: int,
) -> torch.Tensor:
    """Direct port of Fortran ``DYK(M, MP)`` (rcn31.f:3558-3624).

    The Fortran builds, on the kept mesh,

        g(r) = P_M(r) · ( dP_{M'}/dr − P_{M'}(r)/r )

    then forms the two cumulative integrals
    ``X1(r) = ∫_0^r r'^k g(r') dr'`` and
    ``X2(r) = ∫_0^r g(r')/r'^(k+3) dr'``,
    and finally returns

        ∫_0^∞ P_M(r) P_{M'}(r)
              · [ X1(r)/r^(k+2) + r^(k+1)·(X2(∞) − X2(r)) ] dr.

    The Fortran implements the derivative as a per-block constant
    finite difference (``THM1·(P_{M'}[I+1] − P_{M'}[I-1]) − P_{M'}[I]/R[I]``)
    on the doubling-step mesh. We use a mesh-aware centered second-order
    finite difference instead, identical to the one used at
    :func:`multitorch.atomic.hfs.hfs_scf` for ``d(V·r)/dr``.

    Returns the bare integral (no ``α²/2`` prefactor).
    """
    rr = r[: KKK + 1]
    pm = P_M[: KKK + 1]
    pmp = P_MP[: KKK + 1]

    dP_MP_dr = _mesh_derivative(pmp, rr)
    g = pm * (dP_MP_dr - pmp / rr)

    f1 = (rr ** k) * g
    X1 = _cum_trap(f1, rr)

    f2 = g / (rr ** (k + 3))
    X2_cum = _cum_trap(f2, rr)
    X2_total = X2_cum[-1]

    pmpmp = pm * pmp
    integrand = pmpmp * (
        X1 / (rr ** (k + 2))
        + (rr ** (k + 1)) * (X2_total - X2_cum)
    )
    return _trap(integrand, rr)


def compute_vk(
    P_M: torch.Tensor,
    P_MP: torch.Tensor,
    r: torch.Tensor,
    k: int,
    KKK: int | None = None,
) -> torch.Tensor:
    """Cowan ``VK(M, MP)`` at multipole rank ``k``.

    Returns ``α²/2 × (DYK(M,MP) − DYK(MP,M))`` in Rydbergs (rcn31.f:3553).
    """
    if KKK is None:
        KKK = r.shape[0] - 1
    A1 = _dyk_integral(P_M, P_MP, r, k, KKK)
    A2 = _dyk_integral(P_MP, P_M, r, k, KKK)
    return ALPHA_SQ_HALF * (A1 - A2)


# ─────────────────────────────────────────────────────────────
# ⟨r⁻³⟩ for the seed term
# ─────────────────────────────────────────────────────────────


def compute_r_minus_3(
    P_M: torch.Tensor,
    r: torch.Tensor,
    KKK: int | None = None,
) -> torch.Tensor:
    """Hydrogenic ``⟨r⁻³⟩_M = ∫ P_M²(r) / r³ dr`` (atomic units).

    Used to construct the bare seed term ``Z·⟨r⁻³⟩·α²`` in
    :func:`zeta_blume_watson`.
    """
    if KKK is None:
        KKK = r.shape[0] - 1
    rr = r[: KKK + 1]
    pm = P_M[: KKK + 1]
    integrand = (pm ** 2) / (rr ** 3)
    return _trap(integrand, rr)


# ─────────────────────────────────────────────────────────────
# Top-level driver
# ─────────────────────────────────────────────────────────────


def zeta_blume_watson(
    orbitals: Sequence,
    m_idx: int,
    r: torch.Tensor,
    Z: float,
) -> torch.Tensor:
    """Compute the Blume-Watson spin-orbit parameter ζ for one orbital.

    This is a step-for-step port of ``rcn31.f::ZETABW(M, RM3)``
    (lines 3302-3421). All angular momentum quantities use the same
    integer (orbital) ``l`` values that the orbitals carry.

    Parameters
    ----------
    orbitals
        List of :class:`multitorch.atomic.hfs.OrbitalState`. Every entry
        with ``occ > 0`` and a populated ``P`` is treated as occupied;
        ``E > 0`` is treated as continuum and skipped (Fortran's
        ``EE(MP) > 0`` test, line 3331).
    m_idx
        Index into ``orbitals`` of the orbital whose ζ we want.
    r
        Radial mesh shared by all orbitals (the one returned by
        :func:`multitorch.atomic.hfs.build_cowan_mesh`).
    Z
        Nuclear charge — supplied explicitly because ``OrbitalState``
        does not carry it.

    Returns
    -------
    torch.Tensor
        0-d float64 tensor, ζ in Rydbergs.

    Notes
    -----
    For ``l = 0`` orbitals the result is identically zero (Fortran skips
    via ``IF (L(M).EQ.0) GO TO 400``).
    """
    orb_M = orbitals[m_idx]
    L_M = int(orb_M.l)
    if L_M == 0 or orb_M.P is None:
        return torch.zeros((), dtype=DTYPE, device=r.device)

    P_M = orb_M.P
    KKK_M = orb_M.KKK if orb_M.KKK > 0 else r.shape[0] - 1
    FLM = float(L_M)

    # Step 1 — bare central-field seed (rcn31.f:3326).
    # SP = Z · ⟨r⁻³⟩_M · α²
    rm3 = compute_r_minus_3(P_M, r, KKK_M)
    SP = Z * rm3 * (2.0 * ALPHA_SQ_HALF)            # α² = 2 × (α²/2)

    # Constants pulled out of the MP loop (lines 3324-3325).
    E2 = -12.0 * L_M - 6.0
    E3 = 3.0 * math.sqrt((2.0 * FLM + 1.0) / (FLM * (FLM + 1.0)))

    # ─────────────────────────────────────────────────────────
    # Step 2 — outer loop over occupied orbitals MP ≠ M
    # ─────────────────────────────────────────────────────────
    for mp_idx, orb_MP in enumerate(orbitals):
        if mp_idx == m_idx:
            continue
        if orb_MP.P is None or orb_MP.occ <= 0:
            continue
        if orb_MP.E > 0.0:                          # rcn31.f:3331 — continuum
            continue

        LP = int(orb_MP.l)
        FLP = float(LP)
        WP = float(orb_MP.occ)
        P_MP = orb_MP.P
        KKK = min(KKK_M, orb_MP.KKK if orb_MP.KKK > 0 else r.shape[0] - 1)

        # K=0 piece (rcn31.f:3336-3337):
        #   K is set to 0, then SP -= 2·WP·SM(M,MP) at K=0
        SP = SP - 2.0 * WP * compute_sm(P_M, P_MP, r, 0, KKK)

        # ─────────────────────────────────────────────────
        # Build SNKK[K+3] / VKK[K+3] tables (rcn31.f:3339-3353)
        # ─────────────────────────────────────────────────
        KN = abs(L_M - LP) - 2
        KX = L_M + LP

        # We index Python lists by K_loop directly for clarity.
        SNKK: dict[int, torch.Tensor] = {}          # K → SN(M,MP) at that K
        VKK: dict[int, torch.Tensor] = {}           # K → VK(M,MP) at that K

        INV = -1
        for K in range(KN, KX + 1):
            INV = -INV                              # toggles SN/VK branch
            if INV > 0:
                # SN branch
                if K < -1:
                    continue
                if K == KX and L_M == LP:
                    continue
                SNKK[K] = compute_sn(P_M, P_MP, r, K, KKK)
            else:
                # VK branch
                if K < 0:
                    continue
                VKK[K] = compute_vk(P_M, P_MP, r, K, KKK)

        # ─────────────────────────────────────────────────
        # SUM2 — even-parity SN contribution (rcn31.f:3357-3372)
        # K runs from KN+2 to KX-2 (inclusive)
        # ─────────────────────────────────────────────────
        KN_sum2 = KN + 2
        KXM2 = KX - 2
        if KXM2 >= KN_sum2:
            SUM2 = torch.zeros((), dtype=DTYPE, device=r.device)
            for K in range(KN_sum2, KXM2 + 1):
                if K not in SNKK:
                    continue                        # SNKK[K] = 0 (not stored)
                FK = float(K)
                ERAS = wigner6j(FK, 1.0, FK + 1.0, FLM, FLP, FLM) ** 2
                ERAS *= wigner3j(FLM, FK, FLP, 0.0, 0.0, 0.0) ** 2
                ERAS *= (2.0 * FK + 1.0) * (2.0 * FK + 3.0) / (FK + 2.0)
                SUM2 = SUM2 + ERAS * SNKK[K]
            SP = SP - E2 * WP * SUM2

        # ─────────────────────────────────────────────────
        # SUM13 — VK + cross-SN contribution (rcn31.f:3374-3389)
        # K runs from max(KN+2, 2) to KX
        # ─────────────────────────────────────────────────
        KN_sum13 = KN_sum2 if KN_sum2 != 0 else 2
        E13 = FLP * (FLP + 1.0) - FLM * (FLM + 1.0)
        SUM13 = torch.zeros((), dtype=DTYPE, device=r.device)
        for K in range(KN_sum13, KX + 1):
            FK = float(K)
            ERAS = math.sqrt(FK * (FK + 1.0) * (2.0 * FK + 1.0))
            ERAS *= wigner3j(FLM, FK, FLP, 0.0, 0.0, 0.0) ** 2
            ERAS *= wigner6j(FK, 1.0, FK, FLM, FLP, FLM)

            # Cross-SN coupling. SNKK[K-2] / SNKK[K] correspond to the
            # Fortran SNKK(K+1) / SNKK(K+3) entries (which are stored at
            # K_loop = K-2 and K_loop = K respectively).
            sn_lo = SNKK.get(K - 2, torch.zeros((), dtype=DTYPE, device=r.device))
            sn_hi = SNKK.get(K, torch.zeros((), dtype=DTYPE, device=r.device))
            ERAS1 = E13 * (sn_lo / FK - sn_hi / (FK + 1.0))

            # VKK[K-1] = Fortran VKK(K+2) (stored at K_loop = K-1).
            vk_val = VKK.get(K - 1, torch.zeros((), dtype=DTYPE, device=r.device))

            SUM13 = SUM13 + ERAS * (vk_val + ERAS1)

        SP = SP - E3 * WP * SUM13

    # ─────────────────────────────────────────────────────────
    # Step 3 — self-interaction correction for w_M ≥ 2
    # (rcn31.f:3393-3417)
    # ─────────────────────────────────────────────────────────
    W = float(orb_M.occ)
    if W >= 2.0:
        SM0 = compute_sm(P_M, P_M, r, 0, KKK_M)
        SP = SP - (2.0 * W - 3.0) * SM0

        if L_M >= 2:
            E_pre = 6.0 * ((2.0 * FLM + 1.0) ** 2)
            for K in range(2, 2 * L_M - 1 + 1):     # K = 2 .. 2L_M − 2 inclusive
                FK = float(K)
                FKP1 = FK + 1.0
                SUM = 0.0
                # KP runs over {K, K+2}
                for kp_offset in (0, 2):
                    FKP = FK + kp_offset
                    e = wigner6j(FKP1, 1.0, FKP, FLM, FLM, FLM) ** 2
                    e *= wigner3j(FLM, FKP, FLM, 0.0, 0.0, 0.0) ** 2
                    SUM += (FKP1 - FKP) * (2.0 * FKP + 1.0) * e
                SMK = compute_sm(P_M, P_M, r, K, KKK_M)
                SP = SP + E_pre * SUM * (2.0 * FK + 3.0) * SMK

    return SP
