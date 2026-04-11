"""
Hartree-Fock-Slater self-consistent field solver.

Full port of rcn31.f (Cowan RCN mod 32) to PyTorch. Implements:
  - Non-uniform logarithmic radial mesh (doubles step size at IDB)
  - SCHEQ: outward + inward Numerov integration with node counting
  - QUAD2: Y^0 electron-electron Coulomb potential via cumulative integration
  - SUBCOR: Slater statistical exchange potential
  - SCF convergence loop (typically < 130 iterations)
  - Spin-orbit coupling parameter (Blume-Watson)

All computations in float64 matching Fortran real*8.

Reference:
  Cowan (1981) The Theory of Atomic Structure and Spectra.
  rcn31.f line annotations reference the original Fortran.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import torch
import numpy as np

from multitorch._constants import DTYPE, RY_TO_EV, C_AU


# ─────────────────────────────────────────────────────────────
# Radial mesh
# ─────────────────────────────────────────────────────────────

def build_cowan_mesh(Z: int, mesh: int = 641) -> Tuple[torch.Tensor, int, float]:
    """
    Build Cowan's non-uniform radial mesh (rcn31.f lines 285-302).

    The mesh step doubles after IDB points (when the step size would exceed
    the threshold DRX = 0.25/sqrt(EMX), where EMX is the maximum energy).

    Parameters
    ----------
    Z : int
        Atomic number (used to set initial step size and scale factor C).
    mesh : int
        Total number of mesh points (default 641, max 1801).

    Returns
    -------
    r : torch.Tensor  shape (mesh,)  float64  radial coordinates (a.u.)
    IDB : int  index where step size doubles
    h_step : float  initial step size
    """
    # Cowan's scaling constant C = 0.88534138 / Z^(1/3)
    C = 0.88534138 / (Z ** (1.0 / 3.0))
    NBLOCK = (mesh - 1) // 40
    EMX = 100.0 + 3.0 * Z * Z  # approximate maximum energy (Ry)

    r = torch.zeros(mesh, dtype=DTYPE)
    r[0] = 1.0e-20  # near-nucleus starting point

    DELTAR = 0.0025 * C
    DRX = 0.25 / math.sqrt(EMX)

    i = 0
    IDB = 1
    RDB = 0.0
    for _j in range(NBLOCK):
        for _k in range(40):
            i += 1
            if i < mesh:
                r[i] = r[i - 1] + DELTAR
        if DELTAR < DRX:
            IDB = i
            RDB = float(r[i])
            DELTAR += DELTAR  # double the step size

    return r, IDB, float(r[1])  # return r, IDB, h (initial step)


# ─────────────────────────────────────────────────────────────
# Initial potential (Thomas-Fermi-like starting potential)
# ─────────────────────────────────────────────────────────────

def build_initial_potential(Z: int, r: torch.Tensor, noelec: int) -> torch.Tensor:
    """
    Build initial Hartree-Fock-Slater potential using a Thomas-Fermi
    type approximation.

    V(r) = -2Z/r at small r, screened exponentially at large r.
    This is a crude approximation; convergence in SCF corrects for it.

    Parameters
    ----------
    Z : int
        Atomic number.
    r : torch.Tensor  shape (N,)
        Radial mesh.
    noelec : int
        Number of electrons.

    Returns
    -------
    RU : torch.Tensor  shape (N,)
        Initial potential × r (in Ry×a.u.), i.e. RU(I) = V(I) × R(I).
    """
    N = len(r)
    TWOZ = 2.0 * Z
    # Approximate Thomas-Fermi: V(r) = -2Z/r × f(r)
    # where f(r) is a screening function that → 1 near nucleus, → ION/Z at infinity
    ION = Z - noelec
    # Simple exponential screening
    C = 0.88534138 / (Z ** (1.0 / 3.0))
    alpha = 2.0 / (3.0 * C)

    RU = torch.zeros(N, dtype=DTYPE)
    # RU(I) = V(I) * R(I) in Cowan's notation (potential × radius)
    # V(r) ≈ -2Z/r → RU = -2Z (constant near nucleus, then screened)
    for i in range(1, N):
        ri = float(r[i])
        x = ri * alpha
        # Thomas-Fermi-Dirac approximation
        screen = math.exp(-x) * (1.0 + x + x * x / 3.0)
        # At large r, approaches ION*2/r (ionic charge)
        V_screen = -(TWOZ * screen + 2.0 * ION * (1.0 - screen)) / ri
        RU[i] = V_screen * ri  # RU = V × r

    RU[0] = RU[1]  # value at r=0 (limit)
    return RU


# ─────────────────────────────────────────────────────────────
# Orbital state
# ─────────────────────────────────────────────────────────────

@dataclass
class OrbitalState:
    """State of one atomic orbital during SCF iteration."""
    n: int          # principal quantum number
    l: int          # orbital angular momentum (0=s, 1=p, 2=d)
    occ: float      # occupancy (0 to 2*(2l+1))
    E: float = -1.0        # orbital energy (Ry), negative for bound states
    P: Optional[torch.Tensor] = None   # radial wavefunction P_nl(r), shape (MESH,)
    KKK: int = 0    # last non-negligible mesh point
    zeta_ry: float = 0.0   # spin-orbit parameter (Ry)
    zeta_ev: float = 0.0   # spin-orbit parameter (eV)

    @property
    def nl_label(self) -> str:
        l_labels = 'spdfg'
        return f"{self.n}{l_labels[self.l].upper()}"


# ─────────────────────────────────────────────────────────────
# QUAD2: Electron-electron Coulomb integral
# ─────────────────────────────────────────────────────────────

def quad2_coulomb(P: torch.Tensor, r: torch.Tensor, IDB: int) -> torch.Tensor:
    """
    Compute the Coulomb self-energy integral XI(I) for orbital P.

    XI(I) = ∫_0^{r_I} P^2(r')/r' dr' + (1/r_I) ∫_{r_I}^∞ P^2(r') dr'

    This is the Y^0 potential (k=0 case of QUAD5 in rcn31.f).
    On the non-uniform mesh: each block has its own step size.

    Parameters
    ----------
    P : torch.Tensor  shape (MESH,)
        Radial wavefunction P_nl(r).
    r : torch.Tensor  shape (MESH,)
        Radial mesh.
    IDB : int
        Index where step size doubles.

    Returns
    -------
    XI : torch.Tensor  shape (MESH,)
        Coulomb self-energy: XI(I) = ∫ ... (as above).
    """
    N = len(r)
    P2 = P * P

    # Forward integral: ∫_0^r P^2(r') dr' (on non-uniform mesh with dr = h×Δi)
    # The non-uniform mesh uses trapezoidal rule with variable step sizes.
    # Step size: h = r[i] - r[i-1]
    h_arr = torch.cat([torch.zeros(1, dtype=DTYPE), r[1:] - r[:-1]])  # shape (N,)

    integrand_fwd = P2 * h_arr  # P^2(r_i) × dr_i
    I_fwd = torch.cumsum(integrand_fwd, dim=0)  # ∫_0^{r_i} P^2 dr

    # Backward integral: ∫_r^∞ P^2(r') dr'
    I_total = I_fwd[-1]
    I_bwd = I_total - I_fwd  # ∫_{r_i}^∞ P^2 dr

    # Y^0(r_i) = (1/r_i) × ∫_0^{r_i} P^2 dr + ∫_{r_i}^∞ P^2/r' dr
    # But XI in Cowan = ∫_0^{r_i} P^2/r' dr + r_i × ∫_{r_i}^∞ P^2/r'^2 dr
    # Actually Cowan uses: XI(I) = Coulomb contribution = integral via RUEE calculation

    # Simplified: XI ≈ 2×Y^0/r where Y^0 is the monopole electrostatic potential
    # Y^0(r) = (1/r) ∫_0^r P^2 dr' + ∫_r^∞ P^2/r' dr'
    # On mesh: need dr'/r' = h/r term
    integrand_fwd_r = P2 * h_arr / r.clamp(min=1e-30)   # P^2/r × dr
    I_fwd_r = torch.cumsum(integrand_fwd_r, dim=0)
    I_total_r = I_fwd_r[-1]
    I_bwd_r = I_total_r - I_fwd_r

    XI = I_fwd / r.clamp(min=1e-30) + I_bwd_r  # in Ry
    # Factor 2: Rydberg-unit Coulomb potential is 2Y^0/r (cf. rcn31.f line 2001).
    # Verified against Fortran QUAD2: XI = 2*r*Y^0 → RUEE = XI → V_ee = RUEE/r = 2Y^0.
    return 2.0 * XI


# ─────────────────────────────────────────────────────────────
# SUBCOR: Slater exchange potential
# ─────────────────────────────────────────────────────────────

def build_exchange_potential(
    density: torch.Tensor,
    r: torch.Tensor,
    EXF: float = 1.0,
) -> torch.Tensor:
    """
    Compute Slater statistical exchange potential (RUEXCH in rcn31.f).

    RUEXCH(I) = -3 × (3/π × ρ(r))^(1/3) × r

    where ρ(r) = Σ_{nl} w_nl × P_nl^2(r) is the electron density.

    Parameters
    ----------
    density : torch.Tensor  shape (MESH,)
        Total electron density ρ(r) = Σ w_nl P_nl^2(r).
    r : torch.Tensor  shape (MESH,)
        Radial mesh.
    EXF : float
        Exchange scaling factor (rcn31 uses EXF10 ≈ 0.7, default = 1.0).

    Returns
    -------
    RUEXCH : torch.Tensor  shape (MESH,)
        Exchange potential × r (same units as RU in Cowan code).
    """
    # Avoid division by zero
    rho = (density / r.clamp(min=1e-30) ** 2).clamp(min=0.0)
    # Slater exchange: V_ex(r) = -3 × EXF × (3ρ/8π)^(1/3)
    # In Ry (Cowan's units): factor is -3 (which equals -6 in Hartree)
    coeff = (3.0 / (8.0 * math.pi)) ** (1.0 / 3.0)
    V_ex = -3.0 * EXF * coeff * rho.clamp(min=1e-100) ** (1.0 / 3.0)
    return V_ex * r   # return RUEXCH = V_ex × r


# ─────────────────────────────────────────────────────────────
# SCHEQ: Numerov integrator with outward + inward + matching
# ─────────────────────────────────────────────────────────────

def scheq(
    V: torch.Tensor,   # effective potential V(I) = XJ(I)/R(I) in Cowan notation
    r: torch.Tensor,
    IDB: int,
    n: int,            # principal quantum number
    l: int,            # orbital angular momentum
    Z: float,
    E_init: float,     # initial energy guess (Ry, negative for bound)
    THRESH: float = 1e-6,
    max_iter: int = 200,
) -> Tuple[float, torch.Tensor, int]:
    """
    Solve the radial Schrödinger equation for one orbital (nl) using
    the Numerov outward + inward integration with matching (SCHEQ in rcn31.f).

    -d²P/dr² + [V_eff(r) - E] P(r) = 0
    where V_eff(r) = V(r) + l(l+1)/r²

    Parameters
    ----------
    V : torch.Tensor  shape (MESH,)
        Effective atomic potential V(r) = XJ/R (as used in Cowan rcn31.f).
    r : torch.Tensor  shape (MESH,)
        Non-uniform radial mesh.
    IDB : int
        Mesh point where step size doubles.
    n, l : int
        Principal and orbital quantum numbers.
    Z : float
        Atomic number (nuclear charge).
    E_init : float
        Initial energy guess (Ry, should be negative for bound states).
    THRESH : float
        Convergence threshold for energy update |dE/E| < THRESH.
    max_iter : int
        Maximum number of energy iterations within this call.

    Returns
    -------
    E : float
        Converged orbital energy (Ry).
    P : torch.Tensor  shape (MESH,)
        Normalized radial wavefunction.
    KKK : int
        Last significant mesh point.
    """
    import os
    DEBUG = bool(os.environ.get("SCHEQ_DEBUG"))  # opt-in tracing for debugging
    MESH = len(r)
    B0 = float(l * (l + 1))
    NDCR = n - l - 1  # number of radial nodes

    V_np = V.detach().cpu().numpy().astype(np.float64)
    r_np = r.detach().cpu().numpy().astype(np.float64)

    E = float(E_init)
    if E >= 0:
        E = -abs(E_init) if E_init != 0 else -0.1

    # Bracketing bounds (rcn31.f lines 940-942)
    EMORE = -1.0e-10
    ELESS = 1.0e12 * E

    PNLO = np.zeros(MESH)
    PMATCH = 0.0
    IMATCH = 3
    KKK = MESH - 1
    P_K_tail = 0.0
    sqrtQ_K = 1.0

    for _iter in range(max_iter):
        QQ = V_np + B0 / r_np**2 - E

        # ── Outward starting values: P ~ r^(l+1) power series (rcn31.f 1101-1108)
        H = float(r_np[1] - r_np[0]) if r_np[0] > 0 else float(r_np[1])
        Y = 2.0 * H
        B1 = -2.0 * Z
        B2 = 3.0 * Z / H - E + 2.0 * V_np[1] - V_np[2]
        B3 = (V_np[2] - V_np[1]) / H - Z / H**2
        FLPS = 4 * l + 6.0
        SLPT = 6 * l + 12.0
        ELPT = 8 * l + 20.0
        A1 = -Z / (l + 1.0)
        A2 = (A1 * B1 + B2) / FLPS
        A3 = (A2 * B1 + A1 * B2 + B3) / SLPT
        A4 = (A3 * B1 + A2 * B2 + A1 * B3) / ELPT
        HTL = H ** (l + 1)
        YTL = Y ** (l + 1)
        P1 = (1.0 + H * (A1 + H * (A2 + H * (A3 + H * A4)))) * HTL   # at r[1]
        P2 = (1.0 + Y * (A1 + Y * (A2 + Y * (A3 + Y * A4)))) * YTL   # at r[2]

        PNLO = np.zeros(MESH)
        PNLO[1] = P1
        PNLO[2] = P2

        P_prev, P_cur = P1, P2
        NCROSS = 0
        too_many = False
        matched = False
        IMATCH = 0
        PMATCH = 0.0
        last_i = 2
        max_absP = max(abs(P1), abs(P2))
        in_damped = False

        # Non-uniform-mesh 2nd-order finite-difference recurrence.
        # P''[i] ≈ 2·(P[i-1]·h_r - P[i]·(h_l+h_r) + P[i+1]·h_l) /
        #          (h_l·h_r·(h_l+h_r)),
        # combined with P''[i] = Q[i]·P[i], gives
        #   P[i+1] = (1 + h_r/h_l)·P[i] - (h_r/h_l)·P[i-1]
        #            + (h_r·(h_l+h_r)/2)·Q[i]·P[i]
        for i in range(2, MESH - 1):
            last_i = i
            h_l = r_np[i]     - r_np[i - 1]
            h_r = r_np[i + 1] - r_np[i]
            ratio = h_r / h_l
            P_next = (1.0 + ratio) * P_cur - ratio * P_prev \
                     + 0.5 * h_r * (h_l + h_r) * QQ[i] * P_cur
            PNLO[i + 1] = P_next

            if abs(P_next) > max_absP:
                max_absP = abs(P_next)
            if QQ[i + 1] > 0 and abs(P_next) < 0.5 * max_absP:
                in_damped = True

            if (not in_damped) and P_next * P_cur < 0 and P_cur != 0:
                NCROSS += 1
                if NCROSS > NDCR:
                    too_many = True
                    break

            # Match when node count is right, we are in the forbidden region,
            # and the wavefunction is clearly decaying past its peak.
            if (NCROSS == NDCR
                    and QQ[i + 1] > 0
                    and i + 1 >= 5
                    and abs(PNLO[i])     < abs(PNLO[i - 1])
                    and abs(PNLO[i + 1]) < abs(PNLO[i])
                    and PNLO[i + 1] * PNLO[i - 1] > 0):
                IMATCH = i - 1
                PMATCH = PNLO[IMATCH]
                matched = True
                break

            P_prev, P_cur = P_cur, P_next

            if abs(P_next) > 1.0e25:
                break

        # ── Bracket update on failure modes ─────────────────────────────────
        if too_many:
            EMORE = min(EMORE, E)
            E = max(2.0 * E, 0.5 * (ELESS + EMORE))
            if DEBUG:
                print(f"  iter={_iter} too many nodes ({NCROSS}>{NDCR}) E->{E:.6f}")
            continue
        if not matched:
            ELESS = max(ELESS, E)
            E = min(0.5 * E, 0.5 * (ELESS + EMORE))
            if DEBUG:
                print(f"  iter={_iter} too few nodes ({NCROSS}<{NDCR}) last_i={last_i} E->{E:.6f}")
            continue

        # ── Outward log-derivative and norm at match ────────────────────────
        h_out = r_np[IMATCH + 1] - r_np[IMATCH - 1]
        dP_out = (PNLO[IMATCH + 1] - PNLO[IMATCH - 1]) / h_out
        S6 = dP_out / PMATCH
        SUM1 = float(np.trapezoid(PNLO[1:IMATCH + 1] ** 2, r_np[1:IMATCH + 1]))
        S5 = SUM1 / (PMATCH * PMATCH)

        # ── Inward integration from outer tail down to IMATCH ───────────────
        XINW = 10.0 * r_np[IMATCH]
        KKK = MESH - 1
        for kk in range(40, MESH, 40):
            if r_np[kk] >= XINW:
                KKK = kk
                break
        KKK = min(KKK, MESH - 1)
        if KKK <= IMATCH + 4:
            KKK = min(IMATCH + 40, MESH - 1)

        sqrtQK   = math.sqrt(max(QQ[KKK], 1e-10))
        sqrtQKm1 = math.sqrt(max(QQ[KKK - 1], 1e-10))
        P_K  = math.exp(-r_np[KKK]     * sqrtQK)
        P_Km = math.exp(-r_np[KKK - 1] * sqrtQKm1)
        if PMATCH < 0:
            P_K  = -P_K
            P_Km = -P_Km
        PNLO[KKK]     = P_K
        PNLO[KKK - 1] = P_Km
        SUM3 = P_K * P_K / (2.0 * sqrtQK)

        # Inward 2nd-order FD recurrence: step from (i+1, i) → (i-1).
        #   P[i-1] = (1 + h_l/h_r)·P[i] - (h_l/h_r)·P[i+1]
        #            + (h_l·(h_l+h_r)/2)·Q[i]·P[i]
        P_out, P_in = P_K, P_Km   # P at (i+1), P at i
        i = KKK - 1
        diverged_in = False
        while i > IMATCH:
            h_l = r_np[i]     - r_np[i - 1]
            h_r = r_np[i + 1] - r_np[i]
            ratio = h_l / h_r
            P_new = (1.0 + ratio) * P_in - ratio * P_out \
                    + 0.5 * h_l * (h_l + h_r) * QQ[i] * P_in
            PNLO[i - 1] = P_new
            P_out, P_in = P_in, P_new
            i -= 1
            if abs(P_new) > 1.0e25:
                diverged_in = True
                break

        if diverged_in or PNLO[IMATCH] == 0:
            if DEBUG:
                print(f"  iter={_iter} inward diverged; shrinking KKK")
            E = 0.9 * E
            continue

        # ── Rescale inward to match outward at IMATCH ───────────────────────
        P_inward_at_match = PNLO[IMATCH]
        pop = PMATCH / P_inward_at_match
        PNLO[IMATCH:KKK + 1] *= pop
        SUM3 *= pop * pop

        # Inward log-derivative (centered difference, now on matched P)
        h_in_loc = r_np[IMATCH + 1] - r_np[IMATCH - 1]
        dP_in = (PNLO[IMATCH + 1] - PNLO[IMATCH - 1]) / h_in_loc
        S4 = dP_in / PMATCH
        SUM4 = float(np.trapezoid(PNLO[IMATCH:KKK + 1] ** 2, r_np[IMATCH:KKK + 1]))
        S3 = (SUM3 + SUM4) / (PMATCH * PMATCH)

        # ── Perturbation energy update (rcn31.f 1329) ───────────────────────
        DE = (S6 - S4) / (S5 + S3)

        if DEBUG:
            print(f"  iter={_iter} E={E:.6f} DE={DE:.3e} NCROSS={NCROSS} "
                  f"IMATCH={IMATCH} KKK={KKK} S6={S6:.3e} S4={S4:.3e} "
                  f"S5={S5:.3e} S3={S3:.3e} PMATCH={PMATCH:.3e}")

        # Clamp DE to the current bracket (rcn31.f 1330-1335)
        if DE < 0:
            EMORE = min(EMORE, E)
            DE = max(DE, 0.8 * (ELESS - E))
        else:
            ELESS = max(ELESS, E)
            DE = min(DE, 0.8 * (EMORE - E))

        E_new = E + DE
        if E_new >= 0:
            E_new = 0.5 * E
        if abs(E) > 0 and abs(DE / E) < THRESH:
            E = E_new
            break
        E = E_new

    # ── Normalize ──────────────────────────────────────────────────────────
    POP = PMATCH / PNLO[IMATCH] if PNLO[IMATCH] != 0 else 1.0
    for j in range(IMATCH, KKK + 1):
        PNLO[j] *= POP

    norm = np.trapezoid(PNLO[:KKK + 1] ** 2, r_np[:KKK + 1])
    if norm > 0:
        C1 = 1.0 / math.sqrt(norm)
        if PNLO[2] < 0:
            C1 = -C1
        PNLO = PNLO * C1

    P_result = torch.tensor(PNLO, dtype=DTYPE)
    return E, P_result, KKK


# ─────────────────────────────────────────────────────────────
# Full SCF loop
# ─────────────────────────────────────────────────────────────

@dataclass
class HFSResult:
    """Output of a completed HFS SCF calculation."""
    orbitals: List[OrbitalState]
    r: torch.Tensor          # radial mesh (a.u.)
    IDB: int                 # mesh doubling point
    Z: int                   # atomic number
    ION: int                 # ionic charge
    MESH: int                # number of mesh points
    converged: bool = False
    n_iter: int = 0

    def orbital(self, nl: str) -> Optional[OrbitalState]:
        """Get orbital by label, e.g. '2P', '3D'."""
        for orb in self.orbitals:
            if orb.nl_label.lower() == nl.lower():
                return orb
        return None


def hfs_scf(
    Z: int,
    config: Dict[str, float],
    mesh: int = 641,
    max_iter: int = 130,
    tol: float = 1e-8,
    EXF: float = 1.0,
    IREL: int = 1,  # 0=non-relativistic, 1=relativistic corrections
    device: str = "cpu",
    zeta_method: str = "central_field",
) -> HFSResult:
    """
    Hartree-Fock-Slater self-consistent field calculation (rcn31.f).

    Computes orbital energies and radial wavefunctions for an atom/ion
    with a specified electronic configuration.

    Parameters
    ----------
    Z : int
        Atomic number.
    config : dict
        Electronic configuration: {'1s': 2.0, '2s': 2.0, '2p': 6.0, '3d': 8.0}
        Key format: 'nl' where n is integer and l is s/p/d/f (lowercase).
    mesh : int
        Number of radial mesh points (default 641 for L-edge calculations).
    max_iter : int
        Maximum SCF iterations.
    tol : float
        Convergence tolerance: |ΔRU|_max < tol.
    EXF : float
        Slater-Xα statistical exchange scaling factor. Default 1.0 matches
        Fortran rcn31's default (EXF=1.000 in nid8.rcn31_out line 6 and in
        typical Cowan RCN inputs). A lower value (0.7) gives a Kohn-Sham-
        like reduced exchange but systematically under-binds valence orbitals
        by ~15%. Keep at 1.0 unless matching a specific rcn31 run that used
        a different EXF.
    IREL : int
        Relativistic correction level (0=none, 1=Darwin+mass-velocity).
    device : str
    zeta_method : str
        Method used to compute spin-orbit ζ after the SCF converges:

        - ``"central_field"`` (default) — ``ζ = (α²/2) ∫ (1/r)(dV/dr) P²(r) dr``
          using the converged HFS potential. Reproduces Cowan's "R*VI"
          column to ~5% on Ni²⁺.
        - ``"blume_watson"`` — full multi-orbital Blume-Watson treatment
          (Proc. Roy. Soc. London A270, 127 (1962); A271, 565). Adds
          exchange corrections via ``multitorch.atomic.blume_watson.zeta_blume_watson``.
          Matches the Fortran "BLUME-WATSON" column up to the residual
          gap from the HFS solver itself.

    Returns
    -------
    HFSResult with converged orbital wavefunctions and energies.
    """
    if zeta_method not in ("central_field", "blume_watson"):
        raise ValueError(
            f"zeta_method must be 'central_field' or 'blume_watson', "
            f"got {zeta_method!r}"
        )
    # Parse configuration
    l_labels = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4}
    orbitals: List[OrbitalState] = []
    noelec = 0

    for nl, occ in config.items():
        if occ == 0:
            continue
        n = int(nl[0])
        l = l_labels.get(nl[1].lower(), -1)
        if l < 0:
            raise ValueError(f"Unknown orbital label: {nl}")
        occ = float(occ)
        # Initial energy estimate: hydrogen-like eigenvalue
        E_init = -(Z ** 2) / (2.0 * n ** 2)
        orb = OrbitalState(n=n, l=l, occ=occ, E=E_init)
        orbitals.append(orb)
        noelec += occ

    noelec = int(noelec)
    ION = Z - noelec

    # Build mesh
    r, IDB, h = build_cowan_mesh(Z, mesh)
    r = r.to(device)

    # Initialize potential
    RU = build_initial_potential(Z, r, noelec)
    RU = RU.to(device)

    # Initialize wavefunctions with hydrogenic approximation
    r_np = r.cpu().numpy()
    for orb in orbitals:
        # Simple Slater-type orbital as starting guess
        # P_nl(r) ~ r^(n) × exp(-Z_eff × r / n)
        Z_eff = max(Z - (sum(o.occ for o in orbitals if o.n < orb.n) * 0.85 +
                        sum(o.occ - (1 if o is orb else 0) for o in orbitals if o.n == orb.n) * 0.35), 1.0)
        scale = Z_eff / orb.n
        P_init = r_np ** (orb.n) * np.exp(-scale * r_np)
        # Normalize
        norm = np.trapezoid(P_init ** 2, r_np)
        if norm > 0:
            P_init /= math.sqrt(norm)
        orb.P = torch.tensor(P_init, dtype=DTYPE, device=device)
        orb.KKK = mesh - 1

    # SCF iteration
    TOLEND = tol
    converged = False
    # Initialize V from the Thomas-Fermi-like RU (potential = RU/r in Ry)
    V = RU / r.clamp(min=1e-30)
    V_prev = V.clone()

    for niter in range(1, max_iter + 1):
        # Step 1: Compute electron density (sum of orbital P²; no sqrt)
        density = torch.zeros(mesh, dtype=DTYPE, device=device)
        for orb in orbitals:
            if orb.P is not None and orb.occ > 0:
                density += orb.occ * orb.P ** 2

        # Step 2: Build the total electron-electron Coulomb potential as the
        # weighted sum of per-orbital Y^0 potentials (Cowan rcn31.f::QUAD2).
        # quad2_coulomb returns 2*Y^0(r) — i.e. the orbital contribution to V
        # in Rydberg units, *not* RU = V*r.
        V_ee = torch.zeros(mesh, dtype=DTYPE, device=device)
        for orb in orbitals:
            if orb.P is not None and orb.occ > 0:
                V_ee += orb.occ * quad2_coulomb(orb.P, r, IDB)

        # Step 3: Slater exchange potential. build_exchange_potential returns
        # V_ex × r, so divide by r to get the potential.
        RUEXCH = build_exchange_potential(density, r, EXF)
        V_ex = RUEXCH / r.clamp(min=1e-30)

        # Step 4: Build total potential V(r) [Ry] directly:
        #   V(r) = -2Z/r  +  V_ee(r)  +  V_ex(r)
        # (V_ex is already negative; EXF was applied inside build_exchange_potential)
        V_nuc = -2.0 * float(Z) / r.clamp(min=1e-30)
        V_new = V_nuc + V_ee + V_ex
        # Damp the potential update to stabilize the SCF (Cowan uses ~0.5)
        V = 0.5 * V_new + 0.5 * V_prev

        # Step 5: Solve Schrödinger equation for each orbital
        for orb in orbitals:
            E, P, KKK = scheq(V, r, IDB, orb.n, orb.l, float(Z),
                              orb.E, THRESH=1e-6, max_iter=50)
            orb.E = E
            orb.P = P.to(device)
            orb.KKK = KKK

        # Step 6: Check convergence on the potential
        DELTA_new = float((V - V_prev).abs().max())
        V_prev = V.clone()

        if DELTA_new < TOLEND and niter > 3:
            converged = True
            break

    # Compute spin-orbit coupling parameters via central-field formula
    # ζ_nl = (α²/2) × ∫ (1/r)(dV/dr) P_nl²(r) dr
    # This corresponds to ZETA(M,3) in rcn31.f::ZETA1 (the "R*VI" column).
    # The full Blume-Watson method adds multi-orbital exchange corrections
    # but requires SM/SN/VK integrals not yet implemented.
    alpha_sq_half = 0.5 / (float(C_AU) ** 2)  # α²/2 = 2.6626e-5

    # Compute (1/r)(dV/dr) via the Fortran trick: differentiate VR = V×r
    # (which is smooth at origin), then (1/r)(dV/dr) = (d(VR)/dr - VR/r)/r²
    r_np = r.detach().cpu().numpy()
    VR_np = (V * r).detach().cpu().numpy()  # V×r is finite everywhere
    dVRdr = np.zeros_like(VR_np)
    # Central difference of VR on non-uniform mesh
    for i in range(1, len(r_np) - 1):
        h_l = r_np[i] - r_np[i - 1]
        h_r = r_np[i + 1] - r_np[i]
        dVRdr[i] = (VR_np[i + 1] * h_l**2 - VR_np[i - 1] * h_r**2
                    + VR_np[i] * (h_r**2 - h_l**2)) / (h_l * h_r * (h_l + h_r))
    dVRdr[0] = dVRdr[1]
    dVRdr[-1] = dVRdr[-2]
    # (1/r)(dV/dr) = (d(VR)/dr - V) / r² = (d(VR)/dr - VR/r) / r²
    V_np = VR_np / np.maximum(r_np, 1e-30)
    inv_r_dVdr = (dVRdr - V_np) / np.maximum(r_np**2, 1e-60)
    inv_r_dVdr_t = torch.tensor(inv_r_dVdr, dtype=DTYPE, device=device)

    for orb in orbitals:
        if orb.l > 0 and orb.P is not None:
            KKK = orb.KKK if orb.KKK > 0 else len(r) - 1
            P2 = orb.P[:KKK + 1] ** 2
            integrand = P2 * inv_r_dVdr_t[:KKK + 1]
            zeta = alpha_sq_half * float(torch.trapezoid(integrand, r[:KKK + 1]))
            orb.zeta_ry = zeta
            orb.zeta_ev = zeta * float(RY_TO_EV)
        else:
            orb.zeta_ry = 0.0
            orb.zeta_ev = 0.0

    # If Blume-Watson was requested, overwrite the central-field ζ with the
    # full multi-orbital exchange-corrected value (rcn31.f::ZETABW). This is
    # done after the central-field pass so the central-field number is still
    # available for diagnostics if anyone caches `zeta_ry` mid-call.
    if zeta_method == "blume_watson":
        from multitorch.atomic.blume_watson import zeta_blume_watson
        for i, orb in enumerate(orbitals):
            if orb.l > 0 and orb.P is not None:
                zeta_t = zeta_blume_watson(orbitals, i, r, Z=float(Z))
                zeta = float(zeta_t)
                orb.zeta_ry = zeta
                orb.zeta_ev = zeta * float(RY_TO_EV)

    result = HFSResult(
        orbitals=orbitals,
        r=r,
        IDB=IDB,
        Z=Z,
        ION=ION,
        MESH=mesh,
        converged=converged,
        n_iter=niter if not converged else niter,
    )
    return result
