"""
O(3) → Oh subduction machinery for the Fortran-free RAC generator.

Computes:
  1. The 24 proper rotations of the octahedral group O as 3×3 matrices
  2. Wigner D-matrices D^J(R) for integer J and any rotation R
  3. Subduction matrices: D^J → direct sum of Oh irreps
  4. Isoscalar factors (O3→Oh 3jm symbols) for tensor operator coupling

These are the coupling coefficients that appear as ADD entry ``coeff``
values in the .rme_rac file. They describe how angular-momentum-resolved
(J_bra, J_ket) operator blocks tile into point-group-resolved blocks.

Convention
----------
Oh irreps in Butler notation (matching ttrac output):
  0+ = A₁g (dim 1)    ^0+ = A₂g (dim 1)
  1+ = T₁g (dim 3)    ^1+ = T₂g (dim 3)
  2+ = Eᵍ  (dim 2)
  0- = A₁u (dim 1)    ^0- = A₂u (dim 1)
  1- = T₁u (dim 3)    ^1- = T₂u (dim 3)
  2- = Eᵤ  (dim 2)

Parity: integer J → gerade (+), half-integer → ungerade (−).
For L-edge XAS, ground states have even parity (+) and excited states
(with a core hole) have odd parity (−).

Reference
---------
Butler, P. H. (1981). Point Group Symmetry Applications.
Sugano, Tanabe, Kamimura (1970). Multiplets of Transition-Metal Ions.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np

from multitorch.angular.wigner import clebsch_gordan


# ─────────────────────────────────────────────────────────────
# Oh irrep definitions
# ─────────────────────────────────────────────────────────────

# Butler label → (dimension, index for internal arrays)
# Ordering: A1, A2, E, T1, T2 (the 5 irreps of O)
OH_IRREPS = {
    'A1': 0, 'A2': 1, 'E': 2, 'T1': 3, 'T2': 4,
}

OH_IRREP_DIM = {'A1': 1, 'A2': 1, 'E': 2, 'T1': 3, 'T2': 3}

# Double-group (O') extra irreps for half-integer J
OH_DOUBLE_IRREP_DIM = {'E1/2': 2, 'E5/2': 2, 'G3/2': 4}

# Combined dimension lookup (single + double group)
OH_IRREP_DIM_ALL = {**OH_IRREP_DIM, **OH_DOUBLE_IRREP_DIM}

# Butler notation mapping
BUTLER_LABEL = {
    'A1': '0', 'A2': '^0', 'E': '2', 'T1': '1', 'T2': '^1',
}

# Butler labels for double-group irreps
BUTLER_LABEL_DOUBLE = {
    'E1/2': '1/2', 'E5/2': '^1/2', 'G3/2': '3/2',
}

BUTLER_LABEL_ALL = {**BUTLER_LABEL, **BUTLER_LABEL_DOUBLE}

# Reverse: Butler label → irrep name
BUTLER_TO_IRREP = {v: k for k, v in BUTLER_LABEL.items()}
BUTLER_TO_IRREP_ALL = {v: k for k, v in BUTLER_LABEL_ALL.items()}


def butler_label(irrep: str, parity: str) -> str:
    """Convert irrep name + parity to Butler notation (e.g., 'T1', '+' → '1+')."""
    return BUTLER_LABEL_ALL[irrep] + parity


def irrep_from_butler(label: str) -> Tuple[str, str]:
    """Parse Butler label into (irrep_name, parity). E.g., '^1+' → ('T2', '+')."""
    parity = label[-1]
    base = label[:-1]
    return BUTLER_TO_IRREP_ALL[base], parity


# ─────────────────────────────────────────────────────────────
# The 24 proper rotations of the octahedral group O
# ─────────────────────────────────────────────────────────────

def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix via Rodrigues' formula. Axis must be unit vector."""
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    x, y, z = axis
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
    ], dtype=np.float64)


def octahedral_rotations() -> List[np.ndarray]:
    """Return the 24 proper rotations of the octahedral group O.

    Returns
    -------
    list of 24 ndarray, each shape (3, 3)
        Rotation matrices in the order:
        E, 8×C₃, 3×C₂(face), 6×C₄, 6×C₂'(edge)
    """
    rotations = []

    # Identity
    rotations.append(np.eye(3, dtype=np.float64))

    # 8 C₃ rotations: 4 body diagonals × ±2π/3
    diagonals = [
        np.array([1, 1, 1], dtype=np.float64),
        np.array([1, -1, -1], dtype=np.float64),
        np.array([-1, 1, -1], dtype=np.float64),
        np.array([-1, -1, 1], dtype=np.float64),
    ]
    for d in diagonals:
        axis = d / np.linalg.norm(d)
        rotations.append(_rotation_matrix(axis, 2 * math.pi / 3))
        rotations.append(_rotation_matrix(axis, -2 * math.pi / 3))

    # 3 C₂ rotations: face axes (= C₄²)
    face_axes = [
        np.array([1, 0, 0], dtype=np.float64),
        np.array([0, 1, 0], dtype=np.float64),
        np.array([0, 0, 1], dtype=np.float64),
    ]
    for axis in face_axes:
        rotations.append(_rotation_matrix(axis, math.pi))

    # 6 C₄ rotations: face axes × ±π/2
    for axis in face_axes:
        rotations.append(_rotation_matrix(axis, math.pi / 2))
        rotations.append(_rotation_matrix(axis, -math.pi / 2))

    # 6 C₂' rotations: edge axes
    edge_axes = [
        np.array([1, 1, 0], dtype=np.float64),
        np.array([1, -1, 0], dtype=np.float64),
        np.array([1, 0, 1], dtype=np.float64),
        np.array([1, 0, -1], dtype=np.float64),
        np.array([0, 1, 1], dtype=np.float64),
        np.array([0, 1, -1], dtype=np.float64),
    ]
    for e in edge_axes:
        axis = e / np.linalg.norm(e)
        rotations.append(_rotation_matrix(axis, math.pi))

    assert len(rotations) == 24, f"Expected 24 rotations, got {len(rotations)}"
    return rotations


# ─────────────────────────────────────────────────────────────
# Wigner D-matrix for integer J
# ─────────────────────────────────────────────────────────────

def _euler_angles_from_rotation(R: np.ndarray) -> Tuple[float, float, float]:
    """Extract ZYZ Euler angles (α, β, γ) from a 3×3 rotation matrix.

    Convention: R = Rz(α) Ry(β) Rz(γ) (active rotation).
    """
    # β from R[2,2] = cos(β)
    cos_beta = np.clip(R[2, 2], -1.0, 1.0)
    beta = math.acos(cos_beta)

    if abs(math.sin(beta)) > 1e-10:
        alpha = math.atan2(R[1, 2], R[0, 2])
        gamma = math.atan2(R[2, 1], -R[2, 0])
    else:
        # β ≈ 0 or π: gimbal lock
        if cos_beta > 0:  # β ≈ 0
            alpha = math.atan2(-R[0, 1], R[0, 0])
            gamma = 0.0
        else:  # β ≈ π
            alpha = math.atan2(R[0, 1], -R[0, 0])
            gamma = 0.0

    return alpha, beta, gamma


def _small_d(J, m, mp, beta: float) -> float:
    """Small Wigner d-matrix element d^J_{m,m'}(β).

    Works for both integer and half-integer J (and m, mp).
    Uses the explicit sum formula (Edmonds, eq. 4.1.15).
    All factorial arguments (J±m, J±mp) are guaranteed to be non-negative
    integers even when J, m, mp are half-integer.
    """
    cb2 = math.cos(beta / 2.0)
    sb2 = math.sin(beta / 2.0)

    # Convert to int for factorial (guaranteed integer for valid J,m,mp)
    Jm = int(round(J + m))
    Jmm = int(round(J - m))
    Jmp = int(round(J + mp))
    Jmmp = int(round(J - mp))
    m_mp = int(round(m - mp))

    # Sum over s where all factorials are non-negative
    s_min = max(0, m_mp)
    s_max = min(Jm, Jmmp)

    val = 0.0
    for s in range(s_min, s_max + 1):
        num = math.factorial(Jm) * math.factorial(Jmm)
        num *= math.factorial(Jmp) * math.factorial(Jmmp)
        num = math.sqrt(num)

        denom = (math.factorial(Jm - s) * math.factorial(s)
                 * math.factorial(s - m_mp) * math.factorial(Jmmp - s))

        power_c = int(round(2 * J + m - mp - 2 * s))
        power_s = int(round(mp - m + 2 * s))
        sign = (-1) ** (int(round(mp - m)) + s)

        # Handle edge cases where base is 0 and power is 0
        c_term = cb2 ** power_c if power_c > 0 or abs(cb2) > 1e-15 else (1.0 if power_c == 0 else 0.0)
        s_term = sb2 ** power_s if power_s > 0 or abs(sb2) > 1e-15 else (1.0 if power_s == 0 else 0.0)

        val += sign * num * c_term * s_term / denom

    return val


def _m_values(J):
    """Return list of m values: -J, -J+1, ..., J. Works for integer or half-integer J."""
    dim = int(round(2 * J + 1))
    return [J - dim + 1 + i for i in range(dim)]


def wigner_D_matrix(J, R: np.ndarray) -> np.ndarray:
    """Compute the (2J+1)×(2J+1) Wigner D-matrix for rotation R.

    D^J_{m,m'}(R) = e^{-i m α} d^J_{m,m'}(β) e^{-i m' γ}

    Row/column indices correspond to m = -J, -J+1, ..., J.

    Parameters
    ----------
    J : int or float
        Angular momentum quantum number (non-negative integer or half-integer).
    R : ndarray, shape (3, 3)
        Rotation matrix.

    Returns
    -------
    ndarray, shape (2J+1, 2J+1)
        Complex Wigner D-matrix.
    """
    dim = int(round(2 * J + 1))
    alpha, beta, gamma = _euler_angles_from_rotation(R)

    ms = _m_values(J)
    D = np.zeros((dim, dim), dtype=np.complex128)
    for im, m in enumerate(ms):
        for imp, mp in enumerate(ms):
            d_val = _small_d(J, m, mp, beta)
            phase = np.exp(-1j * m * alpha) * np.exp(-1j * mp * gamma)
            D[im, imp] = phase * d_val

    return D


# ─────────────────────────────────────────────────────────────
# Oh irrep matrices (for the 24 rotations)
# ─────────────────────────────────────────────────────────────

def _oh_irrep_matrices() -> Dict[str, List[np.ndarray]]:
    """Return irrep matrices D^Γ(R) for each O irrep and each of the 24 rotations.

    These are real matrices. For 1D irreps (A1, A2), they're scalars.
    For E (2D), they're 2×2 matrices.
    For T1, T2 (3D), they're 3×3 matrices.

    T1 is the standard 3D rotation representation (the rotation matrices
    themselves), restricted to O. T2 is T1 ⊗ A2.

    Returns
    -------
    dict mapping irrep name → list of 24 matrices (same order as octahedral_rotations())
    """
    rotations = octahedral_rotations()
    result = {}

    # A1: trivial representation
    result['A1'] = [np.array([[1.0]]) for _ in rotations]

    # T1: the vector (xyz) representation = the rotation matrices themselves
    result['T1'] = [R.copy() for R in rotations]

    # A2: the sign (alternating) representation of O.
    # A2 character: E=1, C3=1, C2(face)=1, C4=-1, C2'(edge)=-1
    a2_chars = []
    # Character table values indexed by conjugacy class
    a2_by_class = {0: 1.0, 1: 1.0, 2: 1.0, 3: -1.0, 4: -1.0}
    for R in rotations:
        cls = _classify_rotation(R)
        a2_chars.append(a2_by_class[cls])

    result['A2'] = [np.array([[c]]) for c in a2_chars]

    # T2 = T1 ⊗ A2
    result['T2'] = [R * c for R, c in zip(rotations, a2_chars)]

    # E: the 2D irrep. For O, E appears in D^2.
    # We can extract it by projecting D^2 onto E using the character formula.
    # E character: E=2, C3=-1, C2(face)=2, C4=0, C2'(edge)=0
    # More directly: E is the quadrupole representation.
    # Standard basis: {(3z²-r²)/√6, (x²-y²)/√2} or equivalently
    # the real spherical harmonics Y₂⁰ and (Y₂² + Y₂⁻²)/√2.
    # Under the rotation matrices of O, these transform as the E irrep.
    #
    # Build E from D^2: use projection operator method.
    # E character under each class:
    e_char_by_class = {3.0: 2.0, 0.0: -1.0, -1.0: None, 1.0: 0.0}
    # For C2 face vs edge: E gives 2 for C2(face) and 0 for C2'(edge)

    # Actually, let me build E explicitly from real spherical harmonics.
    # The real Y₂ₘ functions evaluated at (x,y,z) on the unit sphere:
    # Y₂⁰ = (3z²-1)/2 × √(5/4π) ... but we just need the transformation.
    #
    # For the E irrep, a convenient basis uses:
    # e₁ = (2z²-x²-y²)/√6  (proportional to Y₂⁰)
    # e₂ = (x²-y²)/√2      (proportional to Re(Y₂²))
    #
    # Under rotation R, a function f(r) → f(R⁻¹r). For quadratic forms
    # x²,y²,z² → transformed by R applied to coordinates.

    E_mats = []
    for R in rotations:
        # Apply R to coordinates: (x,y,z) → R(x,y,z)
        # x'=R[0,0]x+R[0,1]y+R[0,2]z, etc.
        # Then x'²-y'² and 2z'²-x'²-y'² give the new basis components.
        #
        # Rather than expanding, use the D^2 matrix approach.
        # D^2 is 5×5. The E irrep is a 2D subspace.
        # Use the character projection to identify it.
        pass  # Will compute below

    # Compute E matrices via D^2 projection
    D2_mats = [wigner_D_matrix(2, R) for R in rotations]

    # Convert D^2 to real spherical harmonic basis:
    # |m⟩_real for m = -2,-1,0,1,2:
    # |2c⟩ = (|+2⟩ + |-2⟩)/√2
    # |2s⟩ = (|+2⟩ - |-2⟩)/(i√2)
    # |1c⟩ = -(|+1⟩ - |-1⟩)/√2
    # |1s⟩ = (|+1⟩ + |-1⟩)/(i√2)  ... actually sign depends on convention
    # |0⟩  = |0⟩

    # Standard real-to-complex transformation for integer J:
    # U transforms from complex |m⟩ to real |m⟩_real basis
    U2 = _complex_to_real_matrix(2)
    D2_real = [U2 @ D @ U2.conj().T for D in D2_mats]

    # In the real basis, D^2 decomposes into E(2D) ⊕ T2(3D).
    # E spans {Y₂⁰, Re(Y₂²)} and T2 spans {Im(Y₂²), Re(Y₂¹), Im(Y₂¹)}
    # (up to convention — let's identify them by projection).

    # Use character projection to identify the E subspace:
    # P_E = (2/24) Σ_R χ_E(R)* D^2_real(R)
    e_chars = _oh_characters_at_rotations(rotations, 'E')
    P_E = np.zeros((5, 5), dtype=np.float64)
    for i, D in enumerate(D2_real):
        P_E += e_chars[i] * D.real  # D^2 in real basis should be real
    P_E *= 2.0 / 24.0  # dim_E / |O|

    # Diagonalize P_E to find the E subspace (eigenvalue 1)
    eigvals, eigvecs = np.linalg.eigh(P_E)
    # Sort by eigenvalue descending
    idx = np.argsort(-eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # The E subspace has eigenvalue 1 (2 vectors)
    e_basis = eigvecs[:, :2]  # 5×2, columns span the E subspace

    # E matrices: D^E(R) = e_basis^T @ D^2_real(R) @ e_basis
    E_mats = []
    for D in D2_real:
        E_mats.append(e_basis.T @ D.real @ e_basis)

    result['E'] = E_mats

    return result


def _complex_to_real_matrix(J: int) -> np.ndarray:
    """Unitary matrix U that transforms from complex |J,m⟩ to real basis.

    The real spherical harmonics are defined as:
      |J,0⟩_real = |J,0⟩
      |J,m⟩_c = (-1)^m/√2 (|J,m⟩ + (-1)^m |J,-m⟩)   for m > 0
      |J,m⟩_s = (-1)^m/(i√2) (|J,m⟩ - (-1)^m |J,-m⟩) for m > 0

    We order the real basis as: m=0, 1c, 1s, 2c, 2s, ... (ascending |m|)

    Returns U such that |real⟩ = U |complex⟩, where complex basis is
    ordered m = -J, -J+1, ..., J.
    """
    dim = 2 * J + 1
    U = np.zeros((dim, dim), dtype=np.complex128)

    # Map from m to column index in complex basis: m → m + J
    def cidx(m):
        return m + J

    # Real basis ordering: m=0, 1c, 1s, 2c, 2s, ...
    row = 0

    # m = 0
    U[row, cidx(0)] = 1.0
    row += 1

    for m in range(1, J + 1):
        phase = (-1) ** m
        # |m⟩_c = (-1)^m/√2 × (|m⟩ + (-1)^m |-m⟩)
        U[row, cidx(m)] = phase / math.sqrt(2)
        U[row, cidx(-m)] = phase * phase / math.sqrt(2)  # (-1)^m × (-1)^m/√2 = 1/√2
        row += 1

        # |m⟩_s = (-1)^m/(i√2) × (|m⟩ - (-1)^m |-m⟩)
        U[row, cidx(m)] = phase / (1j * math.sqrt(2))
        U[row, cidx(-m)] = -phase * phase / (1j * math.sqrt(2))
        row += 1

    return U


def _oh_characters_at_rotations(
    rotations: List[np.ndarray], irrep: str,
) -> List[float]:
    """Return the character χ^Γ(R) for each of the 24 rotations.

    Uses the O character table with rotation classification by trace.
    Supports both single-group and double-group irreps.

    For double-group irreps, the characters at the 24 O rotations
    are the characters of the double-valued representations evaluated
    at the "unbarred" elements of O'. These are projective characters
    of O but the product χ_Γ*(R) × χ_J(R) is well-defined for
    half-integer J because both have the same cocycle.
    """
    _sqrt2 = math.sqrt(2)

    # O character table: columns = E, C3, C2(face), C4, C2'(edge)
    char_table = {
        'A1': [1, 1, 1, 1, 1],
        'A2': [1, 1, 1, -1, -1],
        'E':  [2, -1, 2, 0, 0],
        'T1': [3, 0, -1, 1, -1],
        'T2': [3, 0, -1, -1, 1],
        # Double-group irreps (half-integer J)
        'E1/2': [2, 1, 0, _sqrt2, 0],
        'E5/2': [2, 1, 0, -_sqrt2, 0],
        'G3/2': [4, -1, 0, 0, 0],
    }

    chars = char_table[irrep]
    result = []

    for R in rotations:
        cls = _classify_rotation(R)
        result.append(chars[cls])

    return result


def _classify_rotation(R: np.ndarray) -> int:
    """Classify a rotation matrix into one of the 5 conjugacy classes of O.

    Returns: 0=E, 1=C3, 2=C2(face), 3=C4, 4=C2'(edge)
    """
    tr = np.trace(R)

    if abs(tr - 3.0) < 0.1:
        return 0  # E
    elif abs(tr - 0.0) < 0.1:
        return 1  # C3 (trace = 1 + 2cos(2π/3) = 1 - 1 = 0)
    elif abs(tr - 1.0) < 0.1:
        return 3  # C4 (trace = 1 + 2cos(π/2) = 1)
    elif abs(tr + 1.0) < 0.1:
        # C2 (trace = -1): distinguish face vs edge
        # Face C2 axis is along a coordinate axis: one diagonal = 1, others = -1
        diag = np.diag(R)
        if np.any(np.abs(diag - 1.0) < 0.1):
            return 2  # C2(face)
        else:
            return 4  # C2'(edge)
    else:
        raise ValueError(f"Cannot classify rotation with trace {tr}")


# ─────────────────────────────────────────────────────────────
# O3→Oh subduction (branching)
# ─────────────────────────────────────────────────────────────

def oh_branching(J) -> Dict[str, int]:
    """Compute the O3→Oh branching for angular momentum J.

    Returns the multiplicity of each Oh irrep in the reduction of D^J.

    Parameters
    ----------
    J : int or float
        Non-negative angular momentum. Integer J branches into single-group
        irreps (A1, A2, E, T1, T2). Half-integer J branches into
        double-group irreps (E1/2, E5/2, G3/2).

    Returns
    -------
    dict mapping irrep name → multiplicity (0 or positive integer)
    """
    rotations = octahedral_rotations()
    is_half_int = abs(J - round(J)) > 0.1

    if is_half_int:
        irrep_list = ['E1/2', 'E5/2', 'G3/2']
    else:
        irrep_list = ['A1', 'A2', 'E', 'T1', 'T2']

    result = {}
    for irrep in irrep_list:
        n = 0.0
        chi_G_list = _oh_characters_at_rotations(rotations, irrep)
        for i, R in enumerate(rotations):
            chi_J = _so3_character(J, R)
            n += chi_G_list[i] * chi_J
        n /= 24.0
        result[irrep] = int(round(n))

    return result


def _so3_character(J, R: np.ndarray) -> float:
    """Character of D^J for rotation R: χ^J(θ) = sin((2J+1)θ/2)/sin(θ/2).

    Works for both integer and half-integer J.
    """
    tr = np.trace(R)
    # cos(θ) = (tr - 1) / 2  for a proper rotation
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = math.acos(cos_theta)

    if abs(theta) < 1e-10:
        return float(2 * J + 1)

    return math.sin((2 * J + 1) * theta / 2.0) / math.sin(theta / 2.0)


# ─────────────────────────────────────────────────────────────
# Subduction matrices
# ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=32)
def oh_subduction_matrix(J: int) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
    """Compute the unitary subduction matrix for D^J → Oh irreps.

    The subduction matrix S has shape (2J+1, 2J+1). It transforms from
    the real spherical harmonic basis to the Oh-adapted basis where
    states are grouped by irrep.

    The returned label list gives, for each column of S, the
    (irrep_name, occurrence_index) pair. Occurrence index distinguishes
    multiple copies of the same irrep (e.g., two T1 copies in D^5).

    Parameters
    ----------
    J : int
        Non-negative integer.

    Returns
    -------
    S : ndarray, shape (2J+1, 2J+1)
        Real orthogonal subduction matrix. Each column is one Oh-adapted
        basis vector expressed in the real |J,m⟩ basis.
    labels : list of (str, int)
        (irrep_name, copy_index) for each column.
    """
    dim = 2 * J + 1
    rotations = octahedral_rotations()
    branching = oh_branching(J)
    irrep_mats = _oh_irrep_matrices()

    # Compute D^J in real spherical harmonic basis
    D_real = _D_real_matrices(J)

    # For each irrep that appears, extract its subspace via character
    # projection, then do a simultaneous block-diagonalization using
    # the row projection operators.
    S_columns = []
    labels = []

    for irrep in ['A1', 'A2', 'E', 'T1', 'T2']:
        mult = branching.get(irrep, 0)
        if mult == 0:
            continue

        dim_gamma = OH_IRREP_DIM[irrep]
        gamma_mats = irrep_mats[irrep]

        # Row projection operators P^Γ_{μν} = (d_Γ/|G|) Σ_R [D^Γ(R)]_{νμ} D^J(R)
        # Build all P^Γ_{μν} for this irrep
        P = np.zeros((dim_gamma, dim_gamma, dim, dim), dtype=np.float64)
        for mu in range(dim_gamma):
            for nu in range(dim_gamma):
                for i_rot in range(24):
                    P[mu, nu] += gamma_mats[i_rot][nu, mu] * D_real[i_rot]
                P[mu, nu] *= dim_gamma / 24.0

        # P[0,0] projects onto the "first partner" subspace (rank = mult)
        eigvals, eigvecs = np.linalg.eigh(P[0, 0])
        mask = eigvals > 0.5
        first_partners = eigvecs[:, mask]  # (dim, mult)

        if first_partners.shape[1] != mult:
            # Retry with tighter threshold
            mask = np.abs(eigvals - 1.0) < 0.1
            first_partners = eigvecs[:, mask]

        assert first_partners.shape[1] == mult, (
            f"Expected {mult} first-partner vectors for {irrep} in D^{J}, "
            f"got {first_partners.shape[1]}"
        )

        # Orthogonalize first partners via QR
        if mult > 1:
            first_partners, _ = np.linalg.qr(first_partners)
            first_partners = first_partners[:, :mult]

        # For each copy, generate all dim_gamma partner functions
        for copy in range(mult):
            v0 = first_partners[:, copy]

            for mu in range(dim_gamma):
                if mu == 0:
                    v = v0.copy()
                else:
                    # P^Γ_{μ,0} shifts from partner 0 to partner μ
                    v = P[mu, 0] @ v0
                    norm = np.linalg.norm(v)
                    if norm > 1e-10:
                        v /= norm

                S_columns.append(v)
                labels.append((irrep, copy))

    # Stack into matrix
    S = np.column_stack(S_columns)

    # Verify orthogonality
    overlap = S.T @ S
    if not np.allclose(overlap, np.eye(dim), atol=1e-8):
        # Try to fix by Gram-Schmidt within each irrep block
        S, _ = np.linalg.qr(S)

    overlap = S.T @ S
    assert np.allclose(overlap, np.eye(dim), atol=1e-8), (
        f"Subduction matrix is not orthogonal for J={J}: "
        f"max error {np.max(np.abs(overlap - np.eye(dim)))}"
    )

    return S, labels


def _D_real_matrices(J: int) -> List[np.ndarray]:
    """Compute D^J in real spherical harmonic basis for all 24 O rotations."""
    rotations = octahedral_rotations()
    if J == 0:
        return [np.array([[1.0]]) for _ in rotations]

    D_complex = [wigner_D_matrix(J, R) for R in rotations]
    U = _complex_to_real_matrix(J)
    result = []
    for D in D_complex:
        D_r = (U @ D @ U.conj().T)
        assert np.allclose(D_r.imag, 0, atol=1e-12)
        result.append(D_r.real)
    return result


# ─────────────────────────────────────────────────────────────
# Complex-basis Oh irrep matrices and subduction
# ─────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _oh_irrep_matrices_complex() -> Dict[str, List[np.ndarray]]:
    """Return Oh irrep matrices in the complex basis matching Wigner D-matrices.

    T1 = D^1_complex(R) (the Wigner D^1 matrix IS the T1 irrep in complex basis)
    T2 = D^1_complex(R) * chi^{A2}(R)
    E  = extracted from D^2_complex via character projection
    A1, A2: unchanged (1D scalars)
    """
    rotations = octahedral_rotations()
    real_mats = _oh_irrep_matrices()

    result = {}

    # A1, A2: 1D, no basis change needed
    result['A1'] = real_mats['A1']
    result['A2'] = real_mats['A2']

    # T1: the Wigner D^1 matrix IS the T1 irrep in the complex basis
    D1_complex = [wigner_D_matrix(1, R) for R in rotations]
    result['T1'] = D1_complex

    # T2 = T1 x A2
    a2_chars = [real_mats['A2'][i][0, 0] for i in range(24)]
    result['T2'] = [D * c for D, c in zip(D1_complex, a2_chars)]

    # E: extract from D^2_complex via character projection
    D2_complex = [wigner_D_matrix(2, R) for R in rotations]
    e_chars = _oh_characters_at_rotations(rotations, 'E')

    P_E = np.zeros((5, 5), dtype=np.complex128)
    for i in range(24):
        P_E += e_chars[i] * D2_complex[i]
    P_E *= 2.0 / 24.0

    eigvals, eigvecs = np.linalg.eigh(P_E)
    idx = np.argsort(-eigvals.real)
    e_basis = eigvecs[:, idx[:2]]  # 5x2

    result['E'] = [e_basis.conj().T @ D @ e_basis for D in D2_complex]

    return result


def _complex_subduction_matrix(J, irrep: str) -> np.ndarray:
    """Compute the complex-basis subduction matrix B^Gamma_J.

    B has shape (2J+1, dim_Gamma * mult) where each group of dim_Gamma
    columns corresponds to one copy of Gamma in D^J.

    Works for both integer J (single-group irreps) and half-integer J
    (double-group irreps).
    """
    is_half_int = abs(J - round(J)) > 0.1
    if is_half_int:
        return _complex_subduction_matrix_half_int(J, irrep)

    dim = int(round(2 * J + 1))
    dim_g = OH_IRREP_DIM[irrep]
    mult = oh_branching(J).get(irrep, 0)
    if mult == 0:
        return np.zeros((dim, 0), dtype=np.complex128)

    rotations = octahedral_rotations()
    D_complex = [wigner_D_matrix(J, R) for R in rotations]
    gamma_mats = _oh_irrep_matrices_complex()[irrep]

    # Build full projection operators P^Gamma_{mu,nu}
    P = np.zeros((dim_g, dim_g, dim, dim), dtype=np.complex128)
    for mu in range(dim_g):
        for nu in range(dim_g):
            for i_rot in range(24):
                P[mu, nu] += gamma_mats[i_rot][mu, nu].conj() * D_complex[i_rot]
            P[mu, nu] *= dim_g / 24.0

    # P[0,0] projects onto the first-partner subspace (dimension = mult)
    eigvals, eigvecs = np.linalg.eigh(P[0, 0])
    mask = eigvals > 0.5
    first_partners = eigvecs[:, mask]  # (dim, mult)

    if first_partners.shape[1] != mult:
        mask = np.abs(eigvals - 1.0) < 0.1
        first_partners = eigvecs[:, mask]

    assert first_partners.shape[1] == mult, (
        f"Expected {mult} first-partner vectors for {irrep} in D^{J}, "
        f"got {first_partners.shape[1]}"
    )

    # Build full subduction matrix: for each copy, generate all partners
    B_columns = []
    for copy in range(mult):
        v0 = first_partners[:, copy]
        for mu in range(dim_g):
            if mu == 0:
                v = v0.copy()
            else:
                v = P[mu, 0] @ v0
                norm = np.linalg.norm(v)
                if norm > 1e-10:
                    v /= norm
            B_columns.append(v)

    B = np.column_stack(B_columns)  # (dim, dim_g * mult)
    return B


# ─────────────────────────────────────────────────────────────
# Double-group irrep matrices and half-integer subduction
# ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _oh_double_irrep_matrices() -> Dict[str, List[np.ndarray]]:
    """Return double-group irrep matrices for all 24 O rotations.

    E1/2 = D^{1/2}(R)  (the fundamental spinor representation)
    G3/2 = D^{3/2}(R)  (the J=3/2 Wigner matrix, which IS G3/2 since
           J=3/2 branches to a single copy of G3/2)
    E5/2 = extracted from D^{5/2}(R) by removing the G3/2 subspace
           (J=5/2 → E5/2 ⊕ G3/2)

    These are projective representations of O (genuine reps of the double
    cover 2O). The Euler-angle branch is fixed by ``wigner_D_matrix``,
    giving a consistent set of matrices suitable for projection operators.
    """
    rotations = octahedral_rotations()

    result = {}

    # E1/2: the spin-1/2 Wigner matrices
    result['E1/2'] = [wigner_D_matrix(0.5, R) for R in rotations]

    # G3/2: the spin-3/2 Wigner matrices
    result['G3/2'] = [wigner_D_matrix(1.5, R) for R in rotations]

    # E5/2: extract from D^{5/2} as the complement of G3/2.
    # J=5/2 → E5/2(dim=2) ⊕ G3/2(dim=4), total dim=6.
    #
    # Character projection fails for double-group irreps (projective chars
    # aren't genuine). Instead, use the full row-projection operator with
    # D^{3/2} as the G3/2 irrep matrices to find the 4D G3/2 subspace,
    # then take its orthogonal complement as the E5/2 subspace.
    D52 = [wigner_D_matrix(2.5, R) for R in rotations]
    D32 = result['G3/2']  # D^{3/2} matrices = G3/2 irrep matrices

    # Row-projection: P^{G3/2} = (4/24) Σ_R Σ_μ D^{3/2}*(R)_{μμ} D^{5/2}(R)
    # = (4/24) Σ_R tr[D^{3/2}*(R)] D^{5/2}(R) (character projection with
    # proper cocycle match since both are from Wigner matrices)
    P_G32 = np.zeros((6, 6), dtype=np.complex128)
    for i in range(24):
        chi = np.trace(D32[i]).conj()
        P_G32 += chi * D52[i]
    P_G32 *= 4.0 / 24.0

    eigvals, eigvecs = np.linalg.eigh(P_G32)
    # G3/2 subspace: eigenvalue ≈ 1 (4 eigenvectors)
    # E5/2 subspace: eigenvalue ≈ 0 (2 eigenvectors)
    idx = np.argsort(eigvals.real)
    e52_basis = eigvecs[:, idx[:2]]  # 6×2 (the null space of P_G32)

    result['E5/2'] = [e52_basis.conj().T @ D @ e52_basis for D in D52]

    return result


def _complex_subduction_matrix_half_int(J: float, irrep: str) -> np.ndarray:
    """Compute complex-basis subduction matrix for half-integer J.

    B has shape (2J+1, dim_Gamma × mult). Uses double-group irrep matrices
    with the projection operator method. The factor of 2 from the barred
    elements of O' cancels with the |O'|=48 denominator, leaving the same
    dim_Γ/24 coefficient as the single-group case.
    """
    dim = int(round(2 * J + 1))
    dim_g = OH_DOUBLE_IRREP_DIM[irrep]
    mult = oh_branching(J).get(irrep, 0)
    if mult == 0:
        return np.zeros((dim, 0), dtype=np.complex128)

    rotations = octahedral_rotations()
    D_complex = [wigner_D_matrix(J, R) for R in rotations]
    gamma_mats = _oh_double_irrep_matrices()[irrep]

    # Build projection operators P^Γ_{μν} = (dim_Γ/24) Σ_R Γ*(R)_{μν} D^J(R)
    P = np.zeros((dim_g, dim_g, dim, dim), dtype=np.complex128)
    for mu in range(dim_g):
        for nu in range(dim_g):
            for i_rot in range(24):
                P[mu, nu] += gamma_mats[i_rot][mu, nu].conj() * D_complex[i_rot]
            P[mu, nu] *= dim_g / 24.0

    # P[0,0] projects onto the first-partner subspace (rank = mult)
    eigvals, eigvecs = np.linalg.eigh(P[0, 0])
    mask = eigvals > 0.5
    first_partners = eigvecs[:, mask]

    if first_partners.shape[1] != mult:
        mask = np.abs(eigvals - 1.0) < 0.1
        first_partners = eigvecs[:, mask]

    assert first_partners.shape[1] == mult, (
        f"Expected {mult} first-partner vectors for {irrep} in D^{J}, "
        f"got {first_partners.shape[1]}"
    )

    # Generate all partners for each copy
    B_columns = []
    for copy in range(mult):
        v0 = first_partners[:, copy]
        for mu in range(dim_g):
            if mu == 0:
                v = v0.copy()
            else:
                v = P[mu, 0] @ v0
                norm = np.linalg.norm(v)
                if norm > 1e-10:
                    v /= norm
            B_columns.append(v)

    B = np.column_stack(B_columns)
    return B


@lru_cache(maxsize=32)
def _a1_vector(k: int) -> np.ndarray:
    """Get A1 component of D^k in Oh (complex basis).

    Returns a (2k+1,) complex array that is the eigenvector of the A1
    character projector with eigenvalue 1.
    """
    if k == 0:
        return np.array([1.0], dtype=np.complex128)
    d = 2 * k + 1
    D_k = [wigner_D_matrix(k, R) for R in octahedral_rotations()]
    chars = _oh_characters_at_rotations(octahedral_rotations(), 'A1')
    P = np.zeros((d, d), dtype=np.complex128)
    for i in range(24):
        P += chars[i] * D_k[i]
    P /= 24.0
    eigvals, eigvecs = np.linalg.eigh(P)
    idx = np.argmax(eigvals.real)
    a1 = eigvecs[:, idx]
    # Fix phase: make the dominant component real and positive
    mx = np.argmax(np.abs(a1))
    a1 *= np.exp(-1j * np.angle(a1[mx]))
    return a1


def _build_coupling_operator(J_bra, J_ket, k, a1: np.ndarray) -> np.ndarray:
    """Build O(M_bra, M_ket) = sum_q a1(q) CG(J_ket, M_ket; k, q; J_bra, M_bra).

    Works for integer and half-integer J_bra, J_ket (k is always integer).
    """
    d_bra = int(round(2 * J_bra + 1))
    d_ket = int(round(2 * J_ket + 1))
    ms_bra = _m_values(J_bra)
    ms_ket = _m_values(J_ket)
    qs = _m_values(k)  # -k, ..., k (always integer)

    O = np.zeros((d_bra, d_ket), dtype=np.complex128)
    for im_ket, m_ket in enumerate(ms_ket):
        for iq, q in enumerate(qs):
            m_bra = m_ket + q
            if abs(m_bra) > J_bra + 1e-10:
                continue
            # Find index of m_bra in ms_bra
            im_bra = int(round(m_bra + J_bra))
            cg = clebsch_gordan(J_ket, m_ket, k, q, J_bra, m_bra)
            O[im_bra, im_ket] += a1[iq] * cg
    return O


# ─────────────────────────────────────────────────────────────
# Real-basis coupling infrastructure (for mult > 1 correctness)
# ─────────────────────────────────────────────────────────────

def _c2r_unitary(J: int) -> np.ndarray:
    """Unitary U that converts standard-m-ordered complex basis to real.

    Convention:  |real⟩ = U |complex⟩
    So D^J_real = U D^J_complex U^†  is real for integer J.

    Row/column ordering is m = -J, -J+1, ..., J (same as Wigner D).
    """
    dim = 2 * J + 1
    U = np.zeros((dim, dim), dtype=np.complex128)
    for m in range(-J, J + 1):
        idx = m + J  # index for m in standard ordering
        if m > 0:
            # cosine-like: Y^real_m = 1/√2 ((-1)^m Y_m + Y_{-m})
            U[idx, m + J] = (-1) ** m / math.sqrt(2)
            U[idx, -m + J] = 1.0 / math.sqrt(2)
        elif m == 0:
            U[idx, J] = 1.0
        else:  # m < 0
            # sine-like: Y^real_m = i/√2 ((-1)^{|m|+1} Y_{|m|} + Y_{-|m|})
            am = abs(m)
            U[idx, am + J] = (-1) ** (am + 1) * 1j / math.sqrt(2)
            U[idx, -am + J] = 1j / math.sqrt(2)
    return U


@lru_cache(maxsize=32)
def _real_D_matrices_std(J: int) -> List[np.ndarray]:
    """D^J in real SH basis with standard m-ordering, for all 24 O rotations."""
    if J == 0:
        return [np.array([[1.0]]) for _ in octahedral_rotations()]
    U = _c2r_unitary(J)
    return [
        (U @ wigner_D_matrix(J, R) @ U.conj().T).real
        for R in octahedral_rotations()
    ]


@lru_cache(maxsize=1)
def _oh_irrep_matrices_real_std() -> Dict[str, List[np.ndarray]]:
    """Oh irrep matrices in real SH basis with standard m-ordering.

    T1 = D^1_real (the l=1 real SH rotation matrix IS the T1 irrep)
    T2 = D^1_real × χ^A2
    E  = extracted from D^2_real via character projection
    A1, A2: unchanged (1D)
    """
    rotations = octahedral_rotations()
    real_mats = _oh_irrep_matrices()

    result = {}
    result['A1'] = real_mats['A1']
    result['A2'] = real_mats['A2']

    D1_real = _real_D_matrices_std(1)
    result['T1'] = [Dr.copy() for Dr in D1_real]

    a2_chars = [real_mats['A2'][i][0, 0] for i in range(24)]
    result['T2'] = [Dr * c for Dr, c in zip(D1_real, a2_chars)]

    # E: extract from D^2_real via character projection
    D2_real = _real_D_matrices_std(2)
    e_chars = _oh_characters_at_rotations(rotations, 'E')
    P_E = np.zeros((5, 5), dtype=np.float64)
    for i in range(24):
        P_E += e_chars[i] * D2_real[i]
    P_E *= 2.0 / 24.0
    eigvals, eigvecs = np.linalg.eigh(P_E)
    idx = np.argsort(-eigvals)
    e_basis = eigvecs[:, idx[:2]]  # 5×2
    result['E'] = [e_basis.T @ D @ e_basis for D in D2_real]

    return result


def _real_subduction_matrix(J: int, irrep: str) -> np.ndarray:
    """Compute real-valued subduction matrix B^Γ_J in standard m-ordering.

    B has shape (2J+1, dim_Γ × mult). All entries are real.
    Uses real Wigner D-matrices and real irrep matrices in the SH basis
    so that coupling traces are naturally real (no phase ambiguity).
    """
    dim = 2 * J + 1
    dim_g = OH_IRREP_DIM[irrep]
    mult = oh_branching(J).get(irrep, 0)
    if mult == 0:
        return np.zeros((dim, 0), dtype=np.float64)

    D_real = _real_D_matrices_std(J)
    gamma_mats = _oh_irrep_matrices_real_std()[irrep]

    # Build projection operators P^Γ_{μν} (real-valued)
    P = np.zeros((dim_g, dim_g, dim, dim), dtype=np.float64)
    for mu in range(dim_g):
        for nu in range(dim_g):
            for i_rot in range(24):
                P[mu, nu] += gamma_mats[i_rot][mu, nu].conj() * D_real[i_rot]
            P[mu, nu] *= dim_g / 24.0

    # P[0,0] projects onto first-partner subspace (rank = mult)
    eigvals, eigvecs = np.linalg.eigh(P[0, 0])
    mask = eigvals > 0.5
    first_partners = eigvecs[:, mask]  # (dim, mult), real

    if first_partners.shape[1] != mult:
        mask = np.abs(eigvals - 1.0) < 0.1
        first_partners = eigvecs[:, mask]

    assert first_partners.shape[1] == mult, (
        f"Expected {mult} first-partner vectors for {irrep} in D^{J}, "
        f"got {first_partners.shape[1]}"
    )

    # Orthogonalize first partners if mult > 1
    if mult > 1:
        first_partners, _ = np.linalg.qr(first_partners)
        first_partners = first_partners[:, :mult]

    # Generate all partners for each copy
    B_columns = []
    for copy in range(mult):
        v0 = first_partners[:, copy]
        for mu in range(dim_g):
            if mu == 0:
                v = v0.copy()
            else:
                v = P[mu, 0] @ v0
                norm = np.linalg.norm(v)
                if norm > 1e-10:
                    v /= norm
            B_columns.append(v)

    B = np.column_stack(B_columns)

    # Verify orthonormality
    gram = B.T @ B
    assert np.allclose(gram, np.eye(dim_g * mult), atol=1e-8), (
        f"Real subduction matrix not orthonormal for {irrep} in D^{J}"
    )

    return B


def _real_coupling_operator(
    J_bra: int, J_ket: int, k: int,
) -> np.ndarray:
    """Build coupling operator in real SH basis (standard m-ordering).

    For A1 operators in the real SH basis:
    - J_bra + J_ket + k even → operator is real
    - J_bra + J_ket + k odd  → operator is purely imaginary

    The imaginary case arises from the parity of real spherical harmonics
    under m → -m inversion. We extract the imaginary part as a real matrix,
    which is equivalent to absorbing an i-factor into the subduction phase
    convention.

    Returns a real (2J_bra+1, 2J_ket+1) matrix in both cases.
    """
    a1 = _a1_vector(k)
    O_complex = _build_coupling_operator(J_bra, J_ket, k, a1)
    U_bra = _c2r_unitary(J_bra)
    U_ket = _c2r_unitary(J_ket)
    O_transformed = U_bra @ O_complex @ U_ket.conj().T

    if (J_bra + J_ket + k) % 2 == 0:
        assert np.allclose(O_transformed.imag, 0, atol=1e-10), (
            f"Even-parity coupling not real for ({J_bra},{J_ket},k={k}): "
            f"max imag = {np.max(np.abs(O_transformed.imag)):.2e}"
        )
        return O_transformed.real
    else:
        assert np.allclose(O_transformed.real, 0, atol=1e-10), (
            f"Odd-parity coupling not imaginary for ({J_bra},{J_ket},k={k}): "
            f"max real = {np.max(np.abs(O_transformed.real)):.2e}"
        )
        return -O_transformed.imag


def _coupling_trace_full_real(
    J_bra: int, J_ket: int, k: int, irrep: str,
) -> np.ndarray:
    """Compute per-copy coupling matrix c(copy_bra, copy_ket) in real basis.

    Returns a real (mult_bra, mult_ket) matrix. This avoids the complex-basis
    issue where coupling traces have imaginary parts for mult > 1.
    """
    B_bra = _real_subduction_matrix(J_bra, irrep)
    B_ket = _real_subduction_matrix(J_ket, irrep)
    if B_bra.shape[1] == 0 or B_ket.shape[1] == 0:
        return np.zeros((0, 0), dtype=np.float64)

    dim_g = OH_IRREP_DIM[irrep]
    mult_bra = B_bra.shape[1] // dim_g
    mult_ket = B_ket.shape[1] // dim_g

    O = _real_coupling_operator(J_bra, J_ket, k)

    proj = B_bra.T @ O @ B_ket

    c = np.zeros((mult_bra, mult_ket), dtype=np.float64)
    for a in range(mult_bra):
        for b in range(mult_ket):
            block = proj[a * dim_g:(a + 1) * dim_g,
                         b * dim_g:(b + 1) * dim_g]
            c[a, b] = np.trace(block) / dim_g

    return c


# ─────────────────────────────────────────────────────────────
# Complex-basis coupling traces (retained for backward compat)
# ─────────────────────────────────────────────────────────────

def _coupling_trace(
    J_bra, J_ket, k: int, irrep: str, a1: np.ndarray,
) -> complex:
    """Compute average per-copy coupling trace.

    For phase determination, we need a single scalar representative.
    This returns the average over all copy-diagonal traces.
    """
    traces = _coupling_trace_full(J_bra, J_ket, k, irrep, a1)
    if traces.size == 0:
        return 0.0
    # Average of diagonal elements (copy_i, copy_i)
    n = min(traces.shape)
    return sum(traces[i, i] for i in range(n)) / n


def _coupling_trace_full(
    J_bra, J_ket, k: int, irrep: str, a1: np.ndarray,
) -> np.ndarray:
    """Compute per-copy coupling matrix c(copy_bra, copy_ket).

    Returns a (mult_bra, mult_ket) complex matrix where:
      c(a, b) = Tr(B_a^dag O B_b) / dim_Gamma

    B_a is the copy-a subduction sub-matrix (2J+1, dim_Gamma).
    For an A1 operator, off-diagonal (a != b) entries are typically zero.
    Works for both integer J (single-group irreps) and half-integer J
    (double-group irreps).
    """
    B_bra = _complex_subduction_matrix(J_bra, irrep)
    B_ket = _complex_subduction_matrix(J_ket, irrep)
    if B_bra.shape[1] == 0 or B_ket.shape[1] == 0:
        return np.zeros((0, 0), dtype=np.complex128)

    dim_g = OH_IRREP_DIM_ALL[irrep]
    mult_bra = B_bra.shape[1] // dim_g
    mult_ket = B_ket.shape[1] // dim_g

    O = _build_coupling_operator(J_bra, J_ket, k, a1)

    # Full projection: (dim_g*mult_bra, dim_g*mult_ket)
    proj = B_bra.conj().T @ O @ B_ket

    # Extract per-copy traces
    c = np.zeros((mult_bra, mult_ket), dtype=np.complex128)
    for a in range(mult_bra):
        for b in range(mult_ket):
            block = proj[a * dim_g:(a + 1) * dim_g,
                         b * dim_g:(b + 1) * dim_g]
            c[a, b] = np.trace(block) / dim_g

    return c


def _determine_phases(
    J_max: int, k: int, a1: np.ndarray,
    raw: Dict[Tuple[str, int, int], complex],
) -> Dict[Tuple[int, str], complex]:
    """Determine phase factors eps(J, Gamma) that make all ADD coefficients real.

    For each irrep, choose J_ref (lowest J with nonzero branching) and set
    eps(J_ref) = 1. Then propagate phases through connected J values using
    BFS: for each unphased J, find an already-phased J' with a nonzero
    coupling trace, and set eps(J) from that.

    This handles cases where J_ref and J are not directly coupled by the
    triangle rule (e.g., A1 with J_ref=0, k=4 cannot couple to J=6
    since |0-6| > k).
    """
    phases: Dict[Tuple[int, str], complex] = {}
    irreps = ['A1', 'A2', 'E', 'T1', 'T2']

    for irrep in irreps:
        J_values = sorted(
            J for J in range(J_max + 1)
            if oh_branching(J).get(irrep, 0) > 0
        )
        if not J_values:
            continue

        J_ref = J_values[0]
        phases[(J_ref, irrep)] = 1.0 + 0j
        phased = {J_ref}
        unphased = [J for J in J_values[1:]]

        # BFS: keep trying until no more progress
        changed = True
        while changed and unphased:
            changed = False
            still_unphased = []
            for J in unphased:
                found = False
                for Jp in sorted(phased):
                    c = raw.get((irrep, Jp, J), 0)
                    if abs(c) > 1e-15:
                        eps_jp = phases[(Jp, irrep)]
                        phases[(J, irrep)] = eps_jp * np.conj(c) / abs(c)
                        phased.add(J)
                        changed = True
                        found = True
                        break
                    c_rev = raw.get((irrep, J, Jp), 0)
                    if abs(c_rev) > 1e-15:
                        eps_jp = phases[(Jp, irrep)]
                        phases[(J, irrep)] = eps_jp * np.conj(c_rev) / abs(c_rev)
                        phased.add(J)
                        changed = True
                        found = True
                        break
                if not found:
                    still_unphased.append(J)
            unphased = still_unphased

    return phases


def _determine_phases_general(
    J_values_all, k: int, a1: np.ndarray,
    raw: Dict[Tuple, complex],
) -> Dict[Tuple, complex]:
    """Generalized phase determination for both integer and half-integer J.

    Like _determine_phases but accepts an explicit list of J values (which
    may be half-integers like [0.5, 1.5, 2.5, ...]) instead of J_max.
    """
    is_half_int = any(abs(J - round(J)) > 0.1 for J in J_values_all)
    if is_half_int:
        irreps = ['E1/2', 'E5/2', 'G3/2']
    else:
        irreps = ['A1', 'A2', 'E', 'T1', 'T2']

    phases: Dict[Tuple, complex] = {}

    for irrep in irreps:
        J_values = sorted(
            J for J in J_values_all
            if oh_branching(J).get(irrep, 0) > 0
        )
        if not J_values:
            continue

        J_ref = J_values[0]
        phases[(J_ref, irrep)] = 1.0 + 0j
        phased = {J_ref}
        unphased = [J for J in J_values[1:]]

        changed = True
        while changed and unphased:
            changed = False
            still_unphased = []
            for J in unphased:
                found = False
                for Jp in sorted(phased):
                    c = raw.get((irrep, Jp, J), 0)
                    if abs(c) > 1e-15:
                        eps_jp = phases[(Jp, irrep)]
                        phases[(J, irrep)] = eps_jp * np.conj(c) / abs(c)
                        phased.add(J)
                        changed = True
                        found = True
                        break
                    c_rev = raw.get((irrep, J, Jp), 0)
                    if abs(c_rev) > 1e-15:
                        eps_jp = phases[(Jp, irrep)]
                        phases[(J, irrep)] = eps_jp * np.conj(c_rev) / abs(c_rev)
                        phased.add(J)
                        changed = True
                        found = True
                        break
                if not found:
                    still_unphased.append(J)
            unphased = still_unphased

        for J in unphased:
            phases[(J, irrep)] = 1.0 + 0j

    return phases


# ─────────────────────────────────────────────────────────────
# O3->Oh ADD coupling coefficients
# ─────────────────────────────────────────────────────────────

def oh_coupling_coefficients(
    J_max: int, k: int, *, l: int = 2,
) -> Dict[Tuple[str, int, int], float]:
    """Compute all O3->Oh ADD coupling coefficients for a given tensor rank k.

    These are the coefficients that appear in .rme_rac ADD entries.
    They describe how J-basis SHELL/SPIN blocks tile into Oh-irrep-resolved
    operator blocks.

    The computation uses complex-basis subduction matrices with
    self-consistent phase determination to produce real coefficients
    matching the Fortran ttrac convention.

    Formula:
      ADD(Gamma, J_bra, J_ket, k) =
        strength(k) * sqrt(dim_Gamma / (2*J_bra + 1))
        * Re(eps*(J_bra, Gamma) * eps(J_ket, Gamma) * c)

    where c = Tr(B_bra^dag O B_ket) / dim_Gamma and eps are self-consistent
    phase factors.

    Parameters
    ----------
    J_max : int
        Maximum angular momentum to consider.
    k : int
        Operator tensor rank (0 for Hamiltonian, 4 for cubic CF, etc.).
    l : int, optional
        Orbital angular momentum (default 2 for d-shell).
        Only affects the operator strength prefactor.

    Returns
    -------
    dict mapping (irrep, J_bra, J_ket) -> float coefficient
        Only non-zero entries included.
    """
    if k == 0:
        strength = 1.0
    else:
        strength = math.sqrt((2 * k + 1) * (2 * l + 2) / (2 * l + 1))

    a1 = _a1_vector(k)
    irreps = ['A1', 'A2', 'E', 'T1', 'T2']

    # Step 1: Compute all raw coupling traces
    raw: Dict[Tuple[str, int, int], complex] = {}
    for irrep in irreps:
        for Jb in range(J_max + 1):
            if oh_branching(Jb).get(irrep, 0) == 0:
                continue
            for Jk in range(J_max + 1):
                if oh_branching(Jk).get(irrep, 0) == 0:
                    continue
                if abs(Jb - Jk) > k or Jb + Jk < k:
                    continue
                c = _coupling_trace(Jb, Jk, k, irrep, a1)
                if abs(c) > 1e-15:
                    raw[(irrep, Jb, Jk)] = c

    # Step 2: Self-consistent phase determination
    phases = _determine_phases(J_max, k, a1, raw)

    # Step 3: Compute ADD coefficients with phases applied
    result: Dict[Tuple[str, int, int], float] = {}
    for (irrep, Jb, Jk), c in raw.items():
        eps_bra = phases.get((Jb, irrep), 1.0)
        eps_ket = phases.get((Jk, irrep), 1.0)
        c_phased = np.conj(eps_bra) * eps_ket * c
        dim_g = OH_IRREP_DIM[irrep]
        add = strength * math.sqrt(dim_g / (2 * Jb + 1)) * c_phased.real
        if abs(add) > 1e-15:
            result[(irrep, Jb, Jk)] = add

    return result


def oh_coupling_coefficients_full(
    J_max, k: int, *, l: int = 2,
) -> Dict:
    """Compute per-copy O3->Oh ADD coupling coefficients.

    Like oh_coupling_coefficients but handles multiplicity > 1.
    Returns dict mapping (irrep, J_bra, copy_bra, J_ket, copy_ket) -> float.
    copy indices are 0-based.

    Parameters
    ----------
    J_max : int or float
        Maximum angular momentum. Integer → single-group irreps,
        half-integer → double-group irreps.
    k : int
        Operator rank.
    l : int
        Orbital angular momentum for normalization (default 2 for d-shell).

    Uses the same complex-basis BFS phase convention as
    oh_coupling_coefficients (guaranteeing mult=1 entries match exactly).
    For mult > 1, computes complex full coupling traces, applies the BFS
    phases, then finds a per-irrep copy-basis rotation that makes all
    traces real before extracting ADD coefficients.

    The assembled Hamiltonian eigenvalues are invariant under copy-basis
    rotations, so any real-valued copy-basis convention produces spectra
    identical to the Fortran (Butler) convention.
    """
    if k == 0:
        strength = 1.0
    else:
        strength = math.sqrt((2 * k + 1) * (2 * l + 2) / (2 * l + 1))

    a1 = _a1_vector(k)
    is_half_int = abs(J_max - round(J_max)) > 0.1

    if is_half_int:
        irreps = ['E1/2', 'E5/2', 'G3/2']
        dim_lookup = OH_DOUBLE_IRREP_DIM
        # J values: 0.5, 1.5, 2.5, ..., J_max
        J_values = [0.5 + i for i in range(int(round(J_max - 0.5)) + 1)]
    else:
        irreps = ['A1', 'A2', 'E', 'T1', 'T2']
        dim_lookup = OH_IRREP_DIM
        J_values = list(range(int(round(J_max)) + 1))

    # Step 1: Compute mult=1 scalar traces for BFS phase determination
    raw_scalar = {}
    for irrep in irreps:
        for Jb in J_values:
            if oh_branching(Jb).get(irrep, 0) == 0:
                continue
            for Jk in J_values:
                if oh_branching(Jk).get(irrep, 0) == 0:
                    continue
                if abs(Jb - Jk) > k or Jb + Jk < k:
                    continue
                c = _coupling_trace(Jb, Jk, k, irrep, a1)
                if abs(c) > 1e-15:
                    raw_scalar[(irrep, Jb, Jk)] = c

    # Step 2: BFS phase determination
    phases = _determine_phases_general(J_values, k, a1, raw_scalar)

    # Step 3: Compute full traces and apply phases
    result = {}

    for irrep in irreps:
        # Collect multiplicity info for each J
        mult_of = {}
        for J in J_values:
            m = oh_branching(J).get(irrep, 0)
            if m > 0:
                mult_of[J] = m

        # Check if any J has mult > 1 for this irrep
        max_mult = max(mult_of.values()) if mult_of else 0

        if max_mult <= 1:
            # Pure mult=1: use scalar traces (matches oh_coupling_coefficients)
            for (ir, Jb, Jk), c in raw_scalar.items():
                if ir != irrep:
                    continue
                eps_bra = phases.get((Jb, irrep), 1.0)
                eps_ket = phases.get((Jk, irrep), 1.0)
                c_phased = np.conj(eps_bra) * eps_ket * c
                dim_g = dim_lookup[irrep]
                add = strength * math.sqrt(dim_g / (2 * Jb + 1)) * c_phased.real
                if abs(add) > 1e-15:
                    result[(irrep, Jb, 0, Jk, 0)] = add
            continue

        # Mult > 1 path: compute complex full traces and phase them
        phased_traces: Dict[Tuple[int, int], np.ndarray] = {}  # (Jb, Jk) -> matrix

        for Jb in mult_of:
            for Jk in mult_of:
                if abs(Jb - Jk) > k or Jb + Jk < k:
                    continue
                c_full = _coupling_trace_full(Jb, Jk, k, irrep, a1)
                if c_full.size == 0:
                    continue
                eps_bra = phases.get((Jb, irrep), 1.0)
                eps_ket = phases.get((Jk, irrep), 1.0)
                c_phased = np.conj(eps_bra) * eps_ket * c_full
                if np.max(np.abs(c_phased)) > 1e-15:
                    phased_traces[(Jb, Jk)] = c_phased

        # Find copy-basis rotations W_J for each J with mult > 1.
        # For mult=1 J values, W_J = [1] (trivial).
        # For mult>1 J values, find a unitary W that makes all phased
        # traces real simultaneously.
        W: Dict[int, np.ndarray] = {}
        for J, m in mult_of.items():
            if m == 1:
                W[J] = np.array([[1.0]])
            else:
                W[J] = _find_real_copy_basis(
                    J, m, irrep, mult_of, phased_traces,
                )

        # Apply W and extract real ADD coefficients
        for (Jb, Jk), c_phased in phased_traces.items():
            c_rot = W[Jb].conj().T @ c_phased @ W[Jk]
            dim_g = dim_lookup[irrep]
            prefactor = strength * math.sqrt(dim_g / (2 * Jb + 1))

            mb, mk = c_rot.shape
            for a in range(mb):
                for b in range(mk):
                    add = prefactor * c_rot[a, b].real
                    if abs(add) > 1e-15:
                        result[(irrep, Jb, a, Jk, b)] = add

    return result


def _find_real_copy_basis(
    J, mult: int, irrep: str,
    mult_of: Dict,
    phased_traces: Dict[Tuple, np.ndarray],
) -> np.ndarray:
    """Find a unitary W for J with mult > 1 that makes all coupling traces real.

    Strategy: collect all cross-J coupling vectors (1×mult rows from
    c(Jb, J) where Jb has mult=1), stack them, and find the unitary W
    that maps them all to real vectors simultaneously.

    For a 2×2 case (mult=2), this amounts to finding the 2×2 unitary
    that aligns the complex column space with ℝ².
    """
    # Collect constraint vectors: rows from c(Jb, J) where Jb has mult=1
    constraint_rows = []
    for Jb, mb in mult_of.items():
        if mb != 1:
            continue
        c = phased_traces.get((Jb, J))
        if c is not None and c.shape == (1, mult):
            constraint_rows.append(c[0])  # 1×mult -> length-mult vector
        # Also check the transpose direction: c(J, Jb) has shape (mult, 1)
        c_rev = phased_traces.get((J, Jb))
        if c_rev is not None and c_rev.shape == (mult, 1):
            constraint_rows.append(c_rev[:, 0].conj())  # conjugate for consistency

    if not constraint_rows:
        # No constraints from mult=1 J values; use diagonal block
        c_diag = phased_traces.get((J, J))
        if c_diag is not None:
            # Diagonalize the Hermitian part
            H = 0.5 * (c_diag + c_diag.conj().T)
            _, W = np.linalg.eigh(H)
            return W
        return np.eye(mult, dtype=np.complex128)

    # Stack constraint rows into a matrix C (n_constraints × mult)
    C = np.array(constraint_rows, dtype=np.complex128)

    # Find W such that C @ W is real for all rows.
    # This means Im(C @ W) = 0, i.e., Im(C) @ Re(W) + Re(C) @ Im(W) = 0.
    #
    # For mult=2 with enough constraints, we can find W from the SVD of C:
    # C = U Σ V^H → W = V makes the right singular vectors real-aligned.
    # Then C @ V = U Σ, and we need to adjust phases of columns of V
    # so that U Σ is real.
    #
    # Practical approach: QR decomposition of C^H gives an orthonormal
    # basis for the column space of C^H. The unitary that maps this
    # basis to real vectors is our W.
    #
    # Simpler: for each column of W, find the phase that makes
    # C @ w_col maximally real.

    U, S, Vh = np.linalg.svd(C, full_matrices=False)

    # W = V = Vh^H, but we need to adjust column phases
    W = Vh.conj().T  # shape (mult, min(n, mult))

    # Pad to full unitary if needed
    if W.shape[1] < mult:
        # Complete the basis
        Q, _ = np.linalg.qr(
            np.hstack([W, np.random.randn(mult, mult - W.shape[1])
                       + 1j * np.random.randn(mult, mult - W.shape[1])])
        )
        W = Q

    # Adjust column phases: for each column w of W, rotate by e^{-iθ}
    # where θ minimizes Σ|Im(C @ w)|². This makes C @ W maximally real.
    for col in range(mult):
        v = C @ W[:, col]  # complex vector
        # Optimal phase: θ = 0.5 * atan2(2*Re(s), Re(a)-Im(a))
        # where s = Σ Re(v_i)*Im(v_i), a = Σ Re(v_i)², b = Σ Im(v_i)²
        # Equivalently: find θ maximizing Σ|Re(v_i e^{-iθ})|²
        # = Σ (Re(v_i)cosθ + Im(v_i)sinθ)²
        # Derivative = 0: tan(2θ) = 2Σ Re·Im / (Σ Re² - Σ Im²)
        re, im = v.real, v.imag
        S_ri = np.sum(re * im)
        D_ri = np.sum(re ** 2) - np.sum(im ** 2)
        theta = 0.5 * np.arctan2(2 * S_ri, D_ri)
        W[:, col] *= np.exp(-1j * theta)

    return W


# ─────────────────────────────────────────────────────────────
# Transition (dipole) coupling coefficients
# ─────────────────────────────────────────────────────────────


def oh_transition_coupling(
    J_max_bra,
    J_max_ket,
    k: int = 1,
    *,
    l: int = 2,
) -> Dict[Tuple[str, float, int, str, float, int], float]:
    """Compute O3→Oh coupling coefficients for a rank-k transition operator.

    Unlike ``oh_coupling_coefficients_full`` which computes intra-irrep
    coupling for *scalar* (A1) operators, this function computes cross-irrep
    coupling for vector (k=1 dipole) or other rank-k operators whose Oh
    irrep is NOT A1.

    Parameters
    ----------
    J_max_bra, J_max_ket : int or float
        Maximum angular momentum for bra/ket spaces. Integer → single-group
        irreps, half-integer → double-group irreps. Both must have the same
        integrality (both integer or both half-integer).
    k : int
        Operator rank (always integer; k=1 for dipole).
    l : int
        Orbital angular momentum for normalization (default 2 for d-shell).

    The CG tensor is built in the *complex* spherical-harmonic basis (where
    it is real-valued) and then:

    * **Integer J**: transformed to the *real* SH basis with a parity-aware
      sign fix, then projected with real subduction matrices.
    * **Half-integer J**: kept in the complex basis and projected with
      complex subduction matrices; RME is extracted from |proj|.

    To get D4h transition ADD coefficients from these Oh-level values
    (dipole k=1, T1u splits into Eu + A2u in D4h):

    * PERP (Eu, dim 2):  ADD = √(2/3) × oh_coupling
    * PARA (A2u, dim 1): ADD = √(1/3) × oh_coupling

    Returns
    -------
    dict mapping (Γ_bra, J_bra, c_bra, Γ_ket, J_ket, c_ket) → float
        Oh-level coupling coefficients. c_bra and c_ket are 0-based copy
        indices.
    """
    from multitorch.angular.wigner import clebsch_gordan

    is_half_bra = abs(J_max_bra - round(J_max_bra)) > 0.1
    is_half_ket = abs(J_max_ket - round(J_max_ket)) > 0.1
    assert is_half_bra == is_half_ket, (
        "Bra and ket must both be integer or both half-integer J")
    is_half_int = is_half_bra

    if is_half_int:
        irreps_bk = ['E1/2', 'E5/2', 'G3/2']
        dim_lookup = OH_DOUBLE_IRREP_DIM
        J_bra_values = [0.5 + i for i in range(int(round(J_max_bra - 0.5)) + 1)]
        J_ket_values = [0.5 + i for i in range(int(round(J_max_ket - 0.5)) + 1)]
    else:
        irreps_bk = ['A1', 'A2', 'E', 'T1', 'T2']
        dim_lookup = OH_IRREP_DIM
        J_bra_values = list(range(int(round(J_max_bra)) + 1))
        J_ket_values = list(range(int(round(J_max_ket)) + 1))

    # Operator is always integer rank
    irreps_op = ['A1', 'A2', 'E', 'T1', 'T2']
    op_branching = oh_branching(k)

    result: Dict[Tuple[str, float, int, str, float, int], float] = {}

    for J_bra in J_bra_values:
        branching_bra = oh_branching(J_bra)
        m_bra = list(_m_values(J_bra))

        for J_ket in J_ket_values:
            if abs(J_bra - J_ket) > k or J_bra + J_ket < k:
                continue
            branching_ket = oh_branching(J_ket)
            m_ket = list(_m_values(J_ket))

            dim_bra = int(round(2 * J_bra + 1))
            dim_op = 2 * k + 1
            dim_ket = int(round(2 * J_ket + 1))

            # ── Build CG tensor in complex |J,m> basis ──
            cg_complex = np.zeros(
                (dim_bra, dim_op, dim_ket), dtype=np.float64)
            for ib, M_bra in enumerate(m_bra):
                for io, M_op in enumerate(range(-k, k + 1)):
                    M_ket = M_bra + M_op
                    if abs(M_ket) > J_ket:
                        continue
                    ik = m_ket.index(M_ket)
                    cg_complex[ib, io, ik] = clebsch_gordan(
                        J_bra, M_bra, k, M_op, J_ket, M_ket)

            if is_half_int:
                # ── Half-integer: project in complex basis ──
                for irr_op in irreps_op:
                    if op_branching.get(irr_op, 0) == 0:
                        continue
                    B_op = _real_subduction_matrix(k, irr_op)
                    dim_g_op = OH_IRREP_DIM[irr_op]

                    for irr_bra in irreps_bk:
                        mult_bra = branching_bra.get(irr_bra, 0)
                        if mult_bra == 0:
                            continue
                        B_bra = _complex_subduction_matrix(J_bra, irr_bra)
                        dim_g_bra = dim_lookup[irr_bra]

                        for irr_ket in irreps_bk:
                            mult_ket = branching_ket.get(irr_ket, 0)
                            if mult_ket == 0:
                                continue
                            B_ket = _complex_subduction_matrix(J_ket, irr_ket)
                            dim_g_ket = dim_lookup[irr_ket]

                            # proj[a, q, b]: complex
                            proj = np.einsum(
                                'ijk,ia,jq,kb->aqb',
                                cg_complex, B_bra, B_op, B_ket)

                            for c_bra in range(mult_bra):
                                for c_ket in range(mult_ket):
                                    rme_sq = 0.0
                                    first_val = 0.0 + 0j
                                    for mu in range(dim_g_bra):
                                        a = c_bra * dim_g_bra + mu
                                        for q in range(dim_g_op):
                                            b = c_ket * dim_g_ket
                                            v = proj[a, q, b]
                                            rme_sq += abs(v) ** 2
                                            if first_val == 0 and abs(v) > 1e-10:
                                                first_val = v

                                    if rme_sq < 1e-20:
                                        continue

                                    # Sign from phase of first nonzero element
                                    sign = 1.0 if first_val.real >= 0 else -1.0
                                    rme = sign * math.sqrt(rme_sq)

                                    result[(irr_bra, J_bra, c_bra,
                                            irr_ket, J_ket, c_ket)] = rme
            else:
                # ── Integer J: transform to real SH basis ──
                U_bra = _c2r_unitary(J_bra)
                U_op = _c2r_unitary(k)
                U_ket = _c2r_unitary(J_ket)

                cg_transformed = np.einsum(
                    'rm,su,No,muo->rsN',
                    U_bra, U_op, U_ket.conj(), cg_complex)

                parity_sum = int(round(J_bra + k + J_ket))
                if parity_sum % 2 == 0:
                    cg_real = cg_transformed.real
                else:
                    cg_real = -cg_transformed.imag

                for irr_op in irreps_op:
                    if op_branching.get(irr_op, 0) == 0:
                        continue
                    B_op = _real_subduction_matrix(k, irr_op)
                    dim_g_op = OH_IRREP_DIM[irr_op]

                    for irr_bra in irreps_bk:
                        mult_bra = branching_bra.get(irr_bra, 0)
                        if mult_bra == 0:
                            continue
                        B_bra = _real_subduction_matrix(J_bra, irr_bra)
                        dim_g_bra = OH_IRREP_DIM[irr_bra]

                        for irr_ket in irreps_bk:
                            mult_ket = branching_ket.get(irr_ket, 0)
                            if mult_ket == 0:
                                continue
                            B_ket = _real_subduction_matrix(J_ket, irr_ket)
                            dim_g_ket = OH_IRREP_DIM[irr_ket]

                            proj = np.einsum(
                                'ijk,ia,jq,kb->aqb',
                                cg_real, B_bra, B_op, B_ket)

                            for c_bra in range(mult_bra):
                                for c_ket in range(mult_ket):
                                    rme_sq = 0.0
                                    first_val = 0.0
                                    for mu in range(dim_g_bra):
                                        a = c_bra * dim_g_bra + mu
                                        for q in range(dim_g_op):
                                            b = c_ket * dim_g_ket
                                            v = proj[a, q, b]
                                            rme_sq += v * v
                                            if first_val == 0.0 and abs(v) > 1e-10:
                                                first_val = v

                                    if rme_sq < 1e-20:
                                        continue

                                    sign = 1.0 if first_val >= 0 else -1.0
                                    rme = sign * math.sqrt(rme_sq)

                                    result[(irr_bra, J_bra, c_bra,
                                            irr_ket, J_ket, c_ket)] = rme

    return result
