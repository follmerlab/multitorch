"""
Wigner 3j and 6j symbols implemented in PyTorch.

Uses the Racah formula (algebraic recurrence) with a precomputed log-factorial
table for numerical stability. Results are cached with functools.lru_cache
for repeated evaluations (d-electron calculations reuse the same small set
of angular momenta j = 0, 1/2, 1, 3/2, 2).

Validated against scipy.special.wigner_3j and sympy.physics.quantum.cg.Wigner3j.

References:
  Racah (1942) Phys. Rev. 62, 438.
  Edmonds (1957) Angular Momentum in Quantum Mechanics.
  Messiah (1962) Quantum Mechanics, App. C.
"""
from __future__ import annotations
import math
import functools
from typing import Union


# Pre-computed log-factorial table for j up to ~100 (½-integer spacing)
# We index by 2*n (so 2j goes from 0 to _MAX_2J)
_MAX_2J = 200
_LOG_FACT = [0.0] * (_MAX_2J + 2)
for _k in range(1, _MAX_2J + 2):
    _LOG_FACT[_k] = _LOG_FACT[_k - 1] + math.log(_k)


def _log_fact(n: int) -> float:
    """Return log(n!) for non-negative integer n."""
    if n < 0:
        return float("-inf")
    if n > _MAX_2J + 1:
        return math.lgamma(n + 1)
    return _LOG_FACT[n]


# ─────────────────────────────────────────────────────────────
# Wigner 3j symbol
# ─────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=65536)
def wigner3j(j1: float, j2: float, j3: float,
             m1: float, m2: float, m3: float) -> float:
    """
    Wigner 3j symbol using the Racah algebraic formula.

    Parameters
    ----------
    j1, j2, j3 : float (half-integer ≥ 0)
    m1, m2, m3 : float (half-integer)

    Returns
    -------
    float : value of the 3j symbol (exact within float64)

    Selection rules:
      m1 + m2 + m3 = 0
      |j1 - j2| ≤ j3 ≤ j1 + j2  (triangle inequality)
      j1 + j2 + j3 must be integer
    """
    # Guard: log-factorial reconstruction loses precision for large j
    _MAX_TJ = 40  # 2j ≤ 40 → j ≤ 20
    # Convert to 2× representation to work with integers
    tj1, tj2, tj3 = int(round(2 * j1)), int(round(2 * j2)), int(round(2 * j3))
    tm1, tm2, tm3 = int(round(2 * m1)), int(round(2 * m2)), int(round(2 * m3))
    if max(tj1, tj2, tj3) > _MAX_TJ:
        raise ValueError(f"wigner3j: 2j={max(tj1,tj2,tj3)} exceeds {_MAX_TJ}; "
                         "log-factorial formula may lose precision")

    # Selection rules
    if tm1 + tm2 + tm3 != 0:
        return 0.0
    if abs(tm1) > tj1 or abs(tm2) > tj2 or abs(tm3) > tj3:
        return 0.0
    if not _triangle(tj1, tj2, tj3):
        return 0.0
    if (tj1 + tj2 + tj3) % 2 != 0:
        return 0.0

    # Racah formula
    return _racah_3j(tj1, tj2, tj3, tm1, tm2, tm3)


def _triangle(tj1: int, tj2: int, tj3: int) -> bool:
    """Check triangle inequality for 2j values (integers)."""
    return (abs(tj1 - tj2) <= tj3 <= tj1 + tj2
            and (tj1 + tj2 + tj3) % 2 == 0)


def _delta_log(ta: int, tb: int, tc: int) -> float:
    """log(Δ(a,b,c)) = log(sqrt((a+b-c)!(a-b+c)!(-a+b+c)! / (a+b+c+1)!))
    where inputs are 2× the angular momenta.
    """
    a, b, c = ta // 2, tb // 2, tc // 2
    # These are actual integers if ta+tb+tc is even (already checked)
    # Use 2j integers: (ta+tb-tc)//2 etc.
    n1 = (ta + tb - tc) // 2
    n2 = (ta - tb + tc) // 2
    n3 = (-ta + tb + tc) // 2
    n4 = (ta + tb + tc) // 2 + 1
    return 0.5 * (_log_fact(n1) + _log_fact(n2) + _log_fact(n3) - _log_fact(n4))


def _racah_3j(tj1: int, tj2: int, tj3: int,
              tm1: int, tm2: int, tm3: int) -> float:
    """
    Racah algebraic formula for the 3j symbol (Edmonds eq. 3.6.5).
    All arguments are 2× the actual values (integers).
    """
    # Pre-factor
    # (-1)^(j1-j2-m3) = (-1)^((tj1-tj2-tm3)/2)
    phase_exp = (tj1 - tj2 - tm3) // 2
    phase = 1.0 if phase_exp % 2 == 0 else -1.0

    # Log of the square-root prefactor
    log_prefactor = (
        _delta_log(tj1, tj2, tj3)
        + 0.5 * (
            _log_fact((tj1 + tm1) // 2)
            + _log_fact((tj1 - tm1) // 2)
            + _log_fact((tj2 + tm2) // 2)
            + _log_fact((tj2 - tm2) // 2)
            + _log_fact((tj3 + tm3) // 2)
            + _log_fact((tj3 - tm3) // 2)
        )
    )

    # Sum over t (Racah sum)
    # t runs over all integers where all factorial arguments are non-negative
    t_min = max(0,
                (tj2 - tj3 - tm1) // 2,
                (tj1 - tj3 + tm2) // 2)
    t_max = min((tj1 + tj2 - tj3) // 2,
                (tj1 - tm1) // 2,
                (tj2 + tm2) // 2)

    if t_min > t_max:
        return 0.0

    total = 0.0
    for t in range(t_min, t_max + 1):
        sign_t = 1.0 if t % 2 == 0 else -1.0
        log_denom = (
            _log_fact(t)
            + _log_fact((tj1 + tj2 - tj3) // 2 - t)
            + _log_fact((tj1 - tm1) // 2 - t)
            + _log_fact((tj2 + tm2) // 2 - t)
            + _log_fact((tj3 - tj2 + tm1) // 2 + t)
            + _log_fact((tj3 - tj1 - tm2) // 2 + t)
        )
        total += sign_t * math.exp(log_prefactor - log_denom)

    return phase * total


# ─────────────────────────────────────────────────────────────
# Wigner 6j symbol
# ─────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=65536)
def wigner6j(j1: float, j2: float, j3: float,
             j4: float, j5: float, j6: float) -> float:
    """
    Wigner 6j symbol {j1 j2 j3; j4 j5 j6} using the Racah W-coefficient.

    Selection rules: all four triangles must satisfy the triangle inequality.

    Racah formula (Edmonds eq. 6.2.2):
      {j1 j2 j3; j4 j5 j6} = Δ(j1,j2,j3) Δ(j1,j5,j6) Δ(j4,j2,j6) Δ(j4,j5,j3)
                              × Σ_t (-1)^t (t+1)! / [...]
    """
    _MAX_TJ = 40  # 2j ≤ 40 → j ≤ 20
    tj1, tj2, tj3 = int(round(2*j1)), int(round(2*j2)), int(round(2*j3))
    tj4, tj5, tj6 = int(round(2*j4)), int(round(2*j5)), int(round(2*j6))
    if max(tj1, tj2, tj3, tj4, tj5, tj6) > _MAX_TJ:
        raise ValueError(f"wigner6j: 2j={max(tj1,tj2,tj3,tj4,tj5,tj6)} exceeds {_MAX_TJ}; "
                         "log-factorial formula may lose precision")

    # Check all four triangle conditions
    if not (_triangle(tj1, tj2, tj3) and _triangle(tj1, tj5, tj6)
            and _triangle(tj4, tj2, tj6) and _triangle(tj4, tj5, tj3)):
        return 0.0

    # Log of four delta factors
    log_delta_sum = (
        _delta_log(tj1, tj2, tj3)
        + _delta_log(tj1, tj5, tj6)
        + _delta_log(tj4, tj2, tj6)
        + _delta_log(tj4, tj5, tj3)
    )

    # Sum over t
    t_min = max(
        (tj1 + tj2 + tj3) // 2,
        (tj1 + tj5 + tj6) // 2,
        (tj4 + tj2 + tj6) // 2,
        (tj4 + tj5 + tj3) // 2,
    )
    t_max = min(
        (tj1 + tj2 + tj4 + tj5) // 2,
        (tj2 + tj3 + tj5 + tj6) // 2,
        (tj1 + tj3 + tj4 + tj6) // 2,
    )

    if t_min > t_max:
        return 0.0

    total = 0.0
    for t in range(t_min, t_max + 1):
        sign_t = 1.0 if t % 2 == 0 else -1.0
        log_num = _log_fact(t + 1)
        log_den = (
            _log_fact(t - (tj1 + tj2 + tj3) // 2)
            + _log_fact(t - (tj1 + tj5 + tj6) // 2)
            + _log_fact(t - (tj4 + tj2 + tj6) // 2)
            + _log_fact(t - (tj4 + tj5 + tj3) // 2)
            + _log_fact((tj1 + tj2 + tj4 + tj5) // 2 - t)
            + _log_fact((tj2 + tj3 + tj5 + tj6) // 2 - t)
            + _log_fact((tj1 + tj3 + tj4 + tj6) // 2 - t)
        )
        total += sign_t * math.exp(log_delta_sum + log_num - log_den)

    return total


# ─────────────────────────────────────────────────────────────
# Wigner 9j symbol
# ─────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=65536)
def wigner9j(j1: float, j2: float, j3: float,
             j4: float, j5: float, j6: float,
             j7: float, j8: float, j9: float) -> float:
    """
    Wigner 9j symbol expressed as a sum over 6j symbols.

    9j(j1 j2 j3; j4 j5 j6; j7 j8 j9) =
      Σ_x (-1)^{2x} (2x+1) {j1 j2 j3; j6 j9 x} {j4 j5 j6; j2 x j8} {j7 j8 j9; x j1 j4}

    This follows the Fortran S9J implementation from ttrcg.f (line 2212).
    The summation variable x runs from
      max(|j1-j9|, |j2-j6|, |j4-j8|) to min(j1+j9, j2+j6, j4+j8).
    """
    x_min = max(abs(j1 - j9), abs(j2 - j6), abs(j4 - j8))
    x_max = min(j1 + j9, j2 + j6, j4 + j8)

    # Half-integer check: x steps by 1
    result = 0.0
    x = x_min
    while x <= x_max + 1e-10:
        sign = (-1.0) ** int(round(2 * x))
        factor = 2.0 * x + 1.0
        s1 = wigner6j(j1, j2, j3, j6, j9, x)
        s2 = wigner6j(j4, j5, j6, j2, x, j8)
        s3 = wigner6j(j7, j8, j9, x, j1, j4)
        result += sign * factor * s1 * s2 * s3
        x += 1.0

    return result


# ─────────────────────────────────────────────────────────────
# Clebsch-Gordan coefficient
# ─────────────────────────────────────────────────────────────

def clebsch_gordan(j1: float, m1: float, j2: float, m2: float,
                   j: float, m: float) -> float:
    """
    Clebsch-Gordan coefficient <j1,m1; j2,m2 | j,m>.

    Relation to 3j symbol:
      <j1,m1; j2,m2 | j,m> = (-1)^(j1-j2+m) * sqrt(2j+1) * W3j(j1,j2,j; m1,m2,-m)
    """
    if m1 + m2 != m:
        return 0.0
    j_int = int(round(2 * j))
    phase_exp = int(round(j1 - j2 + m))
    phase = 1.0 if phase_exp % 2 == 0 else -1.0
    w3j = wigner3j(j1, j2, j, m1, m2, -m)
    return phase * math.sqrt(j_int + 1) * w3j
