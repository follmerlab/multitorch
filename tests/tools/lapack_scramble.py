"""Imitate a different LAPACK: random orthogonal rotations inside degenerate eigen/singular subspaces.

Every basis returned by ``np.linalg.eigh`` / ``np.linalg.svd`` inside a
degenerate cluster is as valid as any other, and LAPACK builds differ in which
they return. Code whose physics depends on that choice is machine-dependent
(2026-09-14: from-scratch D4h was correct on macOS Accelerate and wrong on
exxa). ``install(seed)`` replaces both functions with versions that flip the
sign of every eigen/singular vector at random and apply a random orthogonal
rotation inside every degenerate cluster, seeded by (seed, input matrix) so that
identical inputs get identical outputs as with a real LAPACK, so such a
dependence shows up on any machine. Use in a fresh process (angular
structures are cached).
"""
import numpy as np

_eigh, _svd = np.linalg.eigh, np.linalg.svd
TOL = 1e-8


def _clusters(vals):
    order = np.argsort(vals)
    groups, cur = [], [order[0]]
    for i in order[1:]:
        if abs(vals[i] - vals[cur[-1]]) < TOL * max(1.0, abs(vals[i])):
            cur.append(i)
        else:
            groups.append(cur); cur = [i]
    groups.append(cur)
    return [g for g in groups if len(g) > 1]


import sys as _sys
ONLY = None  # function name of the caller to scramble (None = all)


def _caller_ok():
    if ONLY is None:
        return True
    f = _sys._getframe(2)
    return f.f_code.co_name == ONLY


def install(seed, only=None, flip_signs=True):
    global ONLY
    ONLY = only
    import hashlib

    def _rng_for(a):
        # Deterministic in the input, like a real LAPACK: the same matrix always
        # gets the same scramble, so repeated uncached calls stay consistent.
        h = hashlib.blake2b(np.ascontiguousarray(a).tobytes(), digest_size=8).digest()
        return np.random.default_rng([seed, int.from_bytes(h, "little")])

    rng = None

    def rot(k):
        q, r = np.linalg.qr(rng.standard_normal((k, k)))
        return q * np.sign(np.diag(r))

    def eigh(a, *args, **kw):
        nonlocal rng
        w, v = _eigh(a, *args, **kw)
        rng = _rng_for(a)
        if not _caller_ok():
            return w, v
        v = v.copy()
        if flip_signs:  # LAPACK builds also differ in the sign of non-degenerate eigenvectors
            v = v * rng.choice([-1.0, 1.0], size=v.shape[1])
        for g in _clusters(w):
            R = rot(len(g))
            if np.iscomplexobj(v):
                R = R.astype(complex)
            v[:, g] = v[:, g] @ R
        return w, v

    def svd(a, full_matrices=True, compute_uv=True, **kw):
        nonlocal rng
        out = _svd(a, full_matrices=full_matrices, compute_uv=compute_uv, **kw)
        rng = _rng_for(a)
        if not compute_uv or not _caller_ok():
            return out
        U, S, Vh = out
        U, Vh = U.copy(), Vh.copy()
        if flip_signs:
            sgn = rng.choice([-1.0, 1.0], size=S.shape[0])
            U[:, :len(sgn)] = U[:, :len(sgn)] * sgn
            Vh[:len(sgn), :] = sgn[:, None] * Vh[:len(sgn), :]
        for g in _clusters(S):
            if S[g[0]] < 1e-12:
                continue
            R = rot(len(g)).astype(U.dtype)
            U[:, g] = U[:, g] @ R
            Vh[g, :] = R.conj().T @ Vh[g, :]
        return U, S, Vh

    np.linalg.eigh, np.linalg.svd = eigh, svd
