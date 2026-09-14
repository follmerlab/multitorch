"""Imitate a different LAPACK: random orthogonal rotations inside degenerate eigen/singular subspaces.

Every basis returned by ``np.linalg.eigh`` / ``np.linalg.svd`` inside a
degenerate cluster is as valid as any other, and LAPACK builds differ in which
they return. Code whose physics depends on that choice is machine-dependent
(2026-09-14: from-scratch D4h was correct on macOS Accelerate and wrong on
exxa). ``install(seed)`` replaces both functions with versions that apply a
seeded random orthogonal rotation inside every degenerate cluster, so such a
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


def install(seed, only=None):
    global ONLY
    ONLY = only
    rng = np.random.default_rng(seed)

    def rot(k):
        q, r = np.linalg.qr(rng.standard_normal((k, k)))
        return q * np.sign(np.diag(r))

    def eigh(a, *args, **kw):
        w, v = _eigh(a, *args, **kw)
        if not _caller_ok():
            return w, v
        v = v.copy()
        for g in _clusters(w):
            R = rot(len(g))
            if np.iscomplexobj(v):
                R = R.astype(complex)
            v[:, g] = v[:, g] @ R
        return w, v

    def svd(a, full_matrices=True, compute_uv=True, **kw):
        out = _svd(a, full_matrices=full_matrices, compute_uv=compute_uv, **kw)
        if not compute_uv or not _caller_ok():
            return out
        U, S, Vh = out
        U, Vh = U.copy(), Vh.copy()
        for g in _clusters(S):
            if S[g[0]] < 1e-12:
                continue
            R = rot(len(g)).astype(U.dtype)
            U[:, g] = U[:, g] @ R
            Vh[g, :] = R.conj().T @ Vh[g, :]
        return U, S, Vh

    np.linalg.eigh, np.linalg.svd = eigh, svd
