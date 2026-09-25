"""Imitate a different LAPACK: random orthogonal rotations inside degenerate eigen/singular subspaces.

Every basis returned by ``np.linalg.eigh`` / ``np.linalg.svd`` inside a
degenerate cluster is as valid as any other, and LAPACK builds differ in which
they return. Code whose physics depends on that choice is machine-dependent
(2026-09-14: from-scratch D4h was correct on macOS Accelerate and wrong on
exxa). ``install(seed)`` replaces both functions with versions that randomise the
gauge of every eigen/singular vector and apply a random rotation inside every
degenerate cluster, seeded by (seed, input matrix) so that identical inputs get
identical outputs as with a real LAPACK, so such a dependence shows up on any
machine. Use in a fresh process (angular structures are cached).

**Complex inputs get the full freedom (extended 2026-09-25).** For a real
symmetric matrix the only freedoms are a sign per eigenvector and a real
orthogonal rotation inside a degenerate cluster. For a *complex Hermitian* one
they are an arbitrary **phase** per eigenvector and an arbitrary **unitary**
mixing inside a cluster — strictly more. The half-integer path added two
complex-Hermitian ``eigh`` calls (``double_group.canonical_partner_sets``' rank-4
marker, whose imaginary part reaches 0.42, and
``oh_to_d4h_double_subduction``'s transfer projector), and a real-only scramble
never exercises the freedom they actually have. A scientific audit on
2026-09-25 found this gap and confirmed the path survives the stronger version
(relative spectral change ≤ 5e-14 over three seeds in Oh and D4h), so the
extension protects a property that already holds rather than fixing a defect.
Real inputs keep exactly the old behaviour, so integer-J results and their
determinism are unchanged.
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

    def rot(k, complex_valued=False):
        """Haar-random orthogonal (real) or unitary (complex) k x k matrix."""
        if complex_valued:
            x = rng.standard_normal((k, k)) + 1j * rng.standard_normal((k, k))
            q, r = np.linalg.qr(x)
            d = np.diagonal(r)
            return q * (d / np.abs(d)).conj()
        q, r = np.linalg.qr(rng.standard_normal((k, k)))
        return q * np.sign(np.diag(r))

    def _gauge(n, complex_valued):
        """Per-vector freedom: a sign if real, an arbitrary phase if complex."""
        if complex_valued:
            return np.exp(2j * np.pi * rng.random(n))
        return rng.choice([-1.0, 1.0], size=n)

    def eigh(a, *args, **kw):
        nonlocal rng
        w, v = _eigh(a, *args, **kw)
        rng = _rng_for(a)
        if not _caller_ok():
            return w, v
        v = v.copy()
        cx = np.iscomplexobj(v)
        if flip_signs:  # LAPACK builds also differ in the gauge of non-degenerate eigenvectors
            v = v * _gauge(v.shape[1], cx)
        for g in _clusters(w):
            v[:, g] = v[:, g] @ rot(len(g), cx)
        return w, v

    def svd(a, full_matrices=True, compute_uv=True, **kw):
        nonlocal rng
        out = _svd(a, full_matrices=full_matrices, compute_uv=compute_uv, **kw)
        rng = _rng_for(a)
        if not compute_uv or not _caller_ok():
            return out
        U, S, Vh = out
        U, Vh = U.copy(), Vh.copy()
        cx = np.iscomplexobj(U) or np.iscomplexobj(Vh)
        if flip_signs:
            # A = U diag(S) Vh is preserved by U[:,i] -> U[:,i] p_i together with
            # Vh[i,:] -> conj(p_i) Vh[i,:] for any |p_i| = 1 (a sign is its own conjugate).
            sgn = _gauge(S.shape[0], cx)
            U[:, :len(sgn)] = U[:, :len(sgn)] * sgn
            Vh[:len(sgn), :] = np.conj(sgn)[:, None] * Vh[:len(sgn), :]
        for g in _clusters(S):
            if S[g[0]] < 1e-12:
                continue
            R = rot(len(g), cx).astype(U.dtype)
            U[:, g] = U[:, g] @ R
            Vh[g, :] = R.conj().T @ Vh[g, :]
        return U, S, Vh

    np.linalg.eigh, np.linalg.svd = eigh, svd
