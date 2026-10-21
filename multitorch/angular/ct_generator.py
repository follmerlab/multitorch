"""
Two-configuration (ligand-to-metal charge transfer) L-edge template from scratch (WP-C, C4).

The store and RAC that ttrcg/ttrac write for a pyctm LMCT calculation,
d^n + d^(n+1)L → 2p^5 d^(n+1) + 2p^5 d^(n+2)L, generated without Fortran:

* Store (ttrcg layout, four sections):
  0. dipole d^n → 2p^5 d^(n+1);
  1. dipole d^(n+1)L → 2p^5 d^(n+2)L;
  2. ground manifold: HAMILTONIAN and SHELL (crystal field) of d^n and of
     d^(n+1)L, and the rank-0/2/4 hopping between them;
  3. final manifold: the same under the 2p hole.
  Every block is in the Fortran store basis and gauge
  (:mod:`multitorch.angular.cowan_operators`, each checked elementwise
  against the bundled ttrcg stores). HAMILTONIAN blocks are identity
  placeholders; the returned decomposition holds their operators.
* RAC: per manifold and irrep, HAMILTONIAN/10DQ(/DT/DS) actors of the first
  (kind ``GROUND``) and second (``EXCITE``) configuration, the hybridisation
  actors (``EGHYBR``/``T2GHYBR`` in Oh, ``B1HYBR``/``A1HYBR``/``B2HYBR``/``EHYBR``
  in D4h) coupling them, and the dipole TRANSI actors of configuration 1 then 2.
  The ADD coefficients come from the same projected emitters as the
  single-configuration template (:func:`~multitorch.angular.rac_generator.generate_ledge_template`).

Both parities of the electron count, 2 <= n <= 8: for odd n the RAC coefficients
come from the gauge-fixed double-group basis of
:mod:`multitorch.angular.double_group`, with the two configurations of each parity
sharing one node set and one invariant-channel basis per dipole triad (WP-B2a-5).
d^1 is the one refusal, for a structural reason (:func:`ct_configurations`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from multitorch._constants import DTYPE
from multitorch.angular.cowan_operators import (
    configuration_operators,
    dipole_blocks,
    final_state_hopping_blocks,
    hopping_blocks,
    shell_tensor_blocks,
)
from multitorch.angular import double_group
from multitorch.angular.double_group import DOUBLE_IRREPS, D4H_DOUBLE_IRREP_DIM, d4h_double_layout
from multitorch.angular.point_group import (
    OH_DOUBLE_IRREP_DIM,
    OH_IRREP_DIM,
    _build_coupling_operator,
    _c2r_unitary,
    _real_subduction_matrix,
    butler_label,
    oh_branching,
)
from multitorch.angular.rac_generator import (
    D4H_DOUBLE_TO_BUTLER,
    _d4h_dipole_channels_half,
    _d4h_partner_vector,
    _make_d4h_dipole_adds,
    _make_op_adds,
    _make_op_adds_half,
    _make_oh_dipole_adds,
    _make_oh_op_adds,
    _make_oh_op_adds_half,
    _oh_dipole_channels_half,
    _oh_dipole_pair_elements,
    _operator_real_matrix,
)
from multitorch.angular.symmetry import D4H_IRREP_DIM, D4H_TO_BUTLER, d4h_basis_layout, oh_to_d4h_subduction_matrix
from multitorch.angular.symmetry_plan import plan_for_sym
from multitorch.io.read_rme import ADDEntry, IrrepInfo, RACBlockFull, RACFileFull

Shell = Tuple[int, int]

_S5, _S30, _S42, _S70 = (math.sqrt(x) for x in (5.0, 30.0, 42.0, 70.0))

# Hybridisation actors as Butler route sums (rank, Oh route, strength): pyctm's
# nid8ct.rac (D4h) and eg = b1 + a1, t2g = b2 + e (Oh). With the route phases of
# _ttrac_route each channel is √5 × the projector onto its d orbitals on one
# electron (tests/test_angular/test_ct_generator.py).
HYBR_ROUTES: Dict[str, Dict[str, Tuple[Tuple[int, str, float], ...]]] = {
    'd4h': {
        'B1HYBR': ((0, 'A1', 1 / _S5), (4, 'A1', 3 * _S30 / 10), (4, 'E', -3 * _S42 / 14), (2, 'E', -_S70 / 7)),
        'A1HYBR': ((0, 'A1', 1 / _S5), (4, 'A1', 3 * _S30 / 10), (4, 'E', 3 * _S42 / 14), (2, 'E', _S70 / 7)),
        'B2HYBR': ((0, 'A1', 1 / _S5), (4, 'A1', -_S30 / 5), (4, 'E', 2 * _S42 / 7), (2, 'E', -_S70 / 7)),
        'EHYBR': ((0, 'A1', 2 / _S5), (4, 'A1', -2 * _S30 / 5), (4, 'E', -2 * _S42 / 7), (2, 'E', _S70 / 7)),
    },
    'oh': {
        'EGHYBR': ((0, 'A1', 2 / _S5), (4, 'A1', 3 * _S30 / 5)),
        'T2GHYBR': ((0, 'A1', 3 / _S5), (4, 'A1', -3 * _S30 / 5)),
    },
}
#: ``(operator vector name, store geometry name)`` per group. The Oh row
#: names ``TENDQ`` rather than ``10DQ`` since WP-B5-5b step 2: the first
#: element is what the shared emitter looks the operator vector up by, the
#: second is the store's own string. The two rows now differ only in length.
CF_ACTORS = {'oh': (('HAMILTONIAN', 'HAMILTONIAN'), ('TENDQ', '10DQ')),
             'd4h': (('HAMILTONIAN', 'HAMILTONIAN'), ('TENDQ', '10DQ'), ('DT', 'DT'), ('DS', 'DS'))}


def _ttrac_route(rank: int, oh_label: str) -> np.ndarray:
    """Real-harmonic D^rank vector of the D4h-A1g partner of one Butler route, in ttrac's phase.

    Eigen-decomposition signs are pinned to m = 0 positive; ttrac's rank-2 E
    route has the opposite phase (the Ballhausen pin flips DS and only DS in
    ``_d4h_operator_vector_complex``), which the one-electron projector
    property of the hybridisation actors confirms independently.
    """
    v = (_real_subduction_matrix(rank, oh_label) @ oh_to_d4h_subduction_matrix(oh_label + 'g')['A1g']).flatten()
    if v[rank] < 0:
        v = -v
    return -v if (rank, oh_label) == (2, 'E') else v


def hybridisation_operator_vectors(channel: str, sym: str) -> Dict[int, np.ndarray]:
    """Complex-basis operator vector per rank of one hybridisation actor."""
    out: Dict[int, np.ndarray] = {}
    for rank, oh_label, strength in HYBR_ROUTES[sym][channel]:
        out.setdefault(rank, np.zeros(2 * rank + 1))
        out[rank] = out[rank] + strength * _ttrac_route(rank, oh_label)
    return {k: _c2r_unitary(k).conj().T @ v.astype(np.complex128) for k, v in out.items()}


@dataclass(frozen=True)
class _Config:
    label: str
    section: int
    block_type: str
    shells: Tuple[Shell, ...]
    metal: Optional[int]        # index of the metal d shell, None when closed


def ct_configurations(n: int) -> Tuple[_Config, _Config, _Config, _Config]:
    """(d^n, d^(n+1)L, 2p^5 d^(n+1), 2p^5 d^(n+2)L) in ttrcg's shell order."""
    if n == 1:
        # d^1 has J = 3/2, 5/2 only, which branch to G3/2 and E5/2 + G3/2: no E1/2
        # (Gamma6) at all. Its ligand-hole partner d^2 L has J = 1/2 and does carry
        # E1/2, so that ground irrep exists in the SECOND configuration only. The
        # assembler reads a triad's per-configuration dimensions from the order of its
        # TRANSI blocks and would take the single block for configuration 1, so a
        # one-sided triad cannot be expressed (see `_emit_transi_group`). Supporting
        # it needs a dimension-carrying placeholder for the absent configuration;
        # plan residual 33.
        raise NotImplementedError(
            "charge transfer from scratch does not support 1 metal d electron: the E1/2 "
            "(Gamma6) ground irrep of d^1 + d^2 L exists only in the ligand-hole "
            "configuration, which the assembler's (configuration, copy) block indexing "
            "cannot express. Supported: 2 <= n <= 8 (plan residual 33)")
    if not 2 <= n <= 8:
        raise ValueError(f"charge transfer needs 2 <= n <= 8 metal d electrons, got {n}")
    ex_lh = ((1, 5), (2, 9)) if n == 8 else ((1, 5), (2, n + 2), (2, 9))
    return (
        _Config('gs', 2, 'GROUND', ((2, n),), 0),
        _Config('gs_lh', 2, 'EXCITE', ((2, n + 1), (2, 9)), 0),
        _Config('ex', 3, 'GROUND', ((1, 5), (2, n + 1)), 1),
        _Config('ex_lh', 3, 'EXCITE', ex_lh, None if n == 8 else 1),
    )


def _append(section: List[torch.Tensor], blocks: Dict, keys=None) -> Dict:
    """Append blocks (sorted keys) to a store section; return key → 1-based index."""
    idx = {}
    for key in sorted(blocks) if keys is None else keys:
        section.append(torch.as_tensor(np.ascontiguousarray(blocks[key]), dtype=DTYPE))
        idx[key] = len(section)
    return idx


def _config_store(section: List[torch.Tensor], cfg: _Config, sym: str) -> Dict[str, Dict]:
    ops = configuration_operators(cfg.shells)
    maps = {'ham': {}, 'cf4': {}, 'cf2': {}}
    for J in sorted(ops.dims):
        section.append(torch.eye(ops.dims[J], dtype=DTYPE))
        maps['ham'][J] = len(section)
    if cfg.metal is not None:
        maps['cf4'] = _append(section, shell_tensor_blocks(cfg.shells, cfg.metal, 4))
        if sym == 'd4h':
            maps['cf2'] = _append(section, shell_tensor_blocks(cfg.shells, cfg.metal, 2))
    return maps


def _real_transfer_element(
    vector: Callable[[float, object], np.ndarray], op_vecs: Dict[int, np.ndarray],
) -> Callable[[int, float, object, float, object], float]:
    """⟨v_b|O^(k)|v_k⟩ between real (integer-J) partner-0 vectors, with the i^(J_k−J_b) gauge."""
    cache: Dict[Tuple[int, float, float], np.ndarray] = {}

    def element(k, Jb, kb, Jk, kk):
        if (k, Jb, Jk) not in cache:
            cache[(k, Jb, Jk)] = _operator_real_matrix(Jb, Jk, k, op_vecs[k])
        me = float(vector(Jb, kb) @ cache[(k, Jb, Jk)] @ vector(Jk, kk))
        return (-1.0) ** (int(round(Jk - Jb)) // 2) * me

    return element


def _gauged_transfer_element(
    vector: Callable[[float, object], np.ndarray], op_vecs: Dict[int, np.ndarray],
) -> Callable[[int, float, object, float, object], float]:
    """⟨v_b|O^(k)|v_k⟩ between gauge-fixed double-group partner-0 vectors (half-integer J).

    The hybridisation actors are point-group scalars connecting the two
    configurations of one manifold, so the gauge that has to make them real is
    the one *across* configurations. It does, for free: the partner vectors of
    :mod:`~multitorch.angular.double_group` depend on (J, irrep, copy) and not
    on the configuration, so both configurations of a parity are given one
    shared node set and the reality already verified inside a configuration is
    the same statement. Verified here per element rather than assumed.
    """
    cache: Dict[Tuple[int, float, float], np.ndarray] = {}

    def element(k, Jb, kb, Jk, kk):
        if (k, Jb, Jk) not in cache:
            cache[(k, Jb, Jk)] = _build_coupling_operator(Jb, Jk, k, op_vecs[k])
        z = double_group.gauge_factor(Jb, Jk, k) * (
            vector(Jb, kb).conj() @ cache[(k, Jb, Jk)] @ vector(Jk, kk))
        if abs(z.imag) > double_group.REALITY_TOL * max(abs(z), 1.0):
            raise RuntimeError(f"hybridisation coupling (J={Jb}, {kb}) -> (J={Jk}, {kk}) at rank "
                               f"{k} is not real in the fixed gauge: {z!r}")
        return float(z.real)

    return element


def _scalar_transfer_adds(
    bra: Sequence[Tuple[float, object, int]],
    ket: Sequence[Tuple[float, object, int]],
    element: Callable[[int, float, object, float, object], float],
    ranks: Sequence[int],
    idx_by_rank: Dict[int, Dict[Tuple[float, float], int]],
    dim: int,
) -> List[ADDEntry]:
    """ADD entries of a scalar (A1g) operator between two configurations of one irrep.

    ``bra``/``ket`` list ``(J, key, n_states)`` in block order and ``element``
    (:func:`_real_transfer_element` or :func:`_gauged_transfer_element`) is the
    gauged matrix element between their partner-0 basis vectors, called for
    J_b ≤ J_k only. Same coefficient as the crystal-field emitters:
    √(dim/(2J_b+1)) ⟨v_b|O(J_b, J_k)|v_k⟩, mirrored with (−1)^(J_b−J_k) for
    J_b > J_k (UNCPLA symmetry of the rank-k store blocks), one entry per rank
    into that rank's store block. ``ranks`` are the actor's ranks; its vectors
    are already bound into ``element``.
    """

    def coefficient(k, Jb, kb, Jk, kk):
        if Jb > Jk:
            return (-1.0) ** int(round(Jb - Jk)) * coefficient(k, Jk, kk, Jb, kb)
        return math.sqrt(dim / (2.0 * Jb + 1.0)) * element(k, Jb, kb, Jk, kk)

    adds: List[ADDEntry] = []
    bra_pos = 1
    for Jb, kb, nb in bra:
        ket_pos = 1
        for Jk, kk, nk in ket:
            for k in sorted(ranks):
                idx = idx_by_rank.get(k, {})
                if abs(Jb - Jk) > k or Jb + Jk < k or (Jb, Jk) not in idx:
                    continue
                c = coefficient(k, Jb, kb, Jk, kk)
                if abs(c) > 1e-13:
                    adds.append(ADDEntry(matrix_idx=idx[(Jb, Jk)], bra=bra_pos, ket=ket_pos, nbra=nb, nket=nk, coeff=c))
            ket_pos += nk
        bra_pos += nb
    return adds


def _oh_j_order(dims: Dict[float, int], irrep: str) -> List[Tuple[float, int, int]]:
    return [(J, c, dims[J]) for J in sorted(dims) for c in range(oh_branching(J).get(irrep, 0))]


def generate_ct_ledge_template(n: int, sym: str = 'oh'):
    """Parameter-free (RAC, store, decomposition) for d^n + d^(n+1)L L-edge XAS.

    ``n`` metal d electrons in the ground configuration (2..8; d^1 is refused,
    see :func:`ct_configurations`); ``sym``
    'oh' or 'd4h'. The decomposition holds the four configurations, labelled
    ``gs`` (d^n), ``gs_lh`` (d^(n+1)L), ``ex`` (2p^5 d^(n+1)) and ``ex_lh``
    (2p^5 d^(n+2)L), with zero anchors; fill ``reference`` (eV, operator names of
    :func:`~multitorch.angular.cowan_operators.configuration_operators`) and
    contract with :func:`~multitorch.hamiltonian.parametric.rebuild_hamiltonian_store`.
    Pair with :func:`build_ct_ban` for the BAN.
    """
    from multitorch.hamiltonian.parametric import HamiltonianDecomposition, zero_anchor_config

    if sym not in ('oh', 'd4h'):
        raise ValueError(f"unsupported symmetry {sym!r}; supported: 'oh', 'd4h'")
    configs = ct_configurations(n)
    gs, gs_lh, ex, ex_lh = configs

    store: List[List[torch.Tensor]] = [[], [], [], []]
    dip = [_append(store[0], dipole_blocks(n, False)), _append(store[1], dipole_blocks(n, True))]
    maps = {}
    for cfg in (gs, gs_lh):
        maps[cfg.label] = _config_store(store[2], cfg, sym)
    hop_g = {k: _append(store[2], hopping_blocks(n, k)) for k in (0, 2, 4)}
    for cfg in (ex, ex_lh):
        maps[cfg.label] = _config_store(store[3], cfg, sym)
    hop_f = {k: _append(store[3], final_state_hopping_blocks(n, k)) for k in (0, 2, 4)}

    dims = {cfg.label: configuration_operators(cfg.shells).dims for cfg in configs}
    blocks: List[RACBlockFull] = []
    irreps: List[IrrepInfo] = []
    channels = list(HYBR_ROUTES[sym])
    hybr_vecs = {ch: hybridisation_operator_vectors(ch, sym) for ch in channels}

    # Operator ranks from the group's plan rather than a table in the emitter
    # (WP-B5-5b step 1). Keyed by the store's geometry string, which is what
    # ``CF_ACTORS`` carries as its second element.
    cf_ranks = {name: rank for name, rank, _key, _default
                in plan_for_sym(sym).cf_operators}

    # Half-integer J (odd n): one gauge-fixed double-group basis for the whole
    # calculation, with the two configurations of each parity sharing ONE node
    # set. The partner vectors depend on (J, irrep, copy) only, so a node per
    # parity (J values unioned over that parity's two configurations) is what
    # makes the cross-configuration hybridisation couplings real — the
    # configuration label never enters the angular problem.
    half = double_group.is_half_integer(max(dims['gs']))
    basis = None
    if half:
        basis = double_group.gauged_basis(
            [('g', set(dims['gs']) | set(dims['gs_lh'])),
             ('u', set(dims['ex']) | set(dims['ex_lh']))],
            d4h=(sym == 'd4h'),
        )
    oh_irreps = DOUBLE_IRREPS if half else ('A1', 'A2', 'E', 'T1', 'T2')
    oh_dim = OH_DOUBLE_IRREP_DIM if half else OH_IRREP_DIM
    d4h_layout = d4h_double_layout if half else d4h_basis_layout
    d4h_butler = D4H_DOUBLE_TO_BUTLER if half else D4H_TO_BUTLER

    def d4h_dim(d4h: str) -> int:
        return D4H_DOUBLE_IRREP_DIM[d4h[:-1]] if half else D4H_IRREP_DIM[d4h]

    def manifold(c1: _Config, c2: _Config, parity: str, kind: str, hop):
        if sym == 'oh':
            for irrep in oh_irreps:
                o1, o2 = _oh_j_order(dims[c1.label], irrep), _oh_j_order(dims[c2.label], irrep)
                if not o1 and not o2:
                    continue
                label, dim = butler_label(irrep, '+' if parity == 'g' else '-'), oh_dim[irrep]
                irreps.append(IrrepInfo(name=label, kind=kind, multiplicity=sum(x[2] for x in o1), dim=dim))
                for rac_kind, cfg, order in (('GROUND', c1, o1), ('EXCITE', c2, o2)):
                    if not order:
                        continue
                    m = maps[cfg.label]
                    for op, geometry in CF_ACTORS[sym]:
                        # WP-B5-5b steps 2 and 3: the shared emitters, Oh as
                        # its own target so the subduction step is the identity.
                        shared = dict(ham_idx_map=m['ham'], cf_idx_map=m['cf4'],
                                      cf_idx_map_rank2=m['cf4'], target_group='Oh',
                                      rank=cf_ranks.get(geometry))
                        entries = [(J, irrep, c, 0, n_st) for (J, c, n_st) in order]
                        if half:
                            adds = _make_op_adds_half(irrep + parity, parity, entries,
                                                      op, basis, **shared)
                        else:
                            adds = _make_op_adds(irrep + parity, entries, op, **shared)
                        if adds:
                            size = sum(x[2] for x in order)
                            blocks.append(RACBlockFull(kind=rac_kind, bra_sym=label, op_sym='0+', ket_sym=label,
                                                       geometry=geometry, n_bra=size, n_ket=size, add_entries=adds))
                if o1 and o2:
                    if half:
                        def vector(J, copy, irrep=irrep, parity=parity):
                            return basis.vectors[(parity, J, irrep, copy)]
                    else:
                        def vector(J, copy, irrep=irrep, dim=dim):
                            return np.ascontiguousarray(_real_subduction_matrix(int(round(J)), irrep)[:, copy * dim])
                    for ch in channels:
                        build = _gauged_transfer_element if half else _real_transfer_element
                        adds = _scalar_transfer_adds([(J, c, s) for J, c, s in o1], [(J, c, s) for J, c, s in o2],
                                                     build(vector, hybr_vecs[ch]), sorted(hybr_vecs[ch]), hop, dim)
                        if adds:
                            blocks.append(RACBlockFull(kind='TRANSI', bra_sym=label, op_sym='0+', ket_sym=label, geometry=ch,
                                                       n_bra=sum(x[2] for x in o1), n_ket=sum(x[2] for x in o2), add_entries=adds))
        else:
            l1, l2 = d4h_layout(dims[c1.label], parity=parity), d4h_layout(dims[c2.label], parity=parity)
            for d4h in sorted(set(l1) | set(l2)):
                e1 = [e for e in l1.get(d4h, []) if e[3] == 0]
                e2 = [e for e in l2.get(d4h, []) if e[3] == 0]
                if not e1 and not e2:
                    continue
                label, dim = d4h_butler[d4h], d4h_dim(d4h)
                irreps.append(IrrepInfo(name=label, kind=kind, multiplicity=sum(e[4] for e in e1), dim=dim))
                for rac_kind, cfg, entries in (('GROUND', c1, l1.get(d4h, [])), ('EXCITE', c2, l2.get(d4h, []))):
                    if not entries:
                        continue
                    m = maps[cfg.label]
                    size = sum(e[4] for e in entries) // dim
                    for op, geometry in CF_ACTORS[sym]:
                        adds = (_make_op_adds_half(d4h, parity, entries, op, basis,
                                                   ham_idx_map=m['ham'], cf_idx_map=m['cf4'],
                                                   cf_idx_map_rank2=m['cf2'],
                                                   rank=cf_ranks.get(geometry))
                                if half else
                                _make_op_adds(d4h, entries, op, ham_idx_map=m['ham'],
                                              cf_idx_map=m['cf4'], cf_idx_map_rank2=m['cf2']))
                        if adds:
                            blocks.append(RACBlockFull(kind=rac_kind, bra_sym=label, op_sym='0+', ket_sym=label,
                                                       geometry=geometry, n_bra=size, n_ket=size, add_entries=adds))
                if e1 and e2:
                    if half:
                        def vector(J, key, d4h=d4h, parity=parity):
                            return basis.d4h_partners((parity, J, key[0], key[1]), d4h[:-1])[:, 0]
                    else:
                        def vector(J, key, d4h=d4h):
                            return _d4h_partner_vector(J, key[0], key[1], d4h, 0)
                    for ch in channels:
                        build = _gauged_transfer_element if half else _real_transfer_element
                        adds = _scalar_transfer_adds([(e[0], (e[1], e[2]), e[4]) for e in e1],
                                                     [(e[0], (e[1], e[2]), e[4]) for e in e2],
                                                     build(vector, hybr_vecs[ch]), sorted(hybr_vecs[ch]), hop, dim)
                        if adds:
                            blocks.append(RACBlockFull(kind='TRANSI', bra_sym=label, op_sym='0+', ket_sym=label, geometry=ch,
                                                       n_bra=sum(e[4] for e in e1), n_ket=sum(e[4] for e in e2), add_entries=adds))

    # Dipole TRANSI actors: configuration 1 then 2 for every triad (the assembler reads
    # per-configuration dimensions in that order).
    pairs = ((gs, ex, dip[0]), (gs_lh, ex_lh, dip[1]))
    if sym == 'oh':
        for ig in oh_irreps:
            for ie in oh_irreps:
                orders = [(_oh_j_order(dims[g.label], ig), _oh_j_order(dims[e.label], ie), idx) for g, e, idx in pairs]
                if half:
                    # One invariant-channel basis over the pooled pairs of both
                    # configurations; they feed the same triad and mix coherently.
                    per_config = _oh_dipole_channels_half(ig, ie, orders, basis, ('g', 'u'))
                else:
                    weight: Dict[Tuple[int, int, int], float] = {}
                    for og, oe, idx in orders:
                        for t, w in _oh_dipole_pair_elements(ig, og, ie, oe, idx)[1].items():
                            weight[t] = weight.get(t, 0.0) + w
                    ranking = sorted(weight, key=lambda t: -weight[t])
                    per_config = [[_make_oh_dipole_adds(ig, og, ie, oe, idx, ranked_triples=ranking)] if og and oe else []
                                  for og, oe, idx in orders]
                emitted = [[RACBlockFull(kind='TRANSI', bra_sym=butler_label(ig, '+'), op_sym='1-',
                                         ket_sym=butler_label(ie, '-'), geometry='MULTIPOLE',
                                         n_bra=sum(x[2] for x in og), n_ket=sum(x[2] for x in oe),
                                         add_entries=adds) for adds in chans if adds]
                           for chans, (og, oe, _) in zip(per_config, orders)]
                _emit_transi_group(blocks, emitted)
    else:
        factors = (('Eu', '1-', 'PERP', math.sqrt(2.0 / 3.0)), ('A2u', '^0-', 'PARA', math.sqrt(1.0 / 3.0)))
        lay = [(d4h_layout(dims[g.label], 'g'), d4h_layout(dims[e.label], 'u'), idx) for g, e, idx in pairs]
        g_irreps = sorted(set(lay[0][0]) | set(lay[1][0]))
        u_irreps = sorted(set(lay[0][1]) | set(lay[1][1]))
        for dg in g_irreps:
            for du in u_irreps:
                for target, op_sym, geometry, factor in factors:
                    entries = [([e for e in lg.get(dg, []) if e[3] == 0],
                                [e for e in lu.get(du, []) if e[3] == 0], idx) for lg, lu, idx in lay]
                    if half:
                        per_config = _d4h_dipole_channels_half(dg, du, target, entries, basis, ('g', 'u'))
                    else:
                        per_config = [[_make_d4h_dipole_adds(d4h_gs=dg, gs_entries=eg, d4h_ex=du, ex_entries=eu,
                                                             op_d4h_target=target, multipole_idx=idx, factor=factor)]
                                      if eg and eu else [] for eg, eu, idx in entries]
                    emitted = [[RACBlockFull(kind='TRANSI', bra_sym=d4h_butler[dg], op_sym=op_sym,
                                             ket_sym=d4h_butler[du], geometry=geometry,
                                             n_bra=sum(e[4] for e in eg), n_ket=sum(e[4] for e in eu),
                                             add_entries=adds) for adds in chans if adds]
                               for chans, (eg, eu, _) in zip(per_config, entries)]
                    _emit_transi_group(blocks, emitted)

    manifold(gs, gs_lh, 'g', 'GROUND', hop_g)
    manifold(ex, ex_lh, 'u', 'EXCITE', hop_f)
    rac = RACFileFull(irreps=irreps, blocks=blocks, symmetry=sym)
    rac.index_maps = {'dipole': dip, 'hopping_ground': hop_g, 'hopping_final': hop_f, 'configs': maps}

    decomposition = HamiltonianDecomposition([
        zero_anchor_config(
            cfg.section, cfg.block_type, cfg.shells,
            {J: i - 1 for J, i in maps[cfg.label]['ham'].items()},
            configuration_operators(cfg.shells).blocks, dims[cfg.label],
            label=cfg.label, states=configuration_operators(cfg.shells).states,
        )
        for cfg in configs
    ])
    return rac, store, decomposition


def _emit_transi_group(blocks: List[RACBlockFull], emitted: List[List[RACBlockFull]]) -> None:
    """Append a triad's dipole blocks, grouped by configuration.

    ``emitted`` holds one list of blocks per configuration: one block per
    invariant channel of the triad (a single block for integer J, where every
    triad carries one invariant). The assembler reads the per-configuration
    dimensions of a triad from the order of its TRANSI blocks and then indexes
    the PRMULT copies inside a configuration, so the blocks go out grouped
    ``conf 1 copy 1, conf 1 copy 2, conf 2 copy 1, ...``.

    A triad is emitted only when every configuration has it, with the same
    number of channels; a one-sided triad would be mistaken for configuration 1
    and is not supported.
    """
    counts = {len(group) for group in emitted}
    if counts == {0}:
        return
    if len(counts) != 1:
        first = next(b for group in emitted if group for b in group)
        raise NotImplementedError(
            f"dipole triad {first.bra_sym} {first.op_sym} {first.ket_sym} has "
            f"{sorted(len(g) for g in emitted)} blocks across the configurations; "
            f"the assembler needs the same number in each")
    for group in emitted:
        blocks.extend(group)


def build_ct_ban(rac: RACFileFull, sym: str, *, cf: Optional[dict] = None, delta=0.0, u=0.0, hybridisation=None):
    """BanData for a :func:`generate_ct_ledge_template` RAC.

    ``cf``: Ballhausen ``tendq`` (and ``dt``, ``ds`` in D4h), raw as in the
    single-configuration from-scratch path. ``delta`` = E(d^(n+1)L) − E(d^n)
    (``EG2``), ``u`` = U_pd − U_dd (``EF2 = Δ − u``), pyctm's convention.
    ``hybridisation``: V per channel, as a dict keyed by the group's channel
    names (``{'eg', 't2g'}`` in Oh, ``{'b1', 'a1', 'b2', 'e'}`` in D4h), a list
    in that order, or one value for every channel; applied to the ground and
    final manifolds. The names, the order and the accepted ``cf`` keys are the
    group's :class:`~multitorch.angular.symmetry_plan.SymmetryPlan`, not a list
    kept here.

    *sym* must agree with ``rac.symmetry`` when the RAC carries one.
    """
    from multitorch.angular.symmetry_plan import plan_for_rac
    from multitorch.io.read_ban import BanData, XHAMEntry, XMIXEntry

    # The operator list, the channel list, the public key of each and the
    # default of each all come from the group's SymmetryPlan (WP-B5-4). They
    # used to be restated here, and differently at three other sites; see
    # residual 43 for what that cost.
    plan = plan_for_rac(rac, sym)
    xham = plan.xham_values(cf)
    values = plan.hybr_values(hybridisation)
    # One .ban entry per invariant channel of a triad, in RAC order. The
    # generator writes a triad's blocks grouped by configuration (n_channels
    # per configuration), and the assembler reads the k-th listing as PRMULT
    # copy k of every configuration, so the count divides out the
    # configurations -- ``nconf`` of them, the same number this BAN declares
    # as ``nconf_gs``/``nconf_fs``. LMCT is the only structure the generator
    # emits today; MLCT would raise it and must raise it here too. Half-integer J is the only case with more than one
    # channel (Γ8 ⊗ T1u ⊗ Γ8), and it is the repeated coupling ttrac writes
    # twice in the .ban.
    nconf = 2
    order: List[Tuple[str, str, str]] = []
    counts: Dict[Tuple[str, str, str], int] = {}
    for b in rac.blocks:
        if b.kind == 'TRANSI' and 'HYBR' not in b.geometry and b.add_entries:
            key = (b.bra_sym, b.op_sym, b.ket_sym)
            if key not in counts:
                order.append(key)
            counts[key] = counts.get(key, 0) + 1
    triads = []
    for key in order:
        n_channels, remainder = divmod(counts[key], nconf)
        if remainder or n_channels < 1:
            raise RuntimeError(f"triad {key} has {counts[key]} TRANSI blocks, not a whole number "
                               f"of channels for {nconf} configurations")
        triads.extend([key] * n_channels)
    return BanData(
        nconf_gs=nconf, nconf_fs=nconf,
        xham=[XHAMEntry(values=xham, combos=[(1, i + 1) for i in range(len(xham))])],
        xmix=[XMIXEntry(values=values, combos=[(1, 1, 2), (2, 1, 2)])],
        tran=[(1, 1), (2, 2)], triads=triads,
        eg={1: 0.0, 2: delta}, ef={1: 0.0, 2: delta - u},
        symmetry=sym,
    )
