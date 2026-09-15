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

Integer J only (even n); half-integer J is WP-B2a.
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
from multitorch.angular.point_group import OH_IRREP_DIM, _c2r_unitary, _real_subduction_matrix, butler_label, oh_branching
from multitorch.angular.rac_generator import (
    _d4h_partner_vector,
    _make_d4h_dipole_adds,
    _make_d4h_op_adds,
    _make_oh_dipole_adds,
    _make_oh_op_adds,
    _oh_dipole_pair_elements,
    _operator_real_matrix,
)
from multitorch.angular.symmetry import D4H_IRREP_DIM, D4H_TO_BUTLER, d4h_basis_layout, oh_to_d4h_subduction_matrix
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
CF_ACTORS = {'oh': (('HAMILTONIAN', 'HAMILTONIAN'), ('10DQ', '10DQ')),
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
    if not 1 <= n <= 8:
        raise ValueError(f"charge transfer needs 1 <= n <= 8 metal d electrons, got {n}")
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


def _scalar_transfer_adds(
    bra: Sequence[Tuple[float, object, int]],
    ket: Sequence[Tuple[float, object, int]],
    vector: Callable[[float, object], np.ndarray],
    op_vecs: Dict[int, np.ndarray],
    idx_by_rank: Dict[int, Dict[Tuple[float, float], int]],
    dim: int,
) -> List[ADDEntry]:
    """ADD entries of a scalar (A1g) operator between two configurations of one irrep.

    ``bra``/``ket`` list ``(J, key, n_states)`` in block order and ``vector(J, key)``
    is the partner-0 basis vector. Same coefficient as the crystal-field
    emitters: √(dim/(2J_b+1)) ⟨v_b|O(J_b, J_k)|v_k⟩ with the i^(J_k−J_b) gauge,
    evaluated for J_b ≤ J_k and mirrored with (−1)^(J_b−J_k) (UNCPLA symmetry of
    the rank-k store blocks), one entry per rank into that rank's store block.
    """
    O_cache: Dict[Tuple[int, float, float], np.ndarray] = {}

    def coefficient(k, Jb, kb, Jk, kk):
        if Jb > Jk:
            return (-1.0) ** int(round(Jb - Jk)) * coefficient(k, Jk, kk, Jb, kb)
        if (k, Jb, Jk) not in O_cache:
            O_cache[(k, Jb, Jk)] = _operator_real_matrix(Jb, Jk, k, op_vecs[k])
        me = float(vector(Jb, kb) @ O_cache[(k, Jb, Jk)] @ vector(Jk, kk))
        return (-1.0) ** (int(round(Jk - Jb)) // 2) * math.sqrt(dim / (2.0 * Jb + 1.0)) * me

    adds: List[ADDEntry] = []
    bra_pos = 1
    for Jb, kb, nb in bra:
        ket_pos = 1
        for Jk, kk, nk in ket:
            for k in sorted(op_vecs):
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

    ``n`` metal d electrons in the ground configuration (even, 2..8); ``sym``
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
    if n % 2:
        raise NotImplementedError("charge transfer from scratch supports integer J (even n) only; half-integer J is WP-B2a")
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

    def manifold(c1: _Config, c2: _Config, parity: str, kind: str, hop):
        if sym == 'oh':
            for irrep in ('A1', 'A2', 'E', 'T1', 'T2'):
                o1, o2 = _oh_j_order(dims[c1.label], irrep), _oh_j_order(dims[c2.label], irrep)
                if not o1 and not o2:
                    continue
                label, dim = butler_label(irrep, '+' if parity == 'g' else '-'), OH_IRREP_DIM[irrep]
                irreps.append(IrrepInfo(name=label, kind=kind, multiplicity=sum(x[2] for x in o1), dim=dim))
                for rac_kind, cfg, order in (('GROUND', c1, o1), ('EXCITE', c2, o2)):
                    if not order:
                        continue
                    m = maps[cfg.label]
                    for op, geometry in CF_ACTORS[sym]:
                        adds = _make_oh_op_adds(irrep, order, op, m['ham'], m['cf4'])
                        if adds:
                            size = sum(x[2] for x in order)
                            blocks.append(RACBlockFull(kind=rac_kind, bra_sym=label, op_sym='0+', ket_sym=label,
                                                       geometry=geometry, n_bra=size, n_ket=size, add_entries=adds))
                if o1 and o2:
                    def vector(J, copy, irrep=irrep, dim=dim):
                        return np.ascontiguousarray(_real_subduction_matrix(int(round(J)), irrep)[:, copy * dim])
                    for ch in channels:
                        adds = _scalar_transfer_adds([(J, c, s) for J, c, s in o1], [(J, c, s) for J, c, s in o2],
                                                     vector, hybr_vecs[ch], hop, dim)
                        if adds:
                            blocks.append(RACBlockFull(kind='TRANSI', bra_sym=label, op_sym='0+', ket_sym=label, geometry=ch,
                                                       n_bra=sum(x[2] for x in o1), n_ket=sum(x[2] for x in o2), add_entries=adds))
        else:
            l1, l2 = d4h_basis_layout(dims[c1.label], parity=parity), d4h_basis_layout(dims[c2.label], parity=parity)
            for d4h in sorted(set(l1) | set(l2)):
                e1 = [e for e in l1.get(d4h, []) if e[3] == 0]
                e2 = [e for e in l2.get(d4h, []) if e[3] == 0]
                if not e1 and not e2:
                    continue
                label, dim = D4H_TO_BUTLER[d4h], D4H_IRREP_DIM[d4h]
                irreps.append(IrrepInfo(name=label, kind=kind, multiplicity=sum(e[4] for e in e1), dim=dim))
                for rac_kind, cfg, entries in (('GROUND', c1, l1.get(d4h, [])), ('EXCITE', c2, l2.get(d4h, []))):
                    if not entries:
                        continue
                    m = maps[cfg.label]
                    size = sum(e[4] for e in entries) // dim
                    for op, geometry in CF_ACTORS[sym]:
                        adds = _make_d4h_op_adds(d4h, entries, op, ham_idx_map=m['ham'], cf_idx_map=m['cf4'],
                                                 cf_idx_map_rank2=m['cf2'])
                        if adds:
                            blocks.append(RACBlockFull(kind=rac_kind, bra_sym=label, op_sym='0+', ket_sym=label,
                                                       geometry=geometry, n_bra=size, n_ket=size, add_entries=adds))
                if e1 and e2:
                    def vector(J, key, d4h=d4h):
                        return _d4h_partner_vector(J, key[0], key[1], d4h, 0)
                    for ch in channels:
                        adds = _scalar_transfer_adds([(e[0], (e[1], e[2]), e[4]) for e in e1],
                                                     [(e[0], (e[1], e[2]), e[4]) for e in e2],
                                                     vector, hybr_vecs[ch], hop, dim)
                        if adds:
                            blocks.append(RACBlockFull(kind='TRANSI', bra_sym=label, op_sym='0+', ket_sym=label, geometry=ch,
                                                       n_bra=sum(e[4] for e in e1), n_ket=sum(e[4] for e in e2), add_entries=adds))

    # Dipole TRANSI actors: configuration 1 then 2 for every triad (the assembler reads
    # per-configuration dimensions in that order).
    pairs = ((gs, ex, dip[0]), (gs_lh, ex_lh, dip[1]))
    if sym == 'oh':
        irr = ('A1', 'A2', 'E', 'T1', 'T2')
        for ig in irr:
            for ie in irr:
                orders = [(_oh_j_order(dims[g.label], ig), _oh_j_order(dims[e.label], ie), idx) for g, e, idx in pairs]
                weight: Dict[Tuple[int, int, int], float] = {}
                for og, oe, idx in orders:
                    for t, w in _oh_dipole_pair_elements(ig, og, ie, oe, idx)[1].items():
                        weight[t] = weight.get(t, 0.0) + w
                ranking = sorted(weight, key=lambda t: -weight[t])
                emitted = []
                for og, oe, idx in orders:
                    adds = _make_oh_dipole_adds(ig, og, ie, oe, idx, ranked_triples=ranking) if og and oe else []
                    emitted.append(RACBlockFull(kind='TRANSI', bra_sym=butler_label(ig, '+'), op_sym='1-',
                                                ket_sym=butler_label(ie, '-'), geometry='MULTIPOLE',
                                                n_bra=sum(x[2] for x in og), n_ket=sum(x[2] for x in oe),
                                                add_entries=adds) if adds else None)
                _emit_transi_pair(blocks, emitted)
    else:
        factors = (('Eu', '1-', 'PERP', math.sqrt(2.0 / 3.0)), ('A2u', '^0-', 'PARA', math.sqrt(1.0 / 3.0)))
        lay = [(d4h_basis_layout(dims[g.label], 'g'), d4h_basis_layout(dims[e.label], 'u'), idx) for g, e, idx in pairs]
        g_irreps = sorted(set(lay[0][0]) | set(lay[1][0]))
        u_irreps = sorted(set(lay[0][1]) | set(lay[1][1]))
        for dg in g_irreps:
            for du in u_irreps:
                for target, op_sym, geometry, factor in factors:
                    emitted = []
                    for lg, lu, idx in lay:
                        eg = [e for e in lg.get(dg, []) if e[3] == 0]
                        eu = [e for e in lu.get(du, []) if e[3] == 0]
                        adds = (_make_d4h_dipole_adds(d4h_gs=dg, gs_entries=eg, d4h_ex=du, ex_entries=eu,
                                                      op_d4h_target=target, multipole_idx=idx, factor=factor)
                                if eg and eu else [])
                        emitted.append(RACBlockFull(kind='TRANSI', bra_sym=D4H_TO_BUTLER[dg], op_sym=op_sym,
                                                    ket_sym=D4H_TO_BUTLER[du], geometry=geometry,
                                                    n_bra=sum(e[4] for e in eg), n_ket=sum(e[4] for e in eu),
                                                    add_entries=adds) if adds else None)
                    _emit_transi_pair(blocks, emitted)

    manifold(gs, gs_lh, 'g', 'GROUND', hop_g)
    manifold(ex, ex_lh, 'u', 'EXCITE', hop_f)
    rac = RACFileFull(irreps=irreps, blocks=blocks)
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


def _emit_transi_pair(blocks: List[RACBlockFull], emitted: List[Optional[RACBlockFull]]) -> None:
    """Append a triad's dipole blocks for configurations 1 and 2.

    The assembler reads the per-configuration dimensions of a triad from the
    order of its TRANSI blocks, so a triad is emitted only when both
    configurations have it; one-sided triads would be mistaken for
    configuration 1 and are not supported.
    """
    present = [b for b in emitted if b is not None]
    if not present:
        return
    if len(present) != len(emitted):
        raise NotImplementedError(
            f"dipole triad {present[0].bra_sym} {present[0].op_sym} {present[0].ket_sym} exists in only one configuration")
    blocks.extend(present)


def build_ct_ban(rac: RACFileFull, sym: str, *, cf: Optional[dict] = None, delta=0.0, u=0.0, hybridisation=None):
    """BanData for a :func:`generate_ct_ledge_template` RAC.

    ``cf``: Ballhausen ``tendq`` (and ``dt``, ``ds`` in D4h), raw as in the
    single-configuration from-scratch path. ``delta`` = E(d^(n+1)L) − E(d^n)
    (``EG2``), ``u`` = U_pd − U_dd (``EF2 = Δ − u``), pyctm's convention.
    ``hybridisation``: V per channel, dict ``{'eg', 't2g'}`` (Oh) or
    ``{'b1', 'a1', 'b2', 'e'}`` (D4h), or a list in that order, applied to the
    ground and final manifolds.
    """
    from multitorch.io.read_ban import BanData, XHAMEntry, XMIXEntry

    cf = cf or {}
    names = {'oh': ('eg', 't2g'), 'd4h': ('b1', 'a1', 'b2', 'e')}[sym]
    unknown = set(cf) - ({'tendq'} if sym == 'oh' else {'tendq', 'dt', 'ds'})
    if unknown:
        raise ValueError(f"unknown crystal-field keys {sorted(unknown)} for {sym}")
    if hybridisation is None:
        values = [0.0] * len(names)
    elif isinstance(hybridisation, dict):
        unknown = set(hybridisation) - set(names)
        if unknown:
            raise ValueError(f"unknown hybridisation channels {sorted(unknown)}; {sym} has {names}")
        values = [hybridisation.get(k, 0.0) for k in names]
    else:
        values = list(hybridisation)
        if len(values) != len(names):
            raise ValueError(f"{sym} needs {len(names)} hybridisation values {names}")
    xham = [1.0, cf.get('tendq', 0.0)] + ([cf.get('dt', 0.0), cf.get('ds', 0.0)] if sym == 'd4h' else [])
    triads = []
    for b in rac.blocks:
        if b.kind == 'TRANSI' and 'HYBR' not in b.geometry and (b.bra_sym, b.op_sym, b.ket_sym) not in triads:
            triads.append((b.bra_sym, b.op_sym, b.ket_sym))
    return BanData(
        nconf_gs=2, nconf_fs=2,
        xham=[XHAMEntry(values=xham, combos=[(1, i + 1) for i in range(len(xham))])],
        xmix=[XMIXEntry(values=values, combos=[(1, 1, 2), (2, 1, 2)])],
        tran=[(1, 1), (2, 2)], triads=triads,
        eg={1: 0.0, 2: delta}, ef={1: 0.0, 2: delta - u},
    )
