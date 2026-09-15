"""
Template-based BanData builder for the Track C Phase 5 pipeline (C2).

Scope and approach
------------------
This module modifies a **parsed BanData template** (from a ``.ban`` fixture)
with user-supplied physical parameters.  It does NOT build BanData from
scratch — the structural information (triads, nconf, tran, n_band) comes
from the template, which encodes the angular-momentum selection rules and
configuration topology that Fortran ``ttban`` computed.

This is consistent with the loader-based approach used in C3d
(:mod:`~multitorch.hamiltonian.build_rac`) and C3e
(:mod:`~multitorch.hamiltonian.build_cowan`).

What gets overridden
--------------------
- **Crystal field** (``cf`` dict): maps to ``xham[0].values``.
  Oh symmetry uses ``[1.0, tendq]``; D4h uses ``[1.0, tendq, dt, ds]``.
- **Charge transfer energy** (``delta``): maps to ``eg[2]`` and ``ef[2]``
  (the energy offset of the CT configuration in ground and excited states).
- **Hybridization** (``lmct``/``mlct``): maps to ``xmix[0].values``
  (the mixing matrix elements V between configurations).

Parameters that are NOT overridden here (they route through the COWAN
store instead): ``slater_scale``, ``soc_scale``.  Those are handled by
:func:`~multitorch.atomic.scaled_params.scale_atomic_params` and
:func:`~multitorch.hamiltonian.build_cowan.build_cowan_store_in_memory`.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Union

from multitorch.io.read_ban import BanData, XHAMEntry, XMIXEntry


def modify_ban_params(
    ban: BanData,
    *,
    cf: Optional[Dict[str, Any]] = None,
    delta: Optional[Any] = None,
    u: Optional[Any] = None,
    lmct: Optional[Any] = None,
    mlct: Optional[Any] = None,
) -> BanData:
    """Return a copy of *ban* with user-supplied physical parameters applied.

    Parameters
    ----------
    ban : BanData
        Template BanData parsed from a ``.ban`` fixture file.
    cf : dict, optional
        Crystal-field parameters.  Keys:

        - ``'tendq'`` (float): 10Dq in eV.  Required for Oh and D4h.
        - ``'dt'`` (float): Dt in eV.  D4h only (default 0.0).

        A non-empty ``cf`` sets the whole crystal field: an omitted ``dt`` or
        ``ds`` is 0 (not the fixture template's value), an omitted ``tendq``
        keeps the template's 10Dq. ``None`` or ``{}`` keeps the template;
        :func:`multitorch.api.calc.fixture_defaults` lists it.
        - ``'ds'`` (float): Ds in eV.  D4h only (default 0.0).

        When provided, ``xham[0].values`` is rebuilt as
        ``[1.0, tendq - 35*dt/6, dt, ds]`` (D4h, Ballhausen 10Dq/Dt/Ds
        mapped onto Butler X400/X420/X220 as pyctm does) or ``[1.0, tendq]`` (Oh).
        The leading ``1.0`` is the Hamiltonian (Coulomb + SOC) strength,
        which is always unity.
    delta : float, tensor or dict, optional
        LMCT charge-transfer energy Δ = E(d^(n+1)L̲) − E(d^n), in eV.
        Conventions follow pyctm ``writeBAN``: ``EG2 = Δ``, ``EF2 = Δ − u``.

        - **scalar** or ``{'lmct': Δ}``: sets ``EG2 = Δ`` and
          ``EF2 = Δ − u``, where ``u`` is the argument below or, if not
          given, the template's ``EG2 − EF2`` (so the final-state CT
          energy follows Δ instead of staying at the template value).
        - ``{'eg2': …, 'ef2': …}``: set either offset directly (no ``u``
          logic; combining with ``u`` is an error).
    u : float, tensor or dict, optional
        ``u = U_pd − U_dd`` (pyctm's "Q − U"): how much the core hole lowers
        the ligand-hole configuration, ``EF2 = EG2 − u``. Scalar or
        ``{'lmct': u}``. Without ``delta`` it keeps the template Δ.
    lmct : float, tensor, list or dict, optional
        LMCT hopping integrals V(Γ) in eV, applied to ground and final state
        (the template's XMIX combos).

        - **dict**: by channel name, ``{'eg', 't2g'}`` (Oh) or
          ``{'b1', 'a1', 'b2', 'e'}`` (D4h); missing keys keep the template.
        - **list**: all channels, in that order.
        - **scalar**: the same V for every channel (note that V(e_g) ≈
          2·V(t_2g) physically).
    mlct : optional
        Not supported: no bundled fixture has an MLCT configuration.
        Passing anything but ``None`` raises ``ValueError``.

    Returns
    -------
    BanData
        A shallow copy of *ban* with the specified fields overridden.
        The original *ban* is not modified.

    Notes
    -----
    The ``triads``, ``nconf_gs``, ``nconf_fs``, ``tran``, ``n_band``,
    ``erange``, and ``prmult`` fields are never modified — they encode
    structural information from the angular-momentum selection rules.
    """
    out = copy.copy(ban)

    # Deep-copy the mutable containers we might modify
    out.eg = dict(ban.eg)
    out.ef = dict(ban.ef)
    out.xham = [XHAMEntry(values=list(x.values), combos=list(x.combos))
                for x in ban.xham]
    out.xmix = [XMIXEntry(values=list(x.values), combos=list(x.combos))
                for x in ban.xmix]

    # ── Crystal field ────────────────────────────────────────
    if cf and out.xham:
        allowed = {'tendq', 'dt', 'ds'} if len(out.xham[0].values) >= 4 else {'tendq'}
        unknown = set(cf) - allowed
        if unknown:
            raise ValueError(f"unknown crystal-field keys {sorted(unknown)} for this "
                             f"{'D4h' if len(allowed) == 3 else 'Oh'} fixture; allowed {sorted(allowed)}")
        # Only override if cf contains actual keys; empty dict = no override.
        # Values may be torch tensors (for autograd); do NOT call float().
        n_ops = len(out.xham[0].values)
        vals = out.xham[0].values
        if n_ops >= 3:
            # D4h (Butler chain O3 > Oh > D4h). The XHAM slot for the
            # rank-4 A1g operator is the *effective* cubic field
            #     10Dq_eff = 10Dq - 35*Dt/6
            # because Butler's X400 branch carries the cubic part of the
            # Ballhausen Dt operator (pyctm ``write_BAN.order_cf``; the
            # slot is named ``tendq_eff`` by ``read_ban``). The template
            # stores 10Dq_eff, so recover the raw 10Dq before overriding.
            #
            # A non-empty ``cf`` describes the whole crystal field: Dt and Ds it
            # omits are 0, as on the from-scratch path, not the template's
            # (nid8ct ships Ds = 0.1, so ``cf={'tendq': 1}`` used to keep a
            # tetragonal field; FABLE_HANDOFF N7). An omitted 10Dq keeps the
            # template's raw 10Dq. ``cf={}``/None keeps the fixture as shipped.
            dt_old = vals[2]
            tendq_old = vals[1] + 35.0 * dt_old / 6.0
            tendq_new = cf.get('tendq', tendq_old)
            dt_new = cf.get('dt', 0.0)
            vals[1] = tendq_new - 35.0 * dt_new / 6.0
            vals[2] = dt_new
            if n_ops >= 4:
                vals[3] = cf.get('ds', 0.0)
        elif 'tendq' in cf:
            vals[1] = cf['tendq']

    # ── Charge-transfer energies Δ and u (pyctm: EG2 = Δ, EF2 = Δ − u) ──
    if mlct is not None:
        raise ValueError(
            "mlct is not supported: the fixture has no MLCT configuration"
        )
    if (delta is not None or u is not None) and 2 not in ban.eg:
        raise ValueError("delta/u given but the fixture has no LMCT configuration")
    explicit = isinstance(delta, dict) and ('eg2' in delta or 'ef2' in delta)
    if explicit:
        if u is not None:
            raise ValueError("pass either delta={'eg2','ef2'} or u, not both")
        unknown = set(delta) - {'eg2', 'ef2'}
        if unknown:
            raise ValueError(f"unknown delta keys {sorted(unknown)}")
        if 'eg2' in delta:
            out.eg[2] = delta['eg2']
        if 'ef2' in delta:
            out.ef[2] = delta['ef2']
    elif delta is not None or u is not None:
        d = _lmct_value(delta, 'delta') if delta is not None else ban.eg[2]
        q = _lmct_value(u, 'u') if u is not None else ban.eg[2] - ban.ef.get(2, 0.0)
        out.eg[2] = d
        out.ef[2] = d - q

    # ── Hybridization V ──────────────────────────────────────
    if lmct is not None and out.xmix:
        n_ch = len(out.xmix[0].values)
        if isinstance(lmct, dict):
            names = _HYBR_CHANNELS.get(n_ch)
            if names is None:
                raise ValueError(f"no channel names for a {n_ch}-channel XMIX")
            unknown = set(lmct) - set(names)
            if unknown:
                raise ValueError(
                    f"unknown lmct channels {sorted(unknown)}; this fixture has {names}"
                )
            out.xmix[0].values = [lmct.get(name, v) for name, v in zip(names, out.xmix[0].values)]
        elif isinstance(lmct, (list, tuple)):
            if len(lmct) != n_ch:
                raise ValueError(
                    f"lmct has {len(lmct)} values but template XMIX "
                    f"expects {n_ch}"
                )
            out.xmix[0].values = list(lmct)
        else:
            # scalar (int, float, or torch.Tensor)
            out.xmix[0].values = [lmct] * n_ch

    return out


# XMIX channel order = RAC hybridization blocks EGHYBR, T2GHYBR (Oh) and
# B1HYBR, A1HYBR, B2HYBR, EHYBR (D4h); pyctm write_BAN mix_keys.
_HYBR_CHANNELS = {2: ('eg', 't2g'), 4: ('b1', 'a1', 'b2', 'e')}


def _lmct_value(x, name):
    if isinstance(x, dict):
        unknown = set(x) - {'lmct'}
        if unknown:
            raise ValueError(f"{name} dict accepts only 'lmct' (got {sorted(x)})")
        return x['lmct']
    return x
