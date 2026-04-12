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
        - ``'ds'`` (float): Ds in eV.  D4h only (default 0.0).

        When provided, ``xham[0].values`` is rebuilt as
        ``[1.0, tendq, dt, ds]`` (D4h) or ``[1.0, tendq]`` (Oh).
        The leading ``1.0`` is the Hamiltonian (Coulomb + SOC) strength,
        which is always unity.
    delta : float or dict, optional
        Charge-transfer energy Δ.

        - If a **float**: sets ``eg[2] = delta``.  ``ef[2]`` is left
          unchanged (uses the template value).
        - If a **dict**: may contain ``'eg2'`` and/or ``'ef2'`` keys
          to set the ground-state and excited-state CT offsets
          independently.
    lmct : float or list, optional
        LMCT hybridization strength(s) V.

        - If a **float**: all ``xmix[0].values`` are set to this value.
        - If a **list**: used directly as ``xmix[0].values`` (must match
          the template's XMIX channel count).
    mlct : float or list, optional
        MLCT hybridization strength(s) — same convention as *lmct*.
        Applied to the same ``xmix[0].values`` channels.  If both
        *lmct* and *mlct* are provided, *mlct* overrides *lmct* for
        the second half of the channels (for fixtures with paired
        LMCT+MLCT mixing).

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
        # Only override if cf contains actual keys; empty dict = no override.
        # Values may be torch tensors (for autograd); do NOT call float().
        n_ops = len(out.xham[0].values)
        if 'tendq' in cf:
            out.xham[0].values[1] = cf['tendq']
        if 'dt' in cf and n_ops >= 3:
            out.xham[0].values[2] = cf['dt']
        if 'ds' in cf and n_ops >= 4:
            out.xham[0].values[3] = cf['ds']

    # ── Charge-transfer energy Δ ─────────────────────────────
    if delta is not None:
        if isinstance(delta, dict):
            if 'eg2' in delta:
                out.eg[2] = delta['eg2']
            if 'ef2' in delta:
                out.ef[2] = delta['ef2']
        else:
            # scalar (int, float, or torch.Tensor)
            out.eg[2] = delta

    # ── Hybridization V ──────────────────────────────────────
    if lmct is not None and out.xmix:
        n_ch = len(out.xmix[0].values)
        if isinstance(lmct, (list, tuple)):
            if len(lmct) != n_ch:
                raise ValueError(
                    f"lmct has {len(lmct)} values but template XMIX "
                    f"expects {n_ch}"
                )
            out.xmix[0].values = list(lmct)
        else:
            # scalar (int, float, or torch.Tensor)
            out.xmix[0].values = [lmct] * n_ch

    if mlct is not None and out.xmix:
        n_ch = len(out.xmix[0].values)
        if isinstance(mlct, (list, tuple)):
            half = n_ch // 2
            if len(mlct) != n_ch - half:
                raise ValueError(
                    f"mlct has {len(mlct)} values but template XMIX "
                    f"expects {n_ch - half} for the MLCT half"
                )
            for i, v in enumerate(mlct):
                out.xmix[0].values[half + i] = v
        else:
            # scalar (int, float, or torch.Tensor)
            half = n_ch // 2
            for i in range(half, n_ch):
                out.xmix[0].values[i] = mlct

    return out
