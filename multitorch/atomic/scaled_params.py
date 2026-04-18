"""
Slater / spin-orbit scaling pipeline for the Track C Phase 5 builder.

Why this module exists
----------------------
Cowan's empirical practice is to multiply the bare HFS Slater Fk/Gk
integrals by an empirical reduction factor (typically 0.8 for d-d, 0.85
for p-d) before they enter the multiplet Hamiltonian, and to similarly
re-scale the spin-orbit ζ. These scaling factors are *the* gradient
entry points for the Track C autograd story — they are not baked into
the angular machinery (``compute_*_blocks``) and they are not baked into
the file parser (``read_rcn31_out_params``). They live here because:

  1. The atomic-parameter parser yields plain Python floats, which are
     not autograd-traceable.
  2. The COWAN store builder (C3e) needs torch tensors with the scale
     factors *already applied*, so the angular block can be multiplied
     by a single scalar that already carries the gradient graph.
  3. Exposing ``slater_scale`` and ``soc_scale`` as separate function
     arguments to ``calcXAS`` (in C4) means users can compute
     ``torch.autograd.grad(loss, slater_scale)`` directly.

Contract
--------
``scale_atomic_params(params, slater_scale, soc_scale, ...)`` consumes
an :class:`AtomicParams` (parsed from ``.rcn31_out``) and returns a
:class:`ScaledAtomicParams` whose Fk/Gk/ζ values are torch tensors.

If ``slater_scale`` is a Python float, the returned tensors are leaf
constants (no graph). If it is a torch scalar tensor with
``requires_grad=True``, every Fk/Gk in the output participates in the
autograd graph through that scalar — backward through the resulting
COWAN store will land on ``slater_scale``.

The same applies independently to ``soc_scale`` for ζ.

For the Track C C3f parity test, set both scales to 1.0; the
``.rcn31_out`` values are the ab-initio HFS integrals that the
reference COWAN store was built from, so any scaling would break the
element-wise comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

import torch

from multitorch._constants import DTYPE
from multitorch.atomic.parameter_fixtures import AtomicParams, ConfigParams

ScalarLike = Union[float, int, torch.Tensor]


# ─────────────────────────────────────────────────────────────
# Output dataclasses: torch-tensor mirrors of the parser dataclasses
# ─────────────────────────────────────────────────────────────


@dataclass
class ScaledConfigParams:
    """One configuration's atomic parameters as torch tensors.

    All Fk/Gk/ζ values are 0-dim torch tensors (``DTYPE`` = float64).
    They may carry an autograd graph reaching back to ``slater_scale``
    or ``soc_scale``, depending on whether those were torch scalars
    with ``requires_grad=True``.
    """

    label: str
    nconf: int
    fk: Dict[Tuple[str, str, int], torch.Tensor] = field(default_factory=dict)
    gk: Dict[Tuple[str, str, int], torch.Tensor] = field(default_factory=dict)
    zeta: Dict[str, torch.Tensor] = field(default_factory=dict)

    def f(self, a: str, b: str, k: int) -> torch.Tensor:
        a, b = a.upper(), b.upper()
        if (a, b, k) in self.fk:
            return self.fk[(a, b, k)]
        if (b, a, k) in self.fk:
            return self.fk[(b, a, k)]
        raise KeyError(f"F^{k}({a},{b}) not found in {self.label}")

    def g(self, a: str, b: str, k: int) -> torch.Tensor:
        a, b = a.upper(), b.upper()
        if (a, b, k) in self.gk:
            return self.gk[(a, b, k)]
        if (b, a, k) in self.gk:
            return self.gk[(b, a, k)]
        raise KeyError(f"G^{k}({a},{b}) not found in {self.label}")

    def z(self, shell: str) -> torch.Tensor:
        shell = shell.upper()
        if shell not in self.zeta:
            raise KeyError(f"ζ({shell}) not found in {self.label}")
        return self.zeta[shell]


@dataclass
class ScaledAtomicParams:
    """All scaled atomic parameters from a single .rcn31_out file."""

    configs: List[ScaledConfigParams] = field(default_factory=list)

    def by_nconf(self, nconf: int) -> ScaledConfigParams:
        for c in self.configs:
            if c.nconf == nconf:
                return c
        raise KeyError(
            f"NCONF={nconf} not found "
            f"(have {[c.nconf for c in self.configs]})"
        )

    @property
    def ground(self) -> ScaledConfigParams:
        return self.by_nconf(1)

    @property
    def excited(self) -> ScaledConfigParams:
        return self.by_nconf(2)


# ─────────────────────────────────────────────────────────────
# Scaling driver
# ─────────────────────────────────────────────────────────────


def _as_dtype_tensor(x: ScalarLike) -> torch.Tensor:
    """Coerce a scalar to a 0-d ``DTYPE`` tensor without breaking autograd.

    If ``x`` is already a torch tensor, return it as-is (so its
    ``requires_grad`` flag and any in-graph history are preserved). If
    it's a Python number, wrap it as a constant.
    """
    if isinstance(x, torch.Tensor):
        # Cast dtype if needed but keep the autograd connection.
        if x.dtype != DTYPE:
            return x.to(dtype=DTYPE)
        return x
    return torch.as_tensor(x, dtype=DTYPE)


def _scale_one_config(
    cfg: ConfigParams,
    slater_scale: torch.Tensor,
    soc_scale: torch.Tensor,
    zeta_method: str,
) -> ScaledConfigParams:
    fk_scaled: Dict[Tuple[str, str, int], torch.Tensor] = {}
    for key, val in cfg.fk.items():
        fk_scaled[key] = slater_scale * val

    gk_scaled: Dict[Tuple[str, str, int], torch.Tensor] = {}
    for key, val in cfg.gk.items():
        gk_scaled[key] = slater_scale * val

    if zeta_method == "blume_watson":
        zeta_src = cfg.zeta_bw
    elif zeta_method == "rvi":
        zeta_src = cfg.zeta_rvi
    else:
        raise ValueError(
            f"zeta_method must be 'blume_watson' or 'rvi', got {zeta_method!r}"
        )

    zeta_scaled: Dict[str, torch.Tensor] = {}
    for shell, val in zeta_src.items():
        zeta_scaled[shell] = soc_scale * val

    return ScaledConfigParams(
        label=cfg.label,
        nconf=cfg.nconf,
        fk=fk_scaled,
        gk=gk_scaled,
        zeta=zeta_scaled,
    )


def scale_atomic_params(
    params: AtomicParams,
    slater_scale: ScalarLike = 1.0,
    soc_scale: ScalarLike = 1.0,
    *,
    zeta_method: str = "blume_watson",
) -> ScaledAtomicParams:
    """Multiply Fk/Gk by ``slater_scale`` and ζ by ``soc_scale``.

    Parameters
    ----------
    params : AtomicParams
        Parsed atomic parameters from :func:`read_rcn31_out_params`.
    slater_scale : float or torch.Tensor, default 1.0
        Empirical reduction factor applied to every Fk and Gk integral.
        If a torch tensor with ``requires_grad=True``, the entire
        scaled output carries an autograd graph back to this tensor.
    soc_scale : float or torch.Tensor, default 1.0
        Empirical reduction factor applied to every spin-orbit ζ.
        Same autograd semantics as ``slater_scale``.
    zeta_method : {'blume_watson', 'rvi'}
        Which ζ column from the .rcn31_out file to use.
        ``'blume_watson'`` is the default and the recommended physics.

    Returns
    -------
    ScaledAtomicParams
        Mirror of ``params`` with all values as 0-dim torch tensors.

    Notes
    -----
    For the C3f parity test against a parsed `.rme_rcg` fixture, use
    ``slater_scale=1.0`` and ``soc_scale=1.0`` — the COWAN store in the
    fixture was built from these exact ab-initio rcn31 values without
    further scaling.
    """
    slater_t = _as_dtype_tensor(slater_scale)
    soc_t = _as_dtype_tensor(soc_scale)

    scaled_configs = [
        _scale_one_config(c, slater_t, soc_t, zeta_method)
        for c in params.configs
    ]
    return ScaledAtomicParams(configs=scaled_configs)


# ─────────────────────────────────────────────────────────────
# Batch scaling for parameter sweeps
# ─────────────────────────────────────────────────────────────


def _scale_one_config_batch(
    cfg: ConfigParams,
    slater_scales: torch.Tensor,  # (N,)
    soc_scales: torch.Tensor,     # (N,)
    zeta_method: str,
) -> ScaledConfigParams:
    """Scale one config with a batch of (N,) scale factors.
    
    Returns ScaledConfigParams where each Fk/Gk/ζ is a (N,) tensor
    instead of a scalar. Each sample preserves independent gradients.
    """
    N = slater_scales.shape[0]
    assert soc_scales.shape[0] == N, "slater and soc batches must match"
    
    # Fk scaled: each (N,) = slater_scales * val
    fk_scaled: Dict[Tuple[str, str, int], torch.Tensor] = {}
    for key, val in cfg.fk.items():
        fk_scaled[key] = slater_scales * val  # broadcast: (N,) * scalar → (N,)
    
    # Gk scaled: each (N,) = slater_scales * val
    gk_scaled: Dict[Tuple[str, str, int], torch.Tensor] = {}
    for key, val in cfg.gk.items():
        gk_scaled[key] = slater_scales * val
    
    # Zeta source selection
    if zeta_method == "blume_watson":
        zeta_src = cfg.zeta_bw
    elif zeta_method == "rvi":
        zeta_src = cfg.zeta_rvi
    else:
        raise ValueError(
            f"zeta_method must be 'blume_watson' or 'rvi', got {zeta_method!r}"
        )
    
    # Zeta scaled: each (N,) = soc_scales * val
    zeta_scaled: Dict[str, torch.Tensor] = {}
    for shell, val in zeta_src.items():
        zeta_scaled[shell] = soc_scales * val
    
    return ScaledConfigParams(
        label=cfg.label,
        nconf=cfg.nconf,
        fk=fk_scaled,
        gk=gk_scaled,
        zeta=zeta_scaled,
    )


def batch_scale_atomic_params(
    params: AtomicParams,
    slater_values: torch.Tensor,  # (N,) array of scale factors
    soc_values: torch.Tensor,     # (N,) array of scale factors
    *,
    zeta_method: str = "blume_watson",
) -> ScaledAtomicParams:
    """Batch version: scale atomic params with N different (slater, soc) pairs.
    
    Enables efficient parameter sweeps by computing N scaled parameter sets
    in one operation instead of N sequential calls to scale_atomic_params().
    
    Parameters
    ----------
    params : AtomicParams
        Parsed atomic parameters from :func:`read_rcn31_out_params`.
    slater_values : torch.Tensor, shape (N,)
        Array of slater scale factors (one per spectrum).
        If ``requires_grad=True``, per-sample gradients are preserved.
    soc_values : torch.Tensor, shape (N,)
        Array of spin-orbit scale factors (one per spectrum).
        If ``requires_grad=True``, per-sample gradients are preserved.
    zeta_method : {'blume_watson', 'rvi'}
        Which ζ column from the .rcn31_out file to use.
        
    Returns
    -------
    ScaledAtomicParams
        Scaled parameters where each Fk/Gk/ζ tensor is (N,) instead of 
        scalar. The downstream COWAN rebuild and diagonalization can then
        operate on the full batch efficiently.
        
    Examples
    --------
    >>> # Grid search over 100 parameter combinations
    >>> slater_grid = torch.linspace(0.6, 1.0, 10)
    >>> soc_grid = torch.linspace(0.8, 1.2, 10)
    >>> slater_vals, soc_vals = torch.meshgrid(slater_grid, soc_grid, indexing='ij')
    >>> scaled = batch_scale_atomic_params(
    ...     params, 
    ...     slater_values=slater_vals.flatten(),  # (100,)
    ...     soc_values=soc_vals.flatten()         # (100,)
    ... )
    >>> # Each scaled.ground.fk[key] is now shape (100,)
    
    >>> # Monte Carlo sampling with gradients
    >>> slater_samples = torch.randn(1000, requires_grad=True)
    >>> soc_samples = torch.randn(1000, requires_grad=True)
    >>> scaled = batch_scale_atomic_params(params, slater_samples, soc_samples)
    >>> # Per-sample gradients preserved for loss.backward()
    
    Notes
    -----
    Memory usage scales linearly with batch size N. For Ni d8 L-edge:
    - N=100: ~5 MB
    - N=1000: ~50 MB
    - N=5000: ~250 MB
    
    The returned ScaledAtomicParams can be passed to a batch-aware COWAN
    rebuild function (Phase 2A step 2) to amortize V(11) residual computation.
    """
    # Validate and convert to tensors
    if not isinstance(slater_values, torch.Tensor):
        slater_values = torch.as_tensor(slater_values, dtype=DTYPE)
    if not isinstance(soc_values, torch.Tensor):
        soc_values = torch.as_tensor(soc_values, dtype=DTYPE)
    
    if slater_values.dtype != DTYPE:
        slater_values = slater_values.to(dtype=DTYPE)
    if soc_values.dtype != DTYPE:
        soc_values = soc_values.to(dtype=DTYPE)
    
    # Validate shapes
    if slater_values.ndim != 1:
        raise ValueError(
            f"slater_values must be 1D, got shape {slater_values.shape}"
        )
    if soc_values.ndim != 1:
        raise ValueError(
            f"soc_values must be 1D, got shape {soc_values.shape}"
        )
    if slater_values.shape[0] != soc_values.shape[0]:
        raise ValueError(
            f"Batch size mismatch: slater_values has {slater_values.shape[0]} "
            f"elements but soc_values has {soc_values.shape[0]}"
        )
    
    # Batch scale all configs
    scaled_configs = [
        _scale_one_config_batch(c, slater_values, soc_values, zeta_method)
        for c in params.configs
    ]
    return ScaledAtomicParams(configs=scaled_configs)
